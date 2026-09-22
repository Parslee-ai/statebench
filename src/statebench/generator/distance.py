"""Dependency distance: how far the deciding fact sits from the query that needs it.

The agent-memory survey (arXiv:2602.06052 §7.2.2) names two dimensions as
crucial for memory-centric analysis, and StateBench only measures one of them:

    (1) **dependency distance** — how far apart the required information and
        its later use occur, such as within-turn, cross-turn, or cross-session,
        and (2) **memory correctness under interaction** — whether stored items
        remain faithful, non-contradictory, and policy-consistent as the
        environment evolves.

We are a pure (2) instrument. Every published StateBench number is measured at
one short distance, which means claims of the form "architecture X beats Y" or
"stronger models need less context curation" are claims about a single point on
a curve nobody has plotted. Independent work reports the memory-maintenance gap
*widening* with conversation length while full-context accuracy stays flat, and
reports architecture rankings crossing over as interaction length grows. If
that holds here, several of our published sentences are special cases stated as
general ones.

This module makes distance a generation parameter so the curve can be plotted.

How padding works
-----------------
``pad_timeline`` inserts irrelevant conversation between the last
state-changing event and the first query. Nothing before the padding moves;
nothing about the ground truth changes. The deciding fact and the query stay in
the same causal relationship — only the number of tokens, turns and (optionally)
session boundaries between them grows.

Three invariants make the padded timeline a valid measurement rather than a
different test:

1. **Filler carries no state.** Only ``ConversationTurn`` events are inserted —
   never a ``StateWrite`` or ``Supersession``.
2. **Filler cannot be mistaken for signal.** Every candidate line is checked
   against every ``must_mention`` and ``must_not_mention`` phrase in the
   timeline, with the same boundary-aware matcher the judge uses. A colliding
   line is skipped, not emitted. Without this the padding would silently create
   must-mention hits and must-not-mention violations out of thin air.
3. **Time-sensitive tracks keep their clock.** Session gaps advance the clock by
   days, which would change the ground truth of a track whose whole subject is
   expiry. On those tracks the day gap is dropped and the same token load is
   delivered as turns; the substitution is recorded on the returned
   ``PaddingReport`` rather than applied silently.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import timedelta

from statebench.evaluation.rubric import contains_phrase
from statebench.schema.timeline import (
    ConversationTurn,
    MentionRequirement,
    Query,
    Timeline,
)

#: Tracks whose ground truth depends on elapsed wall-clock time. Advancing the
#: clock across a session gap can flip their correct answer, so they get
#: turn-only padding.
TIME_SENSITIVE_TRACKS = frozenset({"environmental_freshness", "cf_temporal_validity"})


@dataclass(frozen=True)
class DistanceProfile:
    """One point on the dependency-distance axis.

    ``exchanges`` is the number of two-turn user/assistant filler exchanges
    inserted before the query. ``session_gaps`` is how many of those insertion
    points also carry a session boundary and a multi-day jump.
    """

    name: str
    exchanges: int
    session_gaps: int = 0
    gap_days: int = 3
    description: str = ""

    @property
    def turns(self) -> int:
        """Conversation turns added (two per exchange, plus boundary markers)."""
        return self.exchanges * 2 + self.session_gaps


#: Ordered by distance. Names follow the survey's within-turn / cross-turn /
#: cross-session vocabulary; the last two extend past it because that is where
#: the published length-scaling results say the interesting behavior lives.
DISTANCE_PROFILES: tuple[DistanceProfile, ...] = (
    DistanceProfile(
        name="within_turn",
        exchanges=0,
        description="Unpadded. The distance every published StateBench number "
        "was measured at.",
    ),
    DistanceProfile(
        name="cross_turn",
        exchanges=8,
        description="A short detour between the fact and the question.",
    ),
    DistanceProfile(
        name="cross_session",
        exchanges=28,
        session_gaps=1,
        description="The question arrives in a later session, days after the "
        "fact changed.",
    ),
    DistanceProfile(
        name="long_horizon",
        exchanges=96,
        session_gaps=4,
        gap_days=7,
        description="Weeks of intervening work. Roughly the 24x scaling at "
        "which the maintenance gap is reported to widen sharply.",
    ),
)

DISTANCE_BY_NAME = {p.name: p for p in DISTANCE_PROFILES}
DEFAULT_DISTANCE = "within_turn"


def get_profile(name: str) -> DistanceProfile:
    """Look up a profile by name."""
    try:
        return DISTANCE_BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"Unknown distance profile {name!r}. "
            f"Available: {', '.join(DISTANCE_BY_NAME)}"
        ) from None


# --- Filler ---------------------------------------------------------------- #

# Workplace chatter that establishes nothing. Deliberately free of values a
# ground truth might turn on — no amounts, dates, names, addresses or vendor
# names — so collisions are rare; the collision check is still enforced,
# because "rare" is not "never" and a silent collision corrupts scoring.
FILLER_EXCHANGES: tuple[tuple[str, str], ...] = (
    ("The office coffee machine is making that noise again.",
     "I'll note it for facilities."),
    ("Did the all-hands recording ever get posted?",
     "I'll check whether it's been uploaded."),
    ("I'm still catching up on my inbox from being out.",
     "Take your time — let me know what needs triaging."),
    ("The parking garage is closed for resurfacing this week.",
     "Good to know. I'll keep that in mind for scheduling."),
    ("Someone left a whiteboard covered in diagrams in the small room.",
     "I'll ask around before anyone erases it."),
    ("My laptop fan has been loud since the last update.",
     "That's worth raising with IT if it keeps up."),
    ("The new badge readers seem slower than the old ones.",
     "I've heard that from a few people."),
    ("Is anyone actually using the shared calendar for desk booking?",
     "Adoption has been patchy from what I can tell."),
    ("I keep forgetting which conference room has the good speakerphone.",
     "It's worth labelling them properly at some point."),
    ("The kitchen ran out of decaf again.",
     "I'll add it to the supplies list."),
    ("There's a fire drill scheduled at some point this quarter.",
     "Facilities usually gives notice the week before."),
    ("My VPN drops every time I switch networks.",
     "That's a known issue — IT has a ticket open."),
    ("Did the team ever settle on a name for the internal wiki?",
     "Not as far as I know. It's still the placeholder."),
    ("The printer on this floor is jammed.",
     "I'll flag it so nobody wastes a trip."),
    ("It's surprisingly quiet in here today.",
     "A lot of people are working remotely this week."),
    ("I should really clean out my downloads folder.",
     "A recurring reminder might help with that."),
    ("The elevator music has been the same loop for months.",
     "I've stopped noticing it, honestly."),
    ("Someone's plant in the corner is looking rough.",
     "It might just need repotting."),
    ("I never finished the onboarding survey they sent around.",
     "They usually leave those open for a while."),
    ("The window blinds on this side don't close all the way.",
     "I'll mention it to facilities along with the other items."),
    ("Is the loading dock still the best place for deliveries?",
     "As far as I know nothing has changed there."),
    ("My headset battery dies faster than it used to.",
     "Batteries do degrade — a replacement might be due."),
    ("The stairwell door sticks in humid weather.",
     "It's been like that as long as I can remember."),
    ("I keep meaning to tidy up my bookmarks.",
     "That's one of those perpetual tasks."),
)

SESSION_BOUNDARY_MARKERS: tuple[str, ...] = (
    "[session ended]",
    "[new session — returning after a break]",
    "[conversation resumed]",
)


def _ground_truth_phrases(timeline: Timeline) -> list[str]:
    """Every phrase any query in ``timeline`` scores on."""
    phrases: list[str] = []
    for event in timeline.events:
        if not isinstance(event, Query):
            continue
        gt = event.ground_truth
        for item in list(gt.must_mention) + list(gt.must_not_mention):
            phrases.append(item.phrase if isinstance(item, MentionRequirement) else item)
            if isinstance(item, MentionRequirement):
                phrases.extend(item.alternatives)
    return [p for p in phrases if p]


def _collides(text: str, phrases: list[str]) -> bool:
    """True if ``text`` would register as a ground-truth phrase at judging."""
    return any(contains_phrase(text, p) for p in phrases)


def _first_query_index(timeline: Timeline) -> int:
    for i, event in enumerate(timeline.events):
        if isinstance(event, Query):
            return i
    raise ValueError(f"timeline {timeline.id} has no query to pad before")


# --- Padding --------------------------------------------------------------- #

@dataclass
class PaddingReport:
    """What ``pad_timeline`` actually did."""

    profile: str
    exchanges_inserted: int = 0
    turns_inserted: int = 0
    session_gaps_inserted: int = 0
    days_advanced: int = 0
    #: Filler lines skipped because they would have matched a ground-truth
    #: phrase. Non-zero is fine; it means the guard did its job.
    collisions_skipped: int = 0
    #: True when a time-sensitive track had its day gaps dropped.
    clock_preserved_for_time_sensitive: bool = False
    exhausted_filler: bool = False
    notes: list[str] = field(default_factory=list)


def pad_timeline(
    timeline: Timeline,
    profile: DistanceProfile | str,
    rng: random.Random | None = None,
) -> tuple[Timeline, PaddingReport]:
    """Return a copy of ``timeline`` padded to ``profile``'s distance.

    The original is never modified. Ground truth is never modified. Events
    before the insertion point keep their timestamps; the query and anything
    after it shift later by the inserted span.
    """
    if isinstance(profile, str):
        profile = get_profile(profile)
    rng = rng or random.Random(0)

    report = PaddingReport(profile=profile.name)
    if profile.exchanges == 0 and profile.session_gaps == 0:
        return deepcopy(timeline), report

    padded = deepcopy(timeline)
    query_idx = _first_query_index(padded)
    phrases = _ground_truth_phrases(padded)

    time_sensitive = timeline.track in TIME_SENSITIVE_TRACKS
    gap_days = 0 if time_sensitive else profile.gap_days
    if time_sensitive and profile.session_gaps:
        report.clock_preserved_for_time_sensitive = True
        report.notes.append(
            f"track {timeline.track!r} is time-sensitive: session boundaries "
            "inserted without advancing the clock, so expiry ground truth holds"
        )

    # Where the padding starts: immediately after the last event before the
    # query, so the deciding fact is already established and already superseded.
    anchor = padded.events[query_idx - 1] if query_idx else None
    current_time = anchor.ts if anchor is not None else padded.events[query_idx].ts

    # Spread the session gaps evenly through the filler.
    gap_positions: set[int] = set()
    if profile.session_gaps and profile.exchanges:
        step = max(1, profile.exchanges // (profile.session_gaps + 1))
        gap_positions = {
            min(profile.exchanges - 1, step * (i + 1))
            for i in range(profile.session_gaps)
        }
    elif profile.session_gaps:
        gap_positions = {0}

    pool = list(FILLER_EXCHANGES)
    rng.shuffle(pool)
    usable = [ex for ex in pool if not (_collides(ex[0], phrases) or _collides(ex[1], phrases))]
    report.collisions_skipped = len(pool) - len(usable)
    if not usable:
        raise ValueError(
            f"every filler exchange collides with ground truth in {timeline.id}; "
            "padding would corrupt scoring"
        )
    if profile.exchanges > len(usable):
        # Cycling repeats lines. That is honest padding — repetition is what a
        # long conversation looks like — but it is recorded so a reader knows
        # the filler is not all distinct.
        report.exhausted_filler = True
        report.notes.append(
            f"{profile.exchanges} exchanges requested from {len(usable)} distinct "
            "filler lines; the pool cycles"
        )

    filler: list[ConversationTurn] = []
    for i in range(profile.exchanges):
        if i in gap_positions:
            current_time += timedelta(days=gap_days) if gap_days else timedelta(minutes=45)
            report.days_advanced += gap_days
            report.session_gaps_inserted += 1
            filler.append(ConversationTurn(
                ts=current_time,
                speaker="user",
                text=rng.choice(SESSION_BOUNDARY_MARKERS),
            ))

        user_line, assistant_line = usable[i % len(usable)]
        current_time += timedelta(minutes=rng.randint(1, 4))
        filler.append(ConversationTurn(ts=current_time, speaker="user", text=user_line))
        current_time += timedelta(seconds=rng.randint(20, 90))
        filler.append(
            ConversationTurn(ts=current_time, speaker="assistant", text=assistant_line)
        )
        report.exchanges_inserted += 1

    report.turns_inserted = len(filler)

    # Shift the query and everything after it past the filler.
    shift = current_time - (anchor.ts if anchor is not None else current_time)
    tail = padded.events[query_idx:]
    for event in tail:
        event.ts = event.ts + shift + timedelta(minutes=rng.randint(2, 10))

    padded.events = padded.events[:query_idx] + filler + tail

    padded.dependency_distance = profile.name

    return padded, report


def pad_dataset(
    timelines: list[Timeline],
    profile: DistanceProfile | str,
    seed: int = 0,
) -> tuple[list[Timeline], list[PaddingReport]]:
    """Pad a whole dataset, one shared RNG for reproducibility."""
    rng = random.Random(seed)
    padded, reports = [], []
    for timeline in timelines:
        t, r = pad_timeline(timeline, profile, rng=rng)
        padded.append(t)
        reports.append(r)
    return padded, reports
