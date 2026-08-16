"""Templates for the Premise Resistance track.

Every other supersession track asks a *neutral* question and checks whether the
dead value comes back. This track asks a question that has already assumed it:

    "Since we're shipping to 123 Main St, should I use ground or express?"

when the address was changed to 456 Oak Ave three turns ago. Answering the
question as asked is the failure. The correct response rejects the premise
first, then answers from live state.

Why this needs its own track
----------------------------
A correct rejection is *forced to say the dead value out loud* — "you're not at
123 Main St anymore, the address is 456 Oak Ave" — because that is what
rejecting a premise means. Under plain phrase-list ``must_not_mention``
scoring, every correct answer here is a resurrection. Under the negation-aware
rules (``rubric.all_mentions_negated``) the same answer is clean, and only an
un-negated use — "shipping to 123 Main St via ground" — counts.

So the track measures a real capability *and* is unscoreable by the instrument
this benchmark shipped in v1.0. That is the point: it is the sharpest available
test that the v2.0 scoring correction was load-bearing rather than cosmetic.

The maintain half
-----------------
A system that rejects every premise scores perfectly on the false-premise half
and is useless. ``PREMISE_MAINTAIN_TEMPLATES`` hold the wording, entity and
event count fixed and presuppose the *live* value instead, where the correct
behavior is to accept and proceed. Failure there is spurious rejection, scored
as FSR by the ``_maintain`` suffix convention (see ``metrics.MetricsAggregator``),
never blended with the false-premise numbers.

The two halves are written as pairs: ``FALSE_*`` and its ``MAINTAIN_*`` twin
differ only in which value the query embeds.
"""

from dataclasses import dataclass, field


@dataclass
class PremiseTemplate:
    """A premise-resistance scenario.

    The timeline establishes ``stale_value``, supersedes it with
    ``live_value``, then asks a question that presupposes one of them.
    ``presupposes`` selects which, and therefore which behavior is correct.
    """

    name: str
    domain: str
    description: str

    # Fact under test.
    fact_key: str
    stale_value: str  # established first, then superseded
    live_value: str  # supersedes stale_value

    # Natural-language events. ``{stale}`` / ``{live}`` are substituted.
    initial_turn: str
    supersession_turn: str
    ack_turn: str  # assistant acknowledging the change

    # The query, which embeds one of the two values as a settled fact.
    query: str
    presupposes: str  # "stale" (false premise) or "live" (true premise)

    # Ground truth.
    correct_decision: str
    must_mention: list[str]
    # Only meaningful on false-premise cases: the stale value, forbidden when
    # asserted plainly and permitted under negation.
    forbidden: list[str] = field(default_factory=list)
    # Why the premise is false, recorded in GroundTruth.reasoning.
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.presupposes not in ("stale", "live"):
            raise ValueError(
                f"{self.name}: presupposes must be 'stale' or 'live', "
                f"got {self.presupposes!r}"
            )
        if self.presupposes == "live" and self.forbidden:
            raise ValueError(
                f"{self.name}: a true-premise case has no forbidden value — "
                "the premise it embeds is the live one"
            )


# --- Support: shipping address ---------------------------------------------

FALSE_SHIPPING_ADDRESS = PremiseTemplate(
    name="false_shipping_address",
    domain="support",
    description="Query presupposes the old shipping address after a move",
    fact_key="shipping_address",
    stale_value="123 Main St",
    live_value="456 Oak Ave",
    initial_turn="Ship my order to {stale}.",
    supersession_turn="I moved — my address is {live} now, not {stale}.",
    ack_turn="Understood, I've updated your shipping address to {live}.",
    query="Since we're shipping to 123 Main St, should I select ground shipping?",
    presupposes="stale",
    correct_decision="no",
    must_mention=["456 Oak Ave"],
    forbidden=["123 Main St"],
    rationale="The query asserts the superseded address as current. Correct "
    "behavior corrects the premise and answers from 456 Oak Ave.",
)

MAINTAIN_SHIPPING_ADDRESS = PremiseTemplate(
    name="maintain_shipping_address",
    domain="support",
    description="Same scenario, query presupposes the current address",
    fact_key="shipping_address",
    stale_value="123 Main St",
    live_value="456 Oak Ave",
    initial_turn="Ship my order to {stale}.",
    supersession_turn="I moved — my address is {live} now, not {stale}.",
    ack_turn="Understood, I've updated your shipping address to {live}.",
    query="Since we're shipping to 456 Oak Ave, should I select ground shipping?",
    presupposes="live",
    correct_decision="yes",
    must_mention=["456 Oak Ave"],
    rationale="The premise is true — 456 Oak Ave is current. Correcting it "
    "would be a spurious rejection.",
)


# --- Project: deadline ------------------------------------------------------

FALSE_DEADLINE = PremiseTemplate(
    name="false_deadline",
    domain="project",
    description="Query presupposes the original deadline after it moved",
    fact_key="launch_deadline",
    stale_value="March 14",
    live_value="April 2",
    initial_turn="The API migration is due {stale}.",
    supersession_turn="Stakeholders pushed it — the migration is now due {live}.",
    ack_turn="Noted, the migration deadline is {live}.",
    query="Given the March 14 deadline, should I ask the team to work this weekend?",
    presupposes="stale",
    correct_decision="no",
    must_mention=["April 2"],
    forbidden=["March 14"],
    rationale="The deadline moved to April 2, so the urgency the query assumes "
    "does not exist.",
)

MAINTAIN_DEADLINE = PremiseTemplate(
    name="maintain_deadline",
    domain="project",
    description="Same scenario, query presupposes the current deadline",
    fact_key="launch_deadline",
    stale_value="March 14",
    live_value="April 2",
    initial_turn="The API migration is due {stale}.",
    supersession_turn="Stakeholders pushed it — the migration is now due {live}.",
    ack_turn="Noted, the migration deadline is {live}.",
    query="Given the April 2 deadline, should I schedule the freeze for April 1?",
    presupposes="live",
    correct_decision="yes",
    must_mention=["April 2"],
    rationale="April 2 is the live deadline, so the premise holds and the plan "
    "should proceed.",
)


# --- Sales: discount --------------------------------------------------------

FALSE_DISCOUNT = PremiseTemplate(
    name="false_discount",
    domain="sales",
    description="Query presupposes a discount that was revoked",
    fact_key="globex_discount",
    stale_value="20% discount",
    live_value="standard pricing",
    initial_turn="We approved a {stale} for Globex.",
    supersession_turn="Finance revoked it — Globex is on {live} now.",
    ack_turn="Understood, Globex is on {live}.",
    query="Since Globex has the 20% discount, can I send the quote today?",
    presupposes="stale",
    correct_decision="no",
    must_mention=["standard pricing"],
    forbidden=["20% discount"],
    rationale="The discount was revoked. A quote built on it would be wrong.",
)

MAINTAIN_DISCOUNT = PremiseTemplate(
    name="maintain_discount",
    domain="sales",
    description="Same scenario, query presupposes the current pricing",
    fact_key="globex_discount",
    stale_value="20% discount",
    live_value="standard pricing",
    initial_turn="We approved a {stale} for Globex.",
    supersession_turn="Finance revoked it — Globex is on {live} now.",
    ack_turn="Understood, Globex is on {live}.",
    query="Since Globex is on standard pricing, can I send the quote today?",
    presupposes="live",
    correct_decision="yes",
    must_mention=["standard pricing"],
    rationale="Standard pricing is current, so the quote can go out as asked.",
)


# --- HR: leave --------------------------------------------------------------

FALSE_LEAVE = PremiseTemplate(
    name="false_leave",
    domain="hr",
    description="Query presupposes leave dates that were rescheduled",
    fact_key="leave_dates",
    stale_value="the week of June 3",
    live_value="the week of July 8",
    initial_turn="I'm taking leave {stale}.",
    supersession_turn="I had to move it — my leave is {live} instead.",
    ack_turn="Got it, your leave is {live}.",
    query="Since you're out the week of June 3, should I reassign your on-call shift then?",
    presupposes="stale",
    correct_decision="no",
    must_mention=["the week of July 8"],
    forbidden=["the week of June 3"],
    rationale="Leave moved to July 8, so no June 3 cover is needed.",
)

MAINTAIN_LEAVE = PremiseTemplate(
    name="maintain_leave",
    domain="hr",
    description="Same scenario, query presupposes the current leave dates",
    fact_key="leave_dates",
    stale_value="the week of June 3",
    live_value="the week of July 8",
    initial_turn="I'm taking leave {stale}.",
    supersession_turn="I had to move it — my leave is {live} instead.",
    ack_turn="Got it, your leave is {live}.",
    query="Since you're out the week of July 8, should I reassign your on-call shift then?",
    presupposes="live",
    correct_decision="yes",
    must_mention=["the week of July 8"],
    rationale="July 8 is the live leave window, so the cover should be arranged.",
)


# --- Procurement: vendor ----------------------------------------------------

FALSE_VENDOR = PremiseTemplate(
    name="false_vendor",
    domain="procurement",
    description="Query presupposes a vendor selection that was reversed",
    fact_key="hosting_vendor",
    stale_value="CloudPro",
    live_value="DataServices",
    initial_turn="We selected {stale} for hosting.",
    supersession_turn="Legal blocked it — we're going with {live} instead of {stale}.",
    ack_turn="Understood, the hosting vendor is {live}.",
    query="Since CloudPro is our hosting vendor, should I send them the signed SOW?",
    presupposes="stale",
    correct_decision="no",
    must_mention=["DataServices"],
    forbidden=["CloudPro"],
    rationale="The selection was reversed to DataServices, so no SOW goes to "
    "CloudPro.",
)

MAINTAIN_VENDOR = PremiseTemplate(
    name="maintain_vendor",
    domain="procurement",
    description="Same scenario, query presupposes the current vendor",
    fact_key="hosting_vendor",
    stale_value="CloudPro",
    live_value="DataServices",
    initial_turn="We selected {stale} for hosting.",
    supersession_turn="Legal blocked it — we're going with {live} instead of {stale}.",
    ack_turn="Understood, the hosting vendor is {live}.",
    query="Since DataServices is our hosting vendor, should I send them the signed SOW?",
    presupposes="live",
    correct_decision="yes",
    must_mention=["DataServices"],
    rationale="DataServices is the live selection, so the SOW should go out.",
)


PREMISE_RESISTANCE_TEMPLATES = [
    FALSE_SHIPPING_ADDRESS,
    FALSE_DEADLINE,
    FALSE_DISCOUNT,
    FALSE_LEAVE,
    FALSE_VENDOR,
]

PREMISE_MAINTAIN_TEMPLATES = [
    MAINTAIN_SHIPPING_ADDRESS,
    MAINTAIN_DEADLINE,
    MAINTAIN_DISCOUNT,
    MAINTAIN_LEAVE,
    MAINTAIN_VENDOR,
]

# Each false-premise template and its maintain twin, for tests and for paired
# reporting. The pairing is what makes rejection rate interpretable: a system
# that rejects everything moves both numbers.
PREMISE_PAIRS = list(zip(PREMISE_RESISTANCE_TEMPLATES, PREMISE_MAINTAIN_TEMPLATES))


def get_premise_templates(presupposes: str = "stale") -> list[PremiseTemplate]:
    """Templates for one half of the track.

    Args:
        presupposes: ``"stale"`` for the false-premise half, ``"live"`` for the
            maintain half.
    """
    if presupposes == "stale":
        return PREMISE_RESISTANCE_TEMPLATES
    if presupposes == "live":
        return PREMISE_MAINTAIN_TEMPLATES
    raise ValueError(f"presupposes must be 'stale' or 'live', got {presupposes!r}")


def get_premise_templates_by_domain(domain: str) -> list[PremiseTemplate]:
    """All premise templates, both halves, for a given domain."""
    return [
        t
        for t in PREMISE_RESISTANCE_TEMPLATES + PREMISE_MAINTAIN_TEMPLATES
        if t.domain == domain
    ]
