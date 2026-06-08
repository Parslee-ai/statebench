"""Structured-conflict templates: exercise the authority-resolution and Type II
dependency-propagation engine fixes with REAL structured edges (Source.authority on
same-key writes; depends_on chains with supersession) rather than NL — which the
existing authority_hierarchy / repair_propagation tracks use.
"""

from dataclasses import dataclass

# Design contract: same key == same state slot, so two different VALUES under one
# key are a genuine conflict (not complementary info — that belongs under distinct
# keys). These tracks are positive-path: they assert the conflict resolves. The
# no-over-reach direction (cross-key facts coexist; an un-superseded base doesn't
# flag dependents) is covered by engine unit tests, not these tracks.


@dataclass(frozen=True)
class AuthorityConflict:
    """Two same-key facts from different authority levels (no explicit supersedes).
    The higher-authority value must win."""
    domain: str
    key: str
    low_value: str
    low_authority: str   # AuthorityLevel
    high_value: str
    high_authority: str  # AuthorityLevel (must outrank low)
    query: str
    decision: str
    must_mention: list[str]      # tokens of the high-authority (winning) value
    must_not_mention: list[str]  # tokens of the low-authority (losing) value


AUTHORITY_CONFLICTS = [
    AuthorityConflict(
        domain="procurement",
        key="server_purchase_decision",
        low_value="approved the $50k server purchase",
        low_authority="manager",
        high_value="all purchases over $25k are frozen this quarter",
        high_authority="executive",
        query="Should we proceed with the $50k server purchase?",
        decision="no",
        must_mention=["frozen", "over $25k"],
        must_not_mention=["approved", "proceed"],
    ),
    AuthorityConflict(
        domain="sales",
        key="globex_discount_decision",
        low_value="offer Globex a 20% discount",
        low_authority="manager",
        high_value="all discounts are capped at 10% per finance policy",
        high_authority="policy",
        query="Can we offer Globex the 20% discount?",
        decision="no",
        must_mention=["10%", "cap"],
        must_not_mention=["20%"],
    ),
    AuthorityConflict(
        domain="procurement",
        key="vendor_selection_decision",
        low_value="go with VendorX for the contract",
        low_authority="peer",
        high_value="use VendorY per the approved procurement standard",
        high_authority="policy",
        query="Which vendor are we using for the contract?",
        decision="VendorY",
        must_mention=["VendorY"],
        must_not_mention=["VendorX"],
    ),
    AuthorityConflict(
        domain="project",
        key="launch_date_decision",
        low_value="set the launch for next Friday",
        low_authority="peer",
        high_value="the launch is moved to next month by the VP",
        high_authority="executive",
        query="When is the launch?",
        decision="next month",
        must_mention=["next month"],
        must_not_mention=["Friday"],
    ),
]


@dataclass(frozen=True)
class AuthorityMaintain:
    """should-NOT-override (FSR-analog for authority): a lower-authority SPECIFIC
    decision that COMPLIES with a higher-authority general rule (a DIFFERENT slot)
    must remain valid — an over-aggressive authority resolver that retires it is
    wrong. Scored via the _maintain behavioral metric (False Authority Override Rate)."""
    domain: str
    rule_key: str
    rule_value: str
    rule_authority: str       # higher (e.g. policy)
    decision_key: str         # DIFFERENT key from rule_key (different slot)
    decision_value: str
    decision_authority: str   # lower (e.g. manager); complies with the rule
    query: str
    decision: str             # affirms the specific decision still stands
    must_mention: list[str]


AUTHORITY_MAINTAIN = [
    AuthorityMaintain(
        domain="procurement",
        rule_key="travel_policy",
        rule_value="all travel must be booked at least 14 days in advance",
        rule_authority="policy",
        decision_key="sarah_trip_approval",
        decision_value="approved Sarah's conference trip, booked 3 weeks out",
        decision_authority="manager",
        query="Is Sarah's conference trip approved?",
        decision="yes",
        must_mention=["approved|yes"],
    ),
    AuthorityMaintain(
        domain="procurement",
        rule_key="purchase_policy",
        rule_value="purchases over $25k require VP sign-off",
        rule_authority="policy",
        decision_key="monitor_order_approval",
        decision_value="approved the $8k monitor order",
        decision_authority="manager",
        query="Can I place the $8k monitor order?",
        decision="yes",
        must_mention=["approved|yes|$8k"],
    ),
    AuthorityMaintain(
        domain="sales",
        rule_key="discount_policy",
        rule_value="discounts are capped at 10%",
        rule_authority="policy",
        decision_key="acme_discount_approval",
        decision_value="approved an 8% discount for Acme",
        decision_authority="manager",
        query="Is the 8% discount for Acme approved?",
        decision="yes",
        must_mention=["8%|approved|yes"],
    ),
    AuthorityMaintain(
        domain="project",
        rule_key="release_policy",
        rule_value="releases ship only after a full QA pass",
        rule_authority="policy",
        decision_key="hotfix_approval",
        decision_value="approved the hotfix release, which passed full QA",
        decision_authority="manager",
        query="Is the hotfix release approved to ship?",
        decision="yes",
        must_mention=["approved|yes"],
    ),
]


@dataclass(frozen=True)
class DependencyChain:
    """A derived chain base <- mid <- top. The base is superseded; the derived
    values (mid, top) are stale and must be recalculated, not asserted as current."""
    domain: str
    base_key: str
    base_value: str
    mid_key: str
    mid_value: str
    top_key: str
    top_value: str
    new_base_value: str
    query: str
    decision: str
    must_mention: list[str]      # the new base / recalc signal
    must_not_mention: list[str]  # the stale derived (top) value


DEPENDENCY_CHAINS = [
    DependencyChain(
        domain="sales",
        base_key="unit_price",
        base_value="unit price is $10",
        mid_key="subtotal",
        mid_value="subtotal is $100 for 10 units",
        top_key="order_total",
        top_value="order total is $110 including tax",
        new_base_value="unit price is now $12",
        query="What is the current order total?",
        decision="recalculate from $12 unit price",
        must_mention=["$12", "recalculat"],
        must_not_mention=["$110"],
    ),
    DependencyChain(
        domain="hr",
        base_key="base_salary",
        base_value="base salary is $100k",
        mid_key="annual_bonus",
        mid_value="annual bonus is $20k (20% of base)",
        top_key="total_comp",
        top_value="total compensation is $120k",
        new_base_value="base salary is now $130k",
        query="What is the current total compensation?",
        decision="recalculate from $130k base",
        must_mention=["$130k", "recalculat"],
        must_not_mention=["$120k"],
    ),
    DependencyChain(
        domain="project",
        base_key="phase1_duration",
        base_value="phase 1 takes 4 weeks",
        mid_key="phase2_start",
        mid_value="phase 2 starts in week 5",
        top_key="project_end",
        top_value="project ends in week 12",
        new_base_value="phase 1 now takes 6 weeks",
        query="When does the project end now?",
        decision="recalculate from the 6-week phase 1",
        must_mention=["6 week", "recalculat"],
        must_not_mention=["week 12"],
    ),
]
