"""Filler fact generation for compaction stress testing.

Generates realistic but irrelevant StateWrite events to create token pressure
that triggers compaction in Memgine. Facts are deterministic via seed.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import datetime, timedelta

from statebench.schema.state import Source
from statebench.schema.timeline import StateWrite, Write

# Pool of realistic filler fact templates.
# Each tuple: (key_template, value_template_fn)
# Templates produce facts about departments, metrics, and dates
# that are plausible but irrelevant to any benchmark query.

_DEPARTMENTS = [
    "Engineering", "Marketing", "Sales", "Finance", "HR",
    "Legal", "Operations", "Customer Success", "Product", "Design",
    "Data Science", "DevOps", "QA", "Security", "Support",
]

_ValueFn = Callable[[random.Random], str]

_METRICS: list[tuple[str, _ValueFn]] = [
    ("team_size", lambda r: f"Team headcount is {r.randint(5, 200)}"),
    ("quarterly_revenue", lambda r: f"Q{r.randint(1,4)} revenue was ${r.randint(100, 9999)}K"),
    ("sprint_velocity", lambda r: f"Average sprint velocity is {r.randint(15, 80)} points"),
    ("customer_satisfaction", lambda r: f"CSAT score is {r.randint(60, 99)}%"),
    ("incident_count", lambda r: f"Monthly incidents: {r.randint(0, 25)}"),
    ("deployment_frequency", lambda r: f"Deployments per week: {r.randint(1, 30)}"),
    ("meeting_hours", lambda r: f"Average meeting hours per week: {r.randint(4, 25)}"),
    ("open_tickets", lambda r: f"Open support tickets: {r.randint(10, 500)}"),
    ("employee_nps", lambda r: f"Employee NPS: {r.randint(-10, 80)}"),
    ("budget_utilization", lambda r: f"Budget utilization is {r.randint(40, 100)}%"),
    ("project_count", lambda r: f"Active projects: {r.randint(3, 30)}"),
    ("training_hours", lambda r: f"Training hours this quarter: {r.randint(10, 200)}"),
    ("code_coverage", lambda r: f"Code coverage is {r.randint(50, 98)}%"),
    ("response_time_p99", lambda r: f"P99 response time is {r.randint(50, 2000)}ms"),
    ("uptime_sla", lambda r: f"Uptime SLA: {r.uniform(99.0, 99.99):.2f}%"),
    ("backlog_items", lambda r: f"Backlog has {r.randint(20, 500)} items"),
    ("onboarding_time", lambda r: f"Average onboarding time: {r.randint(2, 12)} weeks"),
    ("attrition_rate", lambda r: f"Annual attrition rate: {r.randint(5, 30)}%"),
    ("cost_per_lead", lambda r: f"Cost per lead: ${r.randint(10, 500)}"),
    ("conversion_rate", lambda r: f"Conversion rate: {r.uniform(0.5, 15.0):.1f}%"),
]


def generate_filler_facts(n: int, seed: int = 42) -> list[StateWrite]:
    """Generate n deterministic filler StateWrite events.

    Each event contains a single write to persistent_facts with a realistic
    but irrelevant metric fact. Facts are keyed by department + metric to
    avoid collisions with real benchmark data.

    Args:
        n: Number of filler facts to generate
        seed: Random seed for reproducibility

    Returns:
        List of StateWrite events ready for timeline injection
    """
    rng = random.Random(seed)
    base_ts = datetime(2025, 1, 1, 9, 0, 0)
    events: list[StateWrite] = []

    for i in range(n):
        dept = rng.choice(_DEPARTMENTS)
        metric_key, value_fn = rng.choice(_METRICS)
        key = f"filler_{dept.lower()}_{metric_key}"
        value = f"[{dept}] {value_fn(rng)}"
        ts = base_ts + timedelta(minutes=i * 5)

        write = Write(
            id=f"FILLER-{i:04d}",
            layer="persistent_facts",
            key=key,
            value=value,
            source=Source(type="system", identity="metrics_system", authority="system"),
            scope="global",
        )

        events.append(StateWrite(ts=ts, writes=[write]))

    return events
