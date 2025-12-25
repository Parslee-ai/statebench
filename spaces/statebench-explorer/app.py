"""StateBench Explorer - Interactive inspection of the StateBench benchmark.

This Gradio app allows you to:
1. Browse timelines from parslee/statebench
2. View conversation events and state changes
3. See context built by different memory baselines
4. Inspect ground truth (must mention, must not mention, decision)
"""

import json
import gradio as gr
from datasets import load_dataset

# Load dataset once at startup
print("Loading StateBench dataset...")
ds = load_dataset("parslee/statebench")
print(f"Loaded: train={len(ds['train'])}, validation={len(ds['validation'])}, test={len(ds['test'])}")

# Available baselines
BASELINES = [
    "transcript_replay",
    "no_memory",
    "rolling_summary",
    "fact_extraction",
    "state_based",
]

# Track descriptions
TRACK_INFO = {
    "supersession": "Facts invalidated by newer information",
    "commitment_durability": "Commitments survive interruptions",
    "interruption_resumption": "Context survives topic switches",
    "scope_permission": "Role-based access control",
    "environmental_freshness": "Time-sensitive state expiration",
    "hallucination_resistance": "Only assert established state",
    "scope_leak": "Task-local state stays local",
    "causality": "Multi-constraint dependencies",
    "repair_propagation": "Fixes propagate to dependent facts",
    "brutal_realistic": "Real-world complexity scenarios",
    "supersession_detection": "Infer supersession from natural language",
    "authority_hierarchy": "Respect authority levels",
    "enterprise_privacy": "Cross-tenant isolation",
}


def get_timeline_list(split: str, track_filter: str) -> list[str]:
    """Get list of timeline IDs for dropdown."""
    data = ds[split]

    if track_filter and track_filter != "All":
        # Filter by track
        ids = [row["id"] for row in data if row["track"] == track_filter]
    else:
        ids = [row["id"] for row in data]

    return ids[:100]  # Limit for performance


def parse_timeline(row: dict) -> dict:
    """Parse a timeline row from HF dataset."""
    events = json.loads(row["events"])
    initial_state = json.loads(row["initial_state"])
    actors = json.loads(row["actors"])

    return {
        "id": row["id"],
        "track": row["track"],
        "domain": row["domain"],
        "difficulty": row["difficulty"],
        "detection_mode": row["detection_mode"],
        "events": events,
        "initial_state": initial_state,
        "actors": actors,
        "user_name": row["user_name"],
        "user_authority": row["user_authority"],
    }


def format_events(events: list[dict]) -> str:
    """Format events as readable markdown."""
    lines = []

    for i, event in enumerate(events):
        event_type = event.get("type", "unknown")

        if event_type == "conversation_turn":
            speaker = event.get("speaker", "?")
            text = event.get("text", "")
            emoji = "👤" if speaker == "user" else "🤖"
            lines.append(f"**{emoji} {speaker.title()}**: {text}\n")

        elif event_type == "state_write":
            writes = event.get("writes", [])
            lines.append(f"**📝 State Write**:")
            for w in writes:
                lines.append(f"  - `{w.get('key')}`: {w.get('value')}")
            lines.append("")

        elif event_type == "supersession":
            writes = event.get("writes", [])
            lines.append(f"**🔄 Supersession**:")
            for w in writes:
                supersedes = w.get("supersedes", "")
                lines.append(f"  - `{w.get('key')}`: {w.get('value')}")
                if supersedes:
                    lines.append(f"    *(supersedes: {supersedes})*")
            lines.append("")

        elif event_type == "query":
            prompt = event.get("prompt", "")
            lines.append(f"**❓ Query**: {prompt}\n")

            gt = event.get("ground_truth", {})
            if gt:
                lines.append(f"  - **Expected Decision**: `{gt.get('decision', 'N/A')}`")
                lines.append(f"  - **Decision Type**: {gt.get('decision_type', 'N/A')}")

                must_mention = gt.get("must_mention", [])
                if must_mention:
                    mentions = [m if isinstance(m, str) else m.get("phrase", str(m)) for m in must_mention]
                    lines.append(f"  - **Must Mention**: {mentions}")

                must_not = gt.get("must_not_mention", [])
                if must_not:
                    forbidden = [m if isinstance(m, str) else m.get("phrase", str(m)) for m in must_not]
                    lines.append(f"  - **Must NOT Mention** ⚠️: {forbidden}")
            lines.append("")

    return "\n".join(lines)


def format_initial_state(state: dict) -> str:
    """Format initial state as markdown."""
    lines = ["## Initial State\n"]

    # Identity
    identity = state.get("identity_role", {})
    lines.append(f"**User**: {identity.get('user_name', 'N/A')}")
    lines.append(f"**Authority**: {identity.get('authority', 'N/A')}")
    if identity.get("department"):
        lines.append(f"**Department**: {identity.get('department')}")
    if identity.get("organization"):
        lines.append(f"**Organization**: {identity.get('organization')}")
    lines.append("")

    # Persistent facts
    facts = state.get("persistent_facts", [])
    if facts:
        lines.append("### Persistent Facts")
        for f in facts:
            lines.append(f"- `{f.get('key')}`: {f.get('value')}")
        lines.append("")

    # Working set
    working = state.get("working_set", [])
    if working:
        lines.append("### Working Set")
        for w in working:
            lines.append(f"- `{w.get('key')}`: {w.get('value')}")
        lines.append("")

    return "\n".join(lines)


def build_context_with_baseline(events: list[dict], baseline_name: str) -> str:
    """Build context using specified baseline."""
    try:
        from statebench.baselines import get_baseline
        from statebench.schema.timeline import ConversationTurn, StateWrite, Supersession, Query

        baseline = get_baseline(baseline_name, token_budget=8000)
        baseline.reset()

        # Find the last query
        last_query = None
        for event in events:
            if event.get("type") == "query":
                last_query = event

        if not last_query:
            return "No query found in timeline"

        # Process events up to query
        for event in events:
            event_type = event.get("type")

            if event_type == "query":
                # Build context at query point
                context_result = baseline.build_context(event.get("prompt", ""))
                return context_result.context

            elif event_type == "conversation_turn":
                parsed = ConversationTurn.model_validate(event)
                baseline.process_event(parsed)

            elif event_type == "state_write":
                parsed = StateWrite.model_validate(event)
                baseline.process_event(parsed)

            elif event_type == "supersession":
                parsed = Supersession.model_validate(event)
                baseline.process_event(parsed)

        return "Could not build context"

    except ImportError:
        # Fallback: show raw conversation when statebench not available
        lines = ["*Context building requires statebench package (not available in this Space)*\n"]
        lines.append("**Raw conversation:**\n")
        for event in events:
            if event.get("type") == "conversation_turn":
                speaker = event.get("speaker", "?")
                text = event.get("text", "")
                lines.append(f"- **{speaker}**: {text}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error building context: {str(e)}"


def inspect_timeline(split: str, timeline_id: str, baseline: str):
    """Main inspection function."""
    if not timeline_id:
        return "Select a timeline", "", "", ""

    # Find the timeline
    data = ds[split]
    row = None
    for r in data:
        if r["id"] == timeline_id:
            row = dict(r)
            break

    if not row:
        return f"Timeline {timeline_id} not found", "", "", ""

    # Parse timeline
    timeline = parse_timeline(row)

    # Format metadata
    track_desc = TRACK_INFO.get(timeline["track"], "")
    metadata = f"""## {timeline['id']}

**Track**: {timeline['track']} - *{track_desc}*
**Domain**: {timeline['domain']}
**Difficulty**: {timeline['difficulty']}
**Detection Mode**: {timeline['detection_mode']}
**User**: {timeline['user_name']} ({timeline['user_authority']})
"""

    # Format events
    events_md = format_events(timeline["events"])

    # Format initial state
    state_md = format_initial_state(timeline["initial_state"])

    # Build context
    context = build_context_with_baseline(timeline["events"], baseline)
    context_md = f"## Context ({baseline})\n\n```\n{context}\n```"

    return metadata, events_md, state_md, context_md


def update_timeline_list(split: str, track: str):
    """Update timeline dropdown based on filters."""
    ids = get_timeline_list(split, track)
    return gr.Dropdown(choices=ids, value=ids[0] if ids else None)


# Build the Gradio interface
with gr.Blocks(title="StateBench Explorer") as demo:
    gr.Markdown("""
    # 🔍 StateBench Explorer

    Interactive inspection of the [StateBench](https://huggingface.co/datasets/parslee/statebench) benchmark
    for LLM state correctness.

    **Select a timeline** to view its events, ground truth, and context built by different memory baselines.
    """)

    with gr.Row():
        split_dropdown = gr.Dropdown(
            choices=["test", "validation", "train"],
            value="test",
            label="Split"
        )
        track_dropdown = gr.Dropdown(
            choices=["All"] + list(TRACK_INFO.keys()),
            value="All",
            label="Track Filter"
        )
        baseline_dropdown = gr.Dropdown(
            choices=BASELINES,
            value="transcript_replay",
            label="Baseline for Context"
        )

    timeline_dropdown = gr.Dropdown(
        choices=get_timeline_list("test", "All"),
        label="Timeline ID",
        value=get_timeline_list("test", "All")[0] if get_timeline_list("test", "All") else None
    )

    inspect_btn = gr.Button("🔍 Inspect Timeline", variant="primary")

    with gr.Row():
        with gr.Column(scale=1):
            metadata_output = gr.Markdown(label="Metadata")
            state_output = gr.Markdown(label="Initial State")

        with gr.Column(scale=2):
            events_output = gr.Markdown(label="Events")

    context_output = gr.Markdown(label="Built Context")

    # Event handlers
    split_dropdown.change(
        fn=update_timeline_list,
        inputs=[split_dropdown, track_dropdown],
        outputs=[timeline_dropdown]
    )

    track_dropdown.change(
        fn=update_timeline_list,
        inputs=[split_dropdown, track_dropdown],
        outputs=[timeline_dropdown]
    )

    inspect_btn.click(
        fn=inspect_timeline,
        inputs=[split_dropdown, timeline_dropdown, baseline_dropdown],
        outputs=[metadata_output, events_output, state_output, context_output]
    )

    # Auto-inspect on timeline change
    timeline_dropdown.change(
        fn=inspect_timeline,
        inputs=[split_dropdown, timeline_dropdown, baseline_dropdown],
        outputs=[metadata_output, events_output, state_output, context_output]
    )

    gr.Markdown("""
    ---
    **Resources**: [Dataset](https://huggingface.co/datasets/parslee/statebench) |
    [GitHub](https://github.com/Parslee-ai/statebench) |
    [Paper](https://github.com/Parslee-ai/statebench/blob/main/paper.pdf)
    """)


if __name__ == "__main__":
    demo.launch()
