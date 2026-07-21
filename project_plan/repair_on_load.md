# Repair-on-load — heal dangling tool_calls after an interrupted turn

## The problem

When an orchestrator turn is interrupted at an unlucky moment — machine sleep,
connection drop, request timeout — the saved conversation can end up with a
tool_call that has no matching result:

```
AIMessage:   "call campaign_run_simulation"   ← request  (SAVED to checkpoint)
ToolMessage: <missing>                          ← result   (NEVER written)
```

Most interruptions land safely and the campaign recovers on its own. But when
one lands in this narrow window, the LLM provider rejects the saved history on
the NEXT message with `INVALID_CHAT_HISTORY`, and the session stays stuck —
every future message fails, because the broken pair is persisted on disk.

## What I want to do

Add ONE small helper in `app.py`, run **before every turn**, that:

1. Reads the session's checkpointed messages (piggybacks on the existing
   `get_state` call already at ~app.py:590 — no extra state read).
2. Scans for any tool_call (from an AIMessage) with no answering ToolMessage.
3. If found, injects a synthetic ToolMessage with the **matching tool_call_id**
   (neutral "interrupted, no result" text) via `ACTIVE_GRAPH.update_state(...)`.
4. Proceeds with the turn normally.

This is LangGraph's own documented remedy for `INVALID_CHAT_HISTORY`
(add ToolMessages with matching ids → update_state → resume).

- **Runs before every turn** because a sleep/kill runs no error handler — the
  only place to catch the leftover damage is when the next message arrives. This
  also immediately unsticks a session that was already broken by a prior
  interruption.
- **Prevention vs. repair:** tool-body exceptions are already handled by
  LangGraph's `ToolNode` (`handle_tool_errors` default). This helper covers the
  cases `handle_tool_errors` CANNOT — interruptions where no exception is ever
  thrown (sleep, kill) or an upstream model error before the tool ran.
- **Low risk:** acts only when a pair is genuinely broken; does nothing on clean
  history. ~40 lines, one function, no new dependencies, no change to how tools
  or campaigns behave.

## Where it hooks in

Call it right after the existing `prior_state = get_state(...)` block
(~app.py:590), before `graph_input` is streamed (~app.py:640). `prior_state`
is already fetched there, so it's passed straight in.

## The function

```python
def _heal_dangling_tool_calls(prior_state, session_id: str) -> int:
    """Repair-on-load: if a previous turn was interrupted (machine sleep,
    process kill, or an upstream model error) AFTER the model emitted a
    tool_call but BEFORE the tool's ToolMessage was written, the saved history
    has an AIMessage whose tool_call has no matching ToolMessage. Most LLM
    providers reject that as INVALID_CHAT_HISTORY on the NEXT turn.

    This scans the checkpointed messages for such orphans and injects a
    synthetic ToolMessage (matching tool_call_id) so the pair is valid again.
    Returns the number of orphans healed. Runs before every turn so it also
    unsticks a session already broken by a prior interruption.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    if not prior_state or not getattr(prior_state, "values", None):
        return 0
    messages = prior_state.values.get("messages") or []
    if not messages:
        return 0

    # tool_call_ids that already have a ToolMessage answer
    answered = {
        m.tool_call_id for m in messages
        if isinstance(m, ToolMessage) and getattr(m, "tool_call_id", None)
    }

    # tool_calls (from AIMessages) with no answering ToolMessage
    orphans = []  # (tool_call_id, tool_name)
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                tc_id = tc.get("id")
                if tc_id and tc_id not in answered:
                    orphans.append((tc_id, tc.get("name", "tool")))

    if not orphans:
        return 0

    heal_msgs = [
        ToolMessage(
            content=(
                f"[The previous turn was interrupted (e.g. machine sleep or "
                f"connection loss) before '{name}' returned a result. No result "
                f"was recorded. If this step still needs doing, call it again.]"
            ),
            tool_call_id=tc_id,
        )
        for tc_id, name in orphans
    ]
    ACTIVE_GRAPH.update_state(
        {"configurable": {"thread_id": session_id}},
        {"messages": heal_msgs},
    )
    print(f"[HEAL] repaired {len(orphans)} dangling tool_call(s) for "
          f"session {session_id}: {[n for _, n in orphans]}")
    return len(orphans)
```

## How to call it (one line, in the existing get_state block)

```python
prior_state = await asyncio.to_thread(
    ACTIVE_GRAPH.get_state,
    {"configurable": {"thread_id": session_id}},
)
# NEW: heal any dangling tool_call left by a prior interrupted turn
await asyncio.to_thread(_heal_dangling_tool_calls, prior_state, session_id)
```

## Verification
1. Reproduce: start a campaign, interrupt mid-tool (sleep/kill), then send any
   message → previously `INVALID_CHAT_HISTORY`; now it heals and continues.
2. Log line `[HEAL] repaired N dangling tool_call(s)...` appears exactly when a
   broken pair existed, and never on clean sessions.
3. Clean sessions behave identically (helper returns 0, does nothing).

## Not doing
- Not detecting whether an interrupted LAMMPS run actually finished on disk
  (the "smart" salvage option) — the simple neutral message was chosen.
- Not touching tool bodies (LangGraph's handle_tool_errors already covers
  tool-thrown exceptions).
