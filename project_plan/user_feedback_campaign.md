# Mid-campaign user feedback — steer the next hypothesis (design only)

## Goal

While a campaign is running, let the user send feedback (e.g. "the crack keeps
growing — try higher temperatures" or "focus on smaller crack widths") and have
the orchestrator **incorporate it into the next iteration's hypothesis and knob
choices**. Advisory, not a hard override — the orchestrator interprets the
feedback and adjusts.

## Is it possible? Yes — the architecture already supports it

Two existing facts make this a small, natural addition rather than a rewrite:

1. **The hypothesis is re-formed EVERY iteration.** In the campaign loop, Step 3
   ([campaign.md:198-200](../src/orchestrator/skills/campaign.md#L198)) has the
   orchestrator write a fresh hypothesis + pick knobs each iteration, reading
   from `campaign_get_state()`. So there's already a per-iteration "decide what
   to try next" moment — feedback just becomes another input to it.

2. **The pause/resume/cancel mechanism already does exactly this pattern** —
   a file (`control.json`) is written by an endpoint and read by the loop
   *between iterations* ([app.py:778-794](../src/app.py#L778),
   `campaign_check_stop` criteria #1/#2). Feedback reuses this proven pattern.

So: **the loop already pauses to read a file between iterations and already
re-decides its hypothesis each iteration.** We slot feedback into both.

## Why we can't do it "instantly" (and why that's fine)

The campaign runs as ONE ReAct loop inside a single orchestrator turn, so a chat
message can't reach the orchestrator mid-loop. Feedback therefore lands in a
file and is picked up at the **next iteration boundary** (the current LAMMPS run
finishes first, ~1–5 min). This is the same "applies between iterations" model
as pause/cancel — acceptable and expected.

## The flow

```
User types feedback during a running campaign
        ↓
POST /api/campaign/{id}/feedback     (NEW endpoint, mirrors /pause)
        ↓
writes into control.json:  {"pending_feedback": "<text>"}
        ↓
next iteration, Step 2: campaign_get_state() returns pending_feedback
        ↓  and CLEARS it (so it applies once)
Step 3: orchestrator reads pending_feedback → adjusts hypothesis + knobs
        ↓
iteration proceeds with the feedback-informed choice
```

## The changes needed (3 backend + 1 frontend)

### 1. New endpoint — `app.py`
`POST /api/campaign/{campaign_id}/feedback` with a `{text: str}` body. Writes
`pending_feedback` into that campaign's `control.json`. Mirrors the existing
`_write_campaign_status` helper ([app.py:764](../src/app.py#L764)) — same path
guard, atomic write. ~10 lines.

### 2. `campaign_get_state` — `campaign_tools.py`
- Include `pending_feedback` (from control.json) in the returned dict
  (alongside the existing `control`, `best_so_far`, etc. at ~[line 348](../src/orchestrator/campaign_tools.py#L348)).
- **Clear it after reading** (write control.json back with `pending_feedback`
  removed/empty) so a single feedback message applies to exactly one iteration,
  not every subsequent one. ~8 lines.

### 3. Skill Step 3 rule — `campaign.md`
Add to Step 3: *"If `pending_feedback` is present in state, treat it as
higher-priority user guidance: explain (in your reasoning) how you're
incorporating it, and adjust your hypothesis + knob values accordingly this
iteration."* ~3 lines.

### 4. Frontend (after backend proven) — `index.html`
A small text box in the campaign banner (visible while status=running) that
POSTs to the feedback endpoint. Optional first pass: test the backend by
writing `pending_feedback` into control.json manually / via curl.

## Design decisions locked in
- **Advisory**, not hard override — orchestrator interprets and adjusts; it does
  not force exact knob values. (A stricter "force T=500" override could be a
  later variant.)
- **Applies once** — feedback is cleared after the next iteration consumes it,
  so it's a nudge, not a permanent setting. (Could add a "sticky" option later.)
- **File-based via control.json** — reuses the pause/resume channel; no need to
  interrupt the running loop.

## Edge cases to handle
- Feedback sent between the last iteration and stop → simply never consumed
  (campaign ended); harmless.
- Feedback sent while paused → sits in control.json, consumed on resume's first
  iteration. Fine.
- Multiple feedback messages before the next iteration → last one wins (or
  concatenate; decide at build time). Simplest: last wins.
- Empty/whitespace feedback → ignore (don't set pending_feedback).

## Verification (when built)
1. Start a campaign; mid-run POST feedback ("try higher temperature").
2. Confirm next iteration's Step-3 reasoning (streamed to chat) references the
   feedback and the temperature knob moves accordingly.
3. Confirm the feedback is cleared — the iteration AFTER does NOT re-apply it.
4. Confirm pause/resume/cancel still work (shared control.json not broken).

## Not doing (this design)
- Not a hard knob override (advisory only).
- Not interrupting the current in-flight iteration (applies next boundary).
- Not persisting feedback into trials.jsonl (could later, so hypotheses show
  which trials were user-steered).
