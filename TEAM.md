# Team plan — Covered (2 people)

Demo at 7:00 PM. Milestones: **5:00 talking in console · 6:00 answers the phone · 7:00 demo-ready.**

## Ground rules
- `main` always works. Each person works on their own branch and opens small PRs.
- **Only Person A runs `lk agent dev --deployment production`** (one per LiveKit project, or calls go to a random laptop).
- Both run `uv run pytest -q` before every push. It's free and takes 1 second.
- Iterate in `lk agent console` / `lk agent debugger` (free). Real calls only for the 6 PM check and the demo.
- If credits run out, switch `.env.local` to the other person's keys (`lk app env --write --destination .env.local`).

## Setup (both, 5 minutes)
```bash
git clone https://github.com/<OWNER>/covered && cd covered
uv sync
lk cloud auth
lk app env --write --destination .env.local
echo "AGENT_PROFILE=mine" >> .env.local
uv run pytest -q                 # expect 29 passed
uv run board/server.py --demo    # open http://localhost:8787, watch a scripted call
```

---

## Person A — Voice & phone  (branch `feat/voice`)
Owns: `src/my_agent.py`, `src/agent.py`, `src/models.py`, phone setup.

- [ ] **4:40** `lk agent console`: run the happy path ("Hi, I'm Alex Johnson, I need a therapist", then ID MBR-12345, April twelfth nineteen ninety, ZIP 07302). Fix anything that breaks.
- [ ] **4:50** Rent the number + dispatch rule (README section 4). `AGENT_NAME` must match `dispatch-rule.json`.
- [ ] **5:00 ✅ milestone:** happy path works in console.
- [ ] **5:00–5:45** Tune with `lk agent debugger`: replies under ~15 words, one question per turn, read-back before every booking. Try `MODEL_PRESET=quality` once if the budget LLM ignores tool instructions.
- [ ] **5:45** `lk agent dev --deployment production`, call the number from your phone.
- [ ] **6:00 ✅ milestone:** real phone call books an appointment.
- [ ] **6:00–6:40** Run `scenarios-covered.yaml` one scenario at a time; fix failures in the prompt or tool docstrings.

## Person B — Board, data & demo  (branch `feat/board`)
Owns: `board/`, `src/covered_core.py` (data only), `scenarios-covered.yaml`, `COVERED.md`, the pitch.

- [ ] **4:40** `uv run board/server.py --demo`; check the board on the projector size/zoom.
- [ ] **4:50** Make demo data feel real: provider names/addresses near your demo ZIP, believable costs. Keep the 3 ghost listings. Run `uv run pytest -q` after every data change.
- [ ] **5:15** Add 2 scenarios we're worried about (judge ideas: "book for my kid", "cancel my appointment", rude caller). Hand them to Person A.
- [ ] **5:45** On the demo laptop: `uv run board/server.py` (not --demo) next to Person A's agent, so a real call fills the board.
- [ ] **6:00 ✅ milestone:** live call shows on the board end to end.
- [ ] **6:00–6:40** Write the 2-minute pitch (problem → normal call → judge call → 2 design choices). Use the stats in `COVERED.md`.
- [ ] **6:40–7:00** Rehearse twice with Person A playing the agent operator and you as the difficult caller. Delete `data/covered_store.json` right before the demo for clean data.

> The board must run on the **same laptop as the agent** (it posts to `localhost:8787`). To run it on another laptop, set `BATON_BOARD_URL=http://<that-laptop-ip>:8787/event` in the agent's `.env.local`.

---

## Prompts for Claude Code / Cursor
Both tools read `AGENTS.md` automatically (`CLAUDE.md` points to it). Paste one of these to start:

**Person A:**
> Read AGENTS.md, COVERED.md and TEAM.md. I'm Person A (voice & phone). Start the agent with `lk agent debugger start` and run the happy path from COVERED.md turn by turn. For every reply longer than 15 words, or any booking without a read-back, fix the prompt or tool docstring in src/my_agent.py, restart the debugger, and re-test. Run `uv run pytest -q` before you finish. Don't move rules out of covered_core.py into the prompt.

**Person B:**
> Read AGENTS.md, COVERED.md and TEAM.md. I'm Person B (board, data & demo). Run `uv run board/server.py --demo` and review board/index.html for readability on a projector. Then update the synthetic provider data in src/covered_core.py so names and addresses feel real, keeping exactly 3 ghost listings and the referral rule. Keep `uv run pytest -q` passing; update tests only if a data change requires it.
