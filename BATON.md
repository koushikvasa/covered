# Baton Health — run sheet

**The job:** a home care aide calls at the end of a visit. Baton verifies who they are,
takes the visit report (meds, food, mobility, mood, skin, anything new), confirms it,
files it, and triages it: routine, nurse review, urgent, or emergency.

## Run it (two terminals)

```bash
# .env.local
AGENT_PROFILE=mine
MODEL_PRESET=budget
AGENT_NAME=take2-agent

uv run board/server.py                     # terminal 1 → open http://localhost:8787
lk agent console                           # terminal 2, laptop mic (no phone minutes)
lk agent dev --deployment production       # terminal 2, for real phone calls
```

Board without a call: `uv run board/server.py --demo`.
Fresh demo data: delete `data/baton_store.json` (it reseeds with 2 days of history).
Your own phone recognised by caller ID: `BATON_DEMO_PHONES=+1646XXXXXXX=maria` in `.env.local`.

## Demo data

| Caregiver | PIN | Clients |
|---|---|---|
| Maria Santos | 2468 | Rosa Delgado, James Okafor |
| Devon Price | 1357 | James Okafor |

- **Rosa:** Lisinopril (blood pressure, morning), Metformin (diabetes), Donepezil (memory, bedtime).
  She refused Lisinopril on the last two visits, so a third refusal fires a pattern alert live.
- **James:** Warfarin (blood thinner, evening), Furosemide (water pill, morning). A near fall
  3 days ago, so another near fall or fall fires the fall-risk pattern.

## Test

```bash
uv run pytest -q                                           # 17 tests, no credits
lk agent debugger start && lk agent debugger say "Hi it's Maria, PIN two four six eight"
lk agent simulate text --scenarios scenarios-baton.yaml    # before milestones only
```

## What's enforced in code (not the prompt)

| Rule | Where |
|---|---|
| No client data before PIN or caller ID; only assigned clients | `verify_pin`, `select_client` |
| Only the client's own medications; misheard names need "Did you mean…?" | `baton_core.match_med` |
| Refused, missed or unsure meds, falls and corrections need `confirmed=true` | `record_medication`, `record_observation` |
| Can't file until every item is answered and the read-back is approved | `file_note` |
| Emergencies and abuse disclosures are caught on every caller turn | `on_user_turn_completed` |
| Corrections keep history; red flags are recomputed, never deleted on request | `Visit._reflag` |
| Patterns across visits (3 refusals in a row, 2 falls in 7 days, poor eating) | `baton_core.patterns` |
| Every change saved instantly; a dropped call resumes on callback | `Store`, `on_call_end` |
| "Hold on" pauses the silence check-ins for up to 3 minutes | `hold_on`, `agent.py` |

## 90-second demo
1. "Hi, it's Maria, PIN two four six eight, Rosa's visit."
2. "She had her sugar pill, didn't want the blood pressure one again, ate like half." → confirm → **pattern alert**.
3. "Hold on, she's calling me." … "Okay, back."
4. "She was a bit wobbly getting up." → confirm → "Actually leave that out." → declined, **integrity flag**.
5. Mood, skin, nothing else → read-back → "Yes" → **Filed · Urgent · nurse supervisor alerted**.
