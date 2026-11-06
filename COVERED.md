# Covered — run sheet

**Who calls:** members of a health plan. **The one job:** they hang up with a booked
appointment that is in-network, really available (no ghost listings), cleared under their
plan's rules, and priced before they said yes.

**Why:** insurer "find a doctor" directories are unreliable. A 2023 Senate secret-shopper
study got appointments from only 18% of listed mental health providers; CMS found 45% of
Medicare Advantage directory locations inaccurate. Covered never reads a listing; it only
offers times it can book, and tells you your cost.

## Run (two terminals)

```bash
# .env.local
AGENT_PROFILE=mine          # Covered (baton = the backup idea)
MODEL_PRESET=budget
AGENT_NAME=take2-agent
COVERED_DEMO_PHONES=+1646XXXXXXX=alex   # optional: your phone is recognised as Alex

uv run board/server.py                  # terminal 1 → http://localhost:8787
lk agent console                        # terminal 2, laptop mic, free
lk agent dev --deployment production    # terminal 2, real phone calls
```

Board with no call: `uv run board/server.py --demo`. Fresh data: delete `data/covered_store.json`.

## Synthetic members

| Who | Verify with | Plan | Notes |
|---|---|---|---|
| Alex Johnson | born April 12 1990 | Silver HMO, ZIP 07302 | Specialists need a referral from Dr. Lena Lee. $350 deductible left. Saw Dr. Priya Patel (therapist) before |
| Elena Garcia | born September 3 1948 | Gold PPO, ZIP 07306 | No referrals. Daughter **Maria Garcia** is an authorized rep |

Ghost listings in the directory: Dr. Owen Hart, Bayview Counseling (therapists), Dr. Maya Stone (orthopedics).
Searchable areas: Jersey City, Journal Square, Hoboken, Manhattan, Harlem, Brooklyn.

## Test

```bash
uv run pytest -q                                            # 29 tests, no credits
lk agent debugger start
lk agent debugger say "Hi, I'm Alex Johnson, born April twelfth nineteen ninety"
lk agent simulate text --scenarios scenarios-covered.yaml   # before milestones only
```

## Enforced in code, not the prompt

| Rule | Where |
|---|---|
| No plan detail before name + DOB match; failure never says which part; 3 strikes locks | `verify_identity` |
| Someone else's plan only if the caller is an authorized rep on file | `covered_core.is_authorized_rep` |
| Only providers with a real open slot are offered; ghosts skipped and reported | `covered_core.search` |
| Max 2 options, cheapest first; returning patients get their old provider first | `search(prefer_provider)` |
| Member's own cost: copay, or deductible then coinsurance | `estimate_cost` |
| HMO specialist: no booking until a referral is requested (with a yes) | `find_care`, `book_appointment` |
| Booking only an offered option, only after read-back + yes, never double-booked | `book_appointment`, `Store.book` |
| Coverage answers only from plan data; otherwise offer a specialist | `check_coverage` |
| Every caller turn checked: emergency → 911, self-harm → 988, fraud → decline + flag | `on_user_turn_completed` |
| "Slow down" → one detail per sentence for the rest of the call | `on_user_turn_completed` |
| "Hold on" pauses silence check-ins for up to 3 minutes | `hold_on`, `agent.py` |
| Structured intake record (the spec's JSON) on every booking, handoff or dropped call | `intake()` |

## 2-minute demo
1. **Problem (30 s):** "Insurance directories are full of ghost listings. 82% of the time a secret shopper couldn't get a therapy appointment from them. Covered books you a real one and tells you the price first."
2. **Normal call:** "Hi, it's Alex Johnson, April twelfth nineteen ninety. I need someone to talk to, I've been stressed." → two options with prices, ghosts crossed out on the board → pick one → read-back → booked.
3. **Judge's difficult call:** knee pain (referral rule), "just skip the referral", "not Monday", "I'm in Brooklyn this week", "book it under my husband's insurance", "hold on", silence.
4. **Design choices:** the LLM talks, code decides (identity, cost, rules, booking). Every risky action needs a read-back and a yes.
