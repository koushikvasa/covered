# Covered — run sheet

**Who calls:** members of a health plan. **The one job:** they hang up with a booked
appointment that is in-network, really available (no ghost listings), cleared under their
plan's rules, and priced before they said yes.

**Why:** insurer "find a doctor" directories are unreliable. A 2023 Senate secret-shopper
study got appointments from only 18% of listed mental health providers; CMS found 45% of
Medicare Advantage directory locations inaccurate. In our network, Covered only offers
times it can book, and tells you your cost. Outside it, Covered finds real providers in the
public NPI Registry and sends a request, without claiming a time it can't back.

## Run (two terminals)

```bash
# .env.local
AGENT_PROFILE=mine          # Covered (baton = the backup idea)
MODEL_PRESET=budget
AGENT_NAME=take2-agent

uv run board/server.py                  # terminal 1 → http://localhost:8787
lk agent console                        # terminal 2, laptop mic, free
lk agent dev --deployment production    # terminal 2, real phone calls
```

Board with no call: `uv run board/server.py --demo`. Fresh data: delete `data/covered_store.json`.

## Synthetic members

| Who | Verify with | Plan | Notes |
|---|---|---|---|
| Alex Johnson | Full name + ID MBR-12345 | Silver HMO, ZIP 07302 | Specialists need a referral from Dr. Lena Lee. $350 deductible left. Saw Dr. Priya Patel (therapist) before |
| Elena Garcia | Full name + ID MBR-67890 | Gold PPO, ZIP 07306 | No referrals. Daughter **Maria Garcia** is an authorized rep |

Ghost listings in the directory: Dr. Owen Hart, Bayview Counseling (therapists), Dr. Maya Stone (orthopedics).
Demo network ZIPs (full booking and ghost listings): 07302, 07306, 07030, 10001, 10027, 11201.
Each demo ZIP has two doctors per specialty (16 specialties); `src/covered_directory.py` generates the ones not hand-written.
Any other ZIP gets real providers from the live NPI Registry, offered as a request with no invented time.

## Test

```bash
uv run pytest -q                                            # no credits
lk agent debugger start
lk agent debugger say "Hi, I'm Alex Johnson, I've been stressed and want someone to talk to"
lk agent simulate text --scenarios scenarios-covered.yaml   # before milestones only
```

## Enforced in code, not the prompt

| Rule | Where |
|---|---|
| No plan detail before full name + insurance ID match; failure never says which part; 3 strikes locks | `verify_identity` |
| The caller's full name must fit the member, or be an authorized rep on file, even with a correct ID | `name_matches`, `is_authorized_rep` |
| No search without a ZIP the caller gave; a home ZIP is never assumed | `find_care` |
| Real providers from the live registry are only ever a request: no invented time, cost stated "if in-network" | `_find_live`, `_request_live` |
| Only providers with a real open slot are offered; ghosts skipped and reported | `covered_core.search` |
| Max 2 options from the caller's own ZIP (nearby only if nothing fits), cheapest first; returning patients get their old provider first | `search` |
| Comparing doctors gives background facts only, never who is better | `compare_doctors` |
| Member's own cost. Tier 1: plan copay + practice fee. Tier 2: deductible left, then coinsurance on the allowed amount | `estimate_cost` |
| HMO specialist: no booking until a referral is requested (with a yes) | `find_care`, `book_appointment` |
| Booking only an offered option, only after read-back + yes, never double-booked | `book_appointment`, `Store.book` |
| Coverage answers only from plan data; otherwise offer a specialist | `check_coverage` |
| Every caller turn checked: emergency → 911, self-harm → 988, fraud → decline + flag | `on_user_turn_completed` |
| "Slow down" → one detail per sentence for the rest of the call | `on_user_turn_completed` |
| "Hold on" pauses silence check-ins for up to 3 minutes | `hold_on`, `agent.py` |
| Off-topic questions get no answer; the call returns to its current step, and repeats also offer a person | `handle_off_topic` |
| Structured intake record (the spec's JSON) on every booking, handoff or dropped call | `intake()` |

## 2-minute demo
1. **Problem (30 s):** "Insurance directories are full of ghost listings. 82% of the time a secret shopper couldn't get a therapy appointment from them. Covered books you a real one and tells you the price first."
2. **Normal call:** "Hi, it's Alex Johnson. I need someone to talk to, I've been stressed." → insurance ID MBR-12345 → ZIP 07302 → two options with prices, ghosts crossed out on the board → pick one → read-back → booked.
3. **Judge's difficult call:** knee pain (referral rule), "just skip the referral", "not Monday", "I'm in Brooklyn this week", "book it under my husband's insurance", "hold on", silence.
4. **Design choices:** the LLM talks, code decides (identity, cost, rules, booking). Every risky action needs a read-back and a yes.
