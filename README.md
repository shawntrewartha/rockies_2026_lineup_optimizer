# Rockies 2026 Lineup Optimizer

Determines the run-maximizing batting order for the Colorado Rockies against right-handed and left-handed starting pitchers, using real Statcast pitch-by-pitch data and an exact Markov-chain run-expectancy model — not a simulation, not a machine-learning model, and not a heuristic.

> Full Article: *https://blakestreetbanter.com/2026/08/12/rockies-data-driven-lineup-optimization/*

## What is accomplished?

Given a pool of candidate hitters, the project answers two questions separately for vs-RHP and vs-LHP:

1. **Who should be in the 9-man lineup?** (personnel selection)
2. **What order should they hit in?** (batting order)

It turns out personnel selection matters far more than batting order — swapping in a better hitter is worth roughly 15-20x more than optimizing the order of whoever's already in the lineup. That finding, and the reasoning behind it, is the core of the article.

## Methodology

### 1. Data
`gather_mlb_data.py` pulls pitch-by-pitch Statcast data via [pybaseball](https://github.com/jldbc/pybaseball) for 2023-2026 and saves it locally as a CSV (not committed to this repo — see Setup below).

### 2. Plate-appearance rates, by pitcher hand, with shrinkage
Each candidate batter's outcome rates (BB, 1B, 2B, 3B, HR, Out) are computed separately against right-handed and left-handed pitchers, using every plate appearance available for that player across all seasons in the dataset. Small samples (a bench bat with 30 PA vs LHP, say) are pulled toward the team-wide average for that pitcher hand via Dirichlet-style shrinkage, so a lucky or unlucky short stretch doesn't get mistaken for true talent.

### 3. Run-expectancy Markov chain
The core model represents a half-inning as a Markov chain over 24 states — 8 base configurations (empty, 1st, 2nd, 3rd, 1st&2nd, 1st&3rd, 2nd&3rd, loaded) × 3 out counts (0, 1, 2). Given a batter's outcome probabilities, the model computes the *exact* expected runs produced by any base/out state — no random sampling, no repeated trials, just a closed-form linear-algebra solution (the same technique behind Bukiet, Harold & Palacios's 1997 paper *"A Markov Chain Approach to Baseball."*

This same machinery is used two ways:
- **To evaluate a specific 9-man batting order**: compose the 9 batters' individual transition matrices in order, solve for the long-run steady-state distribution over the 24 states, and convert that into an expected runs-per-game number.
- **To derive linear weights**: compute a run-expectancy table for a hypothetical league-average hitter, then use it to assign each outcome type (a walk, a double, an out, etc.) a single run value — the same idea behind sabermetric stats like wOBA, but derived from this project's own model instead of imported from elsewhere.

### 4. Roster selection
All eligible candidates (current-season regulars, ~14-15 players) are scored using the linear weights above — "expected runs above average per plate appearance," computed separately vs RHP and vs LHP. The top 9 by that score make each lineup. This step is why the vs-RHP and vs-LHP lineups can (and do) end up with different personnel, not just a different order of the same 9 — real platoon behavior, not a simplifying assumption.

### 5. Exhaustive batting-order search
For each hand's selected 9, every distinct batting order is evaluated exactly (40,320 of them — 9!/9, since batting order is cyclic and rotating who leads off a fixed sequence doesn't change the runs/game result). The best, worst, and a naive baseline are reported for comparison.

## Documented simplifying assumptions

The Markov model uses standard, fixed base-running rules rather than modeling every real advancement decision:
- Walks/HBP: standard force-advancement only.
- Singles: runner on 1st → 2nd; runners on 2nd/3rd score.
- Doubles: runner on 1st → 3rd; runners on 2nd/3rd score.
- Triples/HRs: all runners score.
- Outs: no baserunner advancement, and double plays are **not** modeled separately (a GIDP is treated as a single out).

These are common simplifications in this class of model, not simulation noise — they're deterministic rules, documented in code comments in `rockies_lineup_optimizer.py`.

## Project structure

| File | Purpose |
|---|---|
| `gather_mlb_data.py` | Pulls Statcast data via pybaseball and saves it locally as CSV. |
| `rockies_lineup_optimizer.py` | Core library: data loading, split-rate computation with shrinkage, the Markov run-expectancy engine, and the linear-weights roster-selection functions. |
| `rockies_run_search.py` | Runs the full pipeline: builds the candidate pool, selects the best 9 per pitcher hand, and exhaustively searches all valid batting orders for each. |
| `lineup_search_results.txt` | Saved output from an earlier search run, kept for reference. |

## Setup

```bash
pip install pybaseball pandas numpy

python gather_mlb_data.py # pulls Statcast data (takes a while; be nice to Baseball Savant)
python rockies_run_search.py # runs roster selection + full lineup search, prints results
```

The raw Statcast CSV is large (~1.7GB) and is intentionally excluded from this repo via `.gitignore` — regenerate it locally with `gather_mlb_data.py` rather than expecting it to be checked in.

## Known limitations

- **Roster pool is inferred from playing time**, not a live roster feed — it doesn't know about injuries, demotions, or trades unless manually corrected (see the Brenton Doyle removal in `rockies_run_search.py` as an example of a manual override after he left the team).
- **No recency weighting** — a player's 2023 performance counts exactly as much as their 2026 performance when computing rates and scoring candidates.
- **Small-sample splits are real, even with shrinkage** — a bench player's vs-LHP true talent is still uncertain with only 30-90 career PA against lefties; shrinkage reduces that risk, it doesn't eliminate it.
- **Base-running is simplified** (see above) — the model doesn't know about specific runner speed, hit location, or defensive alignment.
