import os
import pandas as pd
import numpy as np
from pybaseball import playerid_reverse_lookup

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Baseball Projects', 'savant_2026_07272026.csv')

def add_batter_names(pa):
    ids = pa['batter'].unique().tolist()
    lookup = playerid_reverse_lookup(ids, key_type='mlbam')
    lookup['full_name'] = (lookup['name_first'].str.title() + ' ' + lookup['name_last'].str.title())
    name_map = lookup.set_index('key_mlbam')['full_name'].to_dict()
    pa = pa.copy()
    pa['batter_name'] = pa['batter'].map(name_map)
    return pa

def add_pitcher_names(pa):
    ids = pa['pitcher'].unique().tolist()
    lookup = playerid_reverse_lookup(ids, key_type='mlbam')
    lookup['full_name'] = (lookup['name_first'].str.title() + ' ' + lookup['name_last'].str.title())
    name_map = lookup.set_index('key_mlbam')['full_name'].to_dict()
    pa = pa.copy()
    pa['pitcher_name'] = pa['pitcher'].map(name_map)
    return pa

## Create outcome mapping for plate appearances
OUTCOME_MAP = {
    'walk': 'BB', 'intent_walk': 'BB', 'hit_by_pitch': 'BB',
    'single': '1B', 'field_error': '1B', 'catcher_interf': 'BB',
    'double': '2B',
    'triple': '3B',
    'home_run': 'HR',
    'strikeout': 'Out', 'strikeout_double_play': 'Out',
    'field_out': 'Out', 'force_out': 'Out',
    'grounded_into_double_play': 'Out', 'double_play': 'Out',
    'fielders_choice_out': 'Out', 'fielders_choice': 'Out',
    'sac_fly': 'Out', 'sac_bunt': 'Out', 'sac_fly_double_play': 'Out',
    'triple_play': 'Out'
}

## Gather plate appearance data
def load_col_plate_appearances():
    df = pd.read_csv(DATA_PATH, index_col=0, low_memory=False)

    # Determine batting team
    is_top = df['inning_topbot'] == 'Top'
    bat_team = pd.Series(np.where(is_top, df['away_team'], df['home_team']), index=df.index)
    # Grab only Colorado Rockies data
    col = df.loc[bat_team == 'COL'].copy()

    # Grab the last pitch of the plate appearance
    pa = col[col['events'].notna()].copy()

    # Keep only the columns we need for the Markov model
    keep_cols = [
        'game_pk', 'game_date', 'game_year', 'at_bat_number',
        'batter', 'pitcher', 'p_throws', 'stand', 'events', 'des'
    ]

    # Remove Duplicates
    pa = pa[keep_cols].drop_duplicates(subset=['game_pk', 'at_bat_number'])

    # Map to a simplified outcome for the Markov model
    pa['outcome'] = pa['events'].map(OUTCOME_MAP)

    # Remove all that dont have a valid outcome
    pa = pa[pa['outcome'].notna()].copy()
    return pa

# Gather the 15 most frequently used batters in 2026
def get_current_roster(pa, n=15):
    pa26 = pa[pa['game_year'] == 2026]
    counts = pa26.groupby('batter_name').size().sort_values(ascending=False)
    roster = counts.head(n).index.tolist()
    return roster, counts

OUTCOMES = ['BB', '1B', '2B', '3B', 'HR', 'Out']
def compute_split_rates(pa, roster, k=150):
    """Per-batter outcome-probability vectors vs RHP and vs LHP.

    Uses all available seasons (2023-2026) for each roster player for sample
    size, then shrinks toward the overall COL-hitter rate for that pitcher
    hand using Dirichlet-style shrinkage: (counts + k*league_rate) / (N + k).
    k is expressed in PA-equivalents of prior strength.
    """
    hand_map = {'R': 'R', 'L': 'L'}
    league = {}
    for hand in ['R', 'L']:
        sub = pa[pa['p_throws'] == hand]
        counts = sub['outcome'].value_counts()
        total = counts.sum()
        league[hand] = np.array([counts.get(o, 0) / total for o in OUTCOMES])

    rates = {}
    debug_rows = []
    for name in roster:
        rates[name] = {}
        for hand in ['R', 'L']:
            sub = pa[(pa['batter_name'] == name) & (pa['p_throws'] == hand)]
            counts = np.array([(sub['outcome'] == o).sum() for o in OUTCOMES], dtype=float)
            n_pa = counts.sum()
            adj = (counts + k * league[hand]) / (n_pa + k)
            rates[name][hand] = dict(zip(OUTCOMES, adj))
            debug_rows.append({'batter': name, 'hand': hand, 'n_pa': int(n_pa), **dict(zip(OUTCOMES, adj))})

    return rates, pd.DataFrame(debug_rows), league

# ---------------------------------------------------------------------------
# Markov chain run-expectancy model (Bukiet/Harold/Palacios style)
#
# State = (base_state, outs). base_state is a 3-bit mask: bit0=runner on 1st,
# bit1=runner on 2nd, bit2=runner on 3rd (0-7). outs is 0, 1, or 2 (a state
# with 3 outs is not represented -- it immediately resets to base_state=0,
# outs=0 for the next batter). State index = outs*8 + base_state (0-23).
#
# Simplifying base-running assumptions (documented, not modeled precisely):
#   - Walk/HBP: standard force-advancement only.
#   - Single: runner on 1st -> 2nd; runners on 2nd/3rd score.
#   - Double: runner on 1st -> 3rd; runners on 2nd/3rd score.
#   - Triple/HR: all runners score.
#   - Out: no advancement, no double-play removal of a lead runner.
# ---------------------------------------------------------------------------

N_STATES = 24


def _bits(base_state):
    return base_state & 1, (base_state >> 1) & 1, (base_state >> 2) & 1


def _advance_walk(base_state):
    b1, b2, b3 = _bits(base_state)
    if not b1:
        return (1 | (b2 << 1) | (b3 << 2)), 0
    if not b2:
        return (1 | 1 << 1 | (b3 << 2)), 0
    if not b3:
        return (1 | 1 << 1 | 1 << 2), 0
    return (1 | 1 << 1 | 1 << 2), 1


def _advance_single(base_state):
    b1, b2, b3 = _bits(base_state)
    runs = b2 + b3
    return (1 | (b1 << 1)), runs


def _advance_double(base_state):
    b1, b2, b3 = _bits(base_state)
    runs = b2 + b3
    return ((1 << 1) | (b1 << 2)), runs


def _advance_triple(base_state):
    b1, b2, b3 = _bits(base_state)
    runs = b1 + b2 + b3
    return (1 << 2), runs


def _advance_hr(base_state):
    b1, b2, b3 = _bits(base_state)
    runs = b1 + b2 + b3 + 1
    return 0, runs


_ADVANCE = {
    'BB': _advance_walk, '1B': _advance_single, '2B': _advance_double,
    '3B': _advance_triple, 'HR': _advance_hr,
}


def build_transition(probs):
    """probs: dict outcome -> probability. Returns (T [24x24], reward [24])."""
    T = np.zeros((N_STATES, N_STATES))
    reward = np.zeros(N_STATES)
    for outs in range(3):
        for base_state in range(8):
            idx = outs * 8 + base_state
            for outcome in ['BB', '1B', '2B', '3B', 'HR']:
                p = probs[outcome]
                if p == 0:
                    continue
                new_base, runs = _ADVANCE[outcome](base_state)
                new_idx = outs * 8 + new_base
                T[idx, new_idx] += p
                reward[idx] += p * runs
            p_out = probs['Out']
            if outs + 1 == 3:
                new_idx = 0  # inning over: outs=0, base_state=0
            else:
                new_idx = (outs + 1) * 8 + base_state
            T[idx, new_idx] += p_out
    return T, reward

def _stationary(M):
    vals, vecs = np.linalg.eig(M.T)
    i = np.argmin(np.abs(vals - 1))
    v = np.real(vecs[:, i])
    v = v / v.sum()
    return v


def evaluate_lineup(order_probs):
    """order_probs: list of 9 outcome-probability dicts, batting order 1-9.
    Returns dict with runs_per_pa, outs_per_pa, pa_per_game, runs_per_game.
    """
    Ts, rewards = zip(*[build_transition(p) for p in order_probs])
    M = Ts[0]
    for T in Ts[1:]:
        M = M @ T
    pi0 = _stationary(M)

    pis = [pi0]
    for T in Ts[:-1]:
        pis.append(pis[-1] @ T)

    runs_per_pa = np.mean([pis[i] @ rewards[i] for i in range(9)])
    outs_per_pa = np.mean([p['Out'] for p in order_probs])
    pa_per_game = 27.0 / outs_per_pa
    runs_per_game = runs_per_pa * pa_per_game

    return {
        'runs_per_pa': runs_per_pa,
        'outs_per_pa': outs_per_pa,
        'pa_per_game': pa_per_game,
        'runs_per_game': runs_per_game,
    }


# ---------------------------------------------------------------------------
# ROSTER-SELECTION STEP: pick the best 9 of the wider candidate pool (e.g.
# 15) instead of just taking the 9 with the most plate appearances.
#
# This is a 3-part process:
#   1. compute_run_expectancy()  -- for every one of the 24 base/out states,
#      how many MORE runs does a league-average hitter's team expect to
#      score before the half-inning ends (3 outs)?
#   2. compute_linear_weights()  -- turn that into a single run-value per
#      outcome type (BB, 1B, 2B, 3B, HR, Out), e.g. "a double is worth
#      about +0.8 runs, an out is worth about -0.25 runs".
#   3. score_batters() / select_top_n() -- multiply each candidate's own
#      outcome rates by those weights to get one number per player ("runs
#      above average per PA"), then keep the best 9.
#
# Only steps 1-2 need the LEAGUE-AVERAGE rates (they're defining a common
# yardstick). Step 3 is what actually looks at each individual batter.
# ---------------------------------------------------------------------------


def compute_run_expectancy(probs):
    """Run expectancy (RE) for all 24 base/out states, given ONE fixed set
    of outcome probabilities (normally the league-average rates for a
    pitcher hand). RE[state] = expected runs scored from that state onward,
    UNTIL THE CURRENT HALF-INNING ENDS (i.e. until the 3rd out), assuming
    every remaining batter this inning hits like `probs`.

    This is the same idea as the famous "RE24" table in sabermetrics --
    we're just computing our own version from our own model's assumptions
    instead of looking one up, so it stays consistent with the rest of this
    file (same simplified base-running rules, etc.).

    THE MATH:
      Start from build_transition(probs), which gives us:
        T      = 24x24 matrix of "if you're in state i, what state do you
                 end up in next, and with what probability?"
        reward = for each state, the runs expected to score on the VERY
                 NEXT play from that state.

      For run expectancy we need a small but important tweak to T. Normally
      (inside evaluate_lineup) a 3rd out just wraps around to "start of the
      next inning" (state 0), because the game keeps going. But here we
      only care about THIS half-inning, so a 3rd out should be a dead end --
      once it happens, no more runs can be added to our count. We call this
      trimmed-down version Q instead of T.

      (Side note: state 0 -- bases empty, 0 outs -- can ONLY be reached via
      a 3rd-out wraparound. No hit or walk ever produces "bases empty",
      since the batter always ends up on base. So the only edits we need to
      make to turn T into Q are on the 8 rows where outs=2: remove the
      "wraps to state 0" transition on those rows.)

      Once Q only moves BETWEEN the 24 states (and "leaks" probability out
      of the system exactly when a 3rd out happens), run expectancy follows
      from a standard identity for this kind of chain:

          RE(s) = reward(s) + sum over s' of [ Q(s, s') * RE(s') ]
          "runs expected from here" = "runs scored on the very next play"
                                       + "runs still expected after that
                                          play, weighted by where you land"

      Rearranged into matrix form: RE = reward + Q @ RE
                                => (I - Q) @ RE = reward
                                => RE = solve(I - Q, reward)
    """
    T, reward = build_transition(probs)

    # Q = T, but with the "3rd out wraps to next inning" transition removed
    # so that reaching 3 outs is a dead end for THIS calculation.
    Q = T.copy()
    for base_state in range(8):
        idx = 2 * 8 + base_state  # outs=2 states live at indices 16-23
        Q[idx, 0] = 0.0           # sever the wraparound to state 0

    identity = np.eye(N_STATES)
    RE = np.linalg.solve(identity - Q, reward)
    return RE


def compute_linear_weights(probs):
    """Turn the 24-state run-expectancy table into ONE run-value per
    outcome type (BB, 1B, 2B, 3B, HR, Out) -- the classic sabermetric
    "linear weights" idea, derived from our own model instead of imported.

    For a single play, the run VALUE of an outcome from a given state is:

        value = (runs scored immediately on the play)
                + RE(state right AFTER the play)
                - RE(state right BEFORE the play)

    e.g. a home run with the bases empty and 0 outs is worth:
        1 (the runner who just scored) + RE(bases empty, 0 outs)
                                        - RE(bases empty, 0 outs)
        = 1 run, exactly what you'd expect.
    But a walk with a runner on 2nd and 0 outs is worth much less, because
    it doesn't score anyone AND barely changes the run-expectancy of the
    situation (the runner on 2nd doesn't even have to move).

    A single state-specific "value" isn't very useful for ranking players,
    so we average it across all 24 states, weighted by pi -- how OFTEN each
    state actually happens in the long run. pi comes from _stationary(),
    the same function evaluate_lineup() already uses.
    """
    RE = compute_run_expectancy(probs)
    T, _ = build_transition(probs)   # only need this to get pi (see below)
    pi = _stationary(T)              # how often each of the 24 states occurs

    weights = {o: 0.0 for o in OUTCOMES}

    for outs in range(3):
        for base_state in range(8):
            idx = outs * 8 + base_state
            state_freq = pi[idx]
            if state_freq <= 0:
                continue

            # Hits/walks never change the out count, so "the state right
            # after the play" is always one of our 24 states -- just look
            # up its RE directly.
            for outcome in ['BB', '1B', '2B', '3B', 'HR']:
                new_base, runs = _ADVANCE[outcome](base_state)
                new_idx = outs * 8 + new_base
                value = runs + RE[new_idx] - RE[idx]
                weights[outcome] += state_freq * value

            # An out is different: if it's the 3rd out, the half-inning is
            # over, so "RE after" is 0 by definition -- no more runs are
            # coming this inning. Otherwise just look up RE at outs+1.
            if outs + 1 == 3:
                re_after = 0.0
            else:
                re_after = RE[(outs + 1) * 8 + base_state]
            value = 0 + re_after - RE[idx]   # an out never scores a run itself
            weights['Out'] += state_freq * value

    return weights


def score_batters(rates, candidates, weights, hand):
    """Give every candidate batter ONE number: their expected runs above
    (or below) league-average, per plate appearance, against `hand`.

    rates:      the dict returned by compute_split_rates() -- each
                candidate's own outcome rates.
    candidates: list of batter names to score (e.g. all 15 in the pool).
    weights:    dict {outcome: run_value} from compute_linear_weights(),
                computed for the SAME hand ('R' or 'L') being scored here.
    hand:       'R' or 'L'.
    """
    scores = {}
    for name in candidates:
        batter_rates = rates[name][hand]
        # Dot product: (this batter's rate of each outcome) x (how many
        # runs that outcome is worth) added up across all outcome types.
        scores[name] = sum(batter_rates[o] * weights[o] for o in OUTCOMES)
    return scores


def select_top_n(scores, n=9):
    """scores: dict {batter_name: value_per_pa}, from score_batters().
    Returns the n highest-scoring names -- this is the actual "who makes
    the 9-man lineup" decision, separate from what ORDER they bat in.
    """
    ranked = sorted(scores.items(), key=lambda item: -item[1])
    return [name for name, _ in ranked[:n]]


if __name__ == '__main__':
    pa = load_col_plate_appearances()
    pa = add_batter_names(pa)

    # Wider candidate pool -- 15 names, not the final 9.
    roster, counts = get_current_roster(pa)
    print('Candidate pool (top 15 by 2026 PA):')
    print(counts.head(15))
    print()

    rates, debug_df, league = compute_split_rates(pa, roster, k=150)
    print('League (all COL hitters) split rates:')
    print('vs R:', dict(zip(OUTCOMES, league['R'].round(3))))
    print('vs L:', dict(zip(OUTCOMES, league['L'].round(3))))
    print()
    print(debug_df.round(3).to_string(index=False))

    # --- sanity checks on the Markov engine (unchanged from before) ---
    print()
    print('Sanity check: identical-batter lineup, order should not matter')
    avg = {o: float(league['R'][i]) for i, o in enumerate(OUTCOMES)}
    r1 = evaluate_lineup([avg] * 9)
    print('  runs/game with all-average-vs-R batters:', round(r1['runs_per_game'], 3))

    # -----------------------------------------------------------------
    # NEW: selection step. Derive linear weights from the LEAGUE-AVERAGE
    # rates (one set per pitcher hand), then use those weights to score
    # and rank all 15 candidates, and keep only the best 9 for each hand.
    # -----------------------------------------------------------------
    print()
    print('--- Selection step ---')
    for hand in ['R', 'L']:
        league_probs = dict(zip(OUTCOMES, league[hand]))
        weights = compute_linear_weights(league_probs)
        print(f'\nLinear weights vs {hand}HP (runs per outcome):')
        print({o: round(weights[o], 3) for o in OUTCOMES})

        # Sanity check: scoring the league-average hitter against its own
        # weights should land right around 0 -- by construction, an
        # average hitter should be worth ~0 runs above average per PA.
        check = sum(league_probs[o] * weights[o] for o in OUTCOMES)
        print(f'  sanity check (league-average score, should be ~0): {check:.5f}')

        scores = score_batters(rates, roster, weights, hand)
        ranked = sorted(scores.items(), key=lambda item: -item[1])
        print(f'  All 15 candidates ranked by value vs {hand}HP (runs above avg / PA):')
        for name, val in ranked:
            print(f'    {val:+.4f}  {name}')

        top9 = select_top_n(scores, n=9)
        print(f'  --> Selected 9 vs {hand}HP: {top9}')

        # Feed the SELECTED 9 (not all 15) into the existing lineup
        # evaluator, using their as-selected order just as a quick check
        # (this is NOT the optimal order -- that still requires the
        # exhaustive search from run_search.py).
        probs = [rates[name][hand] for name in top9]
        res = evaluate_lineup(probs)
        print(f'  Selected 9, as-ranked order: {res["runs_per_game"]:.4f} runs/game')
