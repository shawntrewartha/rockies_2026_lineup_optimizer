import os
import pandas as pd
import numpy as np
from pybaseball import playerid_reverse_lookup

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Baseball Projects', 'savant_2026_07272026.csv')

# Map raw Statcast 'events' values to a simplified outcome bucket used by the
# lineup Markov model. K and other outs are merged since, under this model's
# base-running assumptions, both simply record an out with no advancement.
# GIDP-type events are also treated as a single out (double plays are not
# modeled separately) and reached-on-error/interference are bucketed with 1B
# (batter safe at first, no forced extra advancement). truncated_pa is
# dropped (not a real batting outcome).
OUTCOME_MAP = {
    'walk': 'BB', 'intent_walk': 'BB', 'hit_by_pitch': 'BB',
    'single': '1B', 'field_error': '1B', 'catcher_interf': '1B',
    'double': '2B',
    'triple': '3B',
    'home_run': 'HR',
    'strikeout': 'Out', 'strikeout_double_play': 'Out',
    'field_out': 'Out', 'force_out': 'Out',
    'grounded_into_double_play': 'Out', 'double_play': 'Out',
    'fielders_choice_out': 'Out', 'fielders_choice': 'Out',
    'sac_fly': 'Out', 'sac_bunt': 'Out', 'sac_fly_double_play': 'Out',
}


def load_col_plate_appearances():
    df = pd.read_csv(DATA_PATH, index_col=0, low_memory=False)

    # Determine which team is batting on each pitch
    is_top = df['inning_topbot'] == 'Top'
    bat_team = pd.Series(np.where(is_top, df['away_team'], df['home_team']), index=df.index)

    col = df.loc[bat_team == 'COL'].copy()

    # Each plate appearance ends on the pitch where 'events' is populated
    pa = col[col['events'].notna()].copy()

    keep_cols = [
        'game_pk', 'game_date', 'game_year', 'at_bat_number',
        'batter', 'pitcher', 'p_throws', 'stand', 'events', 'des'
    ]
    pa = pa[keep_cols].drop_duplicates(subset=['game_pk', 'at_bat_number'])

    pa['outcome'] = pa['events'].map(OUTCOME_MAP)
    pa = pa[pa['outcome'].notna()].copy()

    return pa


def add_batter_names(pa):
    ids = pa['batter'].unique().tolist()
    lookup = playerid_reverse_lookup(ids, key_type='mlbam')
    lookup['full_name'] = (lookup['name_first'].str.title() + ' ' + lookup['name_last'].str.title())
    name_map = lookup.set_index('key_mlbam')['full_name'].to_dict()
    pa = pa.copy()
    pa['batter_name'] = pa['batter'].map(name_map)
    return pa


OUTCOMES = ['BB', '1B', '2B', '3B', 'HR', 'Out']


def get_current_roster(pa, n=9):
    pa26 = pa[pa['game_year'] == 2026]
    counts = pa26.groupby('batter_name').size().sort_values(ascending=False)
    roster = counts.head(n).index.tolist()
    return roster, counts


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


if __name__ == '__main__':
    pa = load_col_plate_appearances()
    pa = add_batter_names(pa)

    roster, counts = get_current_roster(pa, n=9)
    print('Roster (top 9 by 2026 PA):')
    print(counts.head(12))
    print()

    rates, debug_df, league = compute_split_rates(pa, roster, k=150)
    print('League (all COL hitters) split rates:')
    print('vs R:', dict(zip(OUTCOMES, league['R'].round(3))))
    print('vs L:', dict(zip(OUTCOMES, league['L'].round(3))))
    print()
    print(debug_df.round(3).to_string(index=False))

    # --- sanity checks on the Markov engine ---
    print()
    print('Sanity check: identical-batter lineup, order should not matter')
    avg = {o: float(league['R'][i]) for i, o in enumerate(OUTCOMES)}
    r1 = evaluate_lineup([avg] * 9)
    print('  runs/game with all-average-vs-R batters:', round(r1['runs_per_game'], 3))

    naive_order = roster  # as-is (2026 PA rank order)
    probsR = [rates[b]['R'] for b in naive_order]
    probsL = [rates[b]['L'] for b in naive_order]
    resR = evaluate_lineup(probsR)
    resL = evaluate_lineup(probsL)
    print()
    print('Naive order (by 2026 PA rank) vs RHP:', round(resR['runs_per_game'], 3), 'runs/game')
    print('Naive order (by 2026 PA rank) vs LHP:', round(resL['runs_per_game'], 3), 'runs/game')
