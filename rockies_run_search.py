import itertools
import time
from rockies_lineup_optimizer import *

pa = load_col_plate_appearances()
pa = add_batter_names(pa)

# Wider candidate pool (default n=15) -- NOT the final 9. Who actually
# makes the lineup is decided per-hand below.
roster, counts = get_current_roster(pa)

# Brenton Doyle is no longer on the Rockies -- remove him from the
# candidate pool entirely so he can't be selected into either lineup.
roster = [name for name in roster if name != 'Brenton Doyle']
print(f'Candidate pool after removing Brenton Doyle ({len(roster)} players):')
print(roster)
print()

rates, debug_df, league = compute_split_rates(pa, roster, k=150)

# ---------------------------------------------------------------------
# Selection step: score all 15 candidates against each pitcher hand using
# linear weights derived from that hand's league-average rates, then keep
# only the best 9 for that hand. vs-RHP and vs-LHP can end up with
# different personnel, not just a different order of the same 9.
# ---------------------------------------------------------------------
lineups = {}
for hand in ['R', 'L']:
    league_probs = dict(zip(OUTCOMES, league[hand]))
    weights = compute_linear_weights(league_probs)
    scores = score_batters(rates, roster, weights, hand)
    lineups[hand] = select_top_n(scores, n=9)
    label = 'RHP' if hand == 'R' else 'LHP'
    print(f'Selected 9 vs {label}: {lineups[hand]}')
print()


def search(hand, lineup):
    """Exhaustive search over the 40,320 distinct orderings of `lineup`
    (a 9-name list already chosen for this specific hand)."""
    anchor = lineup[0]
    rest = lineup[1:]
    probs = {name: rates[name][hand] for name in lineup}

    results = []
    t0 = time.time()
    for perm in itertools.permutations(rest):
        order_names = (anchor,) + perm
        order_probs = [probs[n] for n in order_names]
        rg = evaluate_lineup(order_probs)['runs_per_game']
        results.append((rg, order_names))
    elapsed = time.time() - t0

    results.sort(key=lambda x: -x[0])
    label = 'RHP' if hand == 'R' else 'LHP'
    print(f'--- vs {label}  ({len(results)} distinct lineups, {elapsed:.1f}s) ---')
    print('Best 10:')
    for rg, order in results[:10]:
        print(f'  {rg:.4f} runs/game:  {" -> ".join(order)}')
    print('Worst 3 (for contrast):')
    for rg, order in results[-3:]:
        print(f'  {rg:.4f} runs/game:  {" -> ".join(order)}')

    # "Naive" baseline = the selected 9 batted in the order select_top_n
    # returned them (highest value-per-PA first). This always exists among
    # the searched permutations since anchor=lineup[0], rest=lineup[1:].
    naive_rg = next(rg for rg, order in results if order == tuple(lineup))
    best_rg = results[0][0]
    worst_rg = results[-1][0]
    print(f'Naive (score-ranked, best-to-worst) order: {naive_rg:.4f} runs/game')
    print(f'Spread best-worst: {best_rg-worst_rg:.4f} runs/game ({(best_rg-worst_rg)*162:.1f} runs/162-game season)')
    print(f'Gain best vs naive: {best_rg-naive_rg:.4f} runs/game ({(best_rg-naive_rg)*162:.1f} runs/season)')
    print()
    return results


res_r = search('R', lineups['R'])
res_l = search('L', lineups['L'])
