import itertools
import time
from lineup_optimizer import *

pa = load_col_plate_appearances()
pa = add_batter_names(pa)
roster, counts = get_current_roster(pa, n=9)
rates, debug_df, league = compute_split_rates(pa, roster, k=150)


def search(hand):
    anchor = roster[0]
    rest = roster[1:]
    probs = {name: rates[name][hand] for name in roster}

    results = []
    t0 = time.time()
    for perm in itertools.permutations(rest):
        order_names = (anchor,) + perm
        order_probs = [probs[n] for n in order_names]
        rg = evaluate_lineup(order_probs)['runs_per_game']
        results.append((rg, order_names))
    elapsed = time.time() - t0

    results.sort(key=lambda x: -x[0])
    print(f'--- vs {"RHP" if hand=="R" else "LHP"}  ({len(results)} distinct lineups, {elapsed:.1f}s) ---')
    print('Best 10:')
    for rg, order in results[:10]:
        print(f'  {rg:.4f} runs/game:  {" -> ".join(order)}')
    print('Worst 3 (for contrast):')
    for rg, order in results[-3:]:
        print(f'  {rg:.4f} runs/game:  {" -> ".join(order)}')
    naive_rg = next(rg for rg, order in results if order == tuple(roster))
    best_rg = results[0][0]
    worst_rg = results[-1][0]
    print(f'Naive (2026-PA-rank) order: {naive_rg:.4f} runs/game')
    print(f'Spread best-worst: {best_rg-worst_rg:.4f} runs/game ({(best_rg-worst_rg)*162:.1f} runs/162-game season)')
    print(f'Gain best vs naive: {best_rg-naive_rg:.4f} runs/game ({(best_rg-naive_rg)*162:.1f} runs/season)')
    print()
    return results


res_r = search('R')
res_l = search('L')
