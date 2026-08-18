"""Self-play harness: candidate vs champion across seeds and both seats."""
import sys, statistics
from kaggle_environments import make

def run(a, b, seed):
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run([a, b]); s = env.steps[-1]
    return s[0]["reward"], s[1]["reward"]

def duel(cand, champ, seeds):
    wins = margins = 0; details = []
    for sd in seeds:
        # candidate as P0
        r0, r1 = run(cand, champ, sd)
        details.append((sd, "P0", r0, r1, r0 - r1)); wins += r0 > r1; margins += (r0 - r1)
        # candidate as P1 (seat swap)
        r0, r1 = run(champ, cand, sd)
        details.append((sd, "P1", r1, r0, r1 - r0)); wins += r1 > r0; margins += (r1 - r0)
    return wins, margins, details

if __name__ == "__main__":
    cand = sys.argv[1] if len(sys.argv) > 1 else "metered.py"
    champ = sys.argv[2] if len(sys.argv) > 2 else "main.py"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    seeds = list(range(n))
    wins, margins, details = duel(cand, champ, seeds)
    games = 2 * len(seeds)
    print(f"=== {cand} vs {champ}: {games} games ({len(seeds)} seeds x 2 seats) ===")
    for sd, seat, cm, chm, d in details:
        print(f"  seed {sd:2d} {seat}: cand={cm:8.0f} champ={chm:8.0f} delta={d:+8.0f} {'WIN' if d>0 else ('tie' if d==0 else 'loss')}")
    print(f"win rate: {wins}/{games}  mean margin/game: {margins/games:+.0f}")
