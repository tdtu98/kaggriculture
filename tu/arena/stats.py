"""Winrate statistics for the arena.

Wilson intervals rather than the normal approximation: at the sample sizes we can afford, and at
winrates near 0 or 1, the naive interval is badly wrong and would license acting on noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Z95 = 1.959963984540054


@dataclass(frozen=True)
class Winrate:
    wins: float          # draws count as 0.5
    games: int

    @property
    def rate(self) -> float:
        return self.wins / self.games if self.games else float("nan")

    def wilson(self, z: float = Z95) -> tuple[float, float]:
        """Wilson score interval, which stays inside [0, 1] and behaves near the extremes."""
        n = self.games
        if n == 0:
            return (float("nan"), float("nan"))
        p = self.wins / n
        d = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / d
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (max(0.0, centre - half), min(1.0, centre + half))

    @property
    def half_width(self) -> float:
        lo, hi = self.wilson()
        return (hi - lo) / 2

    def beats_even(self) -> bool:
        """True only when the whole interval sits above 50% — the bar for acting on a result."""
        lo, _ = self.wilson()
        return lo > 0.5

    def __str__(self) -> str:
        lo, hi = self.wilson()
        return f"{100 * self.rate:5.1f}% [{100 * lo:4.1f}, {100 * hi:4.1f}] n={self.games}"


def games_needed(effect: float, z: float = Z95) -> int:
    """Roughly how many games to resolve a `effect`-sized edge over 50%.

    Sanity anchor for sample sizes: separating 52% from 50% needs ~2,400 games, which is why
    `docs/decisions.md` D10 says not to act on a 52%.
    """
    if effect <= 0:
        return 0
    return math.ceil((z * 0.5 / effect) ** 2)
