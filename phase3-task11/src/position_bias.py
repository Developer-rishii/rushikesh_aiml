"""
position_bias.py
=================
Estimates the examination propensity P(examine | position) from logged
data using intervention harvesting: compare click-through rate at each
position on the RANDOMIZED-order slice (where position is decoupled from
relevance) against the non-randomized slice, the standard way real
marketplaces recover propensities without needing true relevance labels.

We fit the classic Position-Based Model P(examine|pos) = 1/pos^eta by
least squares on the randomized slice's observed CTR-by-position curve
(the randomized slice's average relevance-given-position is constant by
construction, so its CTR curve IS the propensity curve up to a scale
constant).

Exposes:
    estimate_propensities(df) -> dict[position -> propensity in (0,1]]
    ips_weight(position, propensities, clip=...) -> inverse-propensity weight
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pbm(pos, eta, scale):
    return scale / np.power(pos, eta)


def estimate_propensities(df: pd.DataFrame, max_position: int = 20) -> dict:
    rand = df[df.is_randomized_slice == 1]
    ctr_by_pos = rand.groupby("position")["click"].mean().reindex(
        range(1, max_position + 1)
    )
    ctr_by_pos = ctr_by_pos.fillna(ctr_by_pos.mean())

    positions = ctr_by_pos.index.values.astype(float)
    ctr = ctr_by_pos.values
    try:
        (eta, scale), _ = curve_fit(_pbm, positions, ctr, p0=[1.0, ctr[0]], maxfev=5000)
    except Exception:
        eta, scale = 1.0, ctr[0]

    propensities = {int(p): float(np.clip(_pbm(p, eta, scale), 1e-3, 1.0)) for p in positions}
    # normalise so position 1 has propensity 1.0 (reference point)
    ref = propensities[1]
    propensities = {p: v / ref for p, v in propensities.items()}
    return propensities, float(eta)


def ips_weight(position: int, propensities: dict, clip: float = 10.0) -> float:
    p = propensities.get(int(position), min(propensities.values()))
    w = 1.0 / max(p, 1e-3)
    return float(min(w, clip))


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "raw_logs.csv")
    props, eta = estimate_propensities(df)
    print(f"Fitted PBM eta = {eta:.3f} (ground truth simulator eta = 1.4)")
    for p in [1, 2, 5, 10, 15, 20]:
        print(f"  position {p:>2}: propensity={props[p]:.3f}  ips_weight={ips_weight(p, props):.2f}")
