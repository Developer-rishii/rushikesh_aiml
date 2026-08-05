"""
Stage B deliverable: "A cost model for the intelligence layer (train + serve)"

Bar (Stage B.1, stated up front per the guide's rule "if you cannot state
the bar, you are not ready to build"):
  - Must output cost per inference, cost per training run, and cost per
    shortlist, in INR, traceable to a unit price table (not guessed).
  - Must separate GPU-served vs CPU-served cost (right-sizing depends on it).
  - Must be reproducible from the experiment log (same inputs -> same cost).
"""
from dataclasses import dataclass


# Unit prices are the one input a FinOps model must NOT hide inside code.
# Sourced as a config table so it can be swapped for real cloud billing
# export data without touching logic.
UNIT_PRICES_INR = dict(
    cpu_instance_hour=6.0,      # e.g. general-purpose CPU node, per hour
    gpu_instance_hour=140.0,    # e.g. T4-class GPU node, per hour
    cpu_inferences_per_hour=1_800_000,   # small GBM model throughput on CPU
    gpu_inferences_per_hour=9_000_000,   # same-size model batched on GPU
    embedding_gpu_inferences_per_hour=600_000,  # embedding model is heavier
    storage_gb_month=2.5,       # cache / precompute store cost
)


@dataclass
class CostBreakdown:
    label: str
    train_cost_inr: float
    serve_cost_per_1000_inr: float
    cost_per_10000_shortlists_inr: float
    inferences: int
    shortlists: int
    hardware: str


def training_cost(train_hours: float, hardware: str = "cpu") -> float:
    rate = UNIT_PRICES_INR["gpu_instance_hour" if hardware == "gpu" else "cpu_instance_hour"]
    return round(train_hours * rate, 2)


def serving_cost_per_1000(hardware: str = "cpu", is_embedding: bool = False) -> float:
    if is_embedding:
        throughput = UNIT_PRICES_INR["embedding_gpu_inferences_per_hour"]
        rate = UNIT_PRICES_INR["gpu_instance_hour"]
    else:
        key = f"{hardware}_inferences_per_hour"
        throughput = UNIT_PRICES_INR[key]
        rate = UNIT_PRICES_INR[f"{hardware}_instance_hour"]
    cost_per_inference = rate / throughput
    return round(cost_per_inference * 1000, 4)


def cache_storage_cost(cached_rows: int, avg_row_bytes: int = 64) -> float:
    gb = (cached_rows * avg_row_bytes) / 1e9
    return round(gb * UNIT_PRICES_INR["storage_gb_month"], 4)


def build_breakdown(
    label: str,
    train_hours: float,
    hardware: str,
    inferences: int,
    shortlists: int,
    is_embedding: bool = False,
    extra_cost_inr: float = 0.0,
) -> CostBreakdown:
    train = training_cost(train_hours, hardware)
    per1000 = serving_cost_per_1000(hardware, is_embedding)
    total_serve = per1000 * inferences / 1000 + extra_cost_inr
    cost_per_shortlist = (train + total_serve) / max(shortlists, 1)
    cost_per_10000_shortlists = cost_per_shortlist * 10000
    return CostBreakdown(
        label=label,
        train_cost_inr=train,
        serve_cost_per_1000_inr=per1000,
        cost_per_10000_shortlists_inr=cost_per_10000_shortlists,
        inferences=inferences,
        shortlists=shortlists,
        hardware=hardware,
    )
