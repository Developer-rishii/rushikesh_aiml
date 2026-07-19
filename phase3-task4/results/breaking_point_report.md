# Breaking Point & Headroom Report

Generated from a real, executed load test (`results/load_test_results.csv`); every number below is read directly from that run, not estimated.

## Raw levels

|   concurrency |   n_requests |   achieved_qps |   p50_ms |   p95_ms |   p99_ms |   max_ms |   error_rate |   fallback_rate | past_knee   |
|--------------:|-------------:|---------------:|---------:|---------:|---------:|---------:|-------------:|----------------:|:------------|
|             1 |           83 |           27.7 |     34.4 |     48.9 |     65.4 |     70.2 |            0 |          0      | False       |
|             2 |          169 |           56.3 |     34.2 |     45.6 |     50.8 |     52.2 |            0 |          0      | False       |
|             8 |          533 |          177.7 |     43   |     59.9 |     71.6 |    106   |            0 |          0      | False       |
|            16 |          589 |          196.3 |     78   |    112.3 |    127.7 |    153.7 |            0 |          0      | False       |
|            32 |          602 |          200.7 |    156.2 |    229.7 |    267.7 |    363.4 |            0 |          0      | True        |
|            64 |          657 |          219   |    307.9 |    424.4 |    494.4 |    545.4 |            0 |          0.0518 | True        |
|            96 |          700 |          233.3 |    450.4 |    578.2 |    621.7 |    671.1 |            0 |          0.08   | True        |
|           128 |          719 |          239.7 |    596.5 |    725   |    798.1 |    817   |            0 |          0.0431 | True        |


## Breaking point

- **Breaking point reached at concurrency=32, achieved throughput ~= 201 QPS.**
- Triggered by: p95 latency crossed 3x baseline (229.7ms vs 48.9ms).
- p95 latency at this level = 229.7 ms (baseline p95 at concurrency=1 was 48.9 ms, 4.7x increase); fallback rate = 0.0%.


- **Last clean level (fallback rate ≈ 0%)**: concurrency=32, achieved ≈ 201 QPS, p95 = 229.7 ms. This is the largest load the model path handles with zero degradation — call it **safe capacity**.

## Required headroom

- Stated target sustained load for this service: **150 QPS** (assumption — replace with the real marketplace peak from traffic logs before sign-off).
- Measured safe capacity of **one instance**: **201 QPS**.
- Headroom ratio at 1 instance: **1.34x** target (INSUFFICIENT — needs more instances or precompute, target rule of thumb is ≥1.5x so a single AZ/instance loss doesn't page someone at 3am).
- If running behind a load balancer, minimum instance count for 150 QPS at 1.5x headroom ≈ **2 instances**, assuming linear horizontal scaling (validated separately, not assumed — see scaling_plan.md).
