
## Run 2026-08-05 12:29:45
- data rows: 100000 (train 75000 / test 25000)
- baseline model: n_estimators=400, max_depth=6, train_s=9.83, version=baseline_v1
- baseline eval: {'ndcg_at_k': 0.5794, 'precision_at_k': 0.5577, 'n_groups': 492}
- baseline cost: CostBreakdown(label='baseline (big model, GPU-served)', train_cost_inr=76.44, serve_cost_per_1000_inr=0.0156, cost_per_shortlist_inr=5.6e-05, inferences=5000000, shortlists=2767600, hardware='gpu')
- train/serve skew check: {'mean_abs_drift': 0.012933132531264815, 'max_abs_drift': 1.4647872858284405, 'pct_rows_stale_feature': 5.8999999999999995}
- small model: n_estimators=60, max_depth=3, train_s=0.89, version=optimized_v1
- small model eval: {'ndcg_at_k': 0.5722, 'precision_at_k': 0.5491, 'n_groups': 492}
- caching: {'total_requests': 100000, 'unique_pairs': 100000, 'cache_hits': 0, 'cache_hit_rate': 0.0, 'inferences_after_cache': 100000}
- precompute vs on-demand: {'n_jobs_total': 1682, 'n_head_jobs': 336, 'head_traffic_share': 0.6462, 'precompute_inferences_per_month': 10080, 'on_demand_inferences': 35381, 'inferences_after_precompute': 45461, 'max_staleness_hours': 24}
- optimized cost: CostBreakdown(label='optimized (small model, CPU, cached, precomputed)', train_cost_inr=0.3, serve_cost_per_1000_inr=0.0033, cost_per_shortlist_inr=0.0, inferences=90800, shortlists=2767600, hardware='cpu')
- BEFORE/AFTER: {'baseline': {'ndcg_at_10': 0.5794, 'serve_cost_per_1000_inr': 0.0156, 'cost_per_shortlist_inr': 5.6e-05, 'train_cost_inr': 76.44}, 'optimized': {'ndcg_at_10': 0.5722, 'serve_cost_per_1000_inr': 0.0033, 'cost_per_shortlist_inr': 0.0, 'train_cost_inr': 0.3}, 'ndcg_delta': -0.0072, 'quality_held_constant': True, 'serve_cost_reduction_pct': 78.85, 'cost_per_shortlist_reduction_pct': 100.0}
- fairness slice (demographic parity proxy): {'group_0': {'mean_score': 1.5922, 'selection_rate_top_quartile': 0.252}, 'group_1': {'mean_score': 1.5962, 'selection_rate_top_quartile': 0.2504}}
- failure injection (model down, 500 requests): {'model_available': False, 'fallback_rows': 499, 'cache_rows': 1, 'heuristic_rows': 499, 'total_rows': 500, 'degraded_gracefully': True}

## Run 2026-08-05 12:45:43
- data rows: 100000 (train 75000 / test 25000)
- baseline model: n_estimators=400, max_depth=6, train_s=9.84, version=baseline_v2
- baseline eval: {'ndcg_at_k': 0.5794, 'precision_at_k': 0.5577, 'n_groups': 492}
- baseline cost: CostBreakdown(label='baseline (big model, GPU-served)', train_cost_inr=76.57, serve_cost_per_1000_inr=0.0156, cost_per_10000_shortlists_inr=0.5584983379101026, inferences=5000000, shortlists=2767600, hardware='gpu')
- train/serve skew check: {'mean_abs_drift': 0.012933132531264815, 'max_abs_drift': 1.4647872858284405, 'pct_rows_stale_feature': 5.8999999999999995}
- small model: n_estimators=60, max_depth=3, train_s=0.92, version=optimized_v2
- small model eval: {'ndcg_at_k': 0.5722, 'precision_at_k': 0.5491, 'n_groups': 492}
- caching: {'total_requests': 100000, 'unique_pairs': 100000, 'cache_hits': 0, 'cache_hit_rate': 0.0, 'inferences_after_cache': 100000}
- precompute vs on-demand: {'n_jobs_total': 1682, 'n_head_jobs': 336, 'head_traffic_share': 0.6462, 'precompute_inferences_per_month': 10080, 'on_demand_inferences': 35381, 'inferences_after_precompute': 45461, 'max_staleness_hours': 24}
- optimized cost: CostBreakdown(label='optimized (small model, CPU, cached, precomputed)', train_cost_inr=0.31, serve_cost_per_1000_inr=0.0033, cost_per_10000_shortlists_inr=0.0022605867899985546, inferences=90800, shortlists=2767600, hardware='cpu')
- BEFORE/AFTER: {'baseline': {'ndcg_at_10': 0.5794, 'serve_cost_per_1000_inr': 0.0156, 'cost_per_10000_shortlists_inr': 0.5585, 'train_cost_inr': 76.57}, 'optimized': {'ndcg_at_10': 0.5722, 'serve_cost_per_1000_inr': 0.0033, 'cost_per_10000_shortlists_inr': 0.0023, 'train_cost_inr': 0.31}, 'ndcg_delta': -0.0072, 'quality_held_constant': True, 'serve_cost_reduction_pct': 78.85, 'cost_per_10000_shortlists_reduction_pct': 99.6}
- fairness slice (demographic parity proxy): {'group_0': {'mean_score': 1.5922, 'selection_rate_top_quartile': 0.252}, 'group_1': {'mean_score': 1.5962, 'selection_rate_top_quartile': 0.2504}}
- failure injection (model down, 500 requests): {'model_available': False, 'fallback_rows': 499, 'cache_rows': 1, 'heuristic_rows': 499, 'total_rows': 500, 'degraded_gracefully': True}
