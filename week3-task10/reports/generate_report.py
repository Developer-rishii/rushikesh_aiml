"""
PlaceMux Quality Sign-Off - Report Generator
==============================================
Auto-generates the sign-off report (Markdown) from actual run output.
"""

import json
import os
from datetime import datetime, timezone

REPORT_DIR = os.path.dirname(os.path.dirname(__file__))
REPORT_DIR = os.path.join(REPORT_DIR, "reports")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "models")


def generate_report():
    eval_path = os.path.join(REPORT_DIR, "evaluation_results.json")
    log_path = os.path.join(MODEL_DIR, "experiment_log.json")

    with open(eval_path) as f:
        ev = json.load(f)
    with open(log_path) as f:
        logs = json.load(f)
    exp = logs[-1]

    overall = ev["overall"]
    pre = ev["pre_monetization"]
    post = ev["post_monetization"]
    guard = ev["guardrail"]
    model_wins = ev["model_vs_baseline"]

    def _row(label, d):
        return f"| {label:20s} | {d.get('precision','-'):>9} | {d.get('recall','-'):>9} | {d.get('fpr','-'):>9} | {d.get('n', d.get('n','-')):>5} |"

    def _delta_row(d):
        return (f"| {'**Delta (mod-base)**':20s} | "
                f"{d.get('precision_delta','-'):>+9} | "
                f"{d.get('recall_delta','-'):>+9} | "
                f"{d.get('fpr_delta','-'):>+9} | "
                f"{'':>5} |")

    hdr = f"| {'':20s} | {'Precision':>9} | {'Recall':>9} | {'FPR':>9} | {'N':>5} |"
    sep = f"|{'-'*22}|{'-'*11}|{'-'*11}|{'-'*11}|{'-'*7}|"

    fi = exp.get("feature_importances", {})
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    fi_table = "\\n".join(f"| {name:30s} | {imp:>10.4f} |" for name, imp in fi_sorted)
    fi_hdr = f"| {'Feature':30s} | {'Importance':>10} |"
    fi_sep = f"|{'-'*32}|{'-'*12}|"

    def _segment_table(seg_dict, label):
        lines = [f"\\n**By {label}:**\\n", hdr, sep]
        for seg_name, seg_data in seg_dict.items():
            lines.append(f"| {f'{seg_name} baseline':20s} | {seg_data['baseline'].get('precision','-'):>9} | {seg_data['baseline'].get('recall','-'):>9} | {seg_data['baseline'].get('fpr','-'):>9} | {seg_data['baseline'].get('n','-'):>5} |")
            lines.append(f"| {f'{seg_name} model':20s} | {seg_data['model'].get('precision','-'):>9} | {seg_data['model'].get('recall','-'):>9} | {seg_data['model'].get('fpr','-'):>9} | {seg_data['model'].get('n','-'):>5} |")
            lines.append(_delta_row(seg_data['delta']))
            lines.append(sep)
        return "\\n".join(lines)

    tier_table = _segment_table(ev.get("by_price_tier", {}), "Price Tier")
    pay_table = _segment_table(ev.get("by_payment_status", {}), "Payment Status")

    def _verdict(flag, metric, pre_val, post_val):
        symbol = "[!] YES" if flag else "[OK] NO"
        return f"- **{metric} degraded?** {symbol} (pre={pre_val}, post={post_val})"

    guard_lines = "\\n".join([
        _verdict(guard["precision_degraded"], "Precision", guard["pre_precision"], guard["post_precision"]),
        _verdict(guard["recall_degraded"], "Recall", guard["pre_recall"], guard["post_recall"]),
        _verdict(guard["fpr_increased"], "FPR increased", guard["pre_fpr"], guard["post_fpr"]),
    ])

    wins = []
    if model_wins.get("precision"): wins.append("precision")
    if model_wins.get("recall"): wins.append("recall")
    if model_wins.get("fpr_lower"): wins.append("lower FPR")
    if wins:
        verdict_str = f"[OK] Trained model **beats baseline** on: {', '.join(wins)}."
    else:
        verdict_str = "[!] Trained model does **not** beat the baseline on any metric. Investigate."

    report = f"""# PlaceMux Quality Sign-Off Report

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
**Test Set Size:** {ev.get('test_set_size', 'N/A')} samples (held-out, never trained/tuned on)

---

## 1. What "Good" Looks Like

This sign-off verifies that the existing matching/recommendation system **still works
correctly** after monetization integration.

---

## 2. Baseline Definition & Numbers

**Baseline rule:** Rank candidates by raw overlap count of (required skills & verified skills).

### Overall Baseline Performance (Test Set)

{hdr}
{sep}
{_row('Baseline', overall['baseline'])}

---

## 3. Trained ML Model

### Labeling Rule (Auditable)

```
label = 1 (good match) IF AND ONLY IF:
  1. Verified-skill coverage of required skills >= 80%
  2. No required skill is missing by more than 1 level
     (for every skill the student HAS: student_level >= req_level - 1)
label = 0 otherwise
```

### Model Details

| Parameter            | Value                             |
|----------------------|-----------------------------------|
| Algorithm            | RandomForestClassifier            |
| n_estimators         | {exp['params'].get('n_estimators', 'N/A')} |
| max_depth            | {exp['params'].get('max_depth', 'N/A')} |
| Train / Val / Test   | {exp['train_size']} / {exp['val_size']} / {exp['test_size']} |
| Train accuracy       | {exp['train_accuracy']} |
| Val P / R / F1       | {exp['val_precision']} / {exp['val_recall']} / {exp['val_f1']} |
| Test P / R / F1      | {exp['test_precision']} / {exp['test_recall']} / {exp['test_f1']} |
| Training time        | {exp['train_time_sec']}s |

### Feature Importances

{fi_hdr}
{fi_sep}
{fi_table}

---

## 4. Model vs Baseline - Pre/Post Monetization

### Overall (Test Set)

{hdr}
{sep}
{_row('Baseline', overall['baseline'])}
{_row('Trained model', overall['model'])}
{_delta_row(overall['delta'])}

### Pre-Monetization (Free Applications)

{hdr}
{sep}
{_row('Baseline', pre['baseline'])}
{_row('Trained model', pre['model'])}
{_delta_row(pre['delta'])}

### Post-Monetization (Paid Applications)

{hdr}
{sep}
{_row('Baseline', post['baseline'])}
{_row('Trained model', post['model'])}
{_delta_row(post['delta'])}

### Guardrail: Post-Monetization Degradation Check

{guard_lines}

### Model vs Baseline Verdict

{verdict_str}

Overall delta: precision {overall['delta']['precision_delta']:+.4f}, recall {overall['delta']['recall_delta']:+.4f}, FPR {overall['delta']['fpr_delta']:+.4f}.

### Segment Breakdowns

{tier_table}

{pay_table}

---

## 5. Worked Example (Explainability Walkthrough)

Use the live API to get a real per-prediction explanation:

```
GET /match/S010/J005
```

---

## 6. Edge Cases Tested

| Edge Case                        | Handled By                          | Test                                |
|----------------------------------|-------------------------------------|-------------------------------------|
| Payment fails mid-application    | `reconciliation.handle_payment_failure()` | `test_student_retains_application_on_failure` |
| Student charged without match    | `reconciliation.reconcile_payments()` | `test_charged_without_match_flagged` |
| Gateway/recorded amount mismatch | `reconciliation.validate_amounts()` | `test_mismatch_detected` |
| Duplicate/partial payments       | `reconciliation.reconcile_payments()` | `test_duplicates_detected` |
| Missing skill scores (NaN)       | `features.build_features()`         | `test_baseline_handles_nan_skills` |
| Zero-overlap JD                  | Handled gracefully                  | `test_baseline_zero_overlap` |

---

## 7. Self-Check Questions

### Can "Quality Sign-Off" be shown working live, not just described?

**YES.** Start the API with `uvicorn api.main:app --port 8000` and hit:
- `/match/S010/J005` for a live prediction with explanation
- `/signoff/report` for the full metrics JSON
- `/signoff/reconciliation` for live mismatch detection

### What happens if a payment fails halfway?

**The student retains their application and is never charged without a match record.**
See `src/reconciliation.py::handle_payment_failure()`. If `gateway_amount > 0` and
`payment_status == "failed"`, a refund is initiated automatically. The application
status is set to `payment_failed`, NOT deleted.

### How do we know our records match what the gateway collected?

**`reconciliation.validate_amounts()` compares every `gateway_amount` to `recorded_amount`
with a configurable tolerance.** Mismatches are flagged with severity.
The `/signoff/reconciliation` endpoint returns all flagged discrepancies live.

### Are we in real-money/test mode?

**This is built and validated in test mode on synthetic data.** The payment amounts
($0, $29.99, $99.99) are realistic but generated.
"""

    out_path = os.path.join(REPORT_DIR, "signoff_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        # Evaluate escaped newlines properly by using normal literal \n in actual output if desired.
        f.write(report.replace('\\n', '\n'))

    print(f"[OK] Sign-off report generated -> reports/signoff_report.md")
    return out_path


if __name__ == "__main__":
    generate_report()
