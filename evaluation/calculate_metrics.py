import csv
from pathlib import Path

INPUT = Path("evaluation/ablation_predictions.csv")
OUTPUT = Path("evaluation/ablation_metrics.csv")

METHODS = ["A", "B", "C", "D"]


def safe_div(a, b):
    return a / b if b else 0.0


def calculate(rows, method):
    decided = []
    review = []
    all_items = []

    for row in rows:
        gt = int(row["gt"])
        pred = int(row[method])

        if gt not in (0, 1):
            raise ValueError(f"Invalid GT: {gt}")
        if pred not in (-1, 0, 1):
            raise ValueError(f"Invalid prediction: {pred}")

        all_items.append((gt, pred))

        if pred == -1:
            review.append((gt, pred))
        else:
            decided.append((gt, pred))

    tp = sum(g == 1 and p == 1 for g, p in decided)
    fp = sum(g == 0 and p == 1 for g, p in decided)
    tn = sum(g == 0 and p == 0 for g, p in decided)
    fn = sum(g == 1 and p == 0 for g, p in decided)

    total = len(all_items)
    gt_positive = sum(g == 1 for g, _ in all_items)
    gt_negative = sum(g == 0 for g, _ in all_items)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    # 확정 판단된 항목 기준
    fpr_decided = safe_div(fp, fp + tn)
    fnr_decided = safe_div(fn, fn + tp)

    # 전체 정답 데이터 기준
    false_alarm_all = safe_div(fp, gt_negative)
    missed_risk_all = safe_div(fn, gt_positive)

    risk_review = sum(g == 1 for g, _ in review)

    return {
        "method": method,
        "total": total,
        "decided": len(decided),
        "review": len(review),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision_decided": round(precision, 4),
        "recall_decided": round(recall, 4),
        "f1_decided": round(f1, 4),
        "fpr_decided": round(fpr_decided, 4),
        "fnr_decided": round(fnr_decided, 4),
        "false_alarm_all": round(false_alarm_all, 4),
        "missed_risk_all": round(missed_risk_all, 4),
        "review_rate": round(safe_div(len(review), total), 4),
        "coverage": round(safe_div(len(decided), total), 4),
        "risk_review_rate": round(
            safe_div(risk_review, gt_positive), 4
        ),
    }


with INPUT.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise ValueError("평가 데이터가 비어 있습니다.")

sample_ids = [row["sample_id"] for row in rows]
if len(sample_ids) != len(set(sample_ids)):
    raise ValueError("sample_id 중복: 평가 단위를 확인하세요.")

results = [calculate(rows, method) for method in METHODS]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

for result in results:
    print(
        f"{result['method']}: "
        f"F1={result['f1_decided']:.4f}, "
        f"FPR={result['fpr_decided']:.4f}, "
        f"누락={result['missed_risk_all']:.4f}, "
        f"보류={result['review_rate']:.4f}, "
        f"Coverage={result['coverage']:.4f}"
    )

print(f"\n저장 완료: {OUTPUT}")