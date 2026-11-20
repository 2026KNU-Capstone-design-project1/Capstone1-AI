import csv
import re
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("evaluation/results")
OUT = ROOT / "recall_priority"
OUT.mkdir(parents=True, exist_ok=True)

SETS = {
    "full_1256": ROOT / "team_ablation/ablation_details.csv",
    "dedup_1203": ROOT / "ab_deduplicated/dedup_details.csv",
}

POLICY_DISPLAY_NAMES = {
    "A": "A",
    "D": "B",
    "E": "C",
    "F": "D",
}

def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def parse_codes(value):
    value = (value or "").strip()
    if not value:
        return set()
    return {
        x.strip()
        for x in re.split(r"[,;|]", value)
        if x.strip()
    }

def predict(row):
    return row["decision"], parse_codes(row["predicted_codes"])

def evaluate(rows, policy):
    groups = defaultdict(dict)

    for row in rows:
        groups[row["image_id"]][row["model"]] = row

    counts = Counter()
    examples = []

    for image_id, group in groups.items():
        if not {"A", "B", "D"} <= group.keys():
            raise ValueError(f"모델 결과 누락: {image_id}")

        a = group["A"]
        b = group["B"]
        d = group["D"]

        gt_hazard = a["gt_hazard"].lower() == "true"
        gt_code = a["gt_code"]

        if policy == "A":
            decision, codes = predict(a)

        elif policy == "D":
            decision, codes = predict(d)

        elif policy == "E":
            # YOLO 우선. YOLO가 정상으로 판단한 경우
            # 공간관계 위험 후보를 추가로 확인한다.
            decision, codes = predict(a)

            if decision != "hazard" and b["decision"] == "hazard":
                decision, codes = predict(b)

        elif policy == "F":
            # 두 방법의 모든 위험 후보를 결합한다.
            a_codes = parse_codes(a["predicted_codes"]) \
                if a["decision"] == "hazard" else set()

            b_codes = parse_codes(b["predicted_codes"]) \
                if b["decision"] == "hazard" else set()

            codes = a_codes | b_codes
            decision = "hazard" if codes else "normal"

        else:
            raise ValueError(policy)

        counts["total"] += 1

        if gt_hazard:
            counts["total_hazard"] += 1
        else:
            counts["total_normal"] += 1

        if decision == "review":
            counts["review"] += 1
            if gt_hazard:
                counts["review_hazard"] += 1
            continue

        counts["covered"] += 1

        if gt_hazard:
            if decision == "hazard" and gt_code in codes:
                counts["tp"] += 1
            else:
                counts["fn"] += 1
                if decision == "hazard":
                    counts["wrong_code"] += 1
        else:
            if decision == "hazard":
                counts["fp"] += 1
            else:
                counts["tn"] += 1

        if policy in ("E", "F"):
            examples.append({
                "image_id": image_id,
                "gt_hazard": gt_hazard,
                "gt_code": gt_code,
                "policy": policy,
                "decision": decision,
                "predicted_codes": ";".join(sorted(codes)),
                "a_decision": a["decision"],
                "b_decision": b["decision"],
            })

    tp, fp = counts["tp"], counts["fp"]
    wrong, fn = counts["wrong_code"], counts["fn"]

    precision = tp / (tp + fp + wrong) if tp + fp + wrong else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) \
        if precision + recall else 0

    return {
        "policy": policy,
        "n_images": counts["total"],
        "total_hazard": counts["total_hazard"],
        "coverage": counts["covered"] / counts["total"],
        "precision_covered": precision,
        "recall_covered": recall,
        "f1_covered": f1,
        "total_hazard_recall": tp / counts["total_hazard"],
        "false_positive_rate": fp / counts["total_normal"],
        "tp": tp,
        "fp": fp,
        "fn_covered": fn,
        "wrong_code": wrong,
        "review": counts["review"],
    }, examples

all_results = []
all_examples = []

for name, path in SETS.items():
    rows = read_rows(path)

    for policy in ("A", "D", "E", "F"):
        result, examples = evaluate(rows, policy)
        result["set"] = name
        all_results.append(result)

        for e in examples:
            e["set"] = name
        all_examples.extend(examples)

# 기존 실험과 기본 지표가 일치하는지 검사
reference = {
    ("full_1256", "A"): (465, 16, 0),
    ("full_1256", "D"): (322, 3, 261),
    ("dedup_1203", "A"): (435, 16, 0),
    ("dedup_1203", "D"): (297, 3, 255),
}

for r in all_results:
    key = (r["set"], r["policy"])
    if key in reference:
        actual = (r["tp"], r["fp"], r["review"])
        if actual != reference[key]:
            raise SystemExit(
                f"기존 평가 불일치: {key} "
                f"기대={reference[key]}, 실제={actual}"
            )

def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

write_csv(OUT / "comparison.csv", all_results)
write_csv(OUT / "prediction_details.csv", all_examples)

print("기존 A/D 결과 일치 검증: PASS")
print()
print("=== 위험 Recall 우선 정책 비교 ===")
print(
    f"{'세트':12} {'정책':4} {'Recall':>9} "
    f"{'Precision':>10} {'F1':>9} "
    f"{'FPR':>8} {'TP':>5} {'FP':>5} {'보류':>5}"
)

for r in all_results:
    print(
        f"{r['set']:12} {r['policy']:4} "
        f"{r['total_hazard_recall']:9.2%} "
        f"{r['precision_covered']:10.2%} "
        f"{r['f1_covered']:9.2%} "
        f"{r['false_positive_rate']:8.2%} "
        f"{r['tp']:5d} {r['fp']:5d} {r['review']:5d}"
    )

print()
print("저장:", OUT / "comparison.csv")
print("저장:", OUT / "prediction_details.csv")
