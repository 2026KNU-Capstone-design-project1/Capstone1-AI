import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

from evidence_fusion import fuse_evidence

ZIP = Path("data/ab_experiment.zip")
OUT = Path("evaluation/results/production_fusion")
OUT.mkdir(parents=True, exist_ok=True)


def load_data(z, name):
    matches = [
        p for p in z.namelist()
        if p.endswith("/" + name + ".json")
        or p == name + ".json"
    ]
    if len(matches) != 1:
        raise ValueError(f"{name}: 파일 확인 필요: {matches}")
    return json.loads(z.read(matches[0]))["items"]


with zipfile.ZipFile(ZIP) as z:
    gt_rows = load_data(z, "gt_labels")
    yolo_rows = load_data(z, "yolo_predictions")
    spatial_rows = load_data(z, "spatial_judgments")


def index_items(rows):
    result = {}
    for item in rows:
        image_id = item["image_id"]
        if image_id in result:
            raise ValueError(f"중복 image_id: {image_id}")
        result[image_id] = item
    return result


gt_map = index_items(gt_rows)
yo_map = index_items(yolo_rows)
sp_map = index_items(spatial_rows)

if not (gt_map.keys() == yo_map.keys() == sp_map.keys()):
    raise ValueError("세 JSON의 image_id 집합이 다릅니다.")


def code_of(item):
    return item.get("hazard_code") or item.get("code")


def normalize_status(status):
    if status == "hazard":
        return "위험"
    if status == "not_evaluable":
        return "판단 불가"
    return status


def make_frame(yo, sp):
    """GT를 전혀 사용하지 않고 운영 함수 입력을 구성한다."""

    detections = []
    for pred in yo.get("predictions", []):
        code = code_of(pred)
        if not code:
            continue

        detections.append({
            "hazard": code,
            "confidence": pred.get("confidence"),
            "bbox": pred.get("bbox"),
        })

    findings = []
    for finding in sp.get("findings", []):
        code = code_of(finding)
        if not code:
            continue

        item = finding.copy()
        item["hazard"] = code
        item["status"] = normalize_status(item.get("status"))
        findings.append(item)

    checks = []
    for check in sp.get("checks", []):
        code = code_of(check)
        if not code:
            continue

        item = check.copy()
        item["hazard"] = code
        item["status"] = normalize_status(item.get("status"))
        checks.append(item)

    return {
        "detections": detections,
        "spatial_findings": findings,
        "rule_checks": checks,
        "evaluation_status": sp.get("evaluation_status"),
        "reason_code": sp.get("reason_code"),
    }


def save_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


item_results = []
image_results = []

status_names = [
    "matched",
    "disagreement",
    "detection_only",
    "spatial_only",
    "undetermined",
]

# 진단용: 매핑 과정에서 누락된 위험 코드 여부 확인
missing_code = Counter()

for image_id in sorted(gt_map):
    yo = yo_map[image_id]
    sp = sp_map[image_id]

    for pred in yo.get("predictions", []):
        if not code_of(pred):
            missing_code["yolo"] += 1

    for finding in sp.get("findings", []):
        if not code_of(finding):
            missing_code["finding"] += 1

    for check in sp.get("checks", []):
        if not code_of(check):
            missing_code["check"] += 1

    frame = make_frame(yo, sp)

    # 실제 운영 융합 함수 호출
    # 이 시점에는 GT를 전달하지 않음
    fused = fuse_evidence(frame)

    # 이후에만 GT를 가져와 결과를 분석
    gt = gt_map[image_id]
    gt_hazard = bool(gt["is_hazard"])
    gt_code = gt.get("hazard_code") or ""

    statuses = []
    codes = []

    for item in fused:
        status = item.get("evidence_status", "unknown")
        code = item.get("hazard", "")

        statuses.append(status)
        codes.append(code)

        item_results.append({
            "image_id": image_id,
            "gt_hazard": gt_hazard,
            "gt_code": gt_code,
            "candidate_code": code,
            "evidence_status": status,
            "candidate_matches_gt": bool(
                gt_hazard and code == gt_code
            ),
            "has_yolo_detection": item.get("detection") is not None,
            "finding_count": len(item.get("spatial_findings") or []),
            "check_count": len(item.get("rule_checks") or []),
            "reason_code": item.get("reason_code") or "",
        })

    image_row = {
        "image_id": image_id,
        "gt_hazard": gt_hazard,
        "gt_code": gt_code,
        "candidate_count": len(fused),
        "statuses": ";".join(sorted(set(statuses))),
        "candidate_codes": ";".join(sorted(set(codes))),
        "has_gt_code_candidate": bool(
            gt_hazard and gt_code in codes
        ),
    }

    for status in status_names:
        image_row["has_" + status] = status in statuses

    image_row["no_candidate"] = len(fused) == 0
    image_results.append(image_row)


save_csv(
    OUT / "fusion_item_details.csv",
    item_results,
    [
        "image_id",
        "gt_hazard",
        "gt_code",
        "candidate_code",
        "evidence_status",
        "candidate_matches_gt",
        "has_yolo_detection",
        "finding_count",
        "check_count",
        "reason_code",
    ],
)

save_csv(
    OUT / "fusion_image_details.csv",
    image_results,
    [
        "image_id",
        "gt_hazard",
        "gt_code",
        "candidate_count",
        "statuses",
        "candidate_codes",
        "has_gt_code_candidate",
        *["has_" + s for s in status_names],
        "no_candidate",
    ],
)


print("=" * 65)
print("운영 fuse_evidence() 오프라인 근거 결합 평가")
print("=" * 65)

print("전체 이미지:", len(image_results))
print("전체 융합 후보:", len(item_results))
print("위험 이미지:", sum(r["gt_hazard"] for r in image_results))
print("정상 이미지:", sum(not r["gt_hazard"] for r in image_results))

print("\n[후보 단위 상태 분포]")
counts = Counter(r["evidence_status"] for r in item_results)
for status, count in counts.most_common():
    print(f"{status:20s} {count:5d}")

print("\n[이미지 단위 상태별 분포]")
print("※ 한 이미지에 여러 상태가 있을 수 있어 합계가 1,256을 넘을 수 있음")

for status in status_names + ["no_candidate"]:
    key = "no_candidate" if status == "no_candidate" else "has_" + status

    hazard_count = sum(
        r["gt_hazard"] and r[key] for r in image_results
    )
    normal_count = sum(
        not r["gt_hazard"] and r[key] for r in image_results
    )

    print(
        f"{status:20s} "
        f"위험 {hazard_count:4d} | 정상 {normal_count:4d}"
    )

correct_candidate_images = sum(
    r["has_gt_code_candidate"] for r in image_results
)

print("\n[참고: 정답 코드 후보 포함]")
print(
    f"{correct_candidate_images}/"
    f"{sum(r['gt_hazard'] for r in image_results)}"
)
print("※ 위험 확정 정확도가 아니라 후보에 정답 코드가 포함된 비율")

print("\n[코드 매핑 누락 진단]")
print(dict(missing_code))

print("\n[결과 파일]")
print(OUT / "fusion_item_details.csv")
print(OUT / "fusion_image_details.csv")

print("\n주의:")
print("- YOLO 신뢰도 0.25 이상으로 저장된 입력을 그대로 사용")
print("- 기존 운영 fuse_evidence() 자체에는 추가 confidence/area 필터 없음")
print("- GT는 fuse_evidence() 실행 후 분석에만 사용")
print("- 이 실험은 운영 입력의 근사 재구성으로 실제 서비스 성능을 확정하지 않음")
