"""
객체 간 공간관계 분석

사진 한 장의 객체(시설물 · 위험 대상 · 장비 ...) 위치로 객체 사이의 공간관계를 계산하고,
AI Hub 시나리오 정의로 만든 참고 규칙으로 위험 근거를 찾는다.
  - 공간관계 : 포함 · 중첩 · 인접 · 이격 (+ 겹침 비율, 거리, 위/아래)
  - 연속성   : 같은 난간 · 펜스 조각 사이가 끊겼는지
  - 결여     : 있어야 할 안전 객체가 없음        (예: 개구부인데 안전난간 · 덮개 없음 → N-04)
  - 위험 인접 : 함께 있으면 안 되는 객체가 가까이 있음 (예: 용접기 옆 단열재 → N-34)
  - 상태 객체 : 객체 자체가 불량 상태로 라벨된 경우  (예: 안전난간이 미설치된 단부 → N-05)
  - 판단 원인 : findings 가 왜 그렇게 나왔는지 (evaluation_status · reason_code · 위험별 checks)
               rule_matched / condition_not_met / rule_not_supported / missing_objects / insufficient_evidence

지금은 AI Hub 라벨의 객체 다각형(정답)을 입력으로 쓴다.
객체 탐지 YOLO 를 학습한 뒤에는 analyze() 에 YOLO 결과만 넣으면 된다.

사람 객체(작업자 · 신호수)는 기본으로 계산에서 뺀다 (--people 로 바꿀 수 있음).
사람이 필요한 규칙(예: 신호수 미배치)은 "판단 불가"로 남긴다.

사용법:
    python spatial_relations.py dataset                    # test 세트
    python spatial_relations.py dataset --split all        # train · val · test 모두
    python spatial_relations.py dataset --people signal    # 신호수까지 포함
결과: output/relations/<split>.jsonl (사진별), output/relations/<split>_summary.csv (규칙별 검증)
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from detector import HAZARDS

# AI Hub 데이터설명서 V1.1 의 객체 클래스 → (이름, 분류)
OBJECTS = {
    "WO-01": ("작업자", "사람"), "WO-02": ("신호수", "사람"), "WO-23": ("용접면을 쓴 작업자", "사람"),
    "WO-03": ("사다리", "장비"), "WO-04": ("안전난간이 설치된 이동식비계", "장비"),
    "WO-05": ("안전난간이 설치되지 않은 이동식비계", "장비"), "WO-06": ("레미콘", "장비"),
    "WO-07": ("굴삭기", "장비"), "WO-08": ("전도방지대가 설치된 이동식크레인", "장비"),
    "WO-09": ("과상승방지봉이 미설치된 시저형고소작업대", "장비"),
    "WO-10": ("과상승방지봉이 설치된 시저형고소작업대", "장비"), "WO-11": ("말비계", "장비"),
    "WO-12": ("전도방지대가 설치된 콘크리트펌프카", "장비"), "WO-13": ("전도방지대가 미설치된 콘크리트펌프카", "장비"),
    "WO-14": ("지게차", "장비"), "WO-15": ("덤프트럭", "장비"), "WO-16": ("항타기", "장비"),
    "WO-17": ("롤러", "장비"), "WO-18": ("타워크레인", "장비"), "WO-19": ("전도방지대가 설치된 사다리차", "장비"),
    "WO-20": ("전도방지대가 설치된 오거크레인", "장비"), "WO-21": ("전도방지대가 미설치된 오거크레인", "장비"),
    "WO-22": ("화물트럭", "장비"),
    "SO-01": ("개구부 안전난간", "시설물"), "SO-02": ("안전난간이 설치된 단부", "시설물"),
    "SO-03": ("정상 작업발판", "시설물"), "SO-04": ("비정상 작업발판", "시설물"),
    "SO-05": ("해치가 닫혀있는 개폐형 작업발판", "시설물"), "SO-06": ("해치가 열려있는 개폐형 작업발판", "시설물"),
    "SO-07": ("개구부 덮개", "시설물"), "SO-08": ("수직방지망", "시설물"), "SO-09": ("낙하물방지망", "시설물"),
    "SO-11": ("출입구 방호선반", "시설물"), "SO-12": ("거푸집", "시설물"),
    "SO-13": ("쐐기목이 미설치된 동바리 받침판", "시설물"), "SO-14": ("안전난간이 미설치된 단부", "시설물"),
    "SO-15": ("쐐기목이 미설치된 버팀대", "시설물"), "SO-16": ("쐐기목이 설치된 버팀대", "시설물"),
    "SO-17": ("쐐기목이 설치된 동바리 받침판", "시설물"), "SO-18": ("낙하물방지망 설치 미흡", "시설물"),
    "SO-19": ("개구부 덮개 미흡", "시설물"),
    "SO-20": ("불티비산방지막", "공구"), "SO-21": ("우레탄폼건", "공구"), "SO-22": ("에어스프레이건", "공구"),
    "SO-23": ("산소절단기", "공구"), "SO-24": ("용접기", "공구"), "SO-25": ("원형톱", "공구"),
    "SO-26": ("공구박스", "공구"), "SO-27": ("롤러브러쉬", "공구"), "SO-28": ("불꽃이 발생하는 원형톱", "공구"),
    "SO-30": ("페인트통", "자재"), "SO-31": ("시멘트포대", "자재"), "SO-32": ("단열재", "자재"),
    "SO-33": ("관", "자재"), "SO-34": ("벽돌", "자재"),
    "SO-35": ("PE 안전펜스", "안전보조장비"), "SO-36": ("고체연료", "안전보조장비"),
    "SO-37": ("철제인양박스", "안전보조장비"), "SO-38": ("항공마대", "안전보조장비"),
    "SO-39": ("이동형환풍기", "안전보조장비"), "SO-40": ("소화기", "안전보조장비"),
    "SO-41": ("라바콘", "안전보조장비"), "SO-42": ("유도봉", "안전보조장비"), "SO-43": ("방독면", "안전보조장비"),
    "SO-44": ("가스경보기", "안전보조장비"), "SO-45": ("타포린형 안전표지판", "안전보조장비"),
    "SO-46": ("위험테이프", "안전보조장비"), "SO-47": ("벤딩형 가림막펜스", "안전보조장비"),
    "DO-01": ("위험물저장소", "위험 대상"), "DO-02": ("수직방지망 설치불량인 갱폼", "위험 대상"),
    "DO-03": ("개구부", "위험 대상"), "DO-04": ("출입구", "위험 대상"), "DO-06": ("방수페인트", "위험 대상"),
}
PEOPLE = {"none": set(), "signal": {"WO-02"}, "all": {"WO-01", "WO-02", "WO-23"}}

# ── 참고 규칙: 데이터설명서의 정상(Y) · 비정상(N) 시나리오 쌍에서 만든 것 (규칙 엔진에서 조정) ──
# 상태 객체: 이 객체가 있으면 그 자체로 위험
STATE_OBJECTS = {
    "SO-04": "N-01", "SO-06": "N-02", "SO-19": "N-03", "SO-14": "N-05", "DO-02": "N-06", "WO-05": "N-07",
    "SO-18": "N-14", "WO-09": "N-23", "SO-13": "N-41", "SO-15": "N-42", "WO-13": "N-45", "WO-21": "N-47",
}
# 결여: 기준 객체가 있는데 필요한 객체가 하나도 없으면 위험
REQUIRED = [
    ({"DO-03"}, {"SO-01", "SO-07", "SO-19"}, "N-04"),   # 개구부 → 안전난간 · 덮개
    ({"DO-04"}, {"SO-11"}, "N-13"),                     # 출입구 → 방호선반
    ({"WO-19"}, {"WO-02"}, "N-18"),                     # 사다리차 → 신호수
    ({"WO-08"}, {"WO-02"}, "N-19"),                     # 이동식크레인 → 신호수
    ({"WO-07"}, {"WO-02"}, "N-27"),                     # 굴삭기 → 신호수 (N-21 · N-22 도 같은 형태)
    ({"WO-15"}, {"WO-02"}, "N-25"),                     # 덤프트럭 → 신호수
    ({"WO-06"}, {"WO-02"}, "N-29"),                     # 레미콘 → 신호수
    ({"WO-12"}, {"WO-02"}, "N-46"),                     # 펌프카 → 신호수
    ({"WO-20"}, {"WO-02"}, "N-48"),                     # 오거크레인 → 신호수
    ({"WO-14"}, {"WO-02"}, "N-49"),                     # 지게차 → 신호수
    ({"SO-22"}, {"SO-39"}, "N-31"),                     # 에어스프레이건 → 이동형환풍기
    ({"SO-24"}, {"SO-40"}, "N-33"),                     # 용접기 → 소화기
    ({"SO-36"}, {"SO-40"}, "N-35"),                     # 고체연료(난로) → 소화기
    ({"SO-23"}, {"SO-40"}, "N-36"),                     # 산소절단기 → 소화기
    ({"DO-01"}, {"SO-40"}, "N-37"),                     # 위험물저장소 → 소화기
    ({"SO-25", "SO-28"}, {"SO-40"}, "N-39"),            # 원형톱 → 소화기
]
# 위험 인접: 기준 객체 근처(gap ≤ NEAR)에 위험 객체가 있으면 위험
HAZARD_NEAR = [
    ({"SO-24"}, {"SO-32"}, "N-34"),                     # 용접기 옆 단열재
    ({"DO-01"}, {"SO-32"}, "N-38"),                     # 위험물저장소 옆 단열재
    ({"SO-25", "SO-28"}, {"SO-32"}, "N-40"),            # 원형톱 옆 단열재
    ({"SO-21"}, {"SO-36"}, "N-32"),                     # 우레탄폼건(실내마감) 옆 고체연료(난로)
    ({"WO-08"}, {"SO-38"}, "N-20"),                     # 크레인 + 항공마대 (철제인양박스 미사용)
    ({"WO-11"}, {"SO-32"}, "N-44"),                     # 말비계 + 적재물(단열재) 위 설치
]
# 연속성을 볼 객체 (여러 조각으로 나뉘어 있을 때 사이 간격)
CONTINUOUS = {"SO-01", "SO-02", "SO-08", "SO-09", "SO-35", "SO-46", "SO-47"}

CONTAIN = 0.8   # 작은 객체의 80% 이상이 큰 객체 안 → 포함
ADJACENT = 0.05  # 박스 사이 거리가 화면 대각선의 5% 이하 → 인접
NEAR = 0.15      # 위험 인접 규칙의 거리 기준 (화면 대각선 대비)
GAP_BREAK = 0.02  # 연속 객체 조각 사이 간격이 화면 너비의 2% 초과 → 끊김


def box_relation(a, b, aspect):
    """두 박스(x, y, w, h · 0~1)의 공간관계. aspect = 가로/세로 비 (거리를 실제 비율로 재기 위해)."""
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0.0, min(ax1, bx1) - max(a[0], b[0])), max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    area_a, area_b = a[2] * a[3], b[2] * b[3]
    a_in_b = inter / area_a if area_a else 0.0
    b_in_a = inter / area_b if area_b else 0.0
    # 박스 사이 빈 거리 (겹치면 0) · 화면 대각선 = 1 로 정규화
    dx = max(0.0, max(a[0], b[0]) - min(ax1, bx1)) * aspect
    dy = max(0.0, max(a[1], b[1]) - min(ay1, by1))
    gap = (dx ** 2 + dy ** 2) ** 0.5 / (aspect ** 2 + 1) ** 0.5
    if max(a_in_b, b_in_a) >= CONTAIN:
        kind = "포함"
    elif inter > 0:
        kind = "중첩"
    elif gap <= ADJACENT:
        kind = "인접"
    else:
        kind = "이격"
    return {
        "type": kind,
        "iou": round(inter / (area_a + area_b - inter), 4) if inter else 0.0,
        "a_in_b": round(a_in_b, 4),
        "b_in_a": round(b_in_a, 4),
        "gap": round(gap, 4),
        # a 가 b 위에 있음: 가로로 겹치고 a 의 아래쪽 절반이 b 의 위쪽보다 위
        "a_above_b": bool(iw > 0 and a[1] + a[3] / 2 < b[1]),
    }


def continuity(objects):
    """같은 연속 객체(난간 · 펜스 · 방지망)가 여러 조각이면 가로 방향 끊김을 본다."""
    out = []
    by_code = defaultdict(list)
    for o in objects:
        if o["code"] in CONTINUOUS:
            by_code[o["code"]].append(o)
    for code, segs in by_code.items():
        if len(segs) < 2:
            continue
        segs = sorted(segs, key=lambda o: o["bbox"][0])
        gaps = [max(0.0, b["bbox"][0] - (a["bbox"][0] + a["bbox"][2])) for a, b in zip(segs, segs[1:])]
        out.append({"code": code, "name": OBJECTS[code][0], "segments": len(segs),
                    "max_gap": round(max(gaps), 4), "broken": max(gaps) > GAP_BREAK})
    return out


RULE_CODES = set(STATE_OBJECTS.values()) | {c for *_, c in REQUIRED} | {c for *_, c in HAZARD_NEAR}

# 규칙 판단 상태 · 원인 (findings 가 비었을 때 왜 비었는지 구분)
#   evaluation_status: hazard(규칙이 위험 확인) / no_hazard(규칙 적용, 조건 불충족) / not_evaluable(판단 불가)
#   reason_code:
#     rule_matched          규칙 조건 충족 → 위험
#     condition_not_met     필요한 객체는 다 탐지됐지만 규칙 조건 불충족 (정상일 가능성)
#     rule_not_supported    그 위험에 대한 규칙이 없음
#     missing_objects       규칙에 필요한 객체가 탐지되지 않음
#     insufficient_evidence 판단 근거 부족 (필요한 객체를 계산에서 뺐거나, 객체가 하나도 없음)
NOT_EVALUABLE_ORDER = ("missing_objects", "rule_not_supported", "insufficient_evidence")


def check_rule(code, present, findings, gaps, objs, detected):
    """위험 코드 하나에 대해 규칙으로 판단한 결과와 그 원인."""
    out = {"code": code, "hazard": HAZARDS[code][0] if code in HAZARDS else code, "detected_by_model": detected}
    mine = [f for f in findings if f["code"] == code]
    if any(f["status"] == "위험" for f in mine):
        out.update(status="hazard", reason_code="rule_matched")
    elif mine:  # 판단 불가 (필요한 객체를 계산에서 뺌)
        out.update(status="not_evaluable", reason_code="insufficient_evidence",
                   missing=sorted({c for f in mine for c in f.get("missing", [])}))
    elif code not in RULE_CODES:
        out.update(status="not_evaluable", reason_code="rule_not_supported")
    else:
        ready, missing, gap = False, set(), None
        for obj, n in STATE_OBJECTS.items():
            if n == code:  # 상태 객체 규칙: 그 불량 객체가 보여야 판단 가능
                missing.add(obj)
        for anchors, _, n in REQUIRED:
            if n == code:
                if anchors & present:
                    ready = True
                else:
                    missing |= anchors
        for anchors, risky, n in HAZARD_NEAR:
            if n == code:
                if anchors & present and risky & present:
                    ready = True
                    ids_a = [o["id"] for o in objs if o["code"] in anchors]
                    ids_b = [o["id"] for o in objs if o["code"] in risky]
                    gap = min(gaps[(a, b)] for a in ids_a for b in ids_b)
                else:
                    missing |= (anchors - present if not anchors & present else set()) | \
                               (risky - present if not risky & present else set())
        if ready:
            out.update(status="no_hazard", reason_code="condition_not_met")
            if gap is not None:
                out["gap"] = gap  # 위험 인접 기준(NEAR)보다 멀었던 거리
        else:
            out.update(status="not_evaluable", reason_code="missing_objects", missing=sorted(missing))
    # 위험 상태 모델과 규칙 판단이 일치하는지 (모델이 탐지한 위험만, 판단 불가면 null)
    out["agrees_with_model"] = (out["status"] == "hazard") if detected and out["status"] != "not_evaluable" else None
    return out


def evaluate(objs, findings, gaps, hazard_codes=()):
    """findings 가 왜 그렇게 나왔는지: 위험별 판단(checks) + 장면 전체 요약(evaluation_status · reason_code).

    hazard_codes: 위험 상태 모델이 이 장면에서 탐지한 위험 코드 (없으면 규칙이 찾은 위험만 본다)
    """
    present = {o["code"] for o in objs}
    detected = list(dict.fromkeys(c for c in hazard_codes if c))
    codes = detected + [c for c in dict.fromkeys(f["code"] for f in findings) if c not in detected]
    checks = [check_rule(c, present, findings, gaps, objs, c in detected) for c in codes]

    statuses = [c["status"] for c in checks]
    if "hazard" in statuses:
        status, reason = "hazard", "rule_matched"
    elif "not_evaluable" in statuses:
        reasons = [c["reason_code"] for c in checks if c["status"] == "not_evaluable"]
        status, reason = "not_evaluable", next(r for r in NOT_EVALUABLE_ORDER if r in reasons)
    elif checks:
        status, reason = "no_hazard", "condition_not_met"
    elif objs:  # 모델도 규칙도 위험을 찾지 않음
        status, reason = "no_hazard", "condition_not_met"
    else:
        status, reason = "not_evaluable", "insufficient_evidence"
    return {"evaluation_status": status, "reason_code": reason, "checks": checks}


def analyze(objects, aspect=16 / 9, people="none", hazard_codes=()):
    """객체 목록 [{code, bbox(x, y, w, h · 0~1)}] → 공간관계 · 연속성 · 규칙 판단.

    YOLO 결과도 같은 형식으로 넣으면 된다 (confidence 가 있으면 그대로 남긴다).
    hazard_codes: 위험 상태 모델이 탐지한 위험 코드 → 위험별로 규칙 판단 원인을 붙인다 (evaluate 참고)
    """
    allowed_people = PEOPLE[people]
    objs = [dict(o, id=i, name=OBJECTS.get(o["code"], (o["code"], ""))[0],
                 category=OBJECTS.get(o["code"], ("", "기타"))[1])
            for i, o in enumerate(
                o for o in objects
                if OBJECTS.get(o["code"], ("", ""))[1] != "사람" or o["code"] in allowed_people)]

    relations = []
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            relations.append({"a": a["id"], "b": b["id"], **box_relation(a["bbox"], b["bbox"], aspect)})

    codes = {o["code"] for o in objs}
    gap = {(r["a"], r["b"]): r["gap"] for r in relations}
    gap.update({(b, a): g for (a, b), g in list(gap.items())})
    findings = []
    for o in objs:
        if o["code"] in STATE_OBJECTS:
            findings.append({"code": STATE_OBJECTS[o["code"]], "type": "상태 객체", "status": "위험", "objects": [o["id"]]})
    for anchors, needed, n_code in REQUIRED:
        for o in (o for o in objs if o["code"] in anchors):
            if needed & codes:
                continue
            # 필요한 객체가 사람인데 사람을 계산에서 뺐으면 판단할 수 없다
            unknown = bool(needed - set(OBJECTS)) or all(OBJECTS[c][1] == "사람" and c not in allowed_people for c in needed)
            findings.append({"code": n_code, "type": "결여", "status": "판단 불가" if unknown else "위험",
                             "objects": [o["id"]], "missing": sorted(needed)})
    for anchors, risky, n_code in HAZARD_NEAR:
        for a in (o for o in objs if o["code"] in anchors):
            for b in (o for o in objs if o["code"] in risky):
                if gap[(a["id"], b["id"])] <= NEAR:
                    findings.append({"code": n_code, "type": "위험 인접", "status": "위험",
                                     "objects": [a["id"], b["id"]], "gap": gap[(a["id"], b["id"])]})
    for f in findings:
        f["hazard"] = HAZARDS[f["code"]][0] if f["code"] in HAZARDS else f["code"]
    return {"objects": objs, "relations": relations, "continuity": continuity(objs), "findings": findings,
            **evaluate(objs, findings, gap, hazard_codes)}


def load_objects(json_path):
    """AI Hub 라벨 json → (상황 코드, 가로/세로 비, 객체 목록). 객체 다각형은 감싸는 박스로 바꾼다."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    width, height = (float(v) for v in data["Raw_Data_Info."]["Resolution"].split(","))
    objects = []
    for ann in data["Learning_Data_Info."]["Annotations"]:
        if ann["type"] != "polygon" or ann["class_ID"] not in OBJECTS:
            continue
        xs, ys = ann["value"][0::2], ann["value"][1::2]
        x0, y0 = max(0.0, min(xs)), max(0.0, min(ys))
        x1, y1 = min(width, max(xs)), min(height, max(ys))
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        objects.append({"code": ann["class_ID"],
                        "bbox": [round(x0 / width, 4), round(y0 / height, 4),
                                 round((x1 - x0) / width, 4), round((y1 - y0) / height, 4)]})
    return data["Raw_Data_Info."]["Situation_ID"], width / height, objects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="labels/<split>/*.json 이 있는 폴더")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--people", default="none", choices=list(PEOPLE),
                        help="사람 객체 포함: none(제외) / signal(신호수만) / all(작업자 포함)")
    parser.add_argument("--out", type=Path, default=Path("output/relations"))
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"{args.split}" + ("" if args.people == "none" else f"_people-{args.people}")
    out_path = args.out / f"{name}.jsonl"

    rel_types, status = Counter(), Counter()
    hit = defaultdict(Counter)  # N 코드 → {N 사진에서 맞춤, N 사진 수, Y 사진에서 잘못 판단, 판단 불가}
    images = 0
    overall = Counter()  # 사진 단위: 위험 사진을 위험으로 / 정상 사진을 위험으로 판단한 수
    with open(out_path, "w", encoding="utf-8") as f:
        for split in splits:
            for json_path in sorted((args.dataset / "labels" / split).glob("*.json")):
                situation, aspect, objects = load_objects(json_path)
                if situation[:2] not in ("N-", "Y-"):
                    continue
                result = analyze(objects, aspect, args.people)
                f.write(json.dumps({"image": json_path.stem, "split": split, "situation": situation, **result},
                                   ensure_ascii=False) + "\n")
                images += 1
                rel_types.update(r["type"] for r in result["relations"])
                flagged = {x["code"] for x in result["findings"] if x["status"] == "위험"}
                unknown = {x["code"] for x in result["findings"] if x["status"] == "판단 불가"}
                status.update(x["status"] for x in result["findings"])
                # 규칙 검증: N-xx 사진은 그 코드로 판단했는지, 같은 번호의 Y-xx 사진은 잘못 판단했는지
                n_code = "N-" + situation[2:]
                kind = "N" if situation.startswith("N-") else "Y"
                overall[kind] += 1
                overall[kind + " 위험 판단"] += bool(flagged)
                if situation.startswith("N-"):
                    hit[n_code]["N 사진"] += 1
                    hit[n_code]["맞춤"] += n_code in flagged
                    hit[n_code]["판단 불가"] += n_code in unknown
                else:
                    hit[n_code]["Y 사진"] += 1
                    hit[n_code]["오경보"] += n_code in flagged

    rule_codes = sorted(set(STATE_OBJECTS.values()) | {c for *_, c in REQUIRED} | {c for *_, c in HAZARD_NEAR})
    summary_path = args.out / f"{name}_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["code", "hazard", "rule", "N 사진", "맞춤", "재현율", "판단 불가", "Y 사진", "오경보", "오경보율"])
        print(f"사진 {images}장 · 사람 객체: {args.people}\n")
        print(f"{'코드':5s} {'규칙':6s} {'재현율':>6s} {'오경보율':>7s} {'판단불가':>6s}  위험 이름")
        for code in rule_codes:
            h = hit[code]
            rule = ("상태 객체" if code in STATE_OBJECTS.values() else
                    "결여" if code in {c for *_, c in REQUIRED} else "위험 인접")
            recall = h["맞춤"] / h["N 사진"] if h["N 사진"] else 0.0
            false_rate = h["오경보"] / h["Y 사진"] if h["Y 사진"] else 0.0
            w.writerow([code, HAZARDS.get(code, ("",))[0], rule, h["N 사진"], h["맞춤"], round(recall, 3),
                        h["판단 불가"], h["Y 사진"], h["오경보"], round(false_rate, 3)])
            print(f"{code:5s} {rule:6s} {recall:6.2f} {false_rate:7.2f} {h['판단 불가']:6d}  {HAZARDS.get(code, ('',))[0]}")
    if overall["N"] and overall["Y"]:
        print(f"\n사진 단위: 위험(N) 사진 중 위험 판단 {overall['N 위험 판단'] / overall['N']:.1%}"
              f" · 정상(Y) 사진 중 위험 판단(오경보) {overall['Y 위험 판단'] / overall['Y']:.1%}")
    print(f"\n공간관계 종류: {dict(rel_types)}")
    print(f"판단 결과: {dict(status)}")
    print(f"\n저장: {out_path}\n      {summary_path}")


if __name__ == "__main__":
    main()
