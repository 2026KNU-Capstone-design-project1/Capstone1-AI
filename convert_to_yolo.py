import argparse
import json
import random
from collections import Counter
from pathlib import Path

from detector import HAZARDS

SPLITS = ("train", "val", "test")

# 위험 그룹 (YOLO 그림 글꼴 문제를 피하려고 이름은 영어)
GROUPS = {
    "fall_protection": ["N-03", "N-04", "N-05", "N-06", "N-07"],              # 추락방지 시설
    "scaffold": ["N-01", "N-02", "N-11", "N-41", "N-42", "N-43", "N-44"],     # 비계·가설구조물
    "falling_object": ["N-13", "N-14", "N-15"],                               # 낙하방지 시설
    "equipment": [f"N-{i}" for i in [*range(17, 31), *range(45, 51)]],        # 건설장비
    "fire": [f"N-{i}" for i in range(31, 41)],                                # 화재 예방
}


def class_table(group):
    """코드 → (클래스 번호), 클래스 이름 목록."""
    if group:
        names = list(GROUPS)
        return {c: i for i, g in enumerate(names) for c in GROUPS[g]}, names
    names = sorted(HAZARDS)
    return {c: i for i, c in enumerate(names)}, names


def to_yolo(box, width, height):
    """AI Hub bbox [x, y, w, h] (px, 왼쪽 위 기준) → YOLO (x중심, y중심, w, h) 0~1. 사진 밖은 잘라낸다."""
    x, y, w, h = box
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return ((x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height)


def convert(json_path, classes, keep_normal):
    """json 하나를 변환한다. (YOLO 줄 목록 또는 None=제외, 사유/코드)"""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    width, height = (float(v) for v in data["Raw_Data_Info."]["Resolution"].split(","))
    situation = data["Raw_Data_Info."]["Situation_ID"]

    if situation.startswith("Y-"):
        return ([], "정상") if keep_normal else (None, "정상 제외 (--no-normal)")
    if not situation.startswith("N-"):
        return None, f"범위 밖 ({situation.split('-')[0]})"
    if situation not in classes:
        return None, "범위 밖 (N 작업자 관련)"

    lines = []
    for ann in data["Learning_Data_Info."]["Annotations"]:
        code = ann["class_ID"]
        if ann["type"] != "bbox" or code not in classes:
            continue
        box = to_yolo(ann["value"], width, height)
        if box:
            lines.append(f"{classes[code]} " + " ".join(f"{v:.6f}" for v in box))
    if not lines:
        return None, "박스 없음"
    return lines, situation


def draw_samples(dataset, items, names, count):
    """변환된 txt 를 사진 위에 다시 그려서 dataset/yolo_check/ 에 저장한다."""
    import cv2

    out = dataset / "yolo_check"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    with_box = [it for it in items if it[2]]
    for img_path, _, lines in random.Random(0).sample(with_box, min(count, len(with_box))):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        for line in lines:
            c, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            p0, p1 = (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2))
            cv2.rectangle(img, p0, p1, (0, 0, 255), 4)
            cv2.putText(img, names[int(c)], (p0[0], max(30, p0[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imwrite(str(out / img_path.name), img)
    print(f"확인용 그림 {min(count, len(with_box))}장: {out}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="images/, labels/ 가 있는 폴더 (split_data.py 실행 후)")
    parser.add_argument("--group", action="store_true", help="세부 코드 대신 위험 그룹 5개로 학습")
    parser.add_argument("--no-normal", action="store_true", help="정상(Y) 사진을 학습에서 제외")
    parser.add_argument("--draw", type=int, default=0, help="변환 결과를 박스로 그려 확인할 장수")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    classes, names = class_table(args.group)
    for split in SPLITS:
        if not (dataset / "labels" / split).is_dir():
            raise SystemExit(f"폴더가 없어요: {dataset / 'labels' / split}\n먼저 split_data.py 를 --run 으로 실행하세요.")

    all_items = []
    boxes = Counter()  # 클래스 번호 → 박스 수
    print(f"클래스 방식: {'위험 그룹' if args.group else '세부 코드'} ({len(names)}개)\n")
    print(f"{'구분':5s} {'학습 사진':>8s} {'위험':>6s} {'정상':>6s} {'제외':>6s}")
    for split in SPLITS:
        img_dir, lbl_dir = dataset / "images" / split, dataset / "labels" / split
        images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        for old in lbl_dir.glob("*.txt"):  # 이전 변환 결과 지우기
            old.unlink()

        listed, reasons = [], Counter()
        for json_path in sorted(lbl_dir.glob("*.json")):
            img = images.get(json_path.stem)
            if img is None:
                reasons["사진 없음"] += 1
                continue
            lines, reason = convert(json_path, classes, not args.no_normal)
            reasons[reason if lines is None else ("위험" if lines else "정상")] += 1
            if lines is None:
                continue
            json_path.with_suffix(".txt").write_text("".join(f"{l}\n" for l in lines), encoding="utf-8")
            listed.append(f"./images/{split}/{img.name}")
            all_items.append((img, split, lines))
            boxes.update(int(l.split()[0]) for l in lines)

        # YOLO 가 읽을 사진 목록 (data.yaml 기준 상대 경로)
        (dataset / f"yolo_{split}.txt").write_text("\n".join(listed) + "\n", encoding="utf-8")
        excluded = sum(n for r, n in reasons.items() if r not in ("위험", "정상"))
        print(f"{split:5s} {len(listed):8d} {reasons['위험']:6d} {reasons['정상']:6d} {excluded:6d}")
        for r, n in sorted(reasons.items()):
            if r not in ("위험", "정상"):
                print(f"        └ 제외: {r} {n}장")

    yaml = [f"path: {dataset}", "train: yolo_train.txt", "val: yolo_val.txt", "test: yolo_test.txt", "", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(names)]
    (dataset / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    print("\n클래스별 박스 수:")
    for i, n in enumerate(names):
        label = n if args.group else f"{n} {HAZARDS[n][0]}"
        print(f"  {i:2d} {boxes[i]:5d}  {label}")
    missing = [names[i] for i in range(len(names)) if not boxes[i]]
    if missing:
        print(f"⚠️ 박스가 하나도 없는 클래스: {missing}")
    print(f"\n완료! 박스 {sum(boxes.values())}개 · 설정 파일: {dataset / 'data.yaml'}")

    if args.draw:
        draw_samples(dataset, all_items, names, args.draw)


if __name__ == "__main__":
    main()
