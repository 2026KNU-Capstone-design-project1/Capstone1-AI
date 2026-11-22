import argparse
import os
import random
from collections import Counter
from pathlib import Path

from spatial_relations import OBJECTS, load_objects

SPLITS = ("train", "val", "test")


def draw_samples(root, items, names, count):
    """변환된 박스를 사진 위에 그려 dataset/objects/check/ 에 저장한다."""
    import cv2

    out = root / "check"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    picked = random.Random(0).sample(items, min(count, len(items)))
    for img_path, lines in picked:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        for line in lines:
            c, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            p0, p1 = (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2))
            cv2.rectangle(img, p0, p1, (0, 200, 0), 4)
            cv2.putText(img, names[int(c)], (p0[0], max(30, p0[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)
        cv2.imwrite(str(out / img_path.name), img)
    print(f"확인용 그림 {len(picked)}장: {out}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="split_data.py 로 나눈 images/, labels/ 가 있는 폴더")
    parser.add_argument("--no-people", action="store_true", help="작업자 · 신호수 객체를 학습에서 제외")
    parser.add_argument("--draw", type=int, default=0, help="변환 결과를 박스로 그려 확인할 장수")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    root = dataset / "objects"
    names = sorted(c for c in OBJECTS if not (args.no_people and OBJECTS[c][1] == "사람"))
    classes = {c: i for i, c in enumerate(names)}

    boxes, all_items = Counter(), []
    print(f"객체 클래스: {len(names)}개 ({'사람 제외' if args.no_people else '사람 포함'})\n")
    print(f"{'구분':5s} {'사진':>6s} {'객체 있음':>8s} {'박스':>7s}")
    for split in SPLITS:
        img_dir, lbl_dir = dataset / "images" / split, dataset / "labels" / split
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            raise SystemExit(f"폴더가 없어요: {img_dir}\n먼저 split_data.py 를 --run 으로 실행하세요.")

        # 사진 폴더 바로가기 (YOLO 는 .../images/... → .../labels/... 로 라벨 위치를 찾는다)
        link = root / "images" / split
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            os.symlink(os.path.relpath(img_dir, link.parent), link)
        out_dir = root / "labels" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.txt"):
            old.unlink()

        images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        listed, with_obj, n_box = [], 0, 0
        for json_path in sorted(lbl_dir.glob("*.json")):
            img = images.get(json_path.stem)
            if img is None:
                continue
            _, _, objects = load_objects(json_path)
            lines = []
            for o in objects:
                if o["code"] not in classes:
                    continue
                x, y, w, h = o["bbox"]
                lines.append(f"{classes[o['code']]} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")
            (out_dir / f"{json_path.stem}.txt").write_text("".join(f"{l}\n" for l in lines), encoding="utf-8")
            listed.append(f"./images/{split}/{img.name}")
            with_obj += bool(lines)
            n_box += len(lines)
            boxes.update(int(l.split()[0]) for l in lines)
            if lines:
                all_items.append((img, lines))
        (root / f"yolo_{split}.txt").write_text("\n".join(listed) + "\n", encoding="utf-8")
        print(f"{split:5s} {len(listed):6d} {with_obj:8d} {n_box:7d}")

    yaml = [f"path: {root}", "train: yolo_train.txt", "val: yolo_val.txt", "test: yolo_test.txt", "", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(names)]
    (root / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    print("\n클래스별 박스 수 (전체):")
    for i, c in enumerate(names):
        print(f"  {i:2d} {boxes[i]:6d}  {c} {OBJECTS[c][0]} ({OBJECTS[c][1]})")
    empty = [names[i] for i in range(len(names)) if not boxes[i]]
    if empty:
        print(f"⚠️ 박스가 하나도 없는 클래스 (데이터에 없음): {empty}")
    print(f"\n완료! 박스 {sum(boxes.values())}개 · 설정 파일: {root / 'data.yaml'}")

    if args.draw:
        draw_samples(root, all_items, names, args.draw)


if __name__ == "__main__":
    main()
