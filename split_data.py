import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
LABEL_EXTS = {".txt", ".json"}

# H-220803_B31_N-11_006_0001 → 현장: H-220803_B31_N-11_006, 상황 코드: N-11
GROUP_RE = re.compile(r"^(.*)_\d+$")
CODE_RE = re.compile(r"_([A-Z]+-\d+)_")


def collect(folder, exts):
    files = {}
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and p.parent.name not in RATIOS:
            if p.stem in files:
                raise RuntimeError(f"이름이 겹치는 파일이 있어요: {files[p.stem]} / {p}")
            files[p.stem] = p
    return files


def pick_split(counts, size):
    """지금까지 나눈 전체 사진 수 기준으로 목표 비율보다 가장 모자란 split 을 고른다."""
    done = sum(counts.values()) + size
    return max(RATIOS, key=lambda n: RATIOS[n] * done - counts[n])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--run", action="store_true", help="실제로 파일을 옮긴다")
    args = parser.parse_args()

    img_dir = args.dataset / "images"
    lbl_dir = args.dataset / "labels"
    for d in (img_dir, lbl_dir):
        if not d.is_dir():
            raise SystemExit(f"폴더가 없어요: {d}")
        for name in RATIOS:
            if (d / name).exists() and any((d / name).iterdir()):
                raise SystemExit(f"이미 나눠진 파일이 있어요: {d / name}")

    images = collect(img_dir, IMAGE_EXTS)
    labels = collect(lbl_dir, LABEL_EXTS)

    pairs = sorted(images.keys() & labels.keys())
    only_img = sorted(images.keys() - labels.keys())
    only_lbl = sorted(labels.keys() - images.keys())
    print(f"짝이 맞는 쌍: {len(pairs)}")
    if only_img:
        print(f"라벨 없는 사진 {len(only_img)}개 (옮기지 않음): {only_img[:5]} ...")
    if only_lbl:
        print(f"사진 없는 라벨 {len(only_lbl)}개 (옮기지 않음): {only_lbl[:5]} ...")
    if not pairs:
        raise SystemExit("짝이 맞는 파일이 없어요.")

    # 상황 코드 → 현장 → 파일 이름
    by_code = defaultdict(lambda: defaultdict(list))
    for stem in pairs:
        m = CODE_RE.search(stem)
        code = m.group(1) if m else "기타"
        g = GROUP_RE.match(stem)
        by_code[code][g.group(1) if g else stem].append(stem)

    rng = random.Random(SEED)
    assign = {}
    counts = {name: 0 for name in RATIOS}
    for code in sorted(by_code):
        groups = [by_code[code][k] for k in sorted(by_code[code])]
        rng.shuffle(groups)
        for stems in groups:
            name = pick_split(counts, len(stems))
            counts[name] += len(stems)
            for stem in stems:
                assign[stem] = name

    for name in RATIOS:
        print(f"{name:5s}: {counts[name]:6d}장 ({counts[name] / len(pairs):.1%})")

    if not args.run:
        print("\n미리 보기만 했어요. 실제로 옮기려면 --run 을 붙여서 다시 실행하세요.")
        return

    for stem, name in assign.items():
        for src, root in ((images[stem], img_dir), (labels[stem], lbl_dir)):
            dst = root / name / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    # 확인: 각 split 에서 사진과 라벨 이름이 정확히 같은지
    for name in RATIOS:
        img_stems = {p.stem for p in (img_dir / name).glob("*") if p.is_file()}
        lbl_stems = {p.stem for p in (lbl_dir / name).glob("*") if p.is_file()}
        if img_stems != lbl_stems:
            raise RuntimeError(f"[{name}] 사진과 라벨 짝이 맞지 않음: {sorted(img_stems ^ lbl_stems)[:5]}")
    print("\n완료! train / val / test 모두 사진과 라벨 짝이 정확히 맞음")


if __name__ == "__main__":
    main()
