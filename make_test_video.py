"""
테스트 영상 만들기

AI Hub 사진은 원래 현장 영상(mp4)에서 뽑은 연속 프레임이다.
같은 현장(앞부분이 같은 사진)의 _0001, _0002 ... 를 순서대로 이어 붙여 다시 영상으로 만든다.
여러 장면을 순서대로 이어서 "정상 → 위험 → 정상" 흐름의 영상을 만들 수 있다.

장면은 두 가지 방법으로 고른다.
  - 상황 코드 (예: N-04)       → 그 코드에서 프레임이 가장 많은 현장을 자동으로 고름
  - 현장 이름 (예: H-220930_A35_N-04_001) → 그 현장을 그대로 사용

프레임 간격은 라벨 json 의 Extraction_Time (원본 영상에서 뽑은 시각) 을 보고 실제 시간대로 맞춘다.
현장에 사진이 1장뿐이면 --still-ms 동안 멈춘 화면으로 보여준다.

결과:
  <출력이름>.mp4          테스트 영상
  <출력이름>.frames.json  프레임 번호 → 원본 사진 이름 (detect_video.py 의 더미 탐지기가 사용)

사용법:
    python make_test_video.py --images 원천데이터폴더 --labels 라벨폴더 --out output/test_videos/test_01.mp4 Y-04 N-04 Y-05 N-03
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2

SIZE = (1920, 1080)
CODE_RE = re.compile(r"^[A-Z]+-\d+$")
NAME_RE = re.compile(r"^(.*_([A-Z]+-\d+)_\d+)_(\d+)$")  # 현장, 상황 코드, 프레임 번호


def index_images(image_root):
    """현장 이름 → [(프레임 번호, 사진 경로)] (프레임 번호 순)."""
    scenes = defaultdict(list)
    for p in Path(image_root).rglob("*"):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        m = NAME_RE.match(p.stem)
        if m:
            scenes[m.group(1)].append((int(m.group(3)), p))
    for frames in scenes.values():
        frames.sort()
    return scenes


def extraction_seconds(label_path):
    """라벨 json 의 Extraction_Time ('00:03:17.25') → 초. 없으면 None."""
    try:
        data = json.loads(label_path.read_text(encoding="utf-8"))
        h, m, s = data["Source_Data_Info."]["Extraction_Time"].split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except (OSError, KeyError, ValueError):
        return None


def pick_scene(item, scenes):
    """상황 코드면 프레임이 가장 많은 현장, 현장 이름이면 그대로."""
    if CODE_RE.match(item):
        candidates = [k for k in scenes if k.split("_")[-2] == item]
        if not candidates:
            raise SystemExit(f"{item} 코드의 사진이 없어요.")
        return max(candidates, key=lambda k: (len(scenes[k]), k))
    if item not in scenes:
        raise SystemExit(f"현장을 찾을 수 없어요: {item}")
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="+", help="상황 코드(N-04) 또는 현장 이름, 영상에 나올 순서대로")
    parser.add_argument("--images", type=Path, required=True, help="원천데이터(jpg) 폴더")
    parser.add_argument("--labels", type=Path, help="라벨 json 폴더 (프레임 간격을 실제 시간대로 맞출 때)")
    parser.add_argument("--out", type=Path, default=Path("output/test_videos/test_01.mp4"))
    parser.add_argument("--fps", type=int, default=10, help="만들 영상의 fps")
    parser.add_argument("--max-frames", type=int, default=40, help="장면당 최대 프레임 수 (고르게 골라냄)")
    parser.add_argument("--frame-ms", type=int, default=700, help="실제 시간을 모를 때 프레임 1장당 보여줄 시간")
    parser.add_argument("--still-ms", type=int, default=3000, help="사진이 1장뿐인 장면을 보여줄 시간")
    args = parser.parse_args()

    scenes = index_images(args.images)
    if not scenes:
        raise SystemExit(f"사진이 없어요: {args.images}")
    labels = {p.stem: p for p in args.labels.rglob("*.json")} if args.labels else {}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, SIZE)
    if not writer.isOpened():
        raise SystemExit(f"영상을 만들 수 없어요: {args.out}")

    frame_names = []  # 영상 프레임 번호 → 원본 사진 이름
    print(f"{'시작':>6}  {'길이':>5}  {'사진':>4}  현장")
    for item in args.scenes:
        scene = pick_scene(item, scenes)
        frames = scenes[scene]
        if len(frames) > args.max_frames:
            step = len(frames) / args.max_frames
            frames = [frames[int(i * step)] for i in range(args.max_frames)]

        # 프레임마다 보여줄 시간(ms): 다음 프레임의 추출 시각과의 차이
        times = [extraction_seconds(labels[p.stem]) if p.stem in labels else None for _, p in frames]
        if len(frames) == 1:
            durations = [args.still_ms]
        else:
            gaps = []
            for a, b in zip(times, times[1:]):
                gap = (b - a) * 1000 if a is not None and b is not None and b > a else None
                gaps.append(gap)
            known = [g for g in gaps if g is not None]
            # 마지막 프레임은 다음 프레임이 없으니 평균 간격만큼 보여준다
            gaps.append(sum(known) / len(known) if known else None)
            durations = [min(max(g or args.frame_ms, 100), 3000) for g in gaps]  # 0.1~3초로 제한

        start_ms = len(frame_names) * 1000 / args.fps
        for (_, path), ms in zip(frames, durations):
            img = cv2.imread(str(path))
            if img is None:
                print(f"  ⚠️ 읽을 수 없는 사진 건너뜀: {path.name}")
                continue
            img = cv2.resize(img, SIZE)
            for _ in range(max(1, round(ms * args.fps / 1000))):
                writer.write(img)
                frame_names.append(path.name)
        length_s = len(frame_names) / args.fps - start_ms / 1000
        print(f"{start_ms / 1000:5.1f}s  {length_s:4.1f}s  {len(frames):4d}  {scene}")

    writer.release()
    map_path = args.out.with_suffix(".frames.json")
    map_path.write_text(json.dumps(frame_names, ensure_ascii=False), encoding="utf-8")
    print(f"\n영상: {args.out} ({len(frame_names)}프레임, {len(frame_names) / args.fps:.1f}초)")
    print(f"매핑: {map_path}")


if __name__ == "__main__":
    main()
