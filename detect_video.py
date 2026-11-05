"""
영상 분석 & 위험 장면 추출

영상을 처음부터 끝까지 읽으면서 위험이 있는 장면만 골라낸다.
  1. 일정 간격(--interval-ms)마다 프레임을 읽는다
  2. detector 로 위험을 확인한다
  3. 같은 위험이 연속으로 잡히면 하나의 "사건"으로 묶고 대표 장면 1장만 남긴다
  4. 대표 장면을 jpg 로 저장한다 (원본 + 박스 그린 버전)
  5. AI 출력 형식 JSON 으로 저장한다 (대표 장면마다 1개)
     - frame_index / frame_time_ms / image_url : 가장 잘 잡힌 대표 프레임
     - relations : 객체 간 공간관계 (--objects-weights 를 줄 때)
     - (--full 이면 assessment_id / site_info / work_info 를 붙인 위험성평가 전체 형식)
     - (명세 밖 추가) 위험마다 start_time_ms / end_time_ms : 사건이 처음·마지막으로 잡힌 시각
     - (명세 밖 추가) 위험마다 frames : 사건 동안 잡힌 프레임 리스트 (번호, 시간, 박스, 신뢰도)
       → 명세 필드만 보내려면 send_result.py --spec-only

--weights 를 주면 학습한 YOLO 모델로 실제 영상을 분석한다 (실제 사용).
--objects-weights 도 주면 대표 장면마다 객체 간 공간관계(relations)를 JSON 에 붙인다.

더미 탐지기(--labels)는 프레임이 어떤 AI Hub 사진인지 알아야 정답을 찾을 수 있어서
프레임 매핑 파일(<영상이름>.frames.json, make_test_video.py 가 만듦)이 필요하다.

사용법:
    python detect_video.py 영상.mp4 --video-id 3 --weights runs/yolo11n/weights/best.pt
    python detect_video.py 영상.mp4 --video-id 3 --weights runs/yolo11n/weights/best.pt \\
        --objects-weights runs/objects/weights/best.pt
    python detect_video.py 영상.mp4 --video-id 3 --labels 라벨폴더       # 더미 탐지기
    전체 형식: --full --site-id SITE-001 --site-name "건설 현장 A" --work-name "굴착 작업" \\
        --work-type "시설물 안전 점검" --work-location "굴착부 주변" [--assessment-id RA-2026-001]
"""
import argparse
import json
from pathlib import Path

import cv2

from detector import (YoloDetector, add_assessment_args, add_detector_args, assessment_kwargs,
                      build_result, draw, make_detector)


def load_frame_map(path):
    """프레임 번호 → AI Hub 사진 이름. 리스트([이름, ...]) 또는 딕셔너리({"0": 이름}) 형식."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return dict(enumerate(data))
    return {int(k): v for k, v in data.items()}


def find_events(cap, fps, detector, frame_map, interval_ms, gap_ms):
    """영상을 훑으며 위험 사건 목록을 만든다.

    사건 = 같은 hazard 가 gap_ms 이내로 계속 잡힌 구간.
    사건마다 confidence 가 가장 높은 프레임을 대표 장면으로 고른다.
    """
    step = max(1, round(fps * interval_ms / 1000))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    open_events = {}  # hazard → 진행 중인 사건
    events = []
    frames = {}       # 대표 후보 프레임 이미지 (frame_index → 이미지)

    index = 0
    while True:
        if index % step:
            # 분석하지 않는 프레임은 디코딩 없이 건너뛴다
            if not cap.grab():
                break
            index += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break

        time_ms = round(index / fps * 1000)
        detections = detector.detect(frame, image_name=frame_map.get(index))

        for det in detections:
            hazard = det["hazard"]
            ev = open_events.get(hazard)
            if ev is None or time_ms - ev["last_ms"] > gap_ms:
                ev = {"hazard": hazard, "start_ms": time_ms, "best": None, "frames": []}
                open_events[hazard] = ev
                events.append(ev)
            ev["last_ms"] = time_ms
            ev["frames"].append({"frame_index": index, "frame_time_ms": time_ms,
                                 "bbox": det["bbox"], "confidence": det["confidence"]})
            if ev["best"] is None or det["confidence"] > ev["best"]["det"]["confidence"]:
                old = ev["best"] and ev["best"]["frame_index"]
                # 예전 대표 프레임을 다른 사건이 안 쓰면 메모리에서 지운다
                if old is not None and not any(
                        e is not ev and e["best"] and e["best"]["frame_index"] == old for e in events):
                    frames.pop(old, None)
                ev["best"] = {"frame_index": index, "frame_time_ms": time_ms, "det": det}
                frames[index] = frame

        if index and index % (step * 50) == 0:
            print(f"  분석 중... {index}/{total} 프레임")
        index += 1

    return events, frames


def analyze_video(video_path, video_id, detector, out_dir=Path("output"), interval_ms=500, gap_ms=2000,
                  frame_map=None, image_url=None, log=print, **assessment):
    """영상 하나를 분석해서 (결과 JSON 리스트, 사건 리스트) 를 돌려준다. ai_server.py 도 이 함수를 쓴다.

    image_url: 대표 장면 jpg 경로 → JSON 에 넣을 image_url 로 바꾸는 함수 (기본: 로컬 경로 그대로)
    assessment: build_result 의 full / assessment_id / site_id / site_name / work_* 인자
    """
    use_yolo = isinstance(detector, YoloDetector)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없어요 (영상 파일이 맞는지 확인): {Path(video_path).name}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        raise ValueError("영상의 FPS 를 읽을 수 없어요.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log(f"영상: {Path(video_path).name} ({total}프레임, {fps:.1f}fps, {total / fps:.1f}초)")

    try:
        events, frames = find_events(cap, fps, detector, frame_map or {}, interval_ms, gap_ms)
    finally:
        cap.release()

    # 대표 장면이 같은 프레임이면 한 결과로 합친다
    by_frame = {}
    for ev in events:
        b = ev["best"]
        by_frame.setdefault(b["frame_index"], {"frame_time_ms": b["frame_time_ms"], "dets": []})
        # 대표 탐지 결과에 사건 구간과 프레임 리스트를 붙인다
        by_frame[b["frame_index"]]["dets"].append({
            **b["det"],
            "start_time_ms": ev["start_ms"],
            "end_time_ms": ev["last_ms"],
            "frames": ev["frames"],
        })

    frame_dir = Path(out_dir) / "frames" / f"video_{video_id}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for frame_index in sorted(by_frame):
        info = by_frame[frame_index]
        raw_path = frame_dir / f"frame_{frame_index}.jpg"
        boxed_path = frame_dir / f"frame_{frame_index}_boxed.jpg"
        cv2.imwrite(str(raw_path), frames[frame_index])
        draw(raw_path, info["dets"], boxed_path)
        # 이미지 업로드(S3) 방식이 정해지기 전까지 image_url 은 로컬 경로 (ai_server 는 서버 주소)
        # 공간관계는 대표 장면에서만 계산한다 (객체 모델이 있을 때)
        relations = detector.relations(frames[frame_index], info["dets"]) if use_yolo else None
        results.append(build_result(
            info["dets"], "video", image_url=image_url(raw_path) if image_url else str(raw_path),
            video_id=video_id, frame_index=frame_index, frame_time_ms=info["frame_time_ms"],
            relations=relations, **assessment,
        ))
    return results, events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--video-id", type=int, required=True, help="백엔드가 정한 영상 번호")
    add_detector_args(parser)
    add_assessment_args(parser)
    parser.add_argument("--frame-map", type=Path, help="(더미) 기본값: <영상이름>.frames.json")
    parser.add_argument("--interval-ms", type=int, default=500, help="몇 ms 마다 프레임을 분석할지")
    parser.add_argument("--gap-ms", type=int, default=2000, help="이 시간 안에 다시 잡히면 같은 사건")
    parser.add_argument("--out", type=Path, default=Path("output"))
    args = parser.parse_args()

    detector = make_detector(args)
    frame_map = {}
    if not isinstance(detector, YoloDetector):
        frame_map_path = args.frame_map or args.video.with_suffix(".frames.json")
        if not frame_map_path.exists():
            raise SystemExit(f"프레임 매핑 파일이 없어요: {frame_map_path}\n"
                             "더미 탐지기는 make_test_video.py 로 만든 영상에서만 동작해요. (실제 영상은 --weights)")
        frame_map = load_frame_map(frame_map_path)

    try:
        results, events = analyze_video(args.video, args.video_id, detector, args.out, args.interval_ms,
                                        args.gap_ms, frame_map=frame_map, **assessment_kwargs(args))
    except ValueError as e:
        raise SystemExit(str(e))

    result_dir = args.out / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    out_path = result_dir / f"video_{args.video_id}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n위험 사건 {len(events)}개 → 대표 장면 {len(results)}장")
    for ev in events:
        b = ev["best"]
        print(f"  {ev['start_ms'] / 1000:6.1f}s ~ {ev['last_ms'] / 1000:6.1f}s  {ev['hazard']}"
              f"  (대표: frame {b['frame_index']}, 프레임 {len(ev['frames'])}개)")
    print(f"\n장면 이미지: {args.out / 'frames' / f'video_{args.video_id}'}/")
    print(f"결과 JSON : {out_path}")


if __name__ == "__main__":
    main()
