"""
백엔드로 결과 보내기

detector.py / detect_video.py 가 만든 결과 JSON 을 백엔드로 POST 한다.
  - 보내기 전에 형식이 맞는지 검사한다 (틀리면 보내지 않음)
    AI 출력 형식(source_type · video_id · frame_* · image_url · detections · relations)과
    위험성평가 전체 형식(assessment_id · site_info · work_info · source · detections) 둘 다 검사 가능
  - 서버가 꺼져 있거나 5xx/429 면 몇 번 다시 시도한다
  - 4xx (형식 오류 등) 는 다시 보내도 같으니 재시도하지 않고 서버 메시지를 보여준다

백엔드 주소는 코드에 적지 않는다. 우선순위: --url > 환경변수 BACKEND_URL > 기본값(localhost:8000, mock_server)

사용법:
    python send_result.py output/results/video_3.json
    python send_result.py output/results/                       # 폴더 안 json 전부
    python send_result.py output/results/video_3.json --batch   # 리스트를 한 번에 보내기
    python send_result.py output/results/video_3.json --dry-run # 검사만, 전송 안 함
    python send_result.py output/results/video_3.json --spec-only  # 사건 구간 정보(start/end/frames) 빼기
    python send_result.py output/results/video_3.json --no-relations  # 공간관계 빼기
    BACKEND_URL=http://192.168.0.10:8080 python send_result.py output/results/video_3.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

DEFAULT_URL = "http://localhost:8000"
DEFAULT_ENDPOINT = "/api/detections"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


SPEC_DETECTION_KEYS = ("hazard", "location", "evidence", "bbox", "confidence")
SOURCE_KEYS = ("source_type", "photo_id", "video_id", "frame_index", "frame_time_ms", "image_url")
FULL_TOP_KEYS = ("assessment_id", "site_info", "work_info", "source", "detections")
EVAL_STATUS = {"hazard", "no_hazard", "not_evaluable"}
REASON_CODES = {"rule_matched", "condition_not_met", "rule_not_supported", "missing_objects", "insufficient_evidence"}


def _is_text(v, nullable=False):
    return (nullable and v is None) or (isinstance(v, str) and bool(v))


def is_full(result):
    """위험성평가 전체 형식(assessment_id · site_info · source ...)이면 True, AI 출력 형식이면 False."""
    return isinstance(result, dict) and "source" in result


def _validate_source(src, p):
    """source_type · photo_id · video_id · frame_* · image_url 검사. p = 오류 메시지 앞에 붙일 경로."""
    missing = [k for k in SOURCE_KEYS if k not in src]
    if missing:
        return [f"{p}필드 없음: {missing}"]
    errors = []
    st = src["source_type"]
    if st == "photo":
        if src["photo_id"] in (None, ""):
            errors.append(f"사진인데 {p}photo_id 가 없음")
        for k in ("video_id", "frame_index", "frame_time_ms"):
            if src[k] is not None:
                errors.append(f"사진인데 {p}{k} 가 null 이 아님")
    elif st == "video":
        if src["photo_id"] is not None:
            errors.append(f"영상인데 {p}photo_id 가 null 이 아님")
        if src["video_id"] in (None, ""):
            errors.append(f"영상인데 {p}video_id 가 없음")
        for k in ("frame_index", "frame_time_ms"):
            if not isinstance(src[k], int):
                errors.append(f"영상인데 {p}{k} 가 정수가 아님: {src[k]!r}")
    else:
        errors.append(f"{p}source_type 이 photo/video 가 아님: {st!r}")
    if not _is_text(src["image_url"], nullable=True):
        errors.append(f"{p}image_url 이 문자열 또는 null 이 아님: {src['image_url']!r}")
    return errors


def _validate_detections(dets):
    if not isinstance(dets, list):
        return ["detections 가 리스트가 아님"]
    errors = []
    for i, det in enumerate(dets):
        for k in ("hazard", "location", "evidence"):
            if not _is_text(det.get(k)):
                errors.append(f"detections[{i}].{k} 가 비어 있음")
        conf = det.get("confidence")
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
            errors.append(f"detections[{i}].confidence 가 0~1 이 아님: {conf!r}")
        if "bbox" not in det:
            errors.append(f"detections[{i}].bbox 필드 없음")
            continue
        b = det["bbox"]
        if b is None:  # 명세상 bbox 는 null 가능
            continue
        try:
            x, y, w, h = (float(b[k]) for k in ("x", "y", "width", "height"))
        except (KeyError, TypeError, ValueError):
            errors.append(f"detections[{i}].bbox 형식 오류: {b!r}")
            continue
        # 반올림 오차를 감안해 0.001 까지 허용
        if min(x, y) < 0 or w <= 0 or h <= 0 or x + w > 1.001 or y + h > 1.001:
            errors.append(f"detections[{i}].bbox 가 0~1 범위를 벗어남: {b}")
    return errors


def _validate_relations(rel):
    if not isinstance(rel, dict):
        return ["relations 가 객체가 아님"]
    errors = [f"relations.{k} 가 리스트가 아님" for k in ("objects", "relations")
              if not isinstance(rel.get(k), list)]
    if errors:
        return errors
    if "evaluation_status" in rel:
        if rel["evaluation_status"] not in EVAL_STATUS:
            errors.append(f"relations.evaluation_status 값이 이상함: {rel['evaluation_status']!r}")
        if rel.get("reason_code") not in REASON_CODES:
            errors.append(f"relations.reason_code 값이 이상함: {rel.get('reason_code')!r}")
        for i, c in enumerate(rel.get("checks", [])):
            if c.get("status") not in EVAL_STATUS or c.get("reason_code") not in REASON_CODES:
                errors.append(f"relations.checks[{i}] 의 status/reason_code 값이 이상함")
    ids = {o.get("id") for o in rel["objects"]}
    for i, r in enumerate(rel["relations"]):
        if r.get("a") not in ids or r.get("b") not in ids:
            errors.append(f"relations.relations[{i}] 의 a/b 가 objects 의 id 가 아님")
    return errors


def validate(result):
    """형식 검사. 문제 목록을 돌려준다 (비어 있으면 통과).

    AI 출력 형식:     {source_type, photo_id, video_id, frame_index, frame_time_ms, image_url, detections, relations?}
    위험성평가 전체 형식: {assessment_id, site_info, work_info, source{...}, detections, relations?}
    """
    if not isinstance(result, dict):
        return ["결과가 객체(dict)가 아님"]
    errors = []
    if is_full(result):
        missing = [k for k in FULL_TOP_KEYS if k not in result]
        if missing:
            return [f"필드 없음: {missing}"]
        if not _is_text(result["assessment_id"]):
            errors.append(f"assessment_id 가 비어 있음: {result['assessment_id']!r}")
        site = result["site_info"]
        if not isinstance(site, dict):
            errors.append("site_info 가 객체가 아님")
        else:
            errors += [f"site_info.{k} 가 비어 있음" for k in ("site_id", "site_name") if not _is_text(site.get(k))]
        work = result["work_info"]
        if work is not None:
            if not isinstance(work, dict):
                errors.append("work_info 가 객체 또는 null 이 아님")
            else:
                for k in ("work_name", "work_type", "location"):
                    if k not in work:
                        errors.append(f"work_info.{k} 필드 없음")
                    elif not _is_text(work[k], nullable=True):
                        errors.append(f"work_info.{k} 가 문자열 또는 null 이 아님: {work[k]!r}")
        if not isinstance(result["source"], dict):
            errors.append("source 가 객체가 아님")
        else:
            errors += _validate_source(result["source"], "source.")
    else:
        if "detections" not in result:
            return ["필드 없음: ['detections']"]
        errors += _validate_source(result, "")
    errors += _validate_detections(result["detections"])
    if "relations" in result:
        errors += _validate_relations(result["relations"])
    return errors


def spec_only(result, keep_relations=True):
    """정해진 필드만 남긴다: 위험별 start_time_ms / end_time_ms / frames 같은 추가 정보를 뺀다.
    keep_relations=False 면 relations(공간관계)도 뺀다."""
    keys = FULL_TOP_KEYS if is_full(result) else SOURCE_KEYS + ("detections",)
    out = {k: result[k] for k in keys}
    out["detections"] = [{k: d.get(k) for k in SPEC_DETECTION_KEYS} for d in result["detections"]]
    if keep_relations and "relations" in result:
        out["relations"] = result["relations"]
    return out


def describe(r):
    """사람이 읽을 요약: 'video 3, 12.3초 (frame 367)' 또는 'photo 사진이름'."""
    src = r["source"] if is_full(r) else r
    if src["source_type"] == "video":
        return f"video {src['video_id']}, {src['frame_time_ms'] / 1000:.1f}초 (frame {src['frame_index']})"
    return f"photo {src['photo_id']}"


def post(url, payload, retries, timeout):
    """POST 하고 (성공 여부, 메시지) 를 돌려준다. 재시도 가능한 오류만 다시 시도."""
    for attempt in range(retries + 1):
        try:
            res = requests.post(url, json=payload, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            reason = f"서버 연결 실패 ({type(e).__name__})"
        else:
            if res.ok:
                return True, f"HTTP {res.status_code}"
            reason = f"HTTP {res.status_code}: {res.text[:300]}"
            if res.status_code not in RETRY_STATUS:
                return False, reason
        if attempt < retries:
            wait = 2 ** attempt
            print(f"    {reason} → {wait}초 뒤 다시 시도 ({attempt + 1}/{retries})")
            time.sleep(wait)
    return False, reason


def load_results(paths):
    """파일/폴더에서 결과를 읽는다. 각 파일은 결과 1개(dict) 또는 리스트."""
    files = []
    for p in paths:
        files.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    items = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        for i, r in enumerate(data if isinstance(data, list) else [data]):
            items.append((f"{f.name}[{i}]", r))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path, help="결과 JSON 파일 또는 폴더")
    parser.add_argument("--url", help=f"백엔드 주소 (기본: 환경변수 BACKEND_URL 또는 {DEFAULT_URL})")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--batch", action="store_true", help="결과 여러 개를 리스트 하나로 한 번에 보내기")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--dry-run", action="store_true", help="형식 검사만 하고 보내지 않기")
    parser.add_argument("--spec-only", action="store_true",
                        help="정해진 필드만 보내기 (위험별 사건 구간 start/end_time_ms, frames 제거)")
    parser.add_argument("--no-relations", action="store_true", help="공간관계(relations) 빼고 보내기")
    args = parser.parse_args()

    base = (args.url or os.environ.get("BACKEND_URL") or DEFAULT_URL).rstrip("/")
    url = base + args.endpoint

    items = load_results(args.paths)
    if not items:
        raise SystemExit("보낼 결과가 없어요.")

    # 1. 형식 검사 — 하나라도 틀리면 아무것도 보내지 않는다
    bad = [(name, errs) for name, r in items if (errs := validate(r))]
    if bad:
        print("형식 오류가 있어서 보내지 않았어요:")
        for name, errs in bad:
            for e in errs:
                print(f"  {name}: {e}")
        sys.exit(1)
    print(f"형식 검사 통과: {len(items)}개")
    if args.spec_only or args.no_relations:
        items = [(name, spec_only(r, keep_relations=not args.no_relations) if args.spec_only
                  else {k: v for k, v in r.items() if k != "relations"}) for name, r in items]
        print("추가 필드를 빼고 보내요:" + (" 사건 구간 정보" if args.spec_only else "")
              + (" relations" if args.no_relations else ""))

    if args.dry_run:
        print(f"(--dry-run) 보낼 주소: {url}")
        return

    # 2. 전송
    print(f"보내는 주소: {url}")
    if args.batch:
        ok, msg = post(url, [r for _, r in items], args.retries, args.timeout)
        print(f"  {'✅' if ok else '❌'} {len(items)}개 한 번에 → {msg}")
        sys.exit(0 if ok else 1)

    failed = 0
    for name, r in items:
        ok, msg = post(url, r, args.retries, args.timeout)
        failed += not ok
        print(f"  {'✅' if ok else '❌'} {name} ({describe(r)}, 위험 {len(r['detections'])}개) → {msg}")
    print(f"\n성공 {len(items) - failed}개 / 실패 {failed}개")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
