"""
AI 서버 (FastAPI)

백엔드가 "이 영상 분석해줘" 하고 부르면, 영상을 분석해서 AI 결과 JSON 을 돌려준다.
  - 모델(best.pt)은 서버를 켤 때 한 번만 불러온다
  - 영상 위치는 세 가지 모두 받는다: 내 컴퓨터 경로 / https:// 링크 / s3://버킷/파일
  - 분석은 detect_video.analyze_video() 를 그대로 쓴다 (명령어로 돌릴 때와 결과가 같다)
  - 위험 장면 사진은 이 서버의 /frames/... 주소로 볼 수 있다 → image_url 에 그 주소를 넣는다
    (S3 업로드 방식이 정해지면 그 주소로 바꾸면 됨)

주소 (기본 포트 8001)
  GET  /health                 서버 · 모델 상태 확인
  POST /predict/video          {"video_id": 3, "video_url": "경로 또는 https:// 또는 s3://"} → 결과 JSON 목록
  POST /predict/video/upload   영상 파일을 직접 올려서 분석 (테스트용)
  POST /predict/photo          사진 파일을 올려서 분석
  GET  /frames/...             위험 장면 사진
  GET  /docs                   브라우저에서 직접 눌러보는 테스트 화면 (Swagger)

응답 형식 (영상 · 사진 동일)
  {"video_id" 또는 "photo_id", "count": 결과 수, "elapsed_ms": 걸린 시간,
   "results": [AI 결과 JSON, ...]}   ← 위험 장면 하나당 JSON 하나

  AI 결과 JSON (기본):
    {source_type, photo_id, video_id, frame_index, frame_time_ms, image_url,
     detections: [{hazard, location, evidence, bbox, confidence, start_time_ms, end_time_ms, frames}],
     relations: {objects, relations, continuity, findings,       ← 객체 간 공간관계 · 규칙 판단
                 evaluation_status, reason_code, checks}}        ← 규칙 판단 상태 · 원인 (위험별 checks)
  요청 옵션: spec_only=true → 위험별 사건 구간 정보(start/end_time_ms, frames) 빼기
             include_relations=false → relations 빼기
             full_format=true → assessment_id · site_info · work_info 를 붙인 위험성평가 전체 형식

설정 (환경변수, 없으면 기본값)
  AI_WEIGHTS          위험 상태 모델     (기본 runs/yolo11n/weights/best.pt)
  AI_OBJECTS_WEIGHTS  객체 모델 → relations (기본 runs/objects/weights/best.pt, 빈 값이면 안 씀)
  AI_CONF             최소 confidence    (기본 0.5)
  AI_PUBLIC_URL       image_url 앞에 붙일 서버 주소 (기본: 요청이 들어온 주소)
  AI_ALLOW_LOCAL_PATH video_url 에 내 컴퓨터 경로를 허용할지
                        local (기본): 이 컴퓨터에서 직접(localhost) 보낸 요청만 허용, 터널 · 다른 컴퓨터는 막음
                        1: 모두 허용 / 0: 모두 막음
  s3:// : 공개 버킷이면 키 없이 읽음 (AWS_REGION 을 주면 더 정확). 비공개 버킷이면
            pip install boto3 + AWS 키를 환경변수(AWS_ACCESS_KEY_ID 등)로 설정 (코드에 적지 말 것)

사용법:
    python ai_server.py                      # 내 컴퓨터에서만 (http://localhost:8001/docs)
    python ai_server.py --host 0.0.0.0       # 같은 와이파이의 다른 컴퓨터도 접속 가능
"""
import argparse
import os
import re
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union
from urllib.parse import quote, urlparse

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from detect_video import analyze_video
from detector import DEFAULT_SITE_ID, DEFAULT_SITE_NAME, YoloDetector, build_result, draw
from send_result import spec_only as to_spec_only

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output" / "server"
FRAMES_DIR = OUT_DIR / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

state = {"detector": None}
model_lock = threading.Lock()  # YOLO 모델은 한 번에 요청 하나씩


def env_path(name, default):
    value = os.environ.get(name, default)
    return (ROOT / value if not Path(value).is_absolute() else Path(value)) if value else None


@asynccontextmanager
async def lifespan(app):
    weights = env_path("AI_WEIGHTS", "runs/yolo11n/weights/best.pt")
    objects = env_path("AI_OBJECTS_WEIGHTS", "runs/objects/weights/best.pt")
    if not weights.exists():
        raise RuntimeError(f"위험 상태 모델이 없어요: {weights}")
    if objects and not objects.exists():
        print(f"⚠️ 객체 모델이 없어서 relations 없이 실행해요: {objects}")
        objects = None
    print("모델 불러오는 중...")
    state["detector"] = YoloDetector(weights, objects, conf=float(os.environ.get("AI_CONF", 0.5)))
    state["weights"], state["objects_weights"] = weights, objects
    print(f"준비 완료 (장치: {state['detector'].device})")
    yield


app = FastAPI(title="Capstone AI 서버", description="건설현장 영상 위험 탐지 + 객체 간 공간관계 → AI 결과 JSON",
              lifespan=lifespan)
app.mount("/frames", StaticFiles(directory=FRAMES_DIR), name="frames")


# ---------------------------------------------------------------- 요청 형식
class AssessmentInfo(BaseModel):
    spec_only: bool = Field(False, description="true 면 위험별 사건 구간 정보(start/end_time_ms, frames) 제외")
    include_relations: bool = Field(True, description="false 면 객체 간 공간관계(relations) 제외")
    full_format: bool = Field(False, description="true 면 assessment_id · site_info · work_info 를 붙인 "
                                                 "위험성평가 전체 형식 (아래 값들은 이때만 사용)")
    assessment_id: Optional[str] = Field(None, description="위험성평가 ID (안 주면 자동: RA-V3-F367)")
    site_id: str = Field(DEFAULT_SITE_ID, description="현장 ID")
    site_name: str = Field(DEFAULT_SITE_NAME, description="현장명")
    work_name: Optional[str] = Field(None, description="작업명")
    work_type: Optional[str] = Field(None, description="작업 유형")
    work_location: Optional[str] = Field(None, description="작업 위치 (JSON 에서는 work_info.location)")


class VideoRequest(AssessmentInfo):
    video_id: Union[int, str] = Field(..., description="백엔드가 정한 영상 번호", examples=[3])
    video_url: str = Field(..., description="영상 위치: 내 컴퓨터 경로 / https://... / s3://버킷/파일",
                           examples=["s3://bucket/videos/3.mp4"])
    interval_ms: int = Field(500, ge=50, description="몇 ms 마다 프레임을 분석할지")
    gap_ms: int = Field(2000, ge=0, description="이 시간 안에 다시 잡히면 같은 사건으로 묶음")


# ---------------------------------------------------------------- 도우미
def safe_name(value):
    """폴더 · 파일 이름에 쓸 수 있게 바꾼다."""
    return re.sub(r"[^\w.-]", "_", str(value))[:80] or "unknown"


def image_url_maker(request):
    """저장된 장면 사진 경로 → 이 서버에서 볼 수 있는 주소."""
    base = (os.environ.get("AI_PUBLIC_URL") or str(request.base_url)).rstrip("/")

    def make(path):
        rel = Path(path).resolve().relative_to(FRAMES_DIR.resolve()).as_posix()
        return f"{base}/frames/{quote(rel)}"
    return make


def aws_credentials_available():
    """boto3 가 설치돼 있고 AWS 키를 찾을 수 있으면 True."""
    try:
        import boto3
    except ImportError:
        return False
    return boto3.Session().get_credentials() is not None


def s3_public_url(bucket, key):
    """s3://버킷/파일 → 키 없이 받을 수 있는 https 주소 (공개 버킷일 때만 열림)."""
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    host = f"s3.{region}.amazonaws.com" if region else "s3.amazonaws.com"
    if "." in bucket:  # 점이 있는 버킷 이름은 경로 방식으로
        return f"https://{host}/{bucket}/{quote(key)}"
    return f"https://{bucket}.{host}/{quote(key)}"


def download(url, target, forbidden=None):
    """url 을 target 파일로 내려받는다."""
    import requests
    try:
        with requests.get(url, stream=True, timeout=(10, 120)) as res:
            if res.status_code == 403 and forbidden:
                raise HTTPException(400, forbidden)
            if not res.ok:
                raise HTTPException(400, f"영상 다운로드 실패: HTTP {res.status_code}")
            with open(target, "wb") as f:
                for chunk in res.iter_content(1 << 20):
                    f.write(chunk)
    except requests.RequestException as e:
        raise HTTPException(400, f"영상 다운로드 실패: {type(e).__name__}")
    return target


PROXY_HEADERS = ("cf-connecting-ip", "cf-ray", "x-forwarded-for", "x-real-ip", "forwarded")


def is_direct_local(request):
    """이 컴퓨터에서 직접 보낸 요청인지 (터널 · 프록시를 거친 요청은 아님)."""
    host = request.client.host if request.client else ""
    return host in ("127.0.0.1", "::1", "localhost") and not any(h in request.headers for h in PROXY_HEADERS)


def local_path_allowed(request):
    mode = os.environ.get("AI_ALLOW_LOCAL_PATH", "local")
    return mode == "1" or (mode == "local" and is_direct_local(request))


def fetch_video(video_url, workdir, request=None):
    """video_url 이 가리키는 영상을 분석할 수 있는 로컬 파일 경로로 만든다."""
    parsed = urlparse(video_url)
    suffix = Path(parsed.path).suffix.lower() if Path(parsed.path).suffix.lower() in VIDEO_SUFFIXES else ".mp4"
    target = Path(workdir) / f"input{suffix}"

    if parsed.scheme == "s3":
        bucket, key = parsed.netloc, parsed.path.lstrip("/")
        if not bucket or not key:
            raise HTTPException(400, f"s3 경로 형식이 아니에요 (s3://버킷/파일): {video_url}")
        # AWS 키가 설정돼 있으면 키로 읽고, 없으면 공개 버킷으로 보고 키 없이 https 로 받는다
        if aws_credentials_available():
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError
            try:
                boto3.client("s3").download_file(bucket, key, str(target))
            except (BotoCoreError, ClientError) as e:
                raise HTTPException(400, f"S3 에서 영상을 받지 못했어요: {e}")
            return target
        return download(s3_public_url(bucket, key), target,
                        forbidden="S3 버킷이 비공개라서 키 없이는 못 읽어요. 버킷(또는 영상 폴더)을 공개 읽기로 하거나, "
                                  "AI 서버에 AWS 키를 설정하거나, https 다운로드 링크(presigned URL)로 보내주세요.")

    if parsed.scheme in ("http", "https"):
        return download(video_url, target)

    if request is None or not local_path_allowed(request):
        raise HTTPException(400, "video_url 은 https:// 또는 s3:// 형식이어야 해요. "
                                 "(내 컴퓨터 경로는 AI 서버 컴퓨터에서 localhost 로 직접 요청할 때만 쓸 수 있어요)")
    path = Path(video_url).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise HTTPException(400, f"영상 파일이 없어요: {video_url}")
    return path


def assessment_args(info):
    return {"full": info.full_format, **{k: getattr(info, k) for k in
            ("assessment_id", "site_id", "site_name", "work_name", "work_type", "work_location")}}


def finish(results, info):
    """요청 옵션에 따라 추가 정보를 뺀다."""
    if info.spec_only:
        results = [to_spec_only(r, keep_relations=info.include_relations) for r in results]
    elif not info.include_relations:
        results = [{k: v for k, v in r.items() if k != "relations"} for r in results]
    return results


def run_video(path, video_id, info, request, interval_ms=500, gap_ms=2000):
    start = time.perf_counter()
    try:
        with model_lock:
            results, events = analyze_video(
                path, video_id, state["detector"], OUT_DIR, interval_ms, gap_ms,
                image_url=image_url_maker(request), log=lambda m: print(f"  [video {video_id}] {m}"),
                **assessment_args(info))
    except ValueError as e:
        raise HTTPException(400, str(e))
    results = finish(results, info)
    elapsed = round((time.perf_counter() - start) * 1000)
    print(f"  [video {video_id}] 위험 사건 {len(events)}개 → 결과 {len(results)}개 ({elapsed / 1000:.1f}초)")
    return {"video_id": video_id, "count": len(results), "elapsed_ms": elapsed, "results": results}


def parse_id(value):
    """폼으로 받은 id: 숫자면 숫자로."""
    return int(value) if str(value).isdigit() else value


# ---------------------------------------------------------------- 주소
@app.get("/health", summary="서버 · 모델 상태 확인")
def health():
    det = state["detector"]
    return {"status": "ok" if det else "loading",
            "device": det.device if det else None,
            "weights": str(state.get("weights")),
            "objects_weights": str(state.get("objects_weights")),
            "relations": bool(det and det.objects_model)}


@app.post("/predict/video", summary="영상 경로(로컬 / https / s3)로 분석")
def predict_video(req: VideoRequest, request: Request):
    print(f"[요청] video {req.video_id}: {req.video_url}")
    with tempfile.TemporaryDirectory() as tmp:
        path = fetch_video(req.video_url, tmp, request)
        return run_video(path, safe_name(req.video_id) if isinstance(req.video_id, str) else req.video_id,
                         req, request, req.interval_ms, req.gap_ms)


@app.post("/predict/video/upload", summary="영상 파일을 직접 올려서 분석 (테스트용)")
def predict_video_upload(request: Request, file: UploadFile = File(...), video_id: str = Form(...),
                         spec_only: bool = Form(False), include_relations: bool = Form(True),
                         full_format: bool = Form(False), assessment_id: Optional[str] = Form(None),
                         site_id: str = Form(DEFAULT_SITE_ID), site_name: str = Form(DEFAULT_SITE_NAME),
                         work_name: Optional[str] = Form(None), work_type: Optional[str] = Form(None),
                         work_location: Optional[str] = Form(None)):
    info = AssessmentInfo(spec_only=spec_only, include_relations=include_relations, full_format=full_format,
                          assessment_id=assessment_id, site_id=site_id, site_name=site_name, work_name=work_name,
                          work_type=work_type, work_location=work_location)
    vid = parse_id(safe_name(video_id))
    print(f"[요청] video {vid}: 업로드 {file.filename}")
    suffix = Path(file.filename or "").suffix.lower()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"input{suffix if suffix in VIDEO_SUFFIXES else '.mp4'}"
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return run_video(path, vid, info, request)


@app.post("/predict/photo", summary="사진 파일을 올려서 분석")
def predict_photo(request: Request, file: UploadFile = File(...), photo_id: Optional[str] = Form(None),
                  spec_only: bool = Form(False), include_relations: bool = Form(True),
                  full_format: bool = Form(False), assessment_id: Optional[str] = Form(None),
                  site_id: str = Form(DEFAULT_SITE_ID), site_name: str = Form(DEFAULT_SITE_NAME),
                  work_name: Optional[str] = Form(None), work_type: Optional[str] = Form(None),
                  work_location: Optional[str] = Form(None)):
    info = AssessmentInfo(spec_only=spec_only, include_relations=include_relations, full_format=full_format,
                          assessment_id=assessment_id, site_id=site_id, site_name=site_name, work_name=work_name,
                          work_type=work_type, work_location=work_location)
    pid = parse_id(safe_name(photo_id or Path(file.filename or "photo").stem))
    print(f"[요청] photo {pid}: {file.filename}")
    suffix = Path(file.filename or "").suffix.lower()
    out_dir = FRAMES_DIR / f"photo_{pid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / f"{pid}{suffix if suffix in ('.jpg', '.jpeg', '.png') else '.jpg'}"
    with open(raw, "wb") as f:
        shutil.copyfileobj(file.file, f)

    start = time.perf_counter()
    det = state["detector"]
    try:
        with model_lock:
            detections = det.detect(str(raw))
            relations = det.relations(str(raw), detections)
        draw(raw, detections, out_dir / f"{pid}_boxed.jpg")
    except Exception as e:  # 사진이 아닌 파일 등
        raise HTTPException(400, f"사진을 분석하지 못했어요: {type(e).__name__}: {e}")
    result = build_result(detections, "photo", image_url=image_url_maker(request)(raw), photo_id=pid,
                          relations=relations, **assessment_args(info))
    result = finish([result], info)[0]
    elapsed = round((time.perf_counter() - start) * 1000)
    print(f"  [photo {pid}] 위험 {len(detections)}개 ({elapsed / 1000:.1f}초)")
    return {"photo_id": pid, "count": 1, "elapsed_ms": elapsed, "results": [result]}


def main():
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 이면 같은 와이파이의 다른 컴퓨터도 접속 가능")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    print(f"AI 서버: http://{args.host}:{args.port}  (테스트 화면: http://localhost:{args.port}/docs)")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
