"""
탐지기 (AI의 "눈")

사진 한 장을 넣으면 위험 목록을 돌려준다.
    detect(frame, image_name) -> [{"hazard", "location", "evidence", "bbox": {x, y, width, height}, "confidence"}]

build_result() 로 AI 결과 JSON (source_type · video_id · frame_* · image_url · detections · relations) 을 만든다.
(--full 이면 assessment_id · site_info · work_info 를 붙인 위험성평가 전체 형식)

탐지기 두 가지 (detect() 형식은 같다)
  - YoloDetector  : 학습한 YOLO 모델(best.pt)로 사진을 직접 보고 위험을 찾는다 (실제 사용)
                    객체 탐지 모델(--objects-weights)도 주면 객체 간 공간관계(relations)를 함께 계산한다
  - DummyDetector : AI Hub 라벨 json 에 적힌 정답 위치를 그대로 돌려준다 (모델 없이 테스트용)

bbox 는 0~1 비율 (x, y = 왼쪽 위 모서리).

사용법 (사진 한 장 테스트):
    python detector.py 사진.jpg --weights runs/yolo11n/weights/best.pt --draw 결과.jpg
    python detector.py 사진.jpg --weights runs/yolo11n/weights/best.pt --objects-weights runs/objects/weights/best.pt
    python detector.py 사진.jpg --labels 라벨폴더                     # 더미 탐지기
    전체 형식: --full --site-id SITE-001 --site-name "건설 현장 A" --work-name "굴착 작업" ...
"""
import argparse
import json
from pathlib import Path

# AI Hub 코드 → (hazard 이름, 위험 대상, 근거 문장)
# hazard 이름은 AI Hub 「건설현장 위험상태 판단 데이터 데이터설명서 V1.1」의 비정상 시나리오 이름
HAZARDS = {
    # 추락 (A)
    "N-01": ("시스템비계 작업발판 설치 불량", "시스템비계 작업발판", "시스템비계 작업발판이 비정상적으로 설치되어 있음"),
    "N-02": ("개폐형 발판이 열려 있는 상황", "개폐형 작업발판", "개폐형 작업발판의 해치가 열려 있어 추락할 수 있음"),
    "N-03": ("개구부 덮개 설치 불량", "개구부", "개구부 덮개가 제대로 설치되어 있지 않음"),
    "N-04": ("개구부 안전난간 설치 불량", "개구부", "개구부 주변에 안전난간이 설치되어 있지 않음"),
    "N-05": ("단부 안전난간 설치 불량", "단부", "단부에 안전난간이 설치되어 있지 않음"),
    "N-06": ("단부 수직방지망 설치 불량", "단부(갱폼)", "단부에 수직방지망이 제대로 설치되어 있지 않음"),
    "N-07": ("이동식비계 안전난간 설치 불량", "이동식비계", "이동식비계에 안전난간이 설치되어 있지 않음"),
    "N-11": ("시스템비계 위 낙하위험물 배치", "시스템비계", "시스템비계 위에 떨어질 수 있는 물건이 놓여 있음"),
    # 낙하 (B)
    "N-13": ("출입구 방호선반 설치 불량", "출입구", "출입구 상부에 방호선반이 제대로 설치되어 있지 않음"),
    "N-14": ("낙하물 방지망 설치 불량", "낙하물 방지망", "낙하물 방지망이 제대로 설치되어 있지 않음"),
    "N-15": ("거푸집 단부 위 적재물 배치", "거푸집 단부", "거푸집 단부 위에 떨어질 수 있는 적재물이 놓여 있음"),
    "N-17": ("사다리차 비정상 과적재", "사다리차", "사다리차에 자재가 비정상적으로 많이 실려 있음"),
    "N-18": ("사다리차 작업 시 신호수 미배치", "사다리차", "사다리차 작업 중 신호수가 배치되어 있지 않음"),
    "N-19": ("이동식 크레인 작업 시 신호수 미배치", "이동식 크레인", "이동식 크레인 작업 중 신호수가 배치되어 있지 않음"),
    "N-20": ("이동식 크레인 작업 시 철제 인양박스 미사용", "이동식 크레인", "철제 인양박스 대신 항공마대 등으로 자재를 인양하고 있음"),
    # 협착 (C)
    "N-21": ("관로공사 시 신호수 미배치", "관로공사 장비", "관로공사 중 신호수가 배치되어 있지 않음"),
    "N-22": ("굴삭기 자재 양중 시 신호수 미배치", "굴삭기", "굴삭기로 자재를 들어 올리는 중 신호수가 배치되어 있지 않음"),
    "N-23": ("시저형 고소작업대 과상승방지봉 설치 높이 미준수", "시저형 고소작업대", "과상승방지봉이 설치 기준 높이를 지키지 않음"),
    "N-24": ("고소작업대 위험반경 내 근로자 작업", "고소작업대", "고소작업대 위험반경 안에서 근로자가 작업하고 있음"),
    "N-25": ("덤프트럭 작업 시 신호수 미배치", "덤프트럭", "덤프트럭 작업 중 신호수가 배치되어 있지 않음"),
    "N-26": ("덤프트럭 위험반경 내 근로자 작업", "덤프트럭", "덤프트럭 위험반경 안에서 근로자가 작업하고 있음"),
    "N-27": ("굴삭기 작업 시 신호수 미배치", "굴삭기", "굴삭기 작업 중 신호수가 배치되어 있지 않음"),
    "N-28": ("굴삭기 수평 위험반경 내 작업", "굴삭기", "굴삭기 수평 위험반경 안에서 작업하고 있음"),
    "N-29": ("레미콘 작업 시 신호수 미배치", "레미콘", "레미콘 작업 중 신호수가 배치되어 있지 않음"),
    "N-30": ("레미콘 위험반경 내 근로자 작업", "레미콘", "레미콘 위험반경 안에서 근로자가 작업하고 있음"),
    # 화재 (D)
    "N-31": ("실내마감 작업 시 이동형 환풍기 미설치", "실내마감 작업 구역", "실내마감 작업 중 이동형 환풍기가 설치되어 있지 않음"),
    "N-32": ("실내마감 작업 시 근처 난로 설치", "실내마감 작업 구역", "실내마감 작업 근처에 난로가 설치되어 있음"),
    "N-33": ("용접기 옆 소화기 미배치", "용접기", "용접기 옆에 소화기가 배치되어 있지 않음"),
    "N-34": ("인화성 자재 옆 용접 작업", "용접 작업 구역", "인화성 자재 옆에서 용접 작업을 하고 있음"),
    "N-35": ("자재야적장 내 난로 설치 시 소화기 미배치", "자재야적장", "난로가 설치된 자재야적장에 소화기가 배치되어 있지 않음"),
    "N-36": ("자재야적장 내 산소절단기 배치 시 소화기 미배치", "산소절단기", "산소절단기가 있는 자재야적장에 소화기가 배치되어 있지 않음"),
    "N-37": ("위험물저장소 외부 소화기 미배치", "위험물저장소", "위험물저장소 외부에 소화기가 배치되어 있지 않음"),
    "N-38": ("위험물저장소 외부 인화성 자재 배치", "위험물저장소", "위험물저장소 외부에 인화성 자재가 놓여 있음"),
    "N-39": ("원형톱 작업 시 소화기 미배치(불티비산방지막)", "원형톱", "원형톱 작업 중 소화기·불티비산방지막이 배치되어 있지 않음"),
    "N-40": ("원형톱 작업 시 인화성 자재 배치", "원형톱", "원형톱 작업 근처에 인화성 자재가 놓여 있음"),
    # 전도 (E)
    "N-41": ("동바리 버팀장치 설치 불량", "동바리", "동바리 받침판에 쐐기목 등 버팀장치가 설치되어 있지 않음"),
    "N-42": ("거푸집 버팀장치 설치 불량", "거푸집 버팀대", "거푸집 버팀대에 쐐기목 등 버팀장치가 설치되어 있지 않음"),
    "N-43": ("말비계 위 적재물 끝단 적치", "말비계", "말비계 끝단에 적재물이 놓여 있음"),
    "N-44": ("말비계 적재물 위 설치", "말비계", "말비계가 적재물 위에 설치되어 있음"),
    "N-45": ("콘크리트 펌프카 안전장치 불량", "콘크리트 펌프카", "콘크리트 펌프카의 전도방지대(안전장치)가 설치되어 있지 않음"),
    "N-46": ("콘크리트 펌프카 작업 시 신호수 미배치", "콘크리트 펌프카", "콘크리트 펌프카 작업 중 신호수가 배치되어 있지 않음"),
    "N-47": ("오거크레인 작업 시 안전장치 불량", "오거크레인", "오거크레인의 전도방지대(안전장치)가 설치되어 있지 않음"),
    "N-48": ("오거크레인 작업 시 신호수 미배치", "오거크레인", "오거크레인 작업 중 신호수가 배치되어 있지 않음"),
    "N-49": ("지게차 신호수 미배치", "지게차", "지게차 작업 중 신호수가 배치되어 있지 않음"),
    "N-50": ("지게차 운전자 시야 초과 자재 과적재", "지게차", "지게차에 운전자 시야를 넘는 높이로 자재가 실려 있음"),
}

KOREAN_FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"


def describe_location(bbox):
    """bbox 위치를 '사진 왼쪽 아래' 같은 말로 바꾼다."""
    if bbox["width"] * bbox["height"] > 0.6:
        return "사진 전체"
    cx = bbox["x"] + bbox["width"] / 2
    cy = bbox["y"] + bbox["height"] / 2
    horiz = "왼쪽" if cx < 1 / 3 else "오른쪽" if cx > 2 / 3 else ""
    vert = "위쪽" if cy < 1 / 3 else "아래쪽" if cy > 2 / 3 else ""
    if not horiz and not vert:
        return "사진 가운데"
    return "사진 " + " ".join(p for p in (horiz, vert) if p)


class DummyDetector:
    """AI Hub 라벨 json 의 정답을 탐지 결과처럼 돌려주는 가짜 탐지기."""

    def __init__(self, label_root, facility_only=True):
        self.facility_only = facility_only
        # 파일 이름(확장자 제외) → json 경로
        self.labels = {p.stem: p for p in Path(label_root).rglob("*.json")}
        if not self.labels:
            raise FileNotFoundError(f"라벨 json 이 없어요: {label_root}")

    def detect(self, frame=None, image_name=None):
        # 더미는 사진 내용(frame)을 보지 않고 이름으로 정답을 찾는다
        if image_name is None:
            raise ValueError("더미 탐지기는 image_name 이 필요해요.")
        path = self.labels.get(Path(image_name).stem)
        if path is None:
            return []

        data = json.loads(path.read_text(encoding="utf-8"))
        width, height = (float(v) for v in data["Raw_Data_Info."]["Resolution"].split(","))

        results = []
        for ann in data["Learning_Data_Info."]["Annotations"]:
            code = ann["class_ID"]
            # 위험 상황은 bbox 로 표시된 N 코드
            if ann["type"] != "bbox" or not code.startswith("N-"):
                continue
            if self.facility_only and code not in HAZARDS:
                continue
            # AI Hub bbox 는 [x, y, 가로, 세로] 픽셀
            x, y, w, h = ann["value"]
            bbox = {
                "x": round(x / width, 4),
                "y": round(y / height, 4),
                "width": round(w / width, 4),
                "height": round(h / height, 4),
            }
            results.append(make_detection(code, bbox, 1.0))  # 더미라서 항상 1.0
        return results


HAZARD_CODES = {name: code for code, (name, _, _) in HAZARDS.items()}


def hazard_code(name):
    """위험 이름 → AI Hub 코드 (표에 없는 코드는 이름이 곧 코드)."""
    return HAZARD_CODES.get(name, name if name in HAZARDS or name.startswith("N-") else None)


def make_detection(code, bbox, confidence):
    """위험 코드 + bbox → 백엔드 detections 항목 하나."""
    hazard, target, evidence = HAZARDS.get(code, (code, "위험 구역", f"{code} 위험 상황이 감지됨"))
    return {
        "hazard": hazard,
        "location": f"{describe_location(bbox)} {target}",
        "evidence": evidence,
        "bbox": bbox,
        "confidence": confidence,
    }


class YoloDetector:
    """학습한 YOLO 모델로 위험을 찾는 실제 탐지기.

    weights         : 위험 상태 모델 (runs/yolo11n/weights/best.pt) → detections
    objects_weights : 객체 탐지 모델 (runs/objects/weights/best.pt) → relations (선택)
    people          : 공간관계에 넣을 사람 객체 (none / signal / all, spatial_relations.PEOPLE)
    """

    def __init__(self, weights, objects_weights=None, conf=0.5, objects_conf=0.25, people="signal", device=None):
        import torch
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.objects_model = YOLO(str(objects_weights)) if objects_weights else None
        self.conf, self.objects_conf, self.people = conf, objects_conf, people
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

    def _boxes(self, model, frame, conf):
        """YOLO 예측 → [(코드, bbox 0~1, confidence)], 가로/세로 비. frame 은 이미지 배열(BGR) 또는 경로."""
        r = model.predict(frame, conf=conf, device=self.device, verbose=False)[0]
        h, w = r.orig_shape
        boxes = []
        for (x0, y0, x1, y1), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)
            bbox = {"x": round(x0 / w, 4), "y": round(y0 / h, 4),
                    "width": round((x1 - x0) / w, 4), "height": round((y1 - y0) / h, 4)}
            boxes.append((r.names[int(k)], bbox, round(c, 4)))
        return boxes, w / h

    def detect(self, frame=None, image_name=None):
        if frame is None:
            frame = str(image_name)
        boxes, _ = self._boxes(self.model, frame, self.conf)
        return [make_detection(code, bbox, c) for code, bbox, c in boxes]

    def relations(self, frame, detections=()):
        """객체 탐지 → spatial_relations.analyze() 결과
        (objects · relations · continuity · findings · evaluation_status · reason_code · checks).
        detections: 같은 장면의 위험 상태 탐지 결과 → 위험별로 규칙 판단 원인(checks)을 붙인다."""
        if self.objects_model is None:
            return None
        from spatial_relations import analyze

        boxes, aspect = self._boxes(self.objects_model, frame, self.objects_conf)
        objects = [{"code": code, "bbox": [b["x"], b["y"], b["width"], b["height"]], "confidence": c}
                   for code, b, c in boxes]
        return analyze(objects, aspect=aspect, people=self.people,
                       hazard_codes=[hazard_code(d["hazard"]) for d in detections])


DEFAULT_SITE_ID = "SITE-001"
DEFAULT_SITE_NAME = "미지정 현장"


def make_assessment_id(source_type, photo_id=None, video_id=None, frame_index=None):
    """위험성평가 ID 를 따로 정하지 않았을 때 자동으로 만든다 (결과마다 겹치지 않게)."""
    if source_type == "video":
        return f"RA-V{video_id}-F{frame_index}"
    return f"RA-P-{photo_id}"


def build_result(detections, source_type, image_url, photo_id=None,
                 video_id=None, frame_index=None, frame_time_ms=None, relations=None,
                 full=False, assessment_id=None, site_id=DEFAULT_SITE_ID, site_name=DEFAULT_SITE_NAME,
                 work_name=None, work_type=None, work_location=None):
    """AI 탐지 결과 JSON 을 만든다. 사진이면 photo_id, 영상이면 video_id/frame_* 가 필수.

    기본 (AI 출력 형식):
      {source_type, photo_id, video_id, frame_index, frame_time_ms, image_url, detections[], relations}
      relations = 객체 간 공간관계 (객체 탐지 모델을 쓸 때만 붙음)

    full=True (위험성평가 전체 형식, 세희 쪽 · 노션 「JSON 필드 및 자료형」):
      {assessment_id, site_info{...}, work_info{...} | null, source{...}, detections[], relations}
      assessment_id 를 안 주면 자동으로 만든다. work_* 를 하나도 안 주면 work_info 는 null.
    """
    if source_type == "photo":
        if photo_id is None:
            raise ValueError("사진은 photo_id 가 필요해요.")
        video_id = frame_index = frame_time_ms = None
    elif source_type == "video":
        if None in (video_id, frame_index, frame_time_ms):
            raise ValueError("영상은 video_id, frame_index, frame_time_ms 가 필요해요.")
        photo_id = None
    else:
        raise ValueError("source_type 은 'photo' 또는 'video' 여야 해요.")

    source = {
        "source_type": source_type,
        "photo_id": photo_id,
        "video_id": video_id,
        "frame_index": frame_index,
        "frame_time_ms": frame_time_ms,
        "image_url": image_url,
    }
    if full:
        work = (work_name, work_type, work_location)
        result = {
            "assessment_id": assessment_id or make_assessment_id(source_type, photo_id, video_id, frame_index),
            "site_info": {"site_id": site_id, "site_name": site_name},
            "work_info": None if work == (None, None, None) else {
                "work_name": work_name, "work_type": work_type, "location": work_location},
            "source": source,
            "detections": detections,
        }
    else:
        result = {**source, "detections": detections}
    if relations is not None:
        result["relations"] = relations
    return result


def add_assessment_args(parser):
    """--full: 위험성평가 전체 형식으로 출력 (+ 현장 · 작업 정보 옵션)."""
    g = parser.add_argument_group("위험성평가 전체 형식 (--full 일 때만 사용)")
    g.add_argument("--full", action="store_true",
                   help="assessment_id · site_info · work_info 를 붙인 전체 형식으로 출력 (기본: AI 출력 형식)")
    g.add_argument("--assessment-id", help="위험성평가 ID (기본: 자동, 예: RA-V3-F367)")
    g.add_argument("--site-id", default=DEFAULT_SITE_ID, help=f"현장 ID (기본 {DEFAULT_SITE_ID})")
    g.add_argument("--site-name", default=DEFAULT_SITE_NAME, help=f"현장명 (기본 {DEFAULT_SITE_NAME})")
    g.add_argument("--work-name", help="작업명 (예: 굴착 작업)")
    g.add_argument("--work-type", help="작업 유형 (예: 시설물 안전 점검)")
    g.add_argument("--work-location", help="작업 위치 (예: 굴착부 주변)")


def assessment_kwargs(args):
    """add_assessment_args 옵션 → build_result 인자."""
    return {"full": args.full, "assessment_id": args.assessment_id, "site_id": args.site_id,
            "site_name": args.site_name, "work_name": args.work_name, "work_type": args.work_type,
            "work_location": args.work_location}


def add_detector_args(parser):
    """detector.py · detect_video.py 공통 옵션."""
    parser.add_argument("--weights", type=Path, help="위험 상태 YOLO 모델 (예: runs/yolo11n/weights/best.pt)")
    parser.add_argument("--objects-weights", type=Path, help="객체 탐지 YOLO 모델 → 공간관계(relations) 추가")
    parser.add_argument("--conf", type=float, default=0.5, help="위험으로 볼 최소 confidence (기본 0.5)")
    parser.add_argument("--people", default="signal", choices=["none", "signal", "all"],
                        help="공간관계에 넣을 사람: none / signal(신호수만, 기본) / all")
    parser.add_argument("--labels", type=Path, help="라벨 json 폴더 (--weights 없이 더미 탐지기로 테스트할 때)")
    parser.add_argument("--all", action="store_true", help="(더미) 표(HAZARDS)에 없는 코드도 포함")


def make_detector(args):
    if args.weights:
        return YoloDetector(args.weights, args.objects_weights, conf=args.conf, people=args.people)
    if args.labels:
        return DummyDetector(args.labels, facility_only=not args.all)
    raise SystemExit("--weights (YOLO 모델) 또는 --labels (더미 탐지기) 중 하나가 필요해요.")


def draw(image_path, detections, out_path):
    """bbox 를 사진에 그려서 저장 (보고서용 위험 장면 이미지)."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.open(image_path).convert("RGB")
    dr = ImageDraw.Draw(im)
    W, H = im.size
    size = max(20, H // 30)
    try:
        font = ImageFont.truetype(KOREAN_FONT, size)
    except OSError:
        font = ImageFont.load_default(size=size)
    for det in detections:
        b = det["bbox"]
        x0, y0 = b["x"] * W, b["y"] * H
        x1, y1 = x0 + b["width"] * W, y0 + b["height"] * H
        dr.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=6)
        label = f'{det["hazard"]} {det["confidence"]:.2f}'
        tx, ty = x0, max(0, y0 - size - 12)
        dr.rectangle(dr.textbbox((tx, ty), label, font=font), fill=(255, 0, 0))
        dr.text((tx, ty), label, fill=(255, 255, 255), font=font)
    im.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    add_detector_args(parser)
    add_assessment_args(parser)
    parser.add_argument("--draw", type=Path, help="bbox 그린 사진 저장 경로")
    args = parser.parse_args()

    detector = make_detector(args)
    if isinstance(detector, YoloDetector):
        detections = detector.detect(str(args.image))
        relations = detector.relations(str(args.image), detections)
    else:
        detections, relations = detector.detect(image_name=args.image.name), None

    # 이미지 업로드 방식이 정해지기 전까지 image_url 은 로컬 경로
    image_path = args.draw or args.image
    result = build_result(detections, "photo", image_url=str(image_path), photo_id=args.image.stem,
                          relations=relations, **assessment_kwargs(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not detections:
        print("위험이 감지되지 않았어요.")

    if args.draw:
        draw(args.image, detections, args.draw)
        print(f"저장: {args.draw}")


if __name__ == "__main__":
    main()
