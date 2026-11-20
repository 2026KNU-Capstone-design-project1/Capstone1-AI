import argparse
import csv
import json
from pathlib import Path

import torch
from ultralytics import YOLO

from detector import HAZARDS
from spatial_relations import OBJECTS


def pick_device(name):
    """auto 면 사용 가능한 가장 빠른 장치를 고른다 (CUDA → Apple MPS → CPU)."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_list(data_yaml, split):
    """data.yaml 의 사진 목록 txt (./images/...) → 사진 경로 목록."""
    root = data_yaml.parent
    lines = (root / f"yolo_{split}.txt").read_text(encoding="utf-8").split()
    return [root / line for line in lines]


def image_auc(model, data_yaml, split, device, imgsz):
    """사진 단위 위험/정상 AUC-ROC. 점수 = 사진에서 가장 높은 박스 신뢰도."""
    scores, labels = [], []
    images = read_list(data_yaml, split)
    for i in range(0, len(images), 32):
        batch = images[i:i + 32]
        for img, res in zip(batch, model.predict(batch, device=device, imgsz=imgsz, conf=0.001, verbose=False)):
            conf = res.boxes.conf
            scores.append(float(conf.max()) if len(conf) else 0.0)
            txt = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
            labels.append(1 if txt.exists() and txt.read_text().strip() else 0)

    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    # AUC = 위험 사진 점수가 정상 사진 점수보다 높을 확률 (동점은 0.5)
    order = sorted(scores)
    rank = {}
    for r, s in enumerate(order, 1):
        rank.setdefault(s, []).append(r)
    avg = {s: sum(r) / len(r) for s, r in rank.items()}
    return (sum(avg[s] for s in pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def evaluate(weights, data_yaml, device, imgsz, out_dir):
    """test 세트 평가 후 json · csv 로 저장한다."""
    model = YOLO(weights)
    m = model.val(data=str(data_yaml), split="test", device=device, imgsz=imgsz,
                  project=str(out_dir), name="test", exist_ok=True, plots=True)
    box = m.box
    f1 = 2 * box.mp * box.mr / (box.mp + box.mr) if box.mp + box.mr else 0.0
    auc = image_auc(model, data_yaml, "test", device, imgsz)
    summary = {
        "weights": str(weights),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "f1": round(float(f1), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "image_auc_roc": round(auc, 4) if auc is not None else None,
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for i, c in enumerate(box.ap_class_index):
        code = m.names[int(c)]
        p, r = float(box.p[i]), float(box.r[i])
        rows.append({
            "class": code,
            "hazard": HAZARDS[code][0] if code in HAZARDS else OBJECTS.get(code, ("",))[0],
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "mAP50": round(float(box.ap50[i]), 4),
            "mAP50-95": round(float(box.ap[i]), 4),
        })
    with open(out_dir / "test_per_class.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== test 세트 성능 =====")
    for k, v in summary.items():
        if k != "weights":
            print(f"  {k:14s} {v}")
    worst = sorted(rows, key=lambda r: r["recall"])[:5]
    print("\n재현율이 가장 낮은 클래스 5개:")
    for r in worst:
        print(f"  {r['class']:6s} {r['hazard']:24s} 재현율 {r['recall']:.3f}  mAP50 {r['mAP50']:.3f}")
    print(f"\n저장: {out_dir / 'test_metrics.json'}, {out_dir / 'test_per_class.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset/data.yaml"), help="convert_to_yolo.py 가 만든 설정 파일")
    parser.add_argument("--model", default="n", help="모델 크기 n / s / m (또는 가중치 .pt 경로)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20, help="검증 성능이 이만큼 안 오르면 조기 종료")
    parser.add_argument("--fraction", type=float, default=1.0, help="학습 사진 중 사용할 비율 (동작 확인용)")
    parser.add_argument("--device", default="auto", help="auto / mps / cpu / 0(GPU)")
    parser.add_argument("--project", type=Path, default=Path("runs"), help="결과를 저장할 폴더")
    parser.add_argument("--name", help="실험 이름 (기본: yolo11<크기>)")
    parser.add_argument("--eval", type=Path, metavar="WEIGHTS", help="학습하지 않고 이 가중치로 test 평가만")
    args = parser.parse_args()

    data_yaml = args.data.resolve()
    if not data_yaml.exists():
        raise SystemExit(f"설정 파일이 없어요: {data_yaml}\n먼저 convert_to_yolo.py 를 실행하세요.")
    device = pick_device(args.device)
    print(f"장치: {device}")

    if args.eval:
        evaluate(args.eval, data_yaml, device, args.imgsz, args.eval.parent.parent)
        return

    weights = args.model if args.model.endswith(".pt") else f"yolo11{args.model}.pt"
    name = args.name or Path(weights).stem
    model = YOLO(weights)
    model.train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.patience, fraction=args.fraction, device=device,
        project=str(args.project.resolve()), name=name, exist_ok=True, seed=42, plots=True,
    )

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    print(f"\n학습 완료! 가장 좋은 가중치: {best}")
    evaluate(best, data_yaml, device, args.imgsz, run_dir)


if __name__ == "__main__":
    main()
