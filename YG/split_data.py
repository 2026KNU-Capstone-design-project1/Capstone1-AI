import random
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHOTO_DIR = HERE / "dataset_photo"
VIDEO_DIR = HERE / "dataset_video"
OUTPUT_DIR = HERE / "dataset"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15 
SEED = 42 

PHOTO_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".mov", ".avi"}


def list_files(folder, exts):
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in exts)


def split(files):
    files = files[:]
    random.shuffle(files)
    n_train = int(len(files) * TRAIN_RATIO)
    n_val = int(len(files) * VAL_RATIO)
    return {
        "training": files[:n_train],
        "validation": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def main():
    random.seed(SEED)
    photos = list_files(PHOTO_DIR, PHOTO_EXTS)
    videos = list_files(VIDEO_DIR, VIDEO_EXTS)
    print(f"사진 {len(photos)}개, 영상 {len(videos)}개 발견")

    for files in (photos, videos):
        for split_name, group in split(files).items():
            target = OUTPUT_DIR / split_name
            target.mkdir(parents=True, exist_ok=True)
            for f in group:
                shutil.copy2(f, target / f.name) 

    for split_name in ("training", "validation", "test"):
        folder = OUTPUT_DIR / split_name
        n_photo = len(list_files(folder, PHOTO_EXTS))
        n_video = len(list_files(folder, VIDEO_EXTS))
        print(f"{split_name}: 사진 {n_photo}개, 영상 {n_video}개")
    print("완료!")


if __name__ == "__main__":
    main()
