"""
data_loader.py
==============
Loads images from final_dataset/, applies CLAHE, detects face, returns
flat normalised float32 arrays.  Multi-threaded for 35K+ datasets.
"""

import os, cv2, numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

IMG_SIZE    = 48
DATASET_DIR = "final_dataset"

_FACE1 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_FACE2 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def _detect_face(gray):
    for casc, sf, mn in [(_FACE1, 1.1, 4), (_FACE2, 1.05, 3)]:
        faces = casc.detectMultiScale(gray, sf, mn, minSize=(20, 20))
        if len(faces):
            return tuple(max(faces, key=lambda r: r[2] * r[3]))
    return None


def preprocess_image(path):

    try:

        img = cv2.imread(path)

        if img is None:
            return None

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        h_img, w_img = gray.shape[:2]

        bbox = _detect_face(gray)

        # ====================================================
        # SAFE FACE CROP
        # ====================================================

        if bbox is not None:

            x, y, w, h = bbox

            # sanitize coords
            x = max(0, x)
            y = max(0, y)

            w = max(1, w)
            h = max(1, h)

            # clamp dimensions
            if x + w > w_img:
                w = w_img - x

            if y + h > h_img:
                h = h_img - y

            # final validation
            if w > 0 and h > 0:

                roi = gray[y:y+h, x:x+w]

                if roi.size > 0:
                    gray = roi

        # ====================================================
        # CLAHE
        # ====================================================

        if gray is None or gray.size == 0:
            return None

        gray = _CLAHE.apply(gray)

        # ====================================================
        # RESIZE
        # ====================================================

        gray = cv2.resize(
            gray,
            (IMG_SIZE, IMG_SIZE)
        )

        return (
            gray.astype(np.float32) / 255.0
        ).flatten()

    except Exception:
        return None


def load_dataset(folder=DATASET_DIR, max_images=35000, n_workers=8):
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [str(p) for p in Path(folder).rglob("*")
             if p.suffix.lower() in supported][:max_images]
    print(f"[DataLoader] Found {len(files)} files in '{folder}'")

    images, paths, failed = [], [], 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(preprocess_image, f): f for f in files}
        done = 0
        for fut in as_completed(futs):
            arr = fut.result()
            done += 1
            if arr is not None:
                images.append(arr)
                paths.append(futs[fut])
            else:
                failed += 1
            if done % 5000 == 0:
                print(f"  ... {done}/{len(files)}")

    print(f"[DataLoader] Loaded {len(images)} faces | Skipped {failed}")
    if not images:
        raise RuntimeError(f"No valid faces found in '{folder}'")
    return np.array(images, dtype=np.float32), paths


def load_single_frame(frame):
    """BGR webcam frame -> flat float32 array."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bbox = _detect_face(gray)
    if bbox:
        x, y, w, h = bbox
        gray = gray[y:y+h, x:x+w]
    gray = _CLAHE.apply(gray)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    return (gray.astype(np.float32) / 255.0).flatten()


def get_face_bbox(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return _detect_face(gray)
