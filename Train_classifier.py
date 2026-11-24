"""
Train a HOG + linear SVM pedestrian classifier on PA-100k.

PA-100k only contains pedestrian images (positives). Negatives (no people) come from:
  1. --neg_dir <folder>   your own folder of person-free images, or
  2. Hugging Face INRIA   (automatic, default): `marcelarosalesj/inria-person`,
                          'no_pedestrians' class, cached in ./inria_negatives
  3. --no_hf              weak fallback: corner patches of PA-100k images

Things done to reduce false positives:
  * grayscale HOG (removes colour as a shortcut between the two datasets)
  * blur / JPEG / contrast augmentation applied to BOTH classes, negatives more strongly,
    so "blurry = person" cannot be learned (PA-100k is blurry, INRIA negatives are sharp)
  * partial-body / wrong-scale windows cut from PA-100k used as extra negatives, so
    misaligned boxes (half a person, too-large windows) are rejected
  * several rounds of hard-negative mining on the negative images

Usage (from the folder containing train.csv, val.csv, test.csv, data/):
    python train_classifier.py

Outputs:
    pedestrian_hog_svm.joblib   full scikit-learn model
    pedestrian_hog_svm.npz      linear weights (used by pedestrian_app.py)
"""
import argparse
import random
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC

WIN = (64, 128)  # (width, height) of the HOG window
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MINE_MAX_SIDE = 960          # negative images are shrunk to this for mining (same as the app)

# (probability of down/up-scale blur, smallest scale factor)
POS_DEG = (0.5, 0.5)
NEG_DEG = (0.9, 0.3)


# --------------------------------------------------------------------------- #
# HOG helpers
# --------------------------------------------------------------------------- #
def make_hog():
    # winSize, blockSize, blockStride, cellSize, nbins  -> 3780 features
    return cv2.HOGDescriptor(WIN, (16, 16), (8, 8), (8, 8), 9)


_HOG = None


def degrade(gray, rng, p_blur, fmin):
    """Random resolution loss / blur / JPEG / contrast change on a 64x128 gray image."""
    out = gray
    if rng.random() < p_blur:
        f = rng.uniform(fmin, 1.0)
        sw, sh = max(8, int(WIN[0] * f)), max(16, int(WIN[1] * f))
        out = cv2.resize(out, (sw, sh), interpolation=cv2.INTER_AREA)
        out = cv2.resize(out, WIN, interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 1.2))
    if rng.random() < 0.3:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(30, 90)])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if rng.random() < 0.4:
        out = cv2.convertScaleAbs(out, alpha=rng.uniform(0.7, 1.3), beta=rng.uniform(-25, 25))
    return out


def hog_feat(img, rng=None, deg=POS_DEG):
    """HOG of a (BGR or gray) image: grayscale -> 64x128 -> optional degradation."""
    global _HOG
    if _HOG is None:
        _HOG = make_hog()
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, WIN, interpolation=cv2.INTER_AREA)
    if rng is not None:
        gray = degrade(gray, rng, *deg)
    return _HOG.compute(gray).ravel().astype(np.float32)


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #
def resolve_image(data_dir: Path, name):
    """Find the image file for a csv entry (handles missing extension / folders)."""
    name = str(name).strip()
    p = data_dir / name
    if p.is_file():
        return p
    for ext in IMG_EXT:
        q = data_dir / (name + ext)
        if q.is_file():
            return q
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                return f
    return None


def read_split(root: Path, split: str):
    csv_path = root / f"{split}.csv"
    data_dir = root / "data"
    df = pd.read_csv(csv_path)
    names = df.iloc[:, 0].tolist()
    # If the csv had no header, the first row was swallowed as the header
    if resolve_image(data_dir, df.columns[0]) is not None:
        names = [df.columns[0]] + names
    paths = []
    for n in names:
        p = resolve_image(data_dir, n)
        if p is not None:
            paths.append(p)
    print(f"[{split}] {len(paths)}/{len(names)} images found")
    return paths


def list_images(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)


def download_hf_negatives(repo: str, cache_dir: Path):
    """
    Download the person-free class of an imagefolder-style HF dataset and save
    the images to cache_dir. Skipped if cache_dir already contains images.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = list_images(cache_dir)
    if existing:
        print(f"Using {len(existing)} cached negatives in {cache_dir}")
        return cache_dir

    try:
        from datasets import ClassLabel, Image as HFImage, load_dataset
    except ImportError:
        raise SystemExit("Missing package: run `pip install datasets` "
                         "(or pass --neg_dir / --no_hf).")

    print(f"Downloading '{repo}' from Hugging Face ...")
    ds = load_dataset(repo)

    saved = 0
    for split_name, split in ds.items():
        img_col = next((k for k, v in split.features.items() if isinstance(v, HFImage)), None)
        lab_col = next((k for k, v in split.features.items() if isinstance(v, ClassLabel)), None)
        if img_col is None or lab_col is None:
            raise SystemExit(f"Could not find image/label columns. Features: {split.features}")

        names = split.features[lab_col].names
        neg_ids = {i for i, n in enumerate(names)
                   if n.lower().startswith(("no", "non", "neg", "background"))}
        print(f"  split '{split_name}': classes={names} -> negative ids={sorted(neg_ids)}")
        if not neg_ids:
            raise SystemExit(f"No negative-looking class among {names}; use --neg_dir instead.")

        labels = split[lab_col]  # column access: does not decode images
        for i, lab in enumerate(labels):
            if lab in neg_ids:
                img = split[i][img_col].convert("RGB")
                img.save(cache_dir / f"{split_name}_{i:06d}.png")
                saved += 1

    print(f"Saved {saved} negative images to {cache_dir}")
    if saved == 0:
        raise SystemExit("No negative images were saved.")
    return cache_dir


# --------------------------------------------------------------------------- #
# Feature workers (run in parallel)
# --------------------------------------------------------------------------- #
def pos_feats(path, flip, seed):
    img = cv2.imread(str(path))
    if img is None:
        return []
    rng = random.Random(seed)
    out = [hog_feat(img, rng, POS_DEG)]
    if flip:
        out.append(hog_feat(cv2.flip(img, 1), rng, POS_DEG))
    return out


def neg_feats(path, n, seed):
    """n random 1:2 crops (log-uniform scale) from a person-free image."""
    img = cv2.imread(str(path))
    if img is None:
        return []
    h, w = img.shape[:2]
    if h < WIN[1] or w < WIN[0]:
        s = max(WIN[0] / w, WIN[1] / h)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
    rng = random.Random(seed)
    smax = max(1.0, min(w / WIN[0], h / WIN[1]))
    out = []
    for _ in range(n):
        s = float(np.exp(rng.uniform(0.0, np.log(smax)))) if smax > 1.0 else 1.0
        ww, hh = min(int(WIN[0] * s), w), min(int(WIN[1] * s), h)
        x = rng.randint(0, w - ww)
        y = rng.randint(0, h - hh)
        out.append(hog_feat(img[y:y + hh, x:x + ww], rng, NEG_DEG))
    return out


def corner_feats(path, n, seed):
    """Weak fallback negatives: small corner patches of a PA-100k image."""
    img = cv2.imread(str(path))
    if img is None:
        return []
    h, w = img.shape[:2]
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        ch = max(16, int(h * rng.uniform(0.15, 0.30)))
        cw = max(8, ch // 2)
        x = 0 if rng.random() < 0.5 else max(0, w - cw)
        y = 0 if rng.random() < 0.5 else max(0, h - ch)
        out.append(hog_feat(img[y:y + ch, x:x + cw], rng, NEG_DEG))
    return out


def partial_feats(path, seed):
    """
    Wrong-alignment negatives made from a pedestrian image:
      - only the top / bottom / left / right part of the person
      - the person shrunk inside a window much larger than them (context padding)
    """
    img = cv2.imread(str(path))
    if img is None:
        return []
    h, w = img.shape[:2]
    rng = random.Random(seed)
    mode = rng.choice(["top", "bottom", "left", "right", "context", "context"])
    if mode == "top":
        crop = img[: int(h * rng.uniform(0.35, 0.6))]
    elif mode == "bottom":
        crop = img[int(h * rng.uniform(0.4, 0.65)):]
    elif mode == "left":
        crop = img[:, : int(w * rng.uniform(0.35, 0.6))]
    elif mode == "right":
        crop = img[:, int(w * rng.uniform(0.4, 0.65)):]
    else:
        ph, pw = int(h * rng.uniform(0.4, 1.0)), int(w * rng.uniform(0.4, 1.0))
        crop = cv2.copyMakeBorder(img, ph, ph, pw, pw, cv2.BORDER_REPLICATE)
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return []
    return [hog_feat(crop, rng, NEG_DEG)]


def mine_hard(path, w, b, max_per_img, seed):
    """Return HOG features of false-positive windows on a person-free image."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    h, wd = img.shape[:2]
    s = MINE_MAX_SIDE / max(h, wd)
    if s < 1.0:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, wd = img.shape[:2]
    if h < WIN[1] or wd < WIN[0]:
        return []
    hog = make_hog()
    hog.setSVMDetector(np.append(w, b).astype(np.float32))
    rects, weights = hog.detectMultiScale(img, 0.0, (8, 8), (0, 0), 1.1, 0.0)
    if len(rects) == 0:
        return []
    order = np.argsort(-np.array(weights).ravel())[:max_per_img]
    rng = random.Random(seed)
    out = []
    for i in order:
        x, y, rw, rh = rects[i]
        x, y = max(0, x), max(0, y)
        crop = img[y:y + rh, x:x + rw]
        if crop.size:
            out.append(hog_feat(crop, rng, NEG_DEG))
    return out


def run(fn, items, n_jobs, desc):
    print(f"  extracting: {desc} ({len(items)} items)")
    res = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(fn)(*it) for it in items)
    return [f for r in res for f in r]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_negatives(split_imgs, split_pa_paths, args, seed_base):
    if split_imgs is not None:
        items = [(p, args.neg_per_image, seed_base + i) for i, p in enumerate(split_imgs)]
        return run(neg_feats, items, args.n_jobs, "negatives (background images)")
    items = [(p, 2, seed_base + i) for i, p in enumerate(split_pa_paths)]
    return run(corner_feats, items, args.n_jobs, "negatives (corner fallback)")


def evaluate(name, clf, Xp, Xn):
    X = np.vstack([Xp, Xn])
    y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
    pred = clf.predict(X)
    print(f"\n===== {name} =====")
    print(confusion_matrix(y, pred))
    print(classification_report(y, pred, target_names=["background", "pedestrian"], digits=4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="PA-100k folder (default: ./PA-100k, or the current folder if it holds train.csv)")
    ap.add_argument("--neg_dir", default=None,
                    help="your own folder of person-free images (overrides Hugging Face)")
    ap.add_argument("--hf_dataset", default="marcelarosalesj/inria-person")
    ap.add_argument("--neg_cache", default="inria_negatives")
    ap.add_argument("--no_hf", action="store_true",
                    help="don't use Hugging Face; weak corner-patch negatives")
    ap.add_argument("--out", default="pedestrian_hog_svm", help="output prefix")
    ap.add_argument("--max_train", type=int, default=20000, help="max positive train images (0 = all)")
    ap.add_argument("--max_eval", type=int, default=3000, help="max positive val/test images")
    ap.add_argument("--neg_per_image", type=int, default=40, help="random windows per negative image")
    ap.add_argument("--partial_ratio", type=float, default=0.5,
                    help="partial/misaligned negatives from PA-100k, as a fraction of #positives")
    ap.add_argument("--mine_rounds", type=int, default=3, help="hard-negative mining rounds")
    ap.add_argument("--mine_per_image", type=int, default=20)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--no_flip", action="store_true")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    if args.root:
        root = Path(args.root)
    elif Path("PA-100k/train.csv").is_file():
        root = Path("PA-100k")
    elif Path("train.csv").is_file():
        root = Path(".")
    else:
        raise SystemExit("Could not find PA-100k. Put the script next to the 'PA-100k' folder "
                         "(or inside it), or pass --root /path/to/PA-100k")
    if not (root / "train.csv").is_file() or not (root / "data").is_dir():
        raise SystemExit(f"'{root}' must contain train.csv, val.csv, test.csv and a 'data' folder.")
    print(f"Using PA-100k root: {root.resolve()}")

    train_p = read_split(root, "train")
    val_p = read_split(root, "val")
    test_p = read_split(root, "test")
    random.shuffle(train_p)
    random.shuffle(val_p)
    random.shuffle(test_p)
    if args.max_train:
        train_p = train_p[:args.max_train]
    val_p = val_p[:args.max_eval]
    test_p = test_p[:args.max_eval]

    # ---- negative images, split by IMAGE (no leakage between train/val/test) ----
    neg_folder = None
    if args.neg_dir:
        neg_folder = Path(args.neg_dir)
    elif not args.no_hf:
        neg_folder = download_hf_negatives(args.hf_dataset, Path(args.neg_cache))

    neg_tr = neg_va = neg_te = None
    if neg_folder is not None:
        negs = list_images(neg_folder)
        if len(negs) < 10:
            raise SystemExit(f"Only {len(negs)} negative images found in {neg_folder}; need more.")
        random.shuffle(negs)
        a, b = int(0.7 * len(negs)), int(0.85 * len(negs))
        neg_tr, neg_va, neg_te = negs[:a], negs[a:b], negs[b:]
        print(f"negative images: train {len(neg_tr)}, val {len(neg_va)}, test {len(neg_te)}")
    else:
        print("WARNING: using weak corner-patch negatives (no negative images).")

    # ---- features ----
    print("\nExtracting HOG features ...")
    Xp = run(pos_feats, [(p, not args.no_flip, i) for i, p in enumerate(train_p)],
             args.n_jobs, "train positives")
    Xn = build_negatives(neg_tr, train_p[::-1], args, 0)

    n_part = int(args.partial_ratio * len(train_p))
    if n_part > 0:
        part = run(partial_feats, [(p, 5_000_000 + i) for i, p in enumerate(train_p[:n_part])],
                   args.n_jobs, "partial / misaligned negatives")
        Xn = Xn + part

    Xp_va = run(pos_feats, [(p, False, 7_000_000 + i) for i, p in enumerate(val_p)],
                args.n_jobs, "val positives")
    Xn_va = build_negatives(neg_va, val_p[::-1], args, 10_000_000)
    Xp_te = run(pos_feats, [(p, False, 8_000_000 + i) for i, p in enumerate(test_p)],
                args.n_jobs, "test positives")
    Xn_te = build_negatives(neg_te, test_p[::-1], args, 20_000_000)

    Xp, Xn = np.array(Xp), np.array(Xn)
    Xp_va, Xn_va = np.array(Xp_va), np.array(Xn_va)
    Xp_te, Xn_te = np.array(Xp_te), np.array(Xn_te)
    print(f"train: {len(Xp)} pos, {len(Xn)} neg | feature dim {Xp.shape[1]}")

    def fit(Xp_, Xn_):
        X = np.vstack([Xp_, Xn_])
        y = np.r_[np.ones(len(Xp_)), np.zeros(len(Xn_))]
        clf = LinearSVC(C=args.C, dual=False, max_iter=5000, class_weight="balanced")
        clf.fit(X, y)
        return clf

    print("\nTraining LinearSVC ...")
    clf = fit(Xp, Xn)
    evaluate("VAL (round 0)", clf, Xp_va, Xn_va)

    # ---- hard-negative mining, several rounds ----
    if neg_tr is not None:
        for r in range(1, args.mine_rounds + 1):
            print(f"\nHard-negative mining, round {r}/{args.mine_rounds} ...")
            w, b = clf.coef_.ravel(), float(clf.intercept_[0])
            items = [(p, w, b, args.mine_per_image, r * 1_000_000 + i)
                     for i, p in enumerate(neg_tr)]
            hard = run(mine_hard, items, args.n_jobs, "hard negatives")
            print(f"  mined {len(hard)} false-positive windows")
            if len(hard) < 20:
                print("  (almost none left, stopping mining)")
                break
            Xn = np.vstack([Xn, np.array(hard)])
            print(f"  retraining on {len(Xn)} negatives ...")
            clf = fit(Xp, Xn)
            evaluate(f"VAL (after mining round {r})", clf, Xp_va, Xn_va)

    evaluate("TEST", clf, Xp_te, Xn_te)

    joblib.dump(clf, f"{args.out}.joblib")
    np.savez(f"{args.out}.npz", w=clf.coef_.ravel().astype(np.float32),
             b=np.float32(clf.intercept_[0]), gray=np.int8(1))
    print(f"\nSaved {args.out}.joblib and {args.out}.npz")


if __name__ == "__main__":
    main()