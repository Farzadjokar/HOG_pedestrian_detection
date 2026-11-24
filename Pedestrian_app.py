"""
Tkinter pedestrian detector (HOG + linear SVM).

- The model (pedestrian_hog_svm.npz or .joblib, produced by train_classifier.py)
  is loaded automatically from the script's folder or the current folder.
- Opening an image detects pedestrians immediately. No settings needed.
- An image pyramid is used, so pedestrians of any size are found: the image is
  repeatedly shrunk (large pedestrians end up inside the 64x128 window) and
  small images are upscaled first (small pedestrians).

Run:  python pedestrian_app.py   ->  click 'Open Image'
Requires: opencv-python, numpy, pillow (+ joblib/scikit-learn only for a .joblib model)
"""
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

WIN = (64, 128)              # HOG window (w, h), must match training
MODEL_NAMES = ("pedestrian_hog_svm.npz", "pedestrian_hog_svm.joblib")

# Fixed, sensible defaults (edit here if you ever want to tune them)
SCORE_THR = 0.7              # SVM score needed for a window to count as a hit
PYRAMID_STEP = 1.15          # shrink factor between pyramid levels
MAX_UPSCALE = 2.0            # how far small images are enlarged
DETECT_MAX_SIDE = 960        # pyramid starts with the longest side at (at most) this size
GROUP_IOU = 0.3              # hits overlapping this much are grouped into one pedestrian
MIN_NEIGHBORS = 3            # a pedestrian needs at least this many overlapping hits
CONTAIN_THR = 0.7            # drop a box if this fraction of it lies inside a stronger box


def make_hog():
    return cv2.HOGDescriptor(WIN, (16, 16), (8, 8), (8, 8), 9)


def find_model():
    for folder in (Path(__file__).resolve().parent, Path.cwd()):
        for name in MODEL_NAMES:
            p = folder / name
            if p.is_file():
                return p
    return None


def load_hog(path: Path):
    if path.suffix == ".npz":
        d = np.load(path)
        if "gray" not in d.files:
            raise ValueError("This model was made by an older train_classifier.py.\n"
                             "Retrain with the new version, then try again.")
        w, b = d["w"].ravel(), float(d["b"])
    else:
        import joblib
        clf = joblib.load(path)
        w, b = clf.coef_.ravel(), float(clf.intercept_[0])
    hog = make_hog()
    if len(w) != hog.getDescriptorSize():
        raise ValueError(f"Weight size {len(w)} != HOG size {hog.getDescriptorSize()}")
    hog.setSVMDetector(np.append(w, b).astype(np.float32))
    return hog


def _overlap(b, arr):
    """IoU and intersection area of box b=(x,y,w,h) with each row of arr."""
    x1 = np.maximum(b[0], arr[:, 0])
    y1 = np.maximum(b[1], arr[:, 1])
    x2 = np.minimum(b[0] + b[2], arr[:, 0] + arr[:, 2])
    y2 = np.minimum(b[1] + b[3], arr[:, 1] + arr[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = b[2] * b[3] + arr[:, 2] * arr[:, 3] - inter
    return inter / np.maximum(union, 1e-9), inter


def group_detections(boxes, scores):
    """
    Merge window hits into pedestrians. A real person produces many overlapping hits
    (neighbouring positions and pyramid levels); isolated hits are treated as noise.
    """
    if not boxes:
        return []
    B = np.array(boxes, dtype=np.float64)
    S = np.array(scores, dtype=np.float64)
    used = np.zeros(len(B), dtype=bool)
    clusters = []
    for i in np.argsort(-S):
        if used[i]:
            continue
        iou, _ = _overlap(B[i], B)
        members = (~used) & (iou > GROUP_IOU)
        members[i] = True
        used |= members
        if members.sum() < MIN_NEIGHBORS:
            continue
        wts = S[members] - SCORE_THR + 0.1
        box = (B[members] * wts[:, None]).sum(0) / wts.sum()
        clusters.append((box, float(S[members].max()), int(members.sum())))

    clusters.sort(key=lambda c: -c[1])
    kept = []
    for box, sc, n in clusters:
        area = box[2] * box[3]
        inside = False
        for kb, _, _ in kept:
            _, inter = _overlap(box, kb[None, :])
            if inter[0] / max(min(area, kb[2] * kb[3]), 1e-9) > CONTAIN_THR:
                inside = True
                break
        if not inside:
            kept.append((box, sc, n))
    return kept


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HOG Pedestrian Detector")
        self.geometry("1000x700")

        self.hog = None
        self.image = None
        self.result = None
        self._tk_img = None
        self.busy = False
        self._job = None

        self._build_ui()
        self.after(100, self._auto_load_model)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        bar = ttk.Frame(self, padding=8)
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(bar, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="Save Result", command=self.save_result).pack(side=tk.LEFT, padx=3)
        self.status = ttk.Label(bar, text="Loading model...", font=("TkDefaultFont", 10, "bold"))
        self.status.pack(side=tk.RIGHT, padx=8)

        self.canvas = tk.Canvas(self, bg="#222")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._show(
            self.result if self.result is not None else self.image))

    def _auto_load_model(self):
        path = find_model()
        if path is None:
            searched = f"{Path(__file__).resolve().parent}\nand\n{Path.cwd()}"
            messagebox.showwarning(
                "Model not found",
                f"Could not find {MODEL_NAMES[0]} (or .joblib) in:\n{searched}\n\n"
                "Run train_classifier.py first, or select the model file now.")
            path = filedialog.askopenfilename(
                title="Select trained model", filetypes=[("Model", "*.npz *.joblib")])
            if not path:
                self.status.config(text="No model loaded", foreground="red")
                return
            path = Path(path)
        try:
            self.hog = load_hog(path)
            self.status.config(text=f"Model: {path.name}", foreground="green")
        except Exception as e:
            self.status.config(text="Model failed to load", foreground="red")
            messagebox.showerror("Model error", str(e))

    # ----------------------------------------------------------- detection
    def detect(self, img):
        """Multi-scale detection: grayscale image pyramid + grouping of overlapping hits."""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # start by shrinking big images / enlarging small ones
        s = min(MAX_UPSCALE, DETECT_MAX_SIDE / max(h, w))

        boxes, scores = [], []
        while True:
            nw, nh = int(round(w * s)), int(round(h * s))
            if nw < WIN[0] or nh < WIN[1]:
                break
            if abs(s - 1.0) < 1e-6:
                level = gray
            else:
                interp = cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA
                level = cv2.resize(gray, (nw, nh), interpolation=interp)

            locs, wts = self.hog.detect(level, SCORE_THR, (8, 8), (0, 0))
            if len(locs):
                pts = np.array(locs).reshape(-1, 2)
                for (x, y), sc in zip(pts, np.array(wts).ravel()):
                    boxes.append([x / s, y / s, WIN[0] / s, WIN[1] / s])
                    scores.append(float(sc))
            s /= PYRAMID_STEP  # next, smaller level (larger pedestrians now fit the window)

        out = img.copy()
        thick = max(2, int(round(max(h, w) / 400)))
        found = group_detections(boxes, scores)
        for box, sc, _ in found:
            x, y, bw, bh = [int(round(v)) for v in box]
            cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), thick)
            cv2.putText(out, f"{sc:.2f}", (x, max(14, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thick / 2 + 0.3,
                        (0, 255, 0), max(1, thick // 2), cv2.LINE_AA)
        return out, len(found)

    def _run_detection(self):
        """Run detection in a worker thread so the window stays responsive."""
        self.busy = True
        self._job = None
        img = self.image

        def work():
            try:
                self._job = ("ok",) + self.detect(img)
            except Exception as e:  # report in the GUI thread
                self._job = ("err", str(e))

        threading.Thread(target=work, daemon=True).start()
        self.after(50, self._poll_detection)

    def _poll_detection(self):
        if self._job is None:
            self.after(50, self._poll_detection)
            return
        self.busy = False
        if self._job[0] == "ok":
            _, self.result, n = self._job
            self.status.config(text=f"Pedestrians detected: {n}", foreground="green")
            self._show(self.result)
        else:
            self.status.config(text="Detection failed", foreground="red")
            messagebox.showerror("Detection error", self._job[1])

    # --------------------------------------------------------------- image
    def open_image(self):
        if self.busy:
            return
        if self.hog is None:
            self._auto_load_model()
            if self.hog is None:
                return
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Could not read image.")
            return
        self.image = img
        self.status.config(text="Detecting...", foreground="black")
        self.update_idletasks()
        self._run_detection()  # automatic

    def _show(self, img):
        if img is None:
            return
        cw, ch = max(self.canvas.winfo_width(), 10), max(self.canvas.winfo_height(), 10)
        h, w = img.shape[:2]
        s = min(cw / w, ch / h)
        disp = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._tk_img = ImageTk.PhotoImage(Image.fromarray(disp))
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._tk_img, anchor="center")

    def save_result(self):
        if self.result is None:
            messagebox.showinfo("Nothing to save", "Open an image first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            ok, buf = cv2.imencode("." + path.rsplit(".", 1)[-1], self.result)
            if ok:
                buf.tofile(path)


if __name__ == "__main__":
    App().mainloop()