# %% [markdown]
# # Cross-Modality Unsupervised Domain Adaptation
# ## NIH ChestX-ray14 (2D CXR, source)  ->  MosMedData (3D Chest CT, target)
#
# Experiments (same pipeline, same seed, same budget):
#   1. Source Only / Transfer Learning Baseline (ImageNet backbone, NIH only)
#   2. DANN (Ganin & Lempitsky, 2016) - MosMed TRAIN images used WITHOUT labels
#   3. (optional, reference only) Target Oracle - supervised on MosMed TRAIN labels.
#      NOT a UDA method; reported only as an upper bound to size the domain gap.
#
# Protocol summary (see the Methodology Review for the full reasoning):
#   * MosMed is split train/val/test (stratified, CT-0 vs CT-1..4) BEFORE anything else.
#   * DANN sees ONLY MosMed TRAIN images; the target train Dataset does not even
#     receive labels, so they cannot leak into the loss.
#   * MosMed VAL is used for checkpoint selection / threshold calibration ONLY when
#     USE_TARGET_LABELS_FOR_MODEL_SELECTION=True (this is reported as a deviation
#     from strict UDA). MosMed TEST is touched exactly once, at the very end.
#   * NIH is split at PATIENT level (no patient in two splits).
#   * CT metrics are VOLUME level (mean of NUM_EVAL_SLICES evenly spaced slices).

# %%
# 0. Kaggle images may miss nibabel; install quietly if needed.
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("nibabel") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nibabel"], check=True)

# %% [markdown]
# ## 1. Imports

# %%
import hashlib
import json
import math
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.autograd import Function
from torch.utils.data import (DataLoader, Dataset, RandomSampler, Subset,
                              WeightedRandomSampler)
from torchvision import models as tv_models
from torchvision import transforms

warnings.filterwarnings("ignore", category=UserWarning)
print(f"PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")

# %% [markdown]
# ## 2. Configuration

# %%
NIH_ROOT = "BURAYA_NIH_DATASET_PATH"
NIH_CSV = "BURAYA_DATA_ENTRY_2017_CSV_PATH"
MOSMED_ROOT = "BURAYA_MOSMED_DATASET_PATH"


@dataclass
class Config:
    # ---------------- Paths ----------------
    NIH_ROOT: str = NIH_ROOT
    NIH_CSV: str = NIH_CSV
    MOSMED_ROOT: str = MOSMED_ROOT
    OUTPUT_DIR: str = "/kaggle/working"
    CT_CACHE_DIR: str = "/kaggle/working/ct_cache"

    # ---------------- Reproducibility ----------------
    SEED: int = 42

    # ---------------- Source label definition ----------------
    # True : Positive = Pneumonia OR Infiltration ; Negative = No Finding
    # False: Positive = Pneumonia                 ; Negative = No Finding
    INCLUDE_INFILTRATION: bool = True

    # ---------------- Source imbalance ----------------
    # True -> WeightedRandomSampler (balanced batches), plain CE.
    # False -> natural sampling + class-weighted CrossEntropyLoss.
    USE_WEIGHTED_SAMPLER: bool = True
    # Images drawn per source "epoch". None = full NIH train split (~2.5k steps @bs32).
    # A fixed budget keeps both experiments within a Kaggle session and makes
    # the epoch-level logs / model selection more frequent. Identical for SO and DANN.
    SOURCE_SAMPLES_PER_EPOCH: Optional[int] = 20000

    # ---------------- NIH split (PATIENT level, stratified) ----------------
    NIH_VAL_RATIO: float = 0.10
    NIH_TEST_RATIO: float = 0.10

    # ---------------- MosMed split (stratified on CT-0 vs CT-1..4) ----------------
    TARGET_TRAIN_RATIO: float = 0.70
    TARGET_VAL_RATIO: float = 0.10
    TARGET_TEST_RATIO: float = 0.20

    # ---------------- CT preprocessing ----------------
    CT_HU_MIN: float = -1000.0
    CT_HU_MAX: float = 400.0
    CT_CENTRAL_MIN_FRAC: float = 0.30      # central axial region used for train & eval
    CT_CENTRAL_MAX_FRAC: float = 0.70
    NUM_EVAL_SLICES: int = 7               # evenly spaced slices per volume at val/test
    VOLUME_AGGREGATION: str = "mean"       # "mean" (protocol default) or "max" (ablation)
    # Volume cache: each volume's central slab is windowed + resized ONCE and stored as
    # uint8 .npy (memory-mapped on read). This is a VOLUME cache, not a "path -> chosen
    # slice" cache, so random slice sampling stays random in every epoch. ~50 KB/slice.
    USE_CT_CACHE: bool = True

    # ---------------- Image ----------------
    IMG_SIZE: int = 224

    # ---------------- Model ----------------
    BACKBONE_NAME: str = "resnet18"        # resnet18/34/50, densenet121, efficientnet_b0
    PRETRAINED: bool = True
    NUM_CLASSES: int = 2
    HEAD_DROPOUT: float = 0.3

    # ---------------- Optimisation (identical for all experiments) ----------------
    BATCH_SIZE: int = 32                   # source and target batch sizes are equal
    NUM_WORKERS: int = 4
    NUM_EPOCHS_SOURCE_ONLY: int = 15
    NUM_EPOCHS_DANN: int = 15
    NUM_EPOCHS_ORACLE: int = 15
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    USE_LR_ANNEALING: bool = True          # lr_p = lr0 / (1 + 10 p)^0.75 (Ganin et al.)
    DOMAIN_LOSS_WEIGHT: float = 1.0
    USE_AMP: bool = True

    # ---------------- Model selection ----------------
    # True : best checkpoint = max MosMed VAL volume-level AUC. Target labels are used
    #        ONLY for checkpoint choice + threshold calibration, never in a gradient.
    #        This is common in UDA papers but is NOT strictly unsupervised and must be
    #        stated as such in the thesis.
    # False: strictly unsupervised. No MosMed label is read before the final test.
    #        UNSUPERVISED_SELECTION decides the checkpoint (see ModelSelector).
    USE_TARGET_LABELS_FOR_MODEL_SELECTION: bool = True
    UNSUPERVISED_SELECTION: str = "source_val_loss"   # or "last_epoch"
    THRESHOLD_METHOD: str = "youden"                  # "youden" or "f1"

    # ---------------- Extra experiments / statistics ----------------
    RUN_TARGET_ORACLE: bool = True
    NUM_BOOTSTRAP: int = 1000

    # ---------------- Domain alignment analysis ----------------
    FEATURE_N_SAMPLES: int = 700           # per domain, equal for both domains & models
    RUN_TSNE: bool = True
    RUN_UMAP: bool = False

    # ---------------- Outputs ----------------
    SOURCE_ONLY_CKPT_PATH: str = "/kaggle/working/source_only_best.pth"
    DANN_CKPT_PATH: str = "/kaggle/working/dann_best.pth"
    ORACLE_CKPT_PATH: str = "/kaggle/working/target_oracle_best.pth"
    RESULTS_CSV_PATH: str = "/kaggle/working/experiment_results.csv"

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    def validate(self) -> None:
        s = self.TARGET_TRAIN_RATIO + self.TARGET_VAL_RATIO + self.TARGET_TEST_RATIO
        if not math.isclose(s, 1.0, abs_tol=1e-6):
            raise ValueError(f"TARGET_*_RATIO must sum to 1.0 (got {s:.4f}).")
        if not (0 < self.NIH_VAL_RATIO < 1 and 0 < self.NIH_TEST_RATIO < 1
                and self.NIH_VAL_RATIO + self.NIH_TEST_RATIO < 1):
            raise ValueError("NIH_VAL_RATIO / NIH_TEST_RATIO must be in (0,1) and sum < 1.")
        if self.CT_HU_MIN >= self.CT_HU_MAX:
            raise ValueError("CT_HU_MIN must be < CT_HU_MAX.")
        if not (0.0 <= self.CT_CENTRAL_MIN_FRAC < self.CT_CENTRAL_MAX_FRAC <= 1.0):
            raise ValueError("Need 0 <= CT_CENTRAL_MIN_FRAC < CT_CENTRAL_MAX_FRAC <= 1.")
        if self.NUM_EVAL_SLICES < 1:
            raise ValueError("NUM_EVAL_SLICES must be >= 1.")
        if self.VOLUME_AGGREGATION not in ("mean", "max"):
            raise ValueError("VOLUME_AGGREGATION must be 'mean' or 'max'.")
        if self.UNSUPERVISED_SELECTION not in ("source_val_loss", "last_epoch"):
            raise ValueError("UNSUPERVISED_SELECTION must be 'source_val_loss' or 'last_epoch'.")
        if self.THRESHOLD_METHOD not in ("youden", "f1"):
            raise ValueError("THRESHOLD_METHOD must be 'youden' or 'f1'.")

    @property
    def use_amp(self) -> bool:
        return self.USE_AMP and self.DEVICE.startswith("cuda")

    @property
    def device_type(self) -> str:
        return "cuda" if self.DEVICE.startswith("cuda") else "cpu"


cfg = Config()
cfg.validate()
print(json.dumps(asdict(cfg), indent=2))

# %% [markdown]
# ## 3. Reproducibility

# %%
def set_seed(seed: int) -> None:
    """Seeds python, numpy, torch (CPU + all CUDA devices) and makes cuDNN deterministic."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader workers: derive numpy/python seeds from the torch worker seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


set_seed(cfg.SEED)

# %% [markdown]
# ## 4. Dataset discovery

# %%
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def _check_path(path: str, what: str, is_dir: bool) -> None:
    ok = os.path.isdir(path) if is_dir else os.path.isfile(path)
    if not ok:
        hint = " (placeholder path not replaced?)" if path.startswith("BURAYA_") else ""
        raise FileNotFoundError(f"{what} not found: '{path}'{hint}")


def discover_nih_images(nih_root: str) -> Dict[str, str]:
    """Indexes every PNG/JPG under NIH_ROOT by file name (NIH is split in images_00x/ folders)."""
    _check_path(nih_root, "NIH_ROOT", is_dir=True)
    index: Dict[str, str] = {}
    for dirpath, _, filenames in os.walk(nih_root):
        for fn in filenames:
            if fn.lower().endswith(IMAGE_EXTS):
                index[fn] = os.path.join(dirpath, fn)
    if not index:
        raise FileNotFoundError(f"No PNG/JPG image found under NIH_ROOT='{nih_root}'.")
    print(f"[NIH] images indexed on disk: {len(index)}")
    return index


def build_nih_labels(nih_csv: str, image_index: Dict[str, str],
                     include_infiltration: bool) -> pd.DataFrame:
    """
    Negative (0): 'No Finding' (in NIH this label never co-occurs with another one).
    Positive (1): any image whose label set contains Pneumonia (or Infiltration).
    Excluded    : images with only OTHER findings (e.g. Effusion, Nodule): they are
                  neither clean negatives nor task positives, so keeping them in either
                  class would corrupt the binary task definition.
    """
    _check_path(nih_csv, "NIH_CSV", is_dir=False)
    df = pd.read_csv(nih_csv)
    for col in ("Image Index", "Finding Labels"):
        if col not in df.columns:
            raise ValueError(f"NIH_CSV is missing column '{col}'. Columns: {list(df.columns)}")

    positive_set = {"Pneumonia"} | ({"Infiltration"} if include_infiltration else set())

    def to_label(s: str) -> float:
        labels = {t.strip() for t in str(s).split("|")}
        if "No Finding" in labels:
            return 0
        if labels & positive_set:
            return 1
        return np.nan

    df["label"] = df["Finding Labels"].map(to_label)
    n_excluded = int(df["label"].isna().sum())
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["filepath"] = df["Image Index"].map(image_index)
    n_missing = int(df["filepath"].isna().sum())
    if n_missing:
        print(f"[NIH][WARN] {n_missing} CSV rows have no image on disk -> dropped.")
    df = df.dropna(subset=["filepath"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No NIH image left after label mapping. Check NIH_ROOT / NIH_CSV.")
    if "Patient ID" in df.columns:
        df["patient_id"] = df["Patient ID"].astype(str)
    else:
        print("[NIH][WARN] 'Patient ID' column missing -> falling back to image-level split "
              "(patient leakage between NIH splits becomes possible).")
        df["patient_id"] = df["Image Index"].astype(str)

    # Positive images with co-existing findings (e.g. Infiltration|Effusion) are kept.
    pos = df[df.label == 1]
    n_comorbid = int(pos["Finding Labels"].str.contains(r"\|").sum())
    print(f"[NIH] positive definition: {sorted(positive_set)} vs 'No Finding'")
    print(f"[NIH] excluded (other findings only): {n_excluded}")
    print(f"[NIH] usable images: {len(df)} | patients: {df.patient_id.nunique()}")
    print(f"[NIH] positives with co-existing findings: {n_comorbid}/{len(pos)}")
    _print_distribution("NIH all", df["label"])
    return df[["filepath", "label", "patient_id"]]


def discover_mosmed_volumes(mosmed_root: str) -> pd.DataFrame:
    """studies/CT-0 -> 0 ; studies/CT-1..CT-4 -> 1. Also accepts CT-x folders directly under root."""
    _check_path(mosmed_root, "MOSMED_ROOT", is_dir=True)
    studies_dir = os.path.join(mosmed_root, "studies")
    if not os.path.isdir(studies_dir):
        studies_dir = mosmed_root
    grade_to_label = {"CT-0": 0, "CT-1": 1, "CT-2": 1, "CT-3": 1, "CT-4": 1}
    records = []
    for grade, label in grade_to_label.items():
        gdir = os.path.join(studies_dir, grade)
        if not os.path.isdir(gdir):
            print(f"[MosMed][WARN] folder missing: {gdir}")
            continue
        for dirpath, _, filenames in os.walk(gdir):
            for fn in sorted(filenames):
                if fn.endswith(".nii") or fn.endswith(".nii.gz"):
                    records.append({"filepath": os.path.join(dirpath, fn),
                                    "label": label, "grade": grade})
    if not records:
        raise FileNotFoundError(f"No .nii/.nii.gz under '{studies_dir}/CT-*'.")
    df = pd.DataFrame(records).sort_values("filepath").reset_index(drop=True)
    print(f"[MosMed] volumes: {len(df)}")
    print(df.groupby("grade").size().to_string())
    _print_distribution("MosMed all", df["label"])
    return df


def _print_distribution(name: str, labels: pd.Series) -> None:
    n = len(labels)
    n_pos = int((labels == 1).sum())
    print(f"  [{name:<14}] n={n:6d} | neg={n - n_pos:6d} ({(n - n_pos) / max(n, 1):6.2%})"
          f" | pos={n_pos:6d} ({n_pos / max(n, 1):6.2%})")

# %% [markdown]
# ## 5. Dataset split

# %%
def _stratified_group_holdout(df: pd.DataFrame, ratio: float, seed: int
                              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Holds out ~ratio of PATIENTS while approximately preserving the class ratio."""
    n_splits = max(2, int(round(1.0 / ratio)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep_idx, hold_idx = next(sgkf.split(df, df["label"], groups=df["patient_id"]))
    return df.iloc[keep_idx], df.iloc[hold_idx]


def split_nih(df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int
              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    PATIENT-level stratified split. NIH has ~30k patients for ~112k images (many
    follow-up films); an image-level split would put the same patient in train and
    val/test and inflate source metrics and checkpoint selection.
    """
    rest, test = _stratified_group_holdout(df, test_ratio, seed)
    train, val = _stratified_group_holdout(rest, val_ratio / (1.0 - test_ratio), seed + 1)
    p_tr, p_va, p_te = (set(x.patient_id) for x in (train, val, test))
    assert not (p_tr & p_va) and not (p_tr & p_te) and not (p_va & p_te), "NIH patient leakage!"
    print("[NIH] patient-level split:")
    for name, s in (("NIH train", train), ("NIH val", val), ("NIH test", test)):
        _print_distribution(name, s["label"])
    return tuple(s.reset_index(drop=True) for s in (train, val, test))  # type: ignore


def split_mosmed(df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float,
                 seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified (CT-0 vs CT-1..4) volume-level split. One volume = one study/patient."""
    train, rest = train_test_split(df, train_size=train_ratio, stratify=df["label"],
                                   random_state=seed)
    val, test = train_test_split(rest, train_size=val_ratio / (val_ratio + test_ratio),
                                 stratify=rest["label"], random_state=seed)
    assert not (set(train.filepath) & set(val.filepath)) and \
        not (set(train.filepath) & set(test.filepath)) and \
        not (set(val.filepath) & set(test.filepath)), "MosMed split leakage!"
    print("[MosMed] stratified split:")
    for name, s in (("MosMed train", train), ("MosMed val", val), ("MosMed test", test)):
        _print_distribution(name, s["label"])
        print(f"      grades: {s.groupby('grade').size().to_dict()}")
    return tuple(s.reset_index(drop=True) for s in (train, val, test))  # type: ignore

# %% [markdown]
# ## 6. Dataset classes
#
# CT handling:
# * volumes are re-oriented to closest-canonical (RAS) so axis 2 is always axial and every
#   slice has the same in-plane orientation (then rotated 90 deg: anterior at the top),
# * NaN/Inf are replaced (NaN/-Inf -> HU_MIN = air, +Inf -> HU_MAX),
# * lung window [CT_HU_MIN, CT_HU_MAX] -> [0, 1] -> uint8 -> 3-channel copy.

# %%
def load_cxr_rgb(path: str) -> Image.Image:
    """Loads a CXR as 8-bit grayscale replicated to 3 channels (handles 16-bit/RGBA PNGs)."""
    try:
        with Image.open(path) as im:
            if im.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
                arr = np.asarray(im, dtype=np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
                im = Image.fromarray(arr.astype(np.uint8))
            return im.convert("L").convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Cannot read CXR image '{path}': {e}") from e


class NIHChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Callable):
        self.paths = df["filepath"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        return self.transform(load_cxr_rgb(self.paths[idx])), self.labels[idx]


def central_slice_range(num_slices: int, min_frac: float, max_frac: float) -> Tuple[int, int]:
    """Inclusive [lo, hi] axial index range of the central lung region."""
    lo = int(math.floor(num_slices * min_frac))
    hi = int(math.ceil(num_slices * max_frac)) - 1
    lo = min(max(lo, 0), num_slices - 1)
    hi = min(max(hi, lo), num_slices - 1)
    return lo, hi


def load_ct_volume_hu(path: str, hu_min: float, hu_max: float) -> np.ndarray:
    """Full volume in canonical RAS orientation as float32 HU, NaN/Inf cleaned. Shape (X, Y, Z)."""
    try:
        img = nib.as_closest_canonical(nib.load(path))
        vol = np.asanyarray(img.dataobj).astype(np.float32)  # applies scl_slope/inter
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Cannot read NIfTI '{path}': {e}") from e
    if vol.ndim == 4:
        vol = vol[..., 0]
    if vol.ndim != 3:
        raise RuntimeError(f"Expected a 3D CT volume, got shape {vol.shape} for '{path}'.")
    vol = np.nan_to_num(vol, nan=hu_min, posinf=hu_max, neginf=hu_min)
    if np.percentile(vol, 1) > -500:
        print(f"[CT][WARN] '{os.path.basename(path)}' 1st-percentile intensity "
              f"{np.percentile(vol, 1):.0f} > -500: data may not be in HU.")
    return vol


def lung_window(x: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    return ((np.clip(x, hu_min, hu_max) - hu_min) / (hu_max - hu_min)).astype(np.float32)


def preprocess_ct_central_slab(path: str, hu_min: float, hu_max: float, min_frac: float,
                               max_frac: float, size: int) -> np.ndarray:
    """Returns the central axial slab as uint8 array (S, size, size), lung-windowed."""
    vol = load_ct_volume_hu(path, hu_min, hu_max)
    lo, hi = central_slice_range(vol.shape[2], min_frac, max_frac)
    slab = np.empty((hi - lo + 1, size, size), dtype=np.uint8)
    for i, z in enumerate(range(lo, hi + 1)):
        s = np.rot90(lung_window(vol[:, :, z], hu_min, hu_max))
        pil = Image.fromarray(np.round(s * 255.0).astype(np.uint8))
        slab[i] = np.asarray(pil.resize((size, size), Image.BILINEAR))
    return slab


def _cache_worker(args) -> Tuple[str, Optional[str]]:
    path, out_path, params = args
    try:
        if not os.path.isfile(out_path):
            slab = preprocess_ct_central_slab(path, *params)
            tmp = out_path + ".tmp.npy"
            np.save(tmp, slab)
            os.replace(tmp, out_path)
        return path, None
    except Exception as e:  # noqa: BLE001
        return path, str(e)


class _CacheJobs(Dataset):
    def __init__(self, jobs):
        self.jobs = jobs

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, i):
        return _cache_worker(self.jobs[i])


class CTSlabStore:
    """
    Gives access to each volume's preprocessed central slab.

    Caching policy (avoids the 'random slice cached once' bug):
      * the cache key is the VOLUME (path + preprocessing params), the value is the full
        central slab; the random slice is chosen at __getitem__ time, every time.
      * slabs live on disk and are memory-mapped, so RAM stays ~constant regardless of
        dataset size and number of workers.
      * USE_CT_CACHE=False: the slab is recomputed from the NIfTI on every access
        (correct but slow, .nii.gz must be decompressed each time).
    """

    def __init__(self, config: Config):
        self.c = config
        self.params = (config.CT_HU_MIN, config.CT_HU_MAX, config.CT_CENTRAL_MIN_FRAC,
                       config.CT_CENTRAL_MAX_FRAC, config.IMG_SIZE)
        if config.USE_CT_CACHE:
            os.makedirs(config.CT_CACHE_DIR, exist_ok=True)
        self._num_slices: Dict[str, int] = {}

    def cache_path(self, path: str) -> str:
        key = hashlib.md5(f"{os.path.abspath(path)}|{self.params}|v2".encode()).hexdigest()
        return os.path.join(self.c.CT_CACHE_DIR, f"{key}.npy")

    def build(self, paths: Sequence[str]) -> List[str]:
        """Pre-computes the cache; returns the paths that FAILED to load (label-agnostic)."""
        if not self.c.USE_CT_CACHE:
            return []
        todo = [p for p in paths if not os.path.isfile(self.cache_path(p))]
        print(f"[CT cache] {len(paths) - len(todo)} cached, {len(todo)} to build "
              f"-> {self.c.CT_CACHE_DIR}")
        failed: List[str] = []
        jobs = [(p, self.cache_path(p), self.params) for p in todo]
        t0 = time.time()
        # Parallelised through DataLoader workers (robust in notebooks: no pickling of
        # functions defined in __main__ is required).
        loader = DataLoader(_CacheJobs(jobs), batch_size=None, shuffle=False,
                            num_workers=min(self.c.NUM_WORKERS, len(jobs)))
        for i, (p, err) in enumerate(loader, 1):
            if err:
                failed.append(p)
                print(f"[CT cache][ERROR] {p}: {err}")
            if i % 100 == 0:
                print(f"[CT cache] {i}/{len(jobs)} ({time.time() - t0:.0f}s)")
        print(f"[CT cache] done in {time.time() - t0:.0f}s, failed: {len(failed)}")
        return failed

    def get(self, path: str) -> np.ndarray:
        if self.c.USE_CT_CACHE:
            cp = self.cache_path(path)
            if not os.path.isfile(cp):
                _, err = _cache_worker((path, cp, self.params))
                if err:
                    raise RuntimeError(f"CT preprocessing failed for '{path}': {err}")
            return np.load(cp, mmap_mode="r")
        return preprocess_ct_central_slab(path, *self.params)

    def num_slices(self, path: str) -> int:
        if path not in self._num_slices:
            if self.c.USE_CT_CACHE:
                self._num_slices[path] = int(self.get(path).shape[0])
            else:  # header only: find the input axis that maps to canonical axis 2 (S/I)
                img = nib.load(path)
                ornt = nib.orientations.io_orientation(img.affine)
                axial_in_axis = int(np.where(ornt[:, 0] == 2)[0][0])
                lo, hi = central_slice_range(img.shape[axial_in_axis],
                                             self.c.CT_CENTRAL_MIN_FRAC, self.c.CT_CENTRAL_MAX_FRAC)
                self._num_slices[path] = hi - lo + 1
        return self._num_slices[path]


def _slab_slice_to_pil(slab: np.ndarray, idx: int) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(slab[idx])).convert("RGB")


class MosMedRandomSliceDataset(Dataset):
    """
    Training-time CT dataset: one RANDOM axial slice from the central slab per call.

    labels=None (DANN): returns ONLY the image. The target disease label is not even
    stored, so it is structurally impossible for it to enter the DANN loss.
    labels given (Target Oracle only): returns (image, label).
    """

    def __init__(self, paths: Sequence[str], store: CTSlabStore, transform: Callable,
                 labels: Optional[Sequence[int]] = None):
        self.paths = list(paths)
        self.store = store
        self.transform = transform
        self.labels = None if labels is None else [int(l) for l in labels]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        slab = self.store.get(self.paths[idx])
        z = int(torch.randint(0, slab.shape[0], (1,)).item())  # fresh draw every call
        x = self.transform(_slab_slice_to_pil(slab, z))
        return x if self.labels is None else (x, self.labels[idx])


def eval_slice_positions(num_slices: int, k: int) -> List[int]:
    """k evenly spaced positions inside the central slab (deterministic)."""
    if num_slices <= 1:
        return [0] * k
    return [int(round(v)) for v in np.linspace(0, num_slices - 1, num=k)]


class MosMedVolumeEvalDataset(Dataset):
    """Val/test: NUM_EVAL_SLICES deterministic slices per volume -> (image, label, volume_idx)."""

    def __init__(self, df: pd.DataFrame, store: CTSlabStore, transform: Callable, k: int):
        self.paths = df["filepath"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.store = store
        self.transform = transform
        self.items: List[Tuple[int, int]] = []
        for v, p in enumerate(self.paths):
            for z in eval_slice_positions(store.num_slices(p), k):
                self.items.append((v, z))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        v, z = self.items[idx]
        x = self.transform(_slab_slice_to_pil(self.store.get(self.paths[v]), z))
        return x, self.labels[v], v

    @property
    def num_volumes(self) -> int:
        return len(self.paths)

# %% [markdown]
# ## 7. Transforms
# CXR and CT get DIFFERENT train augmentations; eval is deterministic and shared.

# %%
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(c: Config) -> Dict[str, transforms.Compose]:
    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    size = (c.IMG_SIZE, c.IMG_SIZE)
    source_train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.95, 1.05)),
        *norm,
    ])
    # CT: no rotation/shear/scale; anatomy in an axial slice is already standardised by
    # the canonical re-orientation. Only L/R flip (lungs are roughly symmetric).
    target_train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(p=0.5),
        *norm,
    ])
    eval_transform = transforms.Compose([transforms.Resize(size), *norm])
    return {"source_train": source_train_transform,
            "target_train": target_train_transform,
            "eval": eval_transform}

# %% [markdown]
# ## 8. DataLoaders

# %%
def make_loader(ds: Dataset, c: Config, *, shuffle: bool = False, sampler=None,
                drop_last: bool = False, seed_offset: int = 0) -> DataLoader:
    nw = c.NUM_WORKERS
    return DataLoader(ds, batch_size=c.BATCH_SIZE, shuffle=shuffle if sampler is None else False,
                      sampler=sampler, drop_last=drop_last, num_workers=nw,
                      pin_memory=c.DEVICE.startswith("cuda"), worker_init_fn=seed_worker,
                      generator=make_generator(c.SEED + seed_offset),
                      persistent_workers=nw > 0)


def class_weights_from_labels(labels: Sequence[int]) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels), minlength=2).astype(np.float64)
    return torch.tensor(counts.sum() / (2.0 * np.clip(counts, 1, None)), dtype=torch.float32)


def build_source_train_loader(train_df: pd.DataFrame, c: Config, tfm) -> DataLoader:
    ds = NIHChestXrayDataset(train_df, tfm)
    n = c.SOURCE_SAMPLES_PER_EPOCH or len(ds)
    g = make_generator(c.SEED + 100)
    if c.USE_WEIGHTED_SAMPLER:
        labels = np.asarray(ds.labels)
        w = (1.0 / np.bincount(labels, minlength=2).clip(min=1))[labels]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=n,
                                        replacement=True, generator=g)
    else:
        sampler = RandomSampler(ds, replacement=False, num_samples=min(n, len(ds)), generator=g)
    return make_loader(ds, c, sampler=sampler, drop_last=True, seed_offset=1)


def build_all_loaders(c: Config, nih: Dict[str, pd.DataFrame], mosmed: Dict[str, pd.DataFrame],
                      store: CTSlabStore) -> Dict[str, DataLoader]:
    t = build_transforms(c)
    loaders = {
        "nih_train": build_source_train_loader(nih["train"], c, t["source_train"]),
        "nih_val": make_loader(NIHChestXrayDataset(nih["val"], t["eval"]), c),
        "nih_test": make_loader(NIHChestXrayDataset(nih["test"], t["eval"]), c),
        # DANN target loader: paths only, NO labels passed.
        "mosmed_train_unlabeled": make_loader(
            MosMedRandomSliceDataset(mosmed["train"]["filepath"], store, t["target_train"]),
            c, shuffle=True, drop_last=True, seed_offset=2),
        "mosmed_val": make_loader(MosMedVolumeEvalDataset(mosmed["val"], store, t["eval"],
                                                          c.NUM_EVAL_SLICES), c),
        "mosmed_test": make_loader(MosMedVolumeEvalDataset(mosmed["test"], store, t["eval"],
                                                           c.NUM_EVAL_SLICES), c),
    }
    if c.RUN_TARGET_ORACLE:
        loaders["mosmed_train_labeled"] = make_loader(
            MosMedRandomSliceDataset(mosmed["train"]["filepath"], store, t["target_train"],
                                     labels=mosmed["train"]["label"]),
            c, shuffle=True, drop_last=True, seed_offset=3)
    if len(loaders["mosmed_train_unlabeled"]) == 0:
        raise RuntimeError("MosMed train split smaller than BATCH_SIZE; lower BATCH_SIZE.")
    for k, v in loaders.items():
        print(f"[Loader] {k:<24} batches={len(v):5d} samples={len(v.dataset):6d}")
    return loaders

# %% [markdown]
# ## 9. Models

# %%
_BACKBONES = {
    # name: (constructor, weights, attribute holding the classifier, how to read in_features)
    "resnet18": (tv_models.resnet18, tv_models.ResNet18_Weights.DEFAULT, "fc"),
    "resnet34": (tv_models.resnet34, tv_models.ResNet34_Weights.DEFAULT, "fc"),
    "resnet50": (tv_models.resnet50, tv_models.ResNet50_Weights.DEFAULT, "fc"),
    "densenet121": (tv_models.densenet121, tv_models.DenseNet121_Weights.DEFAULT, "classifier"),
    "efficientnet_b0": (tv_models.efficientnet_b0, tv_models.EfficientNet_B0_Weights.DEFAULT,
                        "classifier"),
}


def build_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Returns (feature extractor producing a flat vector, feature dim). Add new nets here."""
    name = name.lower()
    if name not in _BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Available: {list(_BACKBONES)}")
    ctor, weights, head_attr = _BACKBONES[name]
    try:
        net = ctor(weights=weights if pretrained else None)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Could not build '{name}' (pretrained={pretrained}). On Kaggle, "
                           f"enable Internet to download ImageNet weights. Original: {e}") from e
    head = getattr(net, head_attr)
    in_features = head.in_features if isinstance(head, nn.Linear) else \
        [m for m in head.modules() if isinstance(m, nn.Linear)][-1].in_features
    setattr(net, head_attr, nn.Identity())
    return net, in_features


class GradientReversalFunction(Function):
    """Identity in the forward pass; multiplies the gradient by -lambda in the backward pass."""

    @staticmethod
    def forward(ctx, x, lambda_: float):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return GradientReversalFunction.apply(x, lambda_)


class LungClassifierNet(nn.Module):
    """
    Image -> Backbone -> features -> ClassifierHead -> disease logits
                                 \\-> GRL -> DomainClassifier -> domain logit (DANN only)
    Same class for every experiment so that Source-Only / DANN / Oracle differ ONLY in the
    training objective, not in capacity.
    """

    def __init__(self, c: Config, with_domain_head: bool):
        super().__init__()
        self.backbone, d = build_backbone(c.BACKBONE_NAME, c.PRETRAINED)
        self.classifier = nn.Sequential(nn.Dropout(c.HEAD_DROPOUT), nn.Linear(d, 256),
                                        nn.ReLU(inplace=True), nn.Dropout(c.HEAD_DROPOUT),
                                        nn.Linear(256, c.NUM_CLASSES))
        self.domain_classifier = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 1)) if with_domain_head else None

    def forward(self, x: torch.Tensor, grl_lambda: float = 0.0) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)
        out = {"features": feats, "class_logits": self.classifier(feats)}
        if self.domain_classifier is not None:
            out["domain_logits"] = self.domain_classifier(grad_reverse(feats, grl_lambda)).squeeze(1)
        return out

# %% [markdown]
# ## 10. Gradient Reversal Layer schedule

# %%
def dann_lambda(p: float) -> float:
    """lambda_p = 2 / (1 + exp(-10 p)) - 1, p = current_step / total_steps in [0, 1]."""
    p = min(max(p, 0.0), 1.0)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


def annealed_lr(base_lr: float, p: float) -> float:
    """Ganin et al. lr schedule: mu_p = mu_0 / (1 + 10 p)^0.75."""
    return base_lr / (1.0 + 10.0 * p) ** 0.75

# %% [markdown]
# ## 11. Training utilities

# %%
class AverageMeter:
    def __init__(self):
        self.sum, self.n = 0.0, 0

    def update(self, v: float, k: int = 1):
        self.sum += float(v) * k
        self.n += k

    @property
    def avg(self) -> float:
        return self.sum / max(self.n, 1)


class RestartingIterator:
    """
    Draws batches forever; when the underlying loader is exhausted a NEW iterator is created
    (new shuffle, new random slices, new augmentations). Unlike itertools.cycle, nothing is
    memorised, so the target domain is not replayed as a frozen list of tensors.
    """

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.it = iter(loader)
        self.restarts = 0

    def __next__(self):
        try:
            return next(self.it)
        except StopIteration:
            self.restarts += 1
            self.it = iter(self.loader)
            return next(self.it)


def make_optimizer(model: nn.Module, c: Config) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=c.LR, weight_decay=c.WEIGHT_DECAY)


def set_lr(optimizer: torch.optim.Optimizer, c: Config, p: float) -> None:
    if c.USE_LR_ANNEALING:
        for g in optimizer.param_groups:
            g["lr"] = annealed_lr(c.LR, p)


def _to_py(obj):
    """Converts numpy scalars/arrays for checkpoint/JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
                    c: Config, validation_metrics: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "config": asdict(c),
                "validation_metrics": _to_py(validation_metrics)}, path)


def load_checkpoint(path: str, model: nn.Module, device: str) -> Dict:
    # weights_only=False: our own file, contains plain-python config/metric dicts.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


class ModelSelector:
    """
    criterion:
      'target_val_auc'  - uses MosMed VAL labels (NOT strictly UDA; must be reported).
      'source_val_loss' - strictly unsupervised w.r.t. target. Caveat: for DANN it tends to
                          favour early epochs (small GRL lambda), i.e. it is biased towards
                          LESS adaptation, because source loss does not measure target risk.
      'last_epoch'      - strictly unsupervised, no selection at all (common in UDA papers).
    """

    def __init__(self, criterion: str):
        self.criterion = criterion
        self.best = -math.inf if criterion == "target_val_auc" else math.inf
        self.best_epoch = -1

    def update(self, epoch: int, metrics: Dict) -> bool:
        if self.criterion == "last_epoch":
            self.best, self.best_epoch = float(epoch), epoch
            return True
        v = metrics[self.criterion]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return False
        better = v > self.best if self.criterion == "target_val_auc" else v < self.best
        if better:
            self.best, self.best_epoch = v, epoch
        return better


def selection_criterion(c: Config) -> str:
    return "target_val_auc" if c.USE_TARGET_LABELS_FOR_MODEL_SELECTION else c.UNSUPERVISED_SELECTION


@torch.no_grad()
def predict_images(model: nn.Module, loader: DataLoader, c: Config) -> Tuple[np.ndarray, np.ndarray, float]:
    """Image-level P(positive), labels and mean CE loss (NIH val/test)."""
    model.eval()
    probs, labels, loss = [], [], AverageMeter()
    for x, y in loader:
        x, y = x.to(c.DEVICE, non_blocking=True), torch.as_tensor(y).to(c.DEVICE)
        with torch.amp.autocast(device_type=c.device_type, enabled=c.use_amp):
            logits = model(x)["class_logits"]
        logits = logits.float()
        loss.update(F.cross_entropy(logits, y).item(), x.size(0))
        probs.append(F.softmax(logits, 1)[:, 1].cpu().numpy())
        labels.append(y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels), loss.avg


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def validation_metrics(model: nn.Module, loaders: Dict[str, DataLoader], c: Config,
                       use_target_labels: bool) -> Dict:
    p, y, loss = predict_images(model, loaders["nih_val"], c)
    out = {"source_val_loss": loss, "source_val_auc": safe_auc(y, p), "target_val_auc": float("nan")}
    # Target VAL labels are read ONLY if the protocol allows it (otherwise not even logged,
    # to rule out informal 'peeking' at target performance during development).
    if use_target_labels:
        tp, ty = predict_volumes(model, loaders["mosmed_val"], c)
        out["target_val_auc"] = safe_auc(ty, tp)
    return out

# %% [markdown]
# ## 12. Source-only training (Experiment 1) - also reused for the Target Oracle

# %%
def train_supervised(model: nn.Module, train_loader: DataLoader, loaders: Dict[str, DataLoader],
                     c: Config, n_epochs: int, ckpt_path: str, tag: str, criterion_name: str,
                     use_target_labels_for_val: bool) -> Tuple[nn.Module, Dict]:
    """
    Plain supervised training: loss = classification loss only.
    Source-Only: train_loader = NIH train; MosMed images are NEVER fed to the network during
    training (MosMed VAL is only forwarded for checkpoint selection if the protocol allows it).
    """
    dev = c.DEVICE
    model.to(dev)
    opt = make_optimizer(model, c)
    scaler = torch.amp.GradScaler(device=c.device_type, enabled=c.use_amp)
    weights = None
    if tag == "Source-Only" and not c.USE_WEIGHTED_SAMPLER:
        weights = class_weights_from_labels(train_loader.dataset.labels).to(dev)
        print(f"[{tag}] weighted CE, class weights = {weights.tolist()}")
    ce = nn.CrossEntropyLoss(weight=weights)
    selector = ModelSelector(criterion_name)
    total_steps, step, history = n_epochs * len(train_loader), 0, []

    for epoch in range(1, n_epochs + 1):
        model.train()
        meter, t0 = AverageMeter(), time.time()
        for x, y in train_loader:
            set_lr(opt, c, step / total_steps)
            x, y = x.to(dev, non_blocking=True), torch.as_tensor(y).to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=c.device_type, enabled=c.use_amp):
                loss = ce(model(x)["class_logits"].float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            meter.update(loss.item(), x.size(0))
            step += 1
        val = validation_metrics(model, loaders, c, use_target_labels_for_val)
        rec = {"epoch": epoch, "cls_loss": meter.avg, "lr": opt.param_groups[0]["lr"], **val}
        history.append(rec)
        improved = selector.update(epoch, val)
        if improved:
            save_checkpoint(ckpt_path, model, opt, epoch, c,
                            {**val, "selection_criterion": criterion_name})
        print(f"[{tag}] ep {epoch:02d}/{n_epochs} | cls_loss {meter.avg:.4f} | "
              f"NIH val loss {val['source_val_loss']:.4f} AUC {val['source_val_auc']:.4f} | "
              f"MosMed val AUC {val['target_val_auc']:.4f} | {time.time() - t0:.0f}s"
              f"{'  <- saved' if improved else ''}")

    ckpt = load_checkpoint(ckpt_path, model, dev)
    print(f"[{tag}] best epoch = {ckpt['epoch']} by '{criterion_name}' -> {ckpt_path}")
    return model, {"best_epoch": ckpt["epoch"], "history": history, "criterion": criterion_name}

# %% [markdown]
# ## 13. DANN training (Experiment 2)

# %%
def train_dann(model: nn.Module, loaders: Dict[str, DataLoader], c: Config,
               criterion_name: str) -> Tuple[nn.Module, Dict]:
    """
    total_loss = CE(class | source) + DOMAIN_LOSS_WEIGHT * BCE(domain | source U target)

    * Target batches come from 'mosmed_train_unlabeled' (images only) -> no target label in
      the loss, by construction.
    * Source and target batches are CONCATENATED into ONE forward pass, so BatchNorm sees a
      mixed batch. Two separate passes would normalise each domain with its own batch
      statistics during training (an implicit AdaBN-style alignment that is not part of
      DANN and makes running statistics inconsistent with training-time behaviour).
    * steps_per_epoch = max(len(source), len(target)); the shorter loader is re-started
      (RestartingIterator), so neither domain truncates the epoch.
    * GRL lambda follows the DANN schedule with p = global_step / total_steps (batch level).

    Note on domain accuracy: ~50% means the domain head cannot separate the domains in its
    current state. It is NOT proof of successful adaptation: features can be made
    domain-confusing by discarding disease information (negative transfer), and a weak or
    lagging domain head also yields ~50%. Always read it together with target AUC.
    """
    dev = c.DEVICE
    model.to(dev)
    opt = make_optimizer(model, c)
    scaler = torch.amp.GradScaler(device=c.device_type, enabled=c.use_amp)
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    src_loader, tgt_loader = loaders["nih_train"], loaders["mosmed_train_unlabeled"]
    src_it, tgt_it = RestartingIterator(src_loader), RestartingIterator(tgt_loader)
    steps_per_epoch = max(len(src_loader), len(tgt_loader))
    total_steps = steps_per_epoch * c.NUM_EPOCHS_DANN
    print(f"[DANN] steps/epoch={steps_per_epoch} (source {len(src_loader)}, target "
          f"{len(tgt_loader)}), total={total_steps}")
    selector = ModelSelector(criterion_name)
    use_tgt_val = criterion_name == "target_val_auc"
    step, history = 0, []

    for epoch in range(1, c.NUM_EPOCHS_DANN + 1):
        model.train()
        m = {k: AverageMeter() for k in ("cls", "dom", "tot", "dacc", "dacc_s", "dacc_t")}
        lam, t0 = 0.0, time.time()
        for _ in range(steps_per_epoch):
            xs, ys = next(src_it)
            xt = next(tgt_it)
            xs, ys = xs.to(dev, non_blocking=True), torch.as_tensor(ys).to(dev, non_blocking=True)
            xt = xt.to(dev, non_blocking=True)
            bs, bt = xs.size(0), xt.size(0)
            p = step / total_steps
            lam = dann_lambda(p)
            set_lr(opt, c, p)
            d_lab = torch.cat([torch.zeros(bs, device=dev), torch.ones(bt, device=dev)])

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=c.device_type, enabled=c.use_amp):
                out = model(torch.cat([xs, xt], 0), grl_lambda=lam)
                cls_loss = ce(out["class_logits"][:bs].float(), ys)
                dom_logits = out["domain_logits"].float()
                dom_loss = bce(dom_logits, d_lab)
                loss = cls_loss + c.DOMAIN_LOSS_WEIGHT * dom_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"[DANN] non-finite loss at step {step} (lambda={lam:.3f}). "
                                   f"Try lower LR or DOMAIN_LOSS_WEIGHT.")
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                correct = ((dom_logits > 0).float() == d_lab).float()
            m["cls"].update(cls_loss.item(), bs)
            m["dom"].update(dom_loss.item(), bs + bt)
            m["tot"].update(loss.item(), bs + bt)
            m["dacc"].update(correct.mean().item(), bs + bt)
            m["dacc_s"].update(correct[:bs].mean().item(), bs)
            m["dacc_t"].update(correct[bs:].mean().item(), bt)
            step += 1

        val = validation_metrics(model, loaders, c, use_tgt_val)
        rec = {"epoch": epoch, "cls_loss": m["cls"].avg, "domain_loss": m["dom"].avg,
               "total_loss": m["tot"].avg, "domain_acc": m["dacc"].avg,
               "domain_acc_source": m["dacc_s"].avg, "domain_acc_target": m["dacc_t"].avg,
               "grl_lambda": lam, "lr": opt.param_groups[0]["lr"], **val}
        history.append(rec)
        improved = selector.update(epoch, val)
        if improved:
            save_checkpoint(c.DANN_CKPT_PATH, model, opt, epoch, c,
                            {**val, "domain_acc": m["dacc"].avg, "grl_lambda": lam,
                             "selection_criterion": criterion_name})
        print(f"[DANN] ep {epoch:02d}/{c.NUM_EPOCHS_DANN} | cls {m['cls'].avg:.4f} | "
              f"dom {m['dom'].avg:.4f} | total {m['tot'].avg:.4f} | dom_acc {m['dacc'].avg:.3f} "
              f"(src {m['dacc_s'].avg:.2f}/tgt {m['dacc_t'].avg:.2f}) | lambda {lam:.3f} | "
              f"NIH val AUC {val['source_val_auc']:.4f} | MosMed val AUC "
              f"{val['target_val_auc']:.4f} | {time.time() - t0:.0f}s"
              f"{'  <- saved' if improved else ''}")

    print(f"[DANN] target loader restarted {tgt_it.restarts}x, source {src_it.restarts}x")
    ckpt = load_checkpoint(c.DANN_CKPT_PATH, model, dev)
    print(f"[DANN] best epoch = {ckpt['epoch']} by '{criterion_name}' -> {c.DANN_CKPT_PATH}")
    return model, {"best_epoch": ckpt["epoch"], "history": history, "criterion": criterion_name}

# %% [markdown]
# ## 14. Volume-level CT inference

# %%
@torch.no_grad()
def predict_volumes(model: nn.Module, loader: DataLoader, c: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Slice probabilities -> one probability per volume (mean, or max for the ablation)."""
    model.eval()
    ds: MosMedVolumeEvalDataset = loader.dataset
    per_vol: List[List[float]] = [[] for _ in range(ds.num_volumes)]
    for x, _, v in loader:
        with torch.amp.autocast(device_type=c.device_type, enabled=c.use_amp):
            logits = model(x.to(c.DEVICE, non_blocking=True))["class_logits"]
        for pv, vi in zip(F.softmax(logits.float(), 1)[:, 1].cpu().numpy(), v.numpy()):
            per_vol[int(vi)].append(float(pv))
    agg = np.mean if c.VOLUME_AGGREGATION == "mean" else np.max
    if any(len(s) == 0 for s in per_vol):
        raise RuntimeError("Some volumes received no slice prediction.")
    return np.array([agg(s) for s in per_vol]), np.asarray(ds.labels)

# %% [markdown]
# ## 15. Evaluation metrics

# %%
def optimal_threshold(y: np.ndarray, p: np.ndarray, method: str) -> float:
    """Threshold fitted on a VALIDATION set (never on test)."""
    if len(np.unique(y)) < 2:
        return 0.5
    if method == "youden":
        fpr, tpr, thr = roc_curve(y, p)
        ok = np.isfinite(thr)
        return float(thr[ok][np.argmax((tpr - fpr)[ok])])
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def binary_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> Dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": thr, "roc_auc": safe_auc(y, p),
            "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "sensitivity": recall_score(y, pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
            "f1": f1_score(y, pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))) if aucs else (np.nan, np.nan)


def paired_bootstrap_auc_diff(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n: int,
                              seed: int) -> Dict:
    """AUC(b) - AUC(a) on the SAME test volumes; two-sided bootstrap p-value."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            diffs.append(roc_auc_score(y[idx], p_b[idx]) - roc_auc_score(y[idx], p_a[idx]))
    diffs = np.asarray(diffs)
    p_val = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
    return {"delta_auc": safe_auc(y, p_b) - safe_auc(y, p_a),
            "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5)),
            "p_value": p_val}


def evaluate_model_on_target(model: nn.Module, loaders: Dict[str, DataLoader], c: Config,
                             name: str) -> Dict:
    """
    Thresholds:
      thr_0.5        : fixed.
      thr_target_val : fitted on MosMed VAL (uses target val labels -> not strict UDA).
      thr_source_val : fitted on NIH VAL (strictly unsupervised w.r.t. target, but the
                       probability scale usually shifts across modalities).
    All thresholds are fitted BEFORE the test set is scored; test labels only enter here.
    """
    vp, vy = predict_volumes(model, loaders["mosmed_val"], c)
    sp, sy, _ = predict_images(model, loaders["nih_val"], c)
    tp, ty = predict_volumes(model, loaders["mosmed_test"], c)
    thr = {"thr_0.5": 0.5,
           "thr_target_val": optimal_threshold(vy, vp, c.THRESHOLD_METHOD),
           "thr_source_val": optimal_threshold(sy, sp, c.THRESHOLD_METHOD)}
    res = {k: binary_metrics(ty, tp, t) for k, t in thr.items()}
    res["auc_ci"] = bootstrap_auc_ci(ty, tp, c.NUM_BOOTSTRAP, c.SEED)
    res["test_probs"], res["test_labels"] = tp, ty
    print(f"\n=== {name} | MosMed TEST (n={len(ty)} volumes, aggregation={c.VOLUME_AGGREGATION}) ===")
    print(f"ROC-AUC {res['thr_0.5']['roc_auc']:.4f} (95% CI {res['auc_ci'][0]:.4f}-{res['auc_ci'][1]:.4f})"
          f" | PR-AUC {res['thr_0.5']['pr_auc']:.4f} (prevalence {ty.mean():.3f})")
    for k in thr:
        r = res[k]
        print(f"  {k:<15} t={r['threshold']:.3f} | ACC {r['accuracy']:.3f} | P {r['precision']:.3f}"
              f" | Sens {r['sensitivity']:.3f} | Spec {r['specificity']:.3f} | F1 {r['f1']:.3f}"
              f" | BalACC {r['balanced_accuracy']:.3f} | CM [[TN {r['tn']} FP {r['fp']}]"
              f" [FN {r['fn']} TP {r['tp']}]]")
    return res

# %% [markdown]
# ## 16. Feature extraction

# %%
def fixed_feature_subsets(nih_test_ds: Dataset, mosmed_test_ds: Dataset, n: int, seed: int
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same random (seeded) indices for EVERY model; equal size per domain.
    NIH TEST images and MosMed TEST slices: neither is used by any training step,
    model selection or threshold fitting. Random (not first-n) sampling matters: the
    NIH CSV is ordered by patient and the eval dataset by volume.
    """
    n = min(n, len(nih_test_ds), len(mosmed_test_ds))
    rng = np.random.default_rng(seed)
    return (np.sort(rng.choice(len(nih_test_ds), n, replace=False)),
            np.sort(rng.choice(len(mosmed_test_ds), n, replace=False)))


@torch.no_grad()
def extract_features(model: nn.Module, ds: Dataset, idx: np.ndarray, c: Config
                     ) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=c.BATCH_SIZE, shuffle=False,
                        num_workers=c.NUM_WORKERS)
    feats, labels = [], []
    for batch in loader:
        with torch.amp.autocast(device_type=c.device_type, enabled=c.use_amp):
            f = model.backbone(batch[0].to(c.DEVICE))
        feats.append(f.float().cpu().numpy())
        labels.append(np.asarray(batch[1]))
    return np.concatenate(feats), np.concatenate(labels)

# %% [markdown]
# ## 17. MMD / CORAL / domain probe
#
# Features are L2-normalised before MMD/CORAL. Otherwise a model could "reduce" both
# distances simply by shrinking feature norms (DANN is known to do this), which says
# nothing about alignment. Raw-feature values are kept in the detailed output.
#
# Interpretation: lower MMD/CORAL = more similar feature distributions. That alone is NOT
# better adaptation. If MMD drops while target AUC drops, the domains were aligned but
# class-discriminative information was destroyed (or the classes were mis-aligned, e.g.
# CT positives mapped onto CXR negatives). Read these numbers only next to target AUC.

# %%
def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def mmd_rbf(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased MMD^2 with a Gaussian kernel; bandwidth = median pairwise distance (pooled)."""
    xt, yt = torch.from_numpy(x).double(), torch.from_numpy(y).double()
    z = torch.cat([xt, yt])
    d2 = torch.cdist(z, z).pow(2)
    sigma2 = torch.median(d2[d2 > 0]).clamp_min(1e-12)
    k = torch.exp(-d2 / (2 * sigma2))
    n, m = len(x), len(y)
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    mmd2 = (kxx.sum() - kxx.diag().sum()) / (n * (n - 1)) + \
           (kyy.sum() - kyy.diag().sum()) / (m * (m - 1)) - 2 * kxy.mean()
    return float(mmd2)


def coral_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Sun & Saenko (2016): ||C_s - C_t||_F^2 / (4 d^2)."""
    d = x.shape[1]
    return float(np.sum((np.cov(x, rowvar=False) - np.cov(y, rowvar=False)) ** 2) / (4 * d * d))


def domain_probe(x: np.ndarray, y: np.ndarray, seed: int) -> Dict:
    """5-fold linear domain classifier on frozen features -> accuracy and proxy A-distance."""
    z = np.concatenate([x, y])
    d = np.r_[np.zeros(len(x)), np.ones(len(y))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    acc = float(cross_val_score(clf, z, d, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                scoring="accuracy").mean())
    err = 1.0 - max(acc, 0.5)
    return {"probe_acc": acc, "proxy_a_distance": 2.0 * (1.0 - 2.0 * err)}


def domain_discrepancy(fs: np.ndarray, ft: np.ndarray, seed: int) -> Dict:
    out = {"mmd": mmd_rbf(l2n(fs), l2n(ft)), "coral": coral_distance(l2n(fs), l2n(ft)),
           "mmd_raw": mmd_rbf(fs, ft), "coral_raw": coral_distance(fs, ft),
           "mean_feat_norm_source": float(np.linalg.norm(fs, axis=1).mean()),
           "mean_feat_norm_target": float(np.linalg.norm(ft, axis=1).mean())}
    out.update(domain_probe(fs, ft, seed))
    return out

# %% [markdown]
# ## 18. Visualization

# %%
def plot_roc(results: Dict[str, Dict], path: str) -> None:
    plt.figure(figsize=(6, 6))
    for name, r in results.items():
        y, p = r["test_labels"], r["test_probs"]
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        lo, hi = r["auc_ci"]
        plt.plot(fpr, tpr, lw=2, label=f"{name}  AUC={safe_auc(y, p):.3f} [{lo:.3f}-{hi:.3f}]")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="chance")
    plt.xlabel("False positive rate (1 - specificity)")
    plt.ylabel("True positive rate (sensitivity)")
    plt.title("MosMed TEST - volume-level ROC")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()


def plot_dann_history(history: List[Dict], path: str) -> None:
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    ax[0].plot(h.epoch, h.cls_loss, label="classification")
    ax[0].plot(h.epoch, h.domain_loss, label="domain")
    ax[0].plot(h.epoch, h.total_loss, label="total")
    ax[0].set_title("DANN losses")
    ax[0].legend()
    ax[1].plot(h.epoch, h.domain_acc, label="overall")
    ax[1].plot(h.epoch, h.domain_acc_source, "--", label="source")
    ax[1].plot(h.epoch, h.domain_acc_target, "--", label="target")
    ax[1].axhline(0.5, color="gray", ls=":")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Domain accuracy (0.5 = confused)")
    ax[1].legend()
    ax[2].plot(h.epoch, h.grl_lambda)
    ax[2].set_title("GRL lambda (end of epoch)")
    ax[3].plot(h.epoch, h.source_val_auc, label="NIH val AUC")
    if h.target_val_auc.notna().any():
        ax[3].plot(h.epoch, h.target_val_auc, label="MosMed val AUC")
    ax[3].set_title("Validation AUC")
    ax[3].legend()
    for a in ax:
        a.set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()


def plot_embeddings(feats: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                    path: str, seed: int, method: str = "tsne") -> None:
    """One panel per model; colour = domain, marker = class (o negative, x positive)."""
    fig, axes = plt.subplots(1, len(feats), figsize=(6 * len(feats), 5.5), squeeze=False)
    for ax, (name, (fs, ys, ft, yt)) in zip(axes[0], feats.items()):
        z = l2n(np.concatenate([fs, ft]))
        if method == "umap":
            import umap  # optional dependency: pip install umap-learn
            emb = umap.UMAP(random_state=seed).fit_transform(z)
        else:
            emb = TSNE(n_components=2, init="pca", random_state=seed,
                       perplexity=min(30, max(5, len(z) // 20))).fit_transform(z)
        es, et = emb[:len(fs)], emb[len(fs):]
        for e, y, col, dom in ((es, ys, "tab:blue", "NIH CXR"), (et, yt, "tab:orange", "MosMed CT")):
            ax.scatter(*e[y == 0].T, s=9, c=col, marker="o", alpha=.5, label=f"{dom} neg")
            ax.scatter(*e[y == 1].T, s=14, c=col, marker="x", alpha=.6, label=f"{dom} pos")
        ax.set_title(f"{name} ({method.upper()})")
        ax.legend(fontsize=7, markerscale=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()

# %% [markdown]
# ## 19. Experiment runner

# %%
def prepare_data(c: Config):
    print("=" * 90 + "\nDATA DISCOVERY\n" + "=" * 90)
    nih_df = build_nih_labels(c.NIH_CSV, discover_nih_images(c.NIH_ROOT), c.INCLUDE_INFILTRATION)
    mosmed_df = discover_mosmed_volumes(c.MOSMED_ROOT)

    store = CTSlabStore(c)
    failed = store.build(mosmed_df["filepath"].tolist())
    if failed:  # label-agnostic exclusion of unreadable files, BEFORE the split
        mosmed_df = mosmed_df[~mosmed_df.filepath.isin(failed)].reset_index(drop=True)
        print(f"[MosMed] {len(failed)} unreadable volumes excluded before splitting.")

    print("=" * 90 + "\nSPLITS\n" + "=" * 90)
    nih = dict(zip(("train", "val", "test"),
                   split_nih(nih_df, c.NIH_VAL_RATIO, c.NIH_TEST_RATIO, c.SEED)))
    mosmed = dict(zip(("train", "val", "test"),
                      split_mosmed(mosmed_df, c.TARGET_TRAIN_RATIO, c.TARGET_VAL_RATIO,
                                   c.TARGET_TEST_RATIO, c.SEED)))
    for k, v in mosmed.items():
        v.to_csv(os.path.join(c.OUTPUT_DIR, f"mosmed_{k}_split.csv"), index=False)
    for k, v in nih.items():
        v.to_csv(os.path.join(c.OUTPUT_DIR, f"nih_{k}_split.csv"), index=False)
    return nih, mosmed, store


def print_protocol(c: Config) -> None:
    crit = selection_criterion(c)
    print("=" * 90 + "\nPROTOCOL\n" + "=" * 90)
    print("* Target labels in any training gradient : NO (DANN target loader has no labels)")
    print("* MosMed images seen during training     : TRAIN split only (DANN); none (Source-Only)")
    print(f"* Checkpoint selection criterion         : {crit}")
    if crit == "target_val_auc":
        print("  -> MosMed VAL labels are used for model selection and threshold calibration.\n"
              "     This is NOT strictly unsupervised; report it as 'UDA with a small labelled\n"
              "     target validation set for model selection'. Set\n"
              "     USE_TARGET_LABELS_FOR_MODEL_SELECTION=False for the strict protocol.")
    else:
        print("  -> Strict UDA: no MosMed label is read before the final test evaluation\n"
              "     (the 'thr_target_val' threshold row is then only a post-hoc analysis).")
    print("* MosMed TEST usage                      : once, final evaluation only")


def run_all_experiments(c: Config) -> Dict:
    c.validate()
    os.makedirs(c.OUTPUT_DIR, exist_ok=True)
    set_seed(c.SEED)
    nih, mosmed, store = prepare_data(c)
    loaders = build_all_loaders(c, nih, mosmed, store)
    print_protocol(c)
    crit = selection_criterion(c)
    use_tgt = crit == "target_val_auc"
    models_: Dict[str, nn.Module] = {}
    info: Dict[str, Dict] = {}

    print("=" * 90 + "\nEXPERIMENT 1 - SOURCE ONLY / TRANSFER LEARNING BASELINE\n" + "=" * 90)
    set_seed(c.SEED)  # identical init + data order for every experiment
    models_["Source Only"], info["Source Only"] = train_supervised(
        LungClassifierNet(c, with_domain_head=False), loaders["nih_train"], loaders, c,
        c.NUM_EPOCHS_SOURCE_ONLY, c.SOURCE_ONLY_CKPT_PATH, "Source-Only", crit, use_tgt)

    print("=" * 90 + "\nEXPERIMENT 2 - DANN\n" + "=" * 90)
    set_seed(c.SEED)
    models_["DANN"], info["DANN"] = train_dann(LungClassifierNet(c, with_domain_head=True),
                                               loaders, c, crit)
    plot_dann_history(info["DANN"]["history"], os.path.join(c.OUTPUT_DIR, "dann_history.png"))

    if c.RUN_TARGET_ORACLE:
        print("=" * 90 + "\nREFERENCE - TARGET ORACLE (supervised on MosMed TRAIN labels; NOT UDA)\n"
              + "=" * 90)
        set_seed(c.SEED)
        models_["Target Oracle (upper bound)"], info["Target Oracle (upper bound)"] = train_supervised(
            LungClassifierNet(c, with_domain_head=False), loaders["mosmed_train_labeled"], loaders,
            c, c.NUM_EPOCHS_ORACLE, c.ORACLE_CKPT_PATH, "Target-Oracle", "target_val_auc", True)

    print("=" * 90 + "\nFINAL EVALUATION ON MosMed TEST (volume level)\n" + "=" * 90)
    target_res = {n: evaluate_model_on_target(m, loaders, c, n) for n, m in models_.items()}
    source_res = {}
    for n, m in models_.items():
        p, y, _ = predict_images(m, loaders["nih_test"], c)
        source_res[n] = safe_auc(y, p)
    plot_roc(target_res, os.path.join(c.OUTPUT_DIR, "roc_mosmed_test.png"))

    print("=" * 90 + "\nDOMAIN ALIGNMENT ANALYSIS (NIH TEST vs MosMed TEST, frozen backbone)\n"
          + "=" * 90)
    nih_ds, mm_ds = loaders["nih_test"].dataset, loaders["mosmed_test"].dataset
    idx_s, idx_t = fixed_feature_subsets(nih_ds, mm_ds, c.FEATURE_N_SAMPLES, c.SEED)
    print(f"[Features] n={len(idx_s)} per domain (same indices for every model)")
    disc, emb_inputs = {}, {}
    for n in ("Source Only", "DANN"):
        fs, ys = extract_features(models_[n], nih_ds, idx_s, c)
        ft, yt = extract_features(models_[n], mm_ds, idx_t, c)
        disc[n] = domain_discrepancy(fs, ft, c.SEED)
        emb_inputs[n] = (fs, ys, ft, yt)
        d = disc[n]
        print(f"[{n:<11}] MMD {d['mmd']:.5f} | CORAL {d['coral']:.3e} | linear domain-probe acc "
              f"{d['probe_acc']:.3f} (PAD {d['proxy_a_distance']:.3f}) | feat-norm S/T "
              f"{d['mean_feat_norm_source']:.2f}/{d['mean_feat_norm_target']:.2f}")
    print("NOTE: lower MMD/CORAL/probe-acc is NOT success by itself; check target AUC "
          "(alignment without discriminability = negative transfer).")
    for method, flag in (("tsne", c.RUN_TSNE), ("umap", c.RUN_UMAP)):
        if flag:
            try:
                plot_embeddings(emb_inputs, os.path.join(c.OUTPUT_DIR, f"{method}_features.png"),
                                c.SEED, method)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] {method} plot skipped: {e}")

    stats = paired_bootstrap_auc_diff(target_res["Source Only"]["test_labels"],
                                      target_res["Source Only"]["test_probs"],
                                      target_res["DANN"]["test_probs"], c.NUM_BOOTSTRAP, c.SEED)
    return {"config": c, "splits": {"nih": nih, "mosmed": mosmed}, "models": models_,
            "train_info": info, "target_results": target_res, "source_test_auc": source_res,
            "discrepancy": disc, "dann_vs_source_only": stats}

# %% [markdown]
# ## 20. Final comparison

# %%
def build_results_table(r: Dict) -> pd.DataFrame:
    c: Config = r["config"]
    thr_key = "thr_target_val" if selection_criterion(c) == "target_val_auc" else "thr_source_val"
    rows, detailed = [], []
    for name, res in r["target_results"].items():
        m, m05 = res[thr_key], res["thr_0.5"]
        d = r["discrepancy"].get(name, {})
        rows.append({
            "Model": name,
            "AUC": m["roc_auc"], "AUC 95% CI": f"{res['auc_ci'][0]:.3f}-{res['auc_ci'][1]:.3f}",
            "PR-AUC": m["pr_auc"], "ACC": m["accuracy"], "F1": m["f1"],
            "Sensitivity": m["sensitivity"], "Specificity": m["specificity"],
            "Balanced ACC": m["balanced_accuracy"], "Threshold": m["threshold"],
            "ACC@0.5": m05["accuracy"], "F1@0.5": m05["f1"],
            "Sens@0.5": m05["sensitivity"], "Spec@0.5": m05["specificity"],
            "MMD": d.get("mmd", np.nan), "CORAL": d.get("coral", np.nan),
            "Domain-probe ACC": d.get("probe_acc", np.nan),
            "NIH test AUC": r["source_test_auc"][name],
            "Best epoch": r["train_info"][name]["best_epoch"],
            "Selection": r["train_info"][name]["criterion"],
        })
        for k in ("thr_0.5", "thr_target_val", "thr_source_val"):
            detailed.append({"Model": name, "threshold_rule": k, **res[k], **d})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("=" * 90 + f"\nFINAL COMPARISON - MosMed TEST, volume level, threshold rule = {thr_key}\n"
          + "=" * 90)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    s = r["dann_vs_source_only"]
    print(f"\nDANN - Source Only: dAUC = {s['delta_auc']:+.4f} (95% CI {s['ci_low']:+.4f} to "
          f"{s['ci_high']:+.4f}, paired bootstrap p = {s['p_value']:.4f})")
    print("Single seed. For the thesis repeat with >= 3-5 seeds and report mean +/- std.")
    os.makedirs(os.path.dirname(c.RESULTS_CSV_PATH) or ".", exist_ok=True)
    df.to_csv(c.RESULTS_CSV_PATH, index=False)
    detailed_path = c.RESULTS_CSV_PATH.replace(".csv", "_detailed.csv")
    pd.DataFrame(detailed).to_csv(detailed_path, index=False)
    with open(os.path.join(c.OUTPUT_DIR, "experiment_protocol.json"), "w") as f:
        json.dump(_to_py({"config": asdict(c), "selection_criterion": selection_criterion(c),
                          "table_threshold_rule": thr_key, "dann_vs_source_only": s,
                          "split_sizes": {f"{dom}_{k}": len(v) for dom in ("nih", "mosmed")
                                          for k, v in r["splits"][dom].items()}}), f, indent=2)
    print(f"\nSaved: {c.RESULTS_CSV_PATH}, {detailed_path}, experiment_protocol.json")
    return df


# %%
if __name__ == "__main__":
    results = run_all_experiments(cfg)
    results_df = build_results_table(results)
    try:
        display(results_df)  # noqa: F821  (available in Jupyter/Kaggle)
    except NameError:
        pass
