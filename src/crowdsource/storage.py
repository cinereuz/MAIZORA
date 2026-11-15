"""Staging queue persistence, hash deduplication, registry, and manifest I/O."""

import glob
import hashlib
import json
import os
import re
import secrets
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
STAGING_DIR = os.path.join(DATA_DIR, "staging")
HISTORY_LOG = os.path.join(DATA_DIR, "crowdsource_history.jsonl")
MANIFEST_CROP = os.path.join(DATA_DIR, "manifest_crop.csv")
MANIFEST_FITUR = os.path.join(DATA_DIR, "manifest_fitur.csv")
SKIPPED_LOG = os.path.join(DATA_DIR, "retrain_skipped.json")

MODELS_DIR = "models"
MODEL_FILES = ["svm_model_v1.joblib", "scaler.joblib", "outlier_stats.joblib"]
MODEL_META = os.path.join(MODELS_DIR, "model_meta.json")
MODEL_HISTORY_DIR = os.path.join(MODELS_DIR, "history")

DATASET_CLASSES = ["sehat", "terkontaminasi"]
STAGING_CATEGORIES = ["sehat", "terkontaminasi", "campuran"]
FEATURE_COLS = [
    "contrast",
    "correlation",
    "energy",
    "homogeneity",
    "hue",
    "saturation",
    "value",
]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Sequential dataset naming convention matching the initial dataset (`normal_001.JPG`, `virus_001.JPG`).
CLASS_PREFIX = {"sehat": "normal_", "terkontaminasi": "virus_"}
DATASET_EXT = ".JPG"

CROWD_REGISTRY = os.path.join(DATA_DIR, "crowd_registry.json")
RAW_HASH_CACHE = os.path.join(DATA_DIR, "raw_hashes.json")
_raw_hashes_mem = None

_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]+$")
_fs_lock = threading.Lock()


def ensure_dirs():
    for c in STAGING_CATEGORIES:
        os.makedirs(os.path.join(STAGING_DIR, c), exist_ok=True)
    for c in DATASET_CLASSES:
        os.makedirs(os.path.join(RAW_DIR, c), exist_ok=True)


def atomic_write_json(path, data, indent=None):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


def log_history(entry):
    entry = {"timestamp": datetime.now().isoformat(timespec="seconds"), **entry}
    with open(HISTORY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_history(limit=None):
    if not os.path.exists(HISTORY_LOG):
        return []
    rows = []
    with open(HISTORY_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows[-limit:] if limit else rows


def validate_id(item_id):
    return bool(item_id) and bool(_SAFE_ID.match(item_id))


def validate_category(category):
    return category in STAGING_CATEGORIES


def decide_staging_category(summary):
    """Determine target staging folder, or None if the detection is not corn."""
    sehat = int(summary.get("sehat", 0))
    kontam = int(summary.get("terkontaminasi", 0))
    total_jagung = sehat + kontam

    # Follow pipeline's Majority Non-Corn Decision Rule: discard non-corn scenes.
    if total_jagung == 0 or summary.get("badge") == "secondary":
        return None
    if sehat > 0 and kontam > 0:
        return "campuran"
    return "sehat" if sehat > 0 else "terkontaminasi"


def _image_hash(img):
    return hashlib.sha1(img.tobytes()).hexdigest()[:12]


def _remove_staging_files(category, item_id):
    folder = os.path.join(STAGING_DIR, category)
    for suffix in (".jpg", "_annotated.jpg", ".json"):
        p = os.path.join(folder, f"{item_id}{suffix}")
        if os.path.exists(p):
            os.remove(p)


def _check_duplicate_hash(img_hash):
    """Check whether `img_hash` already exists in `data/raw/` or `data/staging/`."""
    if img_hash in get_raw_hash_set():
        return "already_in_dataset"
    entry = _load_registry().get(img_hash)
    if entry and os.path.exists(os.path.join(RAW_DIR, entry["label"], entry["filename"])):
        return "already_in_dataset"
    for c in STAGING_CATEGORIES:
        if glob.glob(os.path.join(STAGING_DIR, c, f"*_{img_hash}.json")):
            return "duplicate"
    return None


def sync_raw_hashes():
    """Synchronize pixel hash cache for all images in `data/raw/` and prune duplicates from staging."""
    global _raw_hashes_mem

    ensure_dirs()
    cache = {}
    if os.path.exists(RAW_HASH_CACHE):
        try:
            with open(RAW_HASH_CACHE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    current_files = {}
    for c in DATASET_CLASSES:
        folder = os.path.join(RAW_DIR, c)
        if os.path.isdir(folder):
            for fname in os.listdir(folder):
                if fname.lower().endswith(IMG_EXTS):
                    current_files[f"{c}/{fname}"] = os.path.join(folder, fname)

    dirty = False
    for k in list(cache.keys()):
        if k not in current_files:
            cache.pop(k, None)
            dirty = True

    missing = [(k, path) for k, path in current_files.items() if k not in cache]
    if missing:
        print(f"[Crowdsource] Indexing pixel hashes for {len(missing)} dataset image(s) in data/raw/...")

        def _hash_one(item):
            k, path = item
            img = cv2.imread(path)
            if img is None:
                return k, None
            return k, _image_hash(img)

        with ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4))) as ex:
            for k, h in ex.map(_hash_one, missing):
                if h:
                    cache[k] = h
        dirty = True

    if dirty:
        atomic_write_json(RAW_HASH_CACHE, cache)

    _raw_hashes_mem = set(cache.values())

    for c in STAGING_CATEGORIES:
        for meta_path in glob.glob(os.path.join(STAGING_DIR, c, "*.json")):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                h = meta.get("hash")
                item_id = meta.get("id")
                if h and h in _raw_hashes_mem and item_id:
                    _remove_staging_files(c, item_id)
            except Exception:
                continue

    return _raw_hashes_mem


def get_raw_hash_set():
    global _raw_hashes_mem
    if _raw_hashes_mem is None:
        return sync_raw_hashes()
    return _raw_hashes_mem


def _register_raw_hash(label, filename, img_hash):
    """Add a newly approved image hash to the in-memory and on-disk cache."""
    global _raw_hashes_mem
    if _raw_hashes_mem is None:
        sync_raw_hashes()
    _raw_hashes_mem.add(img_hash)
    try:
        cache = {}
        if os.path.exists(RAW_HASH_CACHE):
            with open(RAW_HASH_CACHE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        cache[f"{label}/{filename}"] = img_hash
        atomic_write_json(RAW_HASH_CACHE, cache)
    except Exception:
        pass


def _load_registry():
    if os.path.exists(CROWD_REGISTRY):
        try:
            with open(CROWD_REGISTRY, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_registry(reg):
    atomic_write_json(CROWD_REGISTRY, reg, indent=2)


def _next_dataset_filename(label, registry=None):
    """Generate the next sequential dataset filename across raw dir, manifest, and registry."""
    prefix = CLASS_PREFIX[label]
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.[A-Za-z]+$", re.IGNORECASE)
    names = set()

    folder = os.path.join(RAW_DIR, label)
    if os.path.isdir(folder):
        names.update(os.listdir(folder))
    if os.path.exists(MANIFEST_CROP):
        try:
            df = pd.read_csv(MANIFEST_CROP, usecols=["file_asal", "label"])
            names.update(df.loc[df["label"] == label, "file_asal"].astype(str))
        except Exception:
            pass
    for e in (registry if registry is not None else _load_registry()).values():
        if e.get("label") == label:
            names.add(e.get("filename", ""))

    max_num = 0
    for n in names:
        m = pattern.match(n)
        if m:
            max_num = max(max_num, int(m.group(1)))
    return f"{prefix}{max_num + 1:03d}{DATASET_EXT}"


def migrate_legacy_crowd_files():
    """Idempotently rename legacy `crowd_*.jpg` files in `data/raw/` to sequential dataset names."""
    legacy = []
    for label in DATASET_CLASSES:
        for p in glob.glob(os.path.join(RAW_DIR, label, "crowd_*")):
            legacy.append((label, p))
    if not legacy:
        return []

    renamed = []
    with _fs_lock:
        reg = _load_registry()
        df_crop = pd.read_csv(MANIFEST_CROP) if os.path.exists(MANIFEST_CROP) else None
        for label, old_path in sorted(legacy):
            old_name = os.path.basename(old_path)
            stem = os.path.splitext(old_name)[0]
            parts = stem.split("_")
            img_hash = parts[-1]
            staging_id = "_".join(parts[1:])
            new_name = _next_dataset_filename(label, reg)
            os.rename(old_path, os.path.join(RAW_DIR, label, new_name))
            reg[img_hash] = {
                "label": label,
                "filename": new_name,
                "staging_id": staging_id,
                "source": None,
                "approved_at": datetime.now().isoformat(timespec="seconds"),
                "migrated_from": old_name,
            }
            if df_crop is not None:
                mask = (df_crop["label"] == label) & (df_crop["file_asal"] == old_name)
                df_crop.loc[mask, "file_asal"] = new_name
            renamed.append((old_name, new_name))
        _save_registry(reg)
        if df_crop is not None:
            df_crop.to_csv(MANIFEST_CROP, index=False)
    for old, new in renamed:
        print(f"[Crowdsource] Renamed legacy dataset file {old} -> {new}")
    return renamed


def save_to_staging(original_img, results, source):
    """Save the original image, annotated preview, and metadata to `data/staging/<category>/`."""
    try:
        summary = results["summary"]
        category = decide_staging_category(summary)
        if category is None:
            return {"saved": False, "reason": "not_corn"}

        img_hash = _image_hash(original_img)
        with _fs_lock:
            ensure_dirs()
            dup_reason = _check_duplicate_hash(img_hash)
            if dup_reason:
                return {"saved": False, "reason": dup_reason}

            item_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}_{img_hash}"
            folder = os.path.join(STAGING_DIR, category)

            cv2.imwrite(
                os.path.join(folder, f"{item_id}.jpg"),
                original_img,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
            annotated = results.get("images", {}).get("annotated")
            if annotated is not None:
                cv2.imwrite(
                    os.path.join(folder, f"{item_id}_annotated.jpg"),
                    annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 88],
                )

            valid_kernels = [
                k for k in results.get("kernels", []) if k.get("prediction") != "bukan_jagung"
            ]
            confs = [float(k.get("confidence", 0)) for k in valid_kernels]
            meta = {
                "id": item_id,
                "category": category,
                "source": source,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "hash": img_hash,
                "summary": {
                    "total_biji": int(summary.get("total_biji", 0)),
                    "sehat": int(summary.get("sehat", 0)),
                    "terkontaminasi": int(summary.get("terkontaminasi", 0)),
                    "bukan_jagung": int(summary.get("bukan_jagung", 0)),
                    "persen_kontaminasi": float(summary.get("persen_kontaminasi", 0)),
                },
                "avg_confidence": round(float(np.mean(confs)), 2) if confs else None,
                "min_confidence": round(float(np.min(confs)), 2) if confs else None,
                "user_feedback": None,
            }
            atomic_write_json(os.path.join(folder, f"{item_id}.json"), meta, indent=2)

        return {"saved": True, "category": category, "id": item_id}
    except Exception as e:  # Auto-save failures must never block user detection responses.
        print(f"[Crowdsource] Failed to save staging item: {e}")
        return {"saved": False, "reason": "error"}


def _find_item(item_id):
    """Locate a staging item by ID. Returns (category, meta_path) or (None, None)."""
    for c in STAGING_CATEGORIES:
        p = os.path.join(STAGING_DIR, c, f"{item_id}.json")
        if os.path.exists(p):
            return c, p
    return None, None


def submit_user_feedback(item_id, is_correct, suggested_label=None):
    """Attach optional user correctness feedback to a queued staging item."""
    if not validate_id(item_id):
        return False
    if suggested_label is not None and suggested_label not in DATASET_CLASSES:
        return False
    with _fs_lock:
        _, meta_path = _find_item(item_id)
        if not meta_path:
            return False
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["user_feedback"] = {
            "is_correct": bool(is_correct),
            "suggested_label": suggested_label,
            "at": datetime.now().isoformat(timespec="seconds"),
        }
        atomic_write_json(meta_path, meta, indent=2)
    return True


def count_images(folder):
    if not os.path.isdir(folder):
        return 0
    return sum(1 for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))


def load_model_meta():
    if os.path.exists(MODEL_META):
        try:
            with open(MODEL_META, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def list_raw_files():
    """Return set of (label, filename) pairs for all images in `data/raw/`."""
    files = set()
    for c in DATASET_CLASSES:
        folder = os.path.join(RAW_DIR, c)
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                if f.lower().endswith(IMG_EXTS):
                    files.add((c, f))
    return files


def load_manifests():
    if os.path.exists(MANIFEST_CROP) and os.path.exists(MANIFEST_FITUR):
        return pd.read_csv(MANIFEST_CROP), pd.read_csv(MANIFEST_FITUR)
    return (
        pd.DataFrame(columns=["file_asal", "file_crop", "label", "area_piksel"]),
        pd.DataFrame(columns=["file_crop", "label"] + FEATURE_COLS),
    )


def load_skipped():
    if os.path.exists(SKIPPED_LOG):
        try:
            with open(SKIPPED_LOG, "r", encoding="utf-8") as f:
                return {tuple(x) for x in json.load(f)}
        except Exception:
            return set()
    return set()


def save_skipped(skipped):
    atomic_write_json(SKIPPED_LOG, sorted(list(skipped)))


def count_pending_retrain():
    """Count images in `data/raw/` whose features have not yet been extracted."""
    try:
        df_crop, _ = load_manifests()
        known = set(zip(df_crop["label"], df_crop["file_asal"]))
        known |= load_skipped()
        return len(list_raw_files() - known)
    except Exception:
        return 0


def list_staging():
    ensure_dirs()
    items = []
    for c in STAGING_CATEGORIES:
        for meta_path in glob.glob(os.path.join(STAGING_DIR, c, "*.json")):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["category"] = c
                items.append(meta)
            except Exception:
                continue
    items.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return items


def get_dataset_stats():
    ensure_dirs()
    raw = {c: count_images(os.path.join(RAW_DIR, c)) for c in DATASET_CLASSES}
    crowd = {c: 0 for c in DATASET_CLASSES}
    for e in _load_registry().values():
        if e.get("label") in crowd and os.path.exists(os.path.join(RAW_DIR, e["label"], e["filename"])):
            crowd[e["label"]] += 1
    staging = {
        c: len(glob.glob(os.path.join(STAGING_DIR, c, "*.json")))
        for c in STAGING_CATEGORIES
    }
    return {
        "raw": raw,
        "raw_total": sum(raw.values()),
        "crowd_approved": crowd,
        "staging": staging,
        "staging_total": sum(staging.values()),
        "pending_retrain": count_pending_retrain(),
        "model": load_model_meta(),
        "history": list(reversed(_read_history(limit=15))),
    }


def staging_image_path(category, item_id, kind="original"):
    if not validate_category(category) or not validate_id(item_id):
        return None
    suffix = "_annotated.jpg" if kind == "annotated" else ".jpg"
    path = os.path.join(STAGING_DIR, category, f"{item_id}{suffix}")
    return path if os.path.exists(path) else None


def approve_item(item_id, label):
    """Move a staging image into `data/raw/<label>/` using sequential dataset naming."""
    if not validate_id(item_id) or label not in DATASET_CLASSES:
        raise ValueError("Invalid id or label")
    with _fs_lock:
        category, meta_path = _find_item(item_id)
        if not meta_path:
            raise FileNotFoundError("Staging item not found")
        folder = os.path.join(STAGING_DIR, category)
        src_img = os.path.join(folder, f"{item_id}.jpg")
        if not os.path.exists(src_img):
            raise FileNotFoundError("Staging image file missing")

        os.makedirs(os.path.join(RAW_DIR, label), exist_ok=True)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        reg = _load_registry()
        new_name = _next_dataset_filename(label, reg)
        dst_img = os.path.join(RAW_DIR, label, new_name)
        shutil.move(src_img, dst_img)

        img_hash = meta.get("hash") or item_id.rsplit("_", 1)[-1]
        reg[img_hash] = {
            "label": label,
            "filename": new_name,
            "staging_id": item_id,
            "source": meta.get("source"),
            "approved_at": datetime.now().isoformat(timespec="seconds"),
        }
        _save_registry(reg)
        _register_raw_hash(label, new_name, img_hash)
        _remove_staging_files(category, item_id)

        log_history(
            {
                "action": "approve",
                "id": item_id,
                "from_category": category,
                "label": label,
                "relabeled": category != label,
                "source": meta.get("source"),
                "filename": new_name,
                "hash": img_hash,
            }
        )
    return {"id": item_id, "label": label, "filename": new_name, "path": dst_img}


def reject_item(item_id):
    if not validate_id(item_id):
        raise ValueError("Invalid id")
    with _fs_lock:
        category, meta_path = _find_item(item_id)
        if not meta_path:
            raise FileNotFoundError("Staging item not found")
        _remove_staging_files(category, item_id)
        log_history({"action": "reject", "id": item_id, "from_category": category})
    return {"id": item_id}
