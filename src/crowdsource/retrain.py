"""Incremental SVM re-training, Mahalanobis outlier stats, and model rollback."""

import os
import shutil
import threading
import time
import traceback
from datetime import datetime

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .storage import (
    DATA_DIR,
    DATASET_CLASSES,
    FEATURE_COLS,
    MANIFEST_CROP,
    MANIFEST_FITUR,
    MODELS_DIR,
    MODEL_FILES,
    MODEL_HISTORY_DIR,
    MODEL_META,
    RAW_DIR,
    SKIPPED_LOG,
    _fs_lock,
    atomic_write_json,
    list_raw_files,
    load_manifests,
    load_model_meta,
    load_skipped,
    log_history,
    save_skipped,
)


def _extract_new_image(pipeline, fpath, label):
    """Segment up to 3 representative kernels from a dataset image and extract features."""
    img = cv2.imread(fpath)
    if img is None:
        return []
    img_std = pipeline.auto_resize_and_crop(img, target_size=(224, 224))
    kernels = pipeline.segment_kernels_otsu(img_std)["kernels"]
    if not kernels:
        return []
    k_data = [
        {"area_piksel": k["area_piksel"], "crop": k["crop"], "features": pipeline.extract_features(k["crop"])}
        for k in kernels
    ]
    if len(k_data) > 3:
        k_data = sorted(
            k_data, key=lambda x: x["features"]["saturation"], reverse=(label == "sehat")
        )[:3]
    return k_data


def compute_outlier_stats(X_scaled, y):
    """Compute global and 5-subcluster Mahalanobis statistics per dataset class."""
    class_stats = {}
    for label in DATASET_CLASSES:
        X_class = X_scaled[y == label]
        mean_vec = X_class.mean(axis=0)
        cov_mat = np.cov(X_class, rowvar=False) + np.eye(X_class.shape[1]) * 1e-5
        cov_inv = np.linalg.inv(cov_mat)
        diffs = X_class - mean_vec
        distances = np.sqrt(np.einsum("ij,jk,ik->i", diffs, cov_inv, diffs))
        thr_global = float(np.percentile(distances, 98.5))

        km = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X_class)
        subclusters = []
        for cid in range(5):
            sub_pts = X_class[km.labels_ == cid]
            if len(sub_pts) < 15:
                continue
            s_mean = np.mean(sub_pts, axis=0)
            s_cov = np.cov(sub_pts, rowvar=False) + np.eye(sub_pts.shape[1]) * 1e-4
            s_inv = np.linalg.inv(s_cov)
            s_diffs = sub_pts - s_mean
            s_dists = np.sqrt(np.sum((s_diffs @ s_inv) * s_diffs, axis=1))
            s_thr = float(np.percentile(s_dists, 98.8))
            subclusters.append(
                {"mean": s_mean, "cov_inv": s_inv, "distance_threshold": max(s_thr, 3.8)}
            )

        class_stats[label] = {
            "mean": mean_vec,
            "cov_inv": cov_inv,
            "distance_threshold": thr_global,
            "percentile": 98.5,
            "n_samples": int(len(X_class)),
            "subclusters": subclusters,
        }
    return class_stats


def _backup_current_models():
    """Copy active models and manifests to `models/history/<timestamp>/` before overwriting."""
    if not all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in MODEL_FILES):
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(MODEL_HISTORY_DIR, stamp)
    os.makedirs(dst, exist_ok=True)
    for f in MODEL_FILES + ["model_meta.json"]:
        src = os.path.join(MODELS_DIR, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst, f))
    for m in (MANIFEST_CROP, MANIFEST_FITUR):
        if os.path.exists(m):
            shutil.copy2(m, os.path.join(dst, os.path.basename(m)))
    return stamp


def list_model_backups():
    if not os.path.isdir(MODEL_HISTORY_DIR):
        return []
    return sorted(os.listdir(MODEL_HISTORY_DIR), reverse=True)


class RetrainManager:
    """Runs incremental model re-training in a background thread and tracks progress."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._lock = threading.Lock()
        self.state = {
            "running": False,
            "stage": "idle",
            "progress": 0,
            "message": "",
            "result": None,
            "error": None,
        }

    def status(self):
        return dict(self.state)

    def _set(self, **kw):
        self.state.update(kw)

    def start(self):
        if not self._lock.acquire(blocking=False):
            return False
        self.state = {
            "running": True,
            "stage": "starting",
            "progress": 0,
            "message": "Preparing...",
            "result": None,
            "error": None,
        }
        threading.Thread(target=self._run_wrapper, daemon=True).start()
        return True

    def _run_wrapper(self):
        try:
            result = self._run()
            self._set(
                running=False,
                stage="done",
                progress=100,
                message="Re-training complete",
                result=result,
            )
        except Exception as e:
            traceback.print_exc()
            self._set(running=False, stage="error", message=str(e), error=str(e))
        finally:
            self._lock.release()

    def _run(self):
        t0 = time.time()
        with _fs_lock:
            raw_files = list_raw_files()
        df_crop, df_fitur = load_manifests()
        skipped = load_skipped()

        # Drop manifest rows whose source images were removed from data/raw/
        crop_keys = list(zip(df_crop["label"], df_crop["file_asal"]))
        keep_mask = [k in raw_files for k in crop_keys]
        removed_crops = set(df_crop.loc[[not m for m in keep_mask], "file_crop"])
        df_crop = df_crop[keep_mask].reset_index(drop=True)
        if removed_crops:
            df_fitur = df_fitur[~df_fitur["file_crop"].isin(removed_crops)].reset_index(drop=True)

        known = set(zip(df_crop["label"], df_crop["file_asal"])) | skipped
        new_files = sorted(raw_files - known)
        self._set(stage="extract", message=f"Extracting features from {len(new_files)} new image(s)...")

        new_crop_rows, new_fitur_rows, n_new_ok = [], [], 0
        for i, (label, fname) in enumerate(new_files, start=1):
            fpath = os.path.join(RAW_DIR, label, fname)
            try:
                k_data = _extract_new_image(self.pipeline, fpath, label)
            except Exception as e:
                print(f"[Retrain] Skip {fpath}: {e}")
                k_data = []
            if not k_data:
                skipped.add((label, fname))
            else:
                n_new_ok += 1
                base = os.path.splitext(fname)[0]
                for idx, k in enumerate(k_data, start=1):
                    crop_fname = f"{base}_k{idx}.png"
                    for sub in ("cropped", "preprocessed"):
                        os.makedirs(os.path.join(DATA_DIR, sub, label), exist_ok=True)
                    cv2.imwrite(os.path.join(DATA_DIR, "cropped", label, crop_fname), k["crop"])
                    prep = cv2.GaussianBlur(
                        cv2.resize(k["crop"], (128, 128), interpolation=cv2.INTER_CUBIC),
                        (3, 3),
                        0,
                    )
                    cv2.imwrite(os.path.join(DATA_DIR, "preprocessed", label, crop_fname), prep)
                    new_crop_rows.append(
                        {
                            "file_asal": fname,
                            "file_crop": crop_fname,
                            "label": label,
                            "area_piksel": int(k["area_piksel"]),
                        }
                    )
                    f = k["features"]
                    new_fitur_rows.append(
                        {"file_crop": crop_fname, "label": label, **{c: f[c] for c in FEATURE_COLS}}
                    )
            self._set(progress=int(5 + 35 * i / max(1, len(new_files))))

        if new_crop_rows:
            df_crop = pd.concat([df_crop, pd.DataFrame(new_crop_rows)], ignore_index=True)
            df_fitur = pd.concat([df_fitur, pd.DataFrame(new_fitur_rows)], ignore_index=True)

        counts = df_fitur["label"].value_counts().to_dict()
        if any(counts.get(c, 0) < 20 for c in DATASET_CLASSES):
            raise RuntimeError(f"Not enough training samples per class: {counts}")

        self._set(
            stage="train",
            progress=45,
            message=f"Training SVM on {len(df_fitur)} kernel samples (GridSearchCV)...",
        )
        X = df_fitur[FEATURE_COLS].values.astype(float)
        y = df_fitur["label"].values

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Evaluate the previous model on the exact same test split for fair comparison.
        old_acc = None
        if self.pipeline.model is not None and self.pipeline.scaler is not None:
            try:
                old_acc = float(
                    accuracy_score(
                        y_test,
                        self.pipeline.model.predict(self.pipeline.scaler.transform(X_test_raw)),
                    )
                )
            except Exception:
                old_acc = None

        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        X_train, X_test = scaler.transform(X_train_raw), scaler.transform(X_test_raw)

        param_grid = {
            "C": [1, 10, 50, 100],
            "gamma": ["scale", 0.05, 0.1, 0.5, 1.0],
            "kernel": ["rbf"],
        }
        grid = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        y_pred = best.predict(X_test)
        new_acc = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, digits=4, output_dict=True)

        self._set(
            stage="outlier",
            progress=85,
            message="Updating novelty detection statistics (Mahalanobis)...",
        )
        outlier_stats = compute_outlier_stats(X_scaled, y)

        self._set(stage="save", progress=92, message="Saving new model version...")
        prev_meta = load_model_meta() or {}
        backup = _backup_current_models()
        version = int(prev_meta.get("version", 1)) + 1

        joblib.dump(best, os.path.join(MODELS_DIR, "svm_model_v1.joblib"))
        joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
        joblib.dump(outlier_stats, os.path.join(MODELS_DIR, "outlier_stats.joblib"))
        df_crop.to_csv(MANIFEST_CROP, index=False)
        df_fitur.to_csv(MANIFEST_FITUR, index=False)
        save_skipped(skipped)

        meta = {
            "version": version,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "accuracy": round(new_acc * 100, 2),
            "previous_accuracy_same_test": round(old_acc * 100, 2) if old_acc is not None else None,
            "best_params": {
                k: (v if isinstance(v, (int, float, str)) else str(v))
                for k, v in grid.best_params_.items()
            },
            "n_samples": int(len(df_fitur)),
            "n_samples_per_class": {k: int(v) for k, v in counts.items()},
            "n_images": int(df_crop["file_asal"].nunique()),
            "n_new_images": int(n_new_ok),
            "n_skipped_new_images": int(len(new_files) - n_new_ok),
            "f1_per_class": {
                c: round(report[c]["f1-score"] * 100, 2) for c in DATASET_CLASSES if c in report
            },
            "backup_of_previous": backup,
            "duration_sec": round(time.time() - t0, 1),
        }
        atomic_write_json(MODEL_META, meta, indent=2)

        self.pipeline.load_models()
        log_history(
            {
                "action": "retrain",
                "version": version,
                "accuracy": meta["accuracy"],
                "previous_accuracy": meta["previous_accuracy_same_test"],
                "n_new_images": n_new_ok,
            }
        )
        return meta

    def rollback(self):
        """Restore the latest backed-up model version and manifests."""
        if self.state.get("running"):
            raise RuntimeError("Re-training is in progress")
        backups = list_model_backups()
        if not backups:
            raise FileNotFoundError("No model backup available")
        latest = backups[0]
        src = os.path.join(MODEL_HISTORY_DIR, latest)
        for f in MODEL_FILES:
            shutil.copy2(os.path.join(src, f), os.path.join(MODELS_DIR, f))
        meta_src = os.path.join(src, "model_meta.json")
        if os.path.exists(meta_src):
            shutil.copy2(meta_src, MODEL_META)
        elif os.path.exists(MODEL_META):
            os.remove(MODEL_META)
        for m in (MANIFEST_CROP, MANIFEST_FITUR):
            b = os.path.join(src, os.path.basename(m))
            if os.path.exists(b):
                shutil.copy2(b, m)
        if os.path.exists(SKIPPED_LOG):
            os.remove(SKIPPED_LOG)
        shutil.rmtree(src, ignore_errors=True)
        self.pipeline.load_models()
        log_history({"action": "rollback", "restored_backup": latest})
        return {"restored": latest, "model": load_model_meta()}
