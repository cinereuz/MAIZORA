"""
Corn Kernel Mold & Disease Detection Pipeline Facade

Orchestrates segmentation, feature extraction, novelty detection, and RBF-SVM
classification while preserving the `CornDiseasePipeline` API for the web app
and research notebooks.
"""

import os

import cv2
import joblib
import numpy as np

from .features import FEATURE_NAMES, FEATURE_PATCH_SIZE, extract_features, preprocess_image
from .novelty import (
    CORN_HUE_MAX,
    CORN_HUE_MIN,
    MAX_KERNEL_ASPECT_RATIO,
    MAX_SINGLE_KERNEL_AREA,
    MAX_SINGLE_KERNEL_SPAN,
    MIN_KERNEL_SOLIDITY,
    NON_CORN_FALLBACK,
    apply_batch_calibrations,
    check_is_corn,
    decision_to_probs,
    mahalanobis_distance,
)
from .segmentation import (
    STANDARD_FRAME_SIZE,
    auto_resize_and_crop,
    extract_warmth_fg_mask,
    segment_fungus_kmeans,
    segment_kernels_otsu,
)

CONTAMINATION_LIGHT_MAX_PCT = 15.0

_ROI_PALETTE_BGR = [
    (255, 195, 0),
    (80, 220, 100),
    (255, 120, 50),
    (180, 105, 255),
    (50, 205, 255),
    (255, 150, 200),
]


def _make_raw_preview(img):
    orig_h, orig_w = img.shape[:2] if img is not None and img.size > 0 else STANDARD_FRAME_SIZE
    max_raw_dim = max(orig_h, orig_w)
    if max_raw_dim > 480:
        r_scale = 480.0 / float(max_raw_dim)
        raw_preview = cv2.resize(
            img,
            (max(1, int(round(orig_w * r_scale))), max(1, int(round(orig_h * r_scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        raw_preview = img.copy() if img is not None else None
    return orig_w, orig_h, raw_preview


def _summarize_batch(sehat_count, kontam_count, bukan_jagung_count, total_kernels):
    """Apply majority non-corn decision rule and quality grading thresholds."""
    total_jagung = sehat_count + kontam_count
    if total_jagung == 0 or bukan_jagung_count > total_jagung:
        contam_pct = 0.0
        quality_status = (
            "Not Corn / No Corn Kernels Detected"
            if total_jagung == 0
            else "Not Corn (Predominantly Non-Corn Objects Detected)"
        )
        status_badge = "secondary"
    else:
        contam_pct = round((kontam_count / total_jagung) * 100, 1)
        if contam_pct == 0.0:
            quality_status = "Excellent Quality (Safe for Consumption & Seed)"
            status_badge = "success"
        elif contam_pct <= CONTAMINATION_LIGHT_MAX_PCT:
            quality_status = "Standard Quality (Light Sorting Required)"
            status_badge = "warning"
        else:
            quality_status = "Poor Quality (Mold Contaminated / Unfit)"
            status_badge = "danger"

    return {
        "total_biji": total_kernels,
        "sehat": sehat_count,
        "terkontaminasi": kontam_count,
        "bukan_jagung": bukan_jagung_count,
        "persen_kontaminasi": contam_pct,
        "status_mutu": quality_status,
        "badge": status_badge,
    }


def _render_kernel_outputs(img, raw_evals):
    """Render segmentation/classification overlay images and collect formatted kernel records."""
    annotated_img = img.copy()
    seg_overlay = img.copy()
    kernel_results = []
    sehat_count = 0
    kontam_count = 0
    bukan_jagung_count = 0

    for k, crop, kmeans_res, features, pred_res in raw_evals:
        label = pred_res["prediction"]
        kid_idx = k["kernel_id"]
        seg_color = _ROI_PALETTE_BGR[(kid_idx - 1) % len(_ROI_PALETTE_BGR)]
        if k.get("contour") is not None:
            cv2.drawContours(seg_overlay, [k["contour"]], -1, seg_color, 2)
        sx, sy, sw, sh = k["bbox"]
        cv2.rectangle(seg_overlay, (sx, sy), (sx + sw, sy + sh), seg_color, 1)
        cv2.putText(
            seg_overlay,
            f"ROI #{kid_idx}",
            (sx, max(sy - 5, 13)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            seg_color,
            1,
            cv2.LINE_AA,
        )

        if label == "sehat":
            sehat_count += 1
            box_color = (46, 204, 113)
            label_text = f"Healthy ({pred_res['confidence']}%)"
        elif label == "terkontaminasi":
            kontam_count += 1
            box_color = (60, 76, 231)
            label_text = f"Mold ({pred_res['confidence']}%)"
        else:
            bukan_jagung_count += 1
            box_color = (150, 150, 150)
            label_text = "Not Corn"

        x, y, w, h = k["bbox"]
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(
            annotated_img,
            f"#{k['kernel_id']}: {label_text}",
            (x, max(y - 6, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            box_color,
            1,
            cv2.LINE_AA,
        )

        kernel_results.append(
            {
                "kernel_id": k["kernel_id"],
                "bbox": [int(v) for v in k["bbox"]],
                "area_piksel": int(k.get("area_piksel", w * h)),
                "solidity": k.get("solidity", 1.0),
                "aspect_ratio": k.get(
                    "aspect_ratio", round(max(w, h) / max(1.0, float(min(w, h))), 2)
                ),
                "prediction": label,
                "is_moldy": (label == "terkontaminasi"),
                "is_corn": (label != "bukan_jagung"),
                "confidence": pred_res["confidence"],
                "decision_score": pred_res.get("decision_score"),
                "features": features,
                "scaled_features": pred_res.get("scaled_features"),
                "mold_ratio": round(kmeans_res["mold_ratio"] * 100, 1) if kmeans_res else 0.0,
                "cluster_centers": kmeans_res.get("cluster_centers", []) if kmeans_res else [],
                "probabilities": pred_res["probabilities"],
                "outlier_distance": pred_res.get("outlier_distance"),
                "outlier_threshold": pred_res.get("outlier_threshold"),
                "outlier_nearest_class": pred_res.get("outlier_nearest_class"),
                "crop_img": crop,
                "kmeans_img": kmeans_res["clustered_bgr"] if kmeans_res else None,
                "mold_mask": kmeans_res["mold_mask"] if kmeans_res else None,
            }
        )

    return annotated_img, seg_overlay, kernel_results, sehat_count, kontam_count, bukan_jagung_count


class CornDiseasePipeline:
    """End-to-end corn kernel mold detection pipeline."""

    def __init__(
        self,
        model_path="models/svm_model_v1.joblib",
        scaler_path="models/scaler.joblib",
        outlier_stats_path="models/outlier_stats.joblib",
        outlier_margin=1.08,
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.outlier_stats_path = outlier_stats_path
        self.outlier_margin = outlier_margin
        self.model = None
        self.scaler = None
        self.outlier_stats = None
        self.feature_names = list(FEATURE_NAMES)
        self.load_models()

    def load_models(self):
        """Load trained SVM model, StandardScaler, and Mahalanobis outlier statistics."""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        else:
            return False

        if os.path.exists(self.outlier_stats_path):
            self.outlier_stats = joblib.load(self.outlier_stats_path)
        else:
            self.outlier_stats = None
            print(
                f"[Warning] '{self.outlier_stats_path}' not found -- 'bukan_jagung' novelty detection disabled."
            )
        return True

    def _extract_warmth_fg_mask(self, img, dark_cool_sat_cutoff=65, fill_holes=True):
        return extract_warmth_fg_mask(
            img, dark_cool_sat_cutoff=dark_cool_sat_cutoff, fill_holes=fill_holes
        )

    def auto_resize_and_crop(self, img, target_size=STANDARD_FRAME_SIZE):
        return auto_resize_and_crop(img, target_size=target_size)

    def preprocess_image(self, img, target_size=FEATURE_PATCH_SIZE):
        return preprocess_image(img, target_size=target_size)

    def segment_kernels_otsu(self, img, min_area=90, max_area=50000, margin=4):
        return segment_kernels_otsu(img, min_area=min_area, max_area=max_area, margin=margin)

    def segment_fungus_kmeans(self, kernel_crop, k=3, mask=None):
        return segment_fungus_kmeans(kernel_crop, k=k, mask=mask)

    def extract_features(self, kernel_img, is_preprocessed=False):
        return extract_features(kernel_img, is_preprocessed=is_preprocessed)

    def _mahalanobis_distance(self, x_scaled, mean_vec, cov_inv):
        return mahalanobis_distance(x_scaled, mean_vec, cov_inv)

    def check_is_corn(self, X_scaled_row, features_dict=None, crop_img=None, k_info=None):
        return check_is_corn(
            X_scaled_row,
            outlier_stats=self.outlier_stats,
            outlier_margin=self.outlier_margin,
            features_dict=features_dict,
            crop_img=crop_img,
            k_info=k_info,
        )

    def predict_single(self, features_dict, crop_img=None, k_info=None):
        """Run multi-layer novelty check followed by RBF-SVM classification."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("SVM model or Scaler has not been loaded.")

        feat_values = np.array([[features_dict[name] for name in self.feature_names]])
        X_scaled = self.scaler.transform(feat_values)
        scaled_dict = {
            name: round(float(val), 3)
            for name, val in zip(self.feature_names, X_scaled[0])
        }
        is_corn, nearest_class, distance, threshold = self.check_is_corn(
            X_scaled[0], features_dict=features_dict, crop_img=crop_img, k_info=k_info
        )

        if not is_corn:
            return {
                "prediction": "bukan_jagung",
                "confidence": None,
                "probabilities": {"sehat": 0.0, "terkontaminasi": 0.0},
                "decision_score": None,
                "scaled_features": scaled_dict,
                "outlier_distance": round(distance, 3) if distance is not None else None,
                "outlier_threshold": round(threshold, 3) if threshold is not None else None,
                "outlier_nearest_class": nearest_class,
            }

        prediction = self.model.predict(X_scaled)[0]
        decision = float(self.model.decision_function(X_scaled)[0])
        prob_dict = decision_to_probs(decision, classes=list(self.model.classes_))

        return {
            "prediction": prediction,
            "confidence": prob_dict.get(prediction, 90.0),
            "probabilities": prob_dict,
            "decision_score": round(decision, 3),
            "scaled_features": scaled_dict,
            "outlier_distance": round(distance, 3) if distance is not None else None,
            "outlier_threshold": round(threshold, 3) if threshold is not None else None,
            "outlier_nearest_class": nearest_class,
        }

    def process_image(self, img, include_kmeans=True):
        """Run full end-to-end standardization, segmentation, extraction, and prediction."""
        orig_w, orig_h, raw_preview = _make_raw_preview(img)

        img = self.auto_resize_and_crop(img, target_size=STANDARD_FRAME_SIZE)
        prep = self.preprocess_image(img)
        seg_otsu = self.segment_kernels_otsu(img)
        kernels = seg_otsu["kernels"]

        if not kernels:
            kernels = [
                {
                    "kernel_id": 1,
                    "bbox": (0, 0, img.shape[1], img.shape[0]),
                    "area_piksel": img.shape[0] * img.shape[1],
                    "crop": img,
                    "mask": np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255,
                    "contour": None,
                }
            ]

        raw_evals = []
        for k in kernels:
            crop = k["crop"]
            kmeans_res = (
                self.segment_fungus_kmeans(crop, k=3, mask=k.get("mask"))
                if include_kmeans
                else None
            )
            features = self.extract_features(crop)
            pred_res = self.predict_single(features, crop_img=crop, k_info=k)
            raw_evals.append((k, crop, kmeans_res, features, pred_res))

        apply_batch_calibrations(raw_evals)

        annotated_img, seg_overlay, kernel_results, sehat_count, kontam_count, bukan_jagung_count = (
            _render_kernel_outputs(img, raw_evals)
        )
        summary = _summarize_batch(sehat_count, kontam_count, bukan_jagung_count, len(kernel_results))

        return {
            "summary": summary,
            "stage_meta": {
                "input": {
                    "orig_width": int(orig_w),
                    "orig_height": int(orig_h),
                    "orig_aspect": f"{round(orig_w / max(1, orig_h), 2)}:1",
                    "std_width": int(img.shape[1]),
                    "std_height": int(img.shape[0]),
                    "was_cropped": bool(orig_w != 224 or orig_h != 224),
                },
                "preprocess": prep.get("stats", {}),
                "segmentation": seg_otsu.get("seg_stats", {}),
            },
            "kernels": kernel_results,
            "images": {
                "raw_input": raw_preview if raw_preview is not None else img,
                "original": img,
                "gray": prep["gray"],
                "blur": prep["blur"],
                "resized_blur": prep["resized_blur"],
                "otsu_mask": seg_otsu["otsu_mask"],
                "seg_overlay": seg_overlay,
                "annotated": annotated_img,
            },
        }

    def process_image_file(self, file_path):
        """Load an image from disk and run `process_image`."""
        img = cv2.imread(file_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read file: {file_path}")
        return self.process_image(img)
