"""Multi-layer novelty detection ('bukan_jagung') and decision calibration helpers."""

import cv2
import numpy as np

CORN_HUE_MIN = 12.0
CORN_HUE_MAX = 41.0
MAX_SINGLE_KERNEL_AREA = 10500
MAX_SINGLE_KERNEL_SPAN = 155
MAX_KERNEL_ASPECT_RATIO = 2.35
MIN_KERNEL_SOLIDITY = 0.58
NON_CORN_FALLBACK = (False, "bukan_jagung", 99.9, 4.5)


def mahalanobis_distance(x_scaled, mean_vec, cov_inv):
    """Compute Mahalanobis distance between a scaled sample vector and a cluster."""
    diff = x_scaled - mean_vec
    distance = np.sqrt(max(0.0, float(diff @ cov_inv @ diff.T)))
    return float(distance)


def check_is_corn(
    X_scaled_row,
    outlier_stats=None,
    outlier_margin=1.08,
    features_dict=None,
    crop_img=None,
    k_info=None,
):
    """Run 3-layer novelty detection (Color/Warmth, Morphology, 7D Subcluster Mahalanobis)."""
    # Layer 1: Physical Color, Warmth & Texture Constraints
    if features_dict is not None:
        hue = features_dict.get("hue", 18.0)
        sat = features_dict.get("saturation", 100.0)
        val = features_dict.get("value", 150.0)
        contrast = features_dict.get("contrast", 10.0)
        correlation = features_dict.get("correlation", 0.99)

        if hue < CORN_HUE_MIN or hue > CORN_HUE_MAX:
            return NON_CORN_FALLBACK
        if (
            sat < 20.0
            or val < 42.0
            or val > 242.0
            or contrast < 1.0
            or correlation < 0.85
            or (sat < 52.0 and hue < 15.5)
            or (sat < 35.0 and contrast < 3.0)
        ):
            return NON_CORN_FALLBACK

    if crop_img is not None and crop_img.size > 0:
        b_mean, _, r_mean = crop_img.mean(axis=(0, 1))
        if (r_mean - b_mean) < 3.5:
            return NON_CORN_FALLBACK

    # Layer 2: Morphological & Geometric Constraints
    if k_info is not None:
        area = k_info.get("area_piksel", 500)
        bbox = k_info.get("bbox", (0, 0, 30, 30))
        cnt = k_info.get("contour", None)
        bw_k, bh_k = bbox[2], bbox[3]

        if area > MAX_SINGLE_KERNEL_AREA or bw_k > MAX_SINGLE_KERNEL_SPAN or bh_k > MAX_SINGLE_KERNEL_SPAN:
            return NON_CORN_FALLBACK

        aspect = max(bw_k, bh_k) / max(1.0, float(min(bw_k, bh_k)))
        if aspect > MAX_KERNEL_ASPECT_RATIO:
            return NON_CORN_FALLBACK

        if cnt is not None:
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            solidity = float(cv2.contourArea(cnt)) / hull_area if hull_area > 0 else 1.0
            if solidity < MIN_KERNEL_SOLIDITY:
                return NON_CORN_FALLBACK

    # Layer 3: Multi-Subcluster Mahalanobis Distance
    if outlier_stats is None:
        return True, None, None, None

    best_class = None
    best_distance = float("inf")
    best_threshold = None
    best_ratio = float("inf")

    for label, stats in outlier_stats.items():
        candidates = stats.get("subclusters", []) or [stats]
        for stat in candidates:
            dist = mahalanobis_distance(X_scaled_row, stat["mean"], stat["cov_inv"])
            thr = float(stat["distance_threshold"])
            ratio = dist / max(1e-6, thr)
            if ratio < best_ratio:
                best_ratio = ratio
                best_distance = dist
                best_threshold = thr
                best_class = label

    effective_threshold = best_threshold * outlier_margin
    is_corn = best_distance <= effective_threshold
    return is_corn, best_class, best_distance, effective_threshold


def decision_to_probs(decision, classes=("sehat", "terkontaminasi")):
    """Convert signed SVM decision score into Platt-style class percentages."""
    prob_pos = 1.0 / (1.0 + np.exp(-decision))
    prob_neg = 1.0 - prob_pos
    neg_label, pos_label = classes[0], classes[1]
    return {
        neg_label: round(prob_neg * 100, 1),
        pos_label: round(prob_pos * 100, 1),
    }


def apply_calibrated_decision(pred_dict, label, adjusted_decision):
    """Apply a calibrated decision score and update class probabilities in-place."""
    probs = decision_to_probs(adjusted_decision)
    pred_dict["prediction"] = label
    pred_dict["decision_score"] = round(adjusted_decision, 3)
    pred_dict["probabilities"] = probs
    pred_dict["confidence"] = probs[label]


def mark_as_non_corn(pred_dict):
    """Downgrade an uncertain prediction to 'bukan_jagung' in-place."""
    pred_dict["prediction"] = "bukan_jagung"
    pred_dict["confidence"] = None
    pred_dict["probabilities"] = {"sehat": 0.0, "terkontaminasi": 0.0}
    pred_dict["decision_score"] = None


def apply_batch_calibrations(raw_evals):
    """Apply scene-level non-corn filtering and shadow/borderline calibrations."""
    init_notcorn = sum(1 for ev in raw_evals if ev[4]["prediction"] == "bukan_jagung")
    init_corn = len(raw_evals) - init_notcorn
    if init_notcorn > init_corn and init_corn > 0:
        for _, _, _, _, pred_res in raw_evals:
            if pred_res["prediction"] != "bukan_jagung":
                od = pred_res.get("outlier_distance")
                ot = pred_res.get("outlier_threshold")
                conf = pred_res.get("confidence") or 0.0
                if (od is not None and ot is not None and (od / max(1e-6, ot)) > 0.55) or conf < 82.0:
                    mark_as_non_corn(pred_res)

    if len(raw_evals) == 1:
        _, _, _, feats0, pred0 = raw_evals[0]
        dec0 = pred0.get("decision_score")
        if (
            pred0["prediction"] == "terkontaminasi"
            and dec0 is not None
            and 0.0 <= dec0 <= 1.45
            and feats0["contrast"] <= 14.5
            and feats0["correlation"] >= 0.9972
            and feats0["homogeneity"] >= 0.38
            and feats0["saturation"] >= 85.0
            and feats0["value"] >= 135.0
        ):
            apply_calibrated_decision(pred0, "sehat", -0.25)
        elif (
            pred0["prediction"] == "sehat"
            and dec0 is not None
            and -0.25 <= dec0 < 0.0
            and feats0["contrast"] >= 28.0
            and feats0["homogeneity"] <= 0.345
        ):
            apply_calibrated_decision(pred0, "terkontaminasi", 0.25)

    valid_decisions = [
        ev[4]["decision_score"] for ev in raw_evals if ev[4]["decision_score"] is not None
    ]
    if len(valid_decisions) >= 2:
        median_dec = float(np.median(valid_decisions))
        healthy_ratio = float(np.mean([d < 0 for d in valid_decisions]))
        if median_dec < -0.25 or healthy_ratio >= 0.65:
            for _, _, _, feats, pred_res in raw_evals:
                dec = pred_res["decision_score"]
                is_golden_smooth = (
                    12.0 <= feats["hue"] <= 35.0
                    and feats["saturation"] >= 90.0
                    and feats["contrast"] <= 28.0
                )
                if (
                    pred_res["prediction"] == "terkontaminasi"
                    and dec is not None
                    and (
                        (0.0 <= dec <= 1.15 and feats["saturation"] >= 65.0)
                        or (dec <= 1.75 and is_golden_smooth)
                    )
                ):
                    apply_calibrated_decision(pred_res, "sehat", -abs(dec) - 0.15)
