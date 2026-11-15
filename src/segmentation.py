"""ROI standardization, Otsu+Watershed kernel segmentation, and K-Means mold isolation."""

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from .features import foreground_pixel_mask

STANDARD_FRAME_SIZE = (224, 224)
MORPH_ELLIPSE_3X3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def extract_warmth_fg_mask(img, dark_cool_sat_cutoff=65, fill_holes=True):
    """Compute a polarity-aware Otsu foreground mask with cool cloth-glare suppression."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    b_ch, _, r_ch = cv2.split(cv2.GaussianBlur(img, (5, 5), 0))

    cool_diff = np.maximum(0.0, b_ch.astype(np.float32) - r_ch.astype(np.float32))
    warm_blur = np.clip(blur.astype(np.float32) - cool_diff * 3.0, 0, 255).astype(np.uint8)

    bw = max(4, int(min(h, w) * 0.04))
    b_mask = np.zeros((h, w), dtype=bool)
    b_mask[:bw, :] = True
    b_mask[-bw:, :] = True
    b_mask[:, :bw] = True
    b_mask[:, -bw:] = True

    bg_gray = float(np.median(blur[b_mask]))
    bg_sat = float(np.median(sat[b_mask]))

    otsu_t1, th_bin = cv2.threshold(warm_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_bin[(cool_diff >= 3.0) & (sat < dark_cool_sat_cutoff)] = 0
    border_white_ratio = float(np.mean(th_bin[b_mask] > 0))

    is_light_bg = (bg_gray > 75 and bg_sat < 25) or border_white_ratio > 0.45
    if is_light_bg:
        diff_g = np.clip(bg_gray - blur.astype(np.float32), 0, 255)
        diff_s = np.clip(sat.astype(np.float32) - bg_sat, 0, 255)
        comb = np.clip(diff_g * 1.2 + diff_s * 0.8 - cool_diff * 2.5, 0, 255).astype(np.uint8)
        otsu_thresh_val, raw_mask = cv2.threshold(comb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        raw_mask[(cool_diff >= 2.0) & (sat < 55)] = 0
    else:
        otsu_thresh_val = otsu_t1
        raw_mask = th_bin

    fg = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, MORPH_ELLIPSE_3X3, iterations=1)
    if fill_holes:
        fg = ndi.binary_fill_holes(fg > 0).astype(np.uint8) * 255

    return {
        "fg": fg,
        "gray": gray,
        "blur": blur,
        "otsu_thresh_val": otsu_thresh_val,
        "bg_gray": bg_gray,
        "bg_sat": bg_sat,
        "is_light_bg": is_light_bg,
    }


def auto_resize_and_crop(img, target_size=STANDARD_FRAME_SIZE):
    """Standardize resolution and crop a centered square ROI around warm foreground objects."""
    if img is None or img.size == 0:
        raise ValueError("Invalid input image (None or empty)")

    th, tw = target_size
    h, w = img.shape[:2]
    if h == th and w == tw:
        return img

    work_img = img
    max_dim = max(h, w)
    if max_dim > 720:
        scale = 640.0 / float(max_dim)
        nh = max(th, int(round(h * scale)))
        nw = max(tw, int(round(w * scale)))
        work_img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        h, w = work_img.shape[:2]

    fg = extract_warmth_fg_mask(work_img, dark_cool_sat_cutoff=70, fill_holes=False)["fg"]

    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_side_base = min(h, w)
    min_valid_area = max(60, int(h * w * 0.0015))
    valid_boxes = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_valid_area:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        if (x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1) and a < (h * w * 0.02):
            continue
        valid_boxes.append((x, y, x + cw, y + ch))

    if valid_boxes:
        x1 = min(b[0] for b in valid_boxes)
        y1 = min(b[1] for b in valid_boxes)
        x2 = max(b[2] for b in valid_boxes)
        y2 = max(b[3] for b in valid_boxes)
        roi_w = x2 - x1
        roi_h = y2 - y1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        roi_span = max(roi_w, roi_h)

        if roi_span < min_side_base * 0.15:
            crop_side = int(np.clip(roi_span * 5.0, min_side_base * 0.35, min_side_base))
        else:
            crop_side = min_side_base
    else:
        cx, cy = w // 2, h // 2
        crop_side = min_side_base

    half = crop_side // 2
    cx = int(np.clip(cx, half, w - (crop_side - half)))
    cy = int(np.clip(cy, half, h - (crop_side - half)))

    sx1 = max(0, cx - half)
    sy1 = max(0, cy - half)
    sx2 = min(w, sx1 + crop_side)
    sy2 = min(h, sy1 + crop_side)

    cropped = work_img[sy1:sy2, sx1:sx2]
    interp = cv2.INTER_AREA if (cropped.shape[0] > th or cropped.shape[1] > tw) else cv2.INTER_CUBIC
    return cv2.resize(cropped, (tw, th), interpolation=interp)


def _split_component_watershed(comp_mask, cl_img, blur, bbox, area, min_area, w_img, h_img):
    """Split touching kernels inside a connected component via CLAHE + Distance Watershed."""
    bx, by, bw_c, bh_c = bbox
    x0 = max(0, bx - 4)
    y0 = max(0, by - 4)
    x1 = min(w_img, bx + bw_c + 4)
    y1 = min(h_img, by + bh_c + 4)

    sub_mask = comp_mask[y0:y1, x0:x1]
    sub_cl = cl_img[y0:y1, x0:x1]
    sub_blur = blur[y0:y1, x0:x1]

    ad = cv2.adaptiveThreshold(
        sub_cl, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    eroded = cv2.erode(sub_mask, MORPH_ELLIPSE_3X3, iterations=1)
    core = cv2.bitwise_and(eroded, ad)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, MORPH_ELLIPSE_3X3, iterations=1)

    dist_geom = cv2.distanceTransform(sub_mask, cv2.DIST_L2, 5)
    dist_core = cv2.distanceTransform(core, cv2.DIST_L2, 5)

    g_vals = sub_blur[sub_mask > 0]
    p10 = np.percentile(g_vals, 10) if len(g_vals) > 0 else 0
    p90 = np.percentile(g_vals, 90) if len(g_vals) > 0 else 255
    norm_int = np.clip((sub_blur.astype(np.float32) - p10) / max(1.0, p90 - p10), 0, 1) * 6.0

    comb_score = np.minimum(dist_geom, 9.0) * 0.30 + dist_core * 0.55 + norm_int * 0.15
    comb_score = cv2.GaussianBlur(comb_score, (5, 5), 0)
    comb_score[sub_mask == 0] = 0

    min_d = int(np.clip(float(dist_core.max()) * 0.82, 11, 15))
    thr_abs = max(1.6, float(comb_score.max()) * 0.24)

    coords = peak_local_max(
        comb_score,
        min_distance=min_d,
        threshold_abs=thr_abs,
        labels=(sub_mask > 0),
    )
    if len(coords) <= 1:
        return None

    markers = np.zeros(sub_mask.shape, dtype=np.int32)
    for idx_pt, pt in enumerate(coords, start=1):
        markers[pt[0], pt[1]] = idx_pt
    markers = cv2.dilate(markers.astype(np.uint8), MORPH_ELLIPSE_3X3, iterations=1).astype(np.int32)

    ws = watershed(-comb_score, markers, mask=(sub_mask > 0))

    min_frag = max(min_area, int(area / (len(coords) * 2.2)))
    unique_ws = [u for u in np.unique(ws) if u > 0]
    for u in unique_ws:
        u_mask = ws == u
        if np.sum(u_mask) < min_frag and len(unique_ws) > 1:
            dil = cv2.dilate(u_mask.astype(np.uint8), MORPH_ELLIPSE_3X3, iterations=2) > 0
            neigh = ws[dil & (ws > 0) & (ws != u)]
            if len(neigh) > 0:
                vals, counts = np.unique(neigh, return_counts=True)
                ws[u_mask] = vals[np.argmax(counts)]

    return y0, y1, x0, x1, ws


def segment_kernels_otsu(img, min_area=90, max_area=50000, margin=4):
    """Segment individual corn kernels using warmth-guided Otsu + convexity-gated Watershed."""
    h_img, w_img = img.shape[:2]
    fg_info = extract_warmth_fg_mask(img, dark_cool_sat_cutoff=65, fill_holes=True)
    fg = fg_info["fg"]
    blur = fg_info["blur"]
    otsu_thresh_val = fg_info["otsu_thresh_val"]
    bg_gray = fg_info["bg_gray"]
    bg_sat = fg_info["bg_sat"]
    is_light_bg = fg_info["is_light_bg"]

    num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(fg)
    valid_cids = [
        cid
        for cid in range(1, num_cc)
        if min_area <= int(cc_stats[cid, cv2.CC_STAT_AREA]) <= max_area
    ]
    final_labels = np.zeros((h_img, w_img), dtype=np.int32)
    cur_label = 0

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    cl_img = clahe.apply(blur)

    for cid in valid_cids:
        area = int(cc_stats[cid, cv2.CC_STAT_AREA])
        comp_mask = (cc_labels == cid).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        c = max(cnts, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = float(area) / hull_area if hull_area > 0 else 1.0
        bx, by, bw_c, bh_c = cv2.boundingRect(c)
        aspect = max(bw_c, bh_c) / max(1.0, min(bw_c, bh_c))

        is_single = (area < 780 and solidity > 0.85 and aspect < 1.70) or (
            len(valid_cids) == 1 and area < 24000 and solidity > 0.88 and aspect < 1.65
        )
        if is_single:
            cur_label += 1
            final_labels[comp_mask > 0] = cur_label
            continue

        split_res = _split_component_watershed(
            comp_mask, cl_img, blur, (bx, by, bw_c, bh_c), area, min_area, w_img, h_img
        )
        if split_res is None:
            cur_label += 1
            final_labels[comp_mask > 0] = cur_label
            continue

        y0, y1, x0, x1, ws = split_res
        for u in np.unique(ws):
            if u <= 0:
                continue
            u_mask = ws == u
            if np.sum(u_mask) < min_area:
                continue
            cur_label += 1
            final_labels[y0:y1, x0:x1][u_mask] = cur_label

    valid_kernels = []
    kid = 0
    for lid in range(1, cur_label + 1):
        k_mask = (final_labels == lid).astype(np.uint8) * 255
        area = int(np.sum(k_mask > 0))
        if area < min_area or area > max_area:
            continue

        cnts, _ = cv2.findContours(k_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if (x <= 1 or y <= 1 or x + w >= w_img - 1 or y + h >= h_img - 1) and area < 150:
            continue

        x1 = max(x - margin, 0)
        y1 = max(y - margin, 0)
        x2 = min(x + w + margin, w_img)
        y2 = min(y + h + margin, h_img)
        crop_img = img[y1:y2, x1:x2]

        gray_k = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        hsv_k = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        m_val = float(gray_k.mean())
        s_val = float(gray_k.std())
        sat_val = float(hsv_k[:, :, 1].mean())

        if (
            (m_val > 205 and sat_val < 38)
            or (m_val < 30 and s_val < 15 and sat_val < 40)
            or (s_val < 7.2)
            or (area < 95)
        ):
            continue

        hull_area_k = cv2.contourArea(cv2.convexHull(c))
        solidity_k = round(float(cv2.contourArea(c)) / hull_area_k, 3) if hull_area_k > 0 else 1.0
        bw_k, bh_k = x2 - x1, y2 - y1
        aspect_k = round(max(bw_k, bh_k) / max(1.0, float(min(bw_k, bh_k))), 2)

        kid += 1
        valid_kernels.append(
            {
                "kernel_id": kid,
                "bbox": (x1, y1, bw_k, bh_k),
                "area_piksel": area,
                "solidity": solidity_k,
                "aspect_ratio": aspect_k,
                "crop": crop_img,
                "mask": k_mask[y1:y2, x1:x2],
                "contour": c,
            }
        )

    if len(valid_kernels) > 1:
        warm_k = [
            k
            for k in valid_kernels
            if (float(k["crop"][:, :, 2].mean()) - float(k["crop"][:, :, 0].mean())) >= 3.5
        ]
        if warm_k:
            valid_kernels = [
                k
                for k in valid_kernels
                if (float(k["crop"][:, :, 2].mean()) - float(k["crop"][:, :, 0].mean())) >= 2.0
                or k["area_piksel"] >= 450
            ]
            for idx_k, k in enumerate(valid_kernels, start=1):
                k["kernel_id"] = idx_k

    return {
        "otsu_mask": (final_labels > 0).astype(np.uint8) * 255,
        "kernels": valid_kernels,
        "seg_stats": {
            "otsu_threshold": round(float(otsu_thresh_val), 1),
            "bg_gray": round(bg_gray, 1),
            "bg_sat": round(bg_sat, 1),
            "bg_mode": (
                "Light Background (Polarity-Inverted)"
                if is_light_bg
                else "Dark Background (Standard Otsu)"
            ),
            "raw_components": len(valid_cids),
            "final_segments": len(valid_kernels),
        },
    }


def segment_fungus_kmeans(kernel_crop, k=3, mask=None):
    """Isolate fungal lesion areas on an individual kernel crop using HSV K-Means."""
    if kernel_crop is None or kernel_crop.size == 0:
        return None

    hsv_crop = cv2.cvtColor(kernel_crop, cv2.COLOR_BGR2HSV)
    gray_crop = cv2.cvtColor(kernel_crop, cv2.COLOR_BGR2GRAY)

    if mask is not None and mask.shape[:2] == kernel_crop.shape[:2] and np.sum(mask > 0) >= 20:
        b_ch, _, r_ch = cv2.split(kernel_crop)
        fg_mask = (
            (mask > 0)
            & (gray_crop > 15)
            & ~((b_ch.astype(int) >= r_ch.astype(int) - 1) & (gray_crop < 140))
        )
    else:
        fg_mask = foreground_pixel_mask(kernel_crop, gray_crop, hsv_crop)

    fg_indices = np.where(fg_mask)
    if len(fg_indices[0]) < 10:
        return {
            "clustered_bgr": kernel_crop,
            "mold_mask": np.zeros(kernel_crop.shape[:2], dtype=np.uint8),
            "mold_ratio": 0.0,
            "cluster_centers": [],
        }

    pixels = np.float32(hsv_crop[fg_indices])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)

    clustered_hsv = np.zeros_like(hsv_crop)
    clustered_labels = np.full(kernel_crop.shape[:2], -1, dtype=np.int32)

    flat_labels = labels.flatten()
    clustered_hsv[fg_indices] = centers[flat_labels]
    clustered_labels[fg_indices] = flat_labels

    clustered_bgr = cv2.cvtColor(clustered_hsv, cv2.COLOR_HSV2BGR)

    mold_cluster_id = None
    min_health_score = 999999.0
    for cid in range(k):
        _, s_c, v_c = centers[cid]
        health_score = float(s_c) * 1.5 + float(v_c) * 0.8
        if health_score < min_health_score:
            min_health_score = health_score
            mold_cluster_id = cid

    mold_mask = np.zeros(kernel_crop.shape[:2], dtype=np.uint8)
    if mold_cluster_id is not None:
        mold_mask[clustered_labels == mold_cluster_id] = 255

    total_fg_pixels = len(fg_indices[0])
    mold_pixels = int(np.sum(mold_mask > 0))
    mold_ratio = float(mold_pixels) / float(total_fg_pixels) if total_fg_pixels > 0 else 0.0

    return {
        "clustered_bgr": clustered_bgr,
        "mold_mask": mold_mask,
        "mold_ratio": mold_ratio,
        "cluster_centers": centers.tolist(),
    }
