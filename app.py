import os
import base64
import cv2
import numpy as np
from dotenv import load_dotenv

# Load .env before importing modules that read os.environ (e.g., ADMIN_PASSWORD).
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from typing import Optional
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Header, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from src.pipeline import CornDiseasePipeline
from src import crowdsource

app = FastAPI(title="Corn Kernel Mold Detector")

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

pipeline = CornDiseasePipeline()
admin_auth = crowdsource.AdminAuth()
retrain_manager = crowdsource.RetrainManager(pipeline)
crowdsource.migrate_legacy_crowd_files()
# Pre-index existing dataset hashes so re-uploaded training photos skip staging/feedback.
crowdsource.sync_raw_hashes()

def mat_to_base64(mat, is_rgb=False):
    """Encodes OpenCV image (BGR or Gray) into base64 JPEG string."""
    if mat is None or mat.size == 0:
        return None
    if is_rgb:
        bgr = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    else:
        bgr = mat
    _, buffer = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

def _decode_image_bytes(raw_bytes, error_detail="Failed to decode image"):
    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail=error_detail)
    return img


def decode_base64_image(b64_string):
    """Strip any data URL prefix and decode a base64 image string into BGR OpenCV format."""
    payload = b64_string.split(",", 1)[1] if "," in b64_string else b64_string
    try:
        img_bytes = base64.b64decode(payload)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")
    return _decode_image_bytes(img_bytes, error_detail="Failed to decode the image frame")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/detect/upload")
async def detect_upload(file: UploadFile = File(...)):
    contents = await file.read()
    img = _decode_image_bytes(contents, error_detail="Invalid or corrupted image file format")
    return run_detection(img, staging_source="upload")


def run_detection(img, staging_source=None):
    """Run the full pipeline and format JSON response with base64 images.

    When `staging_source` is set ("upload" or "webcam"), valid corn detections
    are queued to `data/staging/`.
    """
    try:
        results = pipeline.process_image(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    staging_info = None
    if staging_source:
        staging_info = crowdsource.save_to_staging(img, results, staging_source)

    images_b64 = {
        "raw_input": mat_to_base64(results["images"].get("raw_input", results["images"]["original"])),
        "original": mat_to_base64(results["images"]["original"]),
        "gray": mat_to_base64(results["images"]["gray"]),
        "blur": mat_to_base64(results["images"]["blur"]),
        "resized_blur": mat_to_base64(results["images"].get("resized_blur", results["images"]["blur"])),
        "otsu_mask": mat_to_base64(results["images"]["otsu_mask"]),
        "seg_overlay": mat_to_base64(results["images"].get("seg_overlay", results["images"]["annotated"])),
        "annotated": mat_to_base64(results["images"]["annotated"])
    }

    kernels_formatted = []
    for k in results["kernels"]:
        k_dict = {
            "kernel_id": k["kernel_id"],
            "bbox": k["bbox"],
            "area_piksel": k.get("area_piksel"),
            "solidity": k.get("solidity"),
            "aspect_ratio": k.get("aspect_ratio"),
            "prediction": k["prediction"],
            "is_moldy": k["is_moldy"],
            "is_corn": k.get("is_corn"),
            "confidence": k["confidence"],
            "decision_score": k.get("decision_score"),
            "mold_ratio": k["mold_ratio"],
            "cluster_centers": k.get("cluster_centers", []),
            "probabilities": k["probabilities"],
            "features": k["features"],
            "scaled_features": k.get("scaled_features"),
            "outlier_distance": k.get("outlier_distance"),
            "outlier_threshold": k.get("outlier_threshold"),
            "outlier_nearest_class": k.get("outlier_nearest_class"),
            "crop_b64": mat_to_base64(k["crop_img"]),
            "kmeans_b64": mat_to_base64(k["kmeans_img"]) if k["kmeans_img"] is not None else None,
            "mold_mask_b64": mat_to_base64(k["mold_mask"]) if k["mold_mask"] is not None else None
        }
        kernels_formatted.append(k_dict)

    return {
        "status": "success",
        "summary": results["summary"],
        "stage_meta": results.get("stage_meta", {}),
        "images": images_b64,
        "kernels": kernels_formatted,
        "staging": staging_info
    }


# ================================================================
# Realtime Webcam Endpoints
# ================================================================

class FrameRequest(BaseModel):
    image: str  # Data URL from canvas.toDataURL(), e.g. "data:image/jpeg;base64,..."


def run_detection_lightweight(img):
    """Fast detection path for live webcam frames (skips K-Means and stage previews)."""
    try:
        results = pipeline.process_image(img, include_kmeans=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    kernels_light = [
        {
            "kernel_id": k["kernel_id"],
            "bbox": k["bbox"],
            "prediction": k["prediction"],
            "confidence": k["confidence"],
        }
        for k in results["kernels"]
    ]

    return {
        "status": "success",
        "summary": results["summary"],
        "annotated": mat_to_base64(results["images"]["annotated"]),
        "kernels": kernels_light
    }


@app.post("/api/detect/frame")
def detect_frame(req: FrameRequest):
    """Process a single live webcam frame. Uses sync `def` so FastAPI runs CPU work in its thread pool."""
    img = decode_base64_image(req.image)
    return run_detection_lightweight(img)


@app.post("/api/detect/frame/full")
def detect_frame_full(req: FrameRequest):
    """Run full pipeline analysis on the final webcam frame after stopping the camera."""
    img = decode_base64_image(req.image)
    return run_detection(img, staging_source="webcam")


# ================================================================
# Crowdsourcing Dataset & Continuous Training
# ================================================================

class FeedbackRequest(BaseModel):
    id: str
    is_correct: bool
    suggested_label: Optional[str] = None


@app.post("/api/staging/feedback")
def staging_feedback(req: FeedbackRequest):
    """Optional anonymous user feedback on a queued staging detection."""
    if not crowdsource.submit_user_feedback(req.id, req.is_correct, req.suggested_label):
        raise HTTPException(status_code=404, detail="Staging item not found or invalid label")
    return {"status": "success"}


# ---------- Admin Authentication ----------
class LoginRequest(BaseModel):
    password: str


def require_admin(token: Optional[str]):
    if not admin_auth.is_valid(token):
        raise HTTPException(status_code=401, detail="Admin login required")


@app.post("/api/admin/login")
def admin_login(req: LoginRequest):
    if not admin_auth.enabled:
        raise HTTPException(
            status_code=503,
            detail="Admin login is disabled: ADMIN_PASSWORD is not configured in the server's .env file",
        )
    token = admin_auth.login(req.password)
    if not token:
        raise HTTPException(status_code=401, detail="Wrong admin password")
    return {"status": "success", "token": token}


@app.post("/api/admin/logout")
def admin_logout(x_admin_token: Optional[str] = Header(None)):
    admin_auth.logout(x_admin_token)
    return {"status": "success"}


@app.get("/api/admin/check")
def admin_check(x_admin_token: Optional[str] = Header(None)):
    return {"valid": admin_auth.is_valid(x_admin_token)}


# ---------- Staging Curation ----------
@app.get("/api/admin/stats")
def admin_stats(x_admin_token: Optional[str] = Header(None)):
    require_admin(x_admin_token)
    return crowdsource.get_dataset_stats()


@app.get("/api/admin/staging")
def admin_staging(x_admin_token: Optional[str] = Header(None)):
    require_admin(x_admin_token)
    return {"items": crowdsource.list_staging()}


@app.get("/api/admin/staging/image/{category}/{item_id}")
def admin_staging_image(
    category: str,
    item_id: str,
    kind: str = "original",
    token: Optional[str] = Query(None),
    x_admin_token: Optional[str] = Header(None),
):
    # Accept token via query string as well since <img> tags cannot send custom headers.
    require_admin(x_admin_token or token)
    path = crowdsource.staging_image_path(category, item_id, kind)
    if not path:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/jpeg")


class ApproveRequest(BaseModel):
    id: str
    label: str  # Final admin-verified class: "sehat" or "terkontaminasi"


@app.post("/api/admin/staging/approve")
def admin_approve(req: ApproveRequest, x_admin_token: Optional[str] = Header(None)):
    require_admin(x_admin_token)
    try:
        return {"status": "success", **crowdsource.approve_item(req.id, req.label)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


class RejectRequest(BaseModel):
    id: str


@app.post("/api/admin/staging/reject")
def admin_reject(req: RejectRequest, x_admin_token: Optional[str] = Header(None)):
    require_admin(x_admin_token)
    try:
        return {"status": "success", **crowdsource.reject_item(req.id)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------- Re-training ----------
@app.post("/api/admin/retrain")
def admin_retrain(x_admin_token: Optional[str] = Header(None)):
    """Start background re-training (progress polled via /api/admin/retrain/status)."""
    require_admin(x_admin_token)
    if not retrain_manager.start():
        raise HTTPException(status_code=409, detail="Re-training is already running")
    return {"status": "started"}


@app.get("/api/admin/retrain/status")
def admin_retrain_status(x_admin_token: Optional[str] = Header(None)):
    require_admin(x_admin_token)
    return retrain_manager.status()


@app.post("/api/admin/rollback")
def admin_rollback(x_admin_token: Optional[str] = Header(None)):
    require_admin(x_admin_token)
    try:
        return {"status": "success", **retrain_manager.rollback()}
    except (RuntimeError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)