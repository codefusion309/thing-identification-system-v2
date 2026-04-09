"""
Thing Identification System - FastAPI Backend (Embedding Edition)
Entry point: starts the API server on http://127.0.0.1:5000

Key change from previous version:
  POST /correct  no longer triggers retraining.
                 It updates the class embedding instantly (~50ms).

Endpoints:
    GET  /          - Health check
    GET  /status    - Index build status
    GET  /classes   - List of known classes + image counts
    POST /train     - Build embedding index from data/ folder (one-time or rebuild)
    POST /identify  - Identify a thing from an uploaded image
    POST /correct   - Correct a wrong result (instant, no retraining)
"""

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import uuid
import os

from trainer import Trainer
from identifier import Identifier

# ── Configuration ─────────────────────────────────────────────
DATA_DIR        = "data"
EMBEDDINGS_PATH = "saved_model/embeddings.pt"
COUNTS_PATH     = "saved_model/counts.json"

# ── App & shared instances ────────────────────────────────────
app        = FastAPI(title="Thing Identification System")
trainer    = Trainer(DATA_DIR, EMBEDDINGS_PATH, COUNTS_PATH)
identifier = Identifier(EMBEDDINGS_PATH, COUNTS_PATH)

index_status = {
    "status":  "idle",
    "message": "No index built yet. Add photos to data/ then POST /train.",
    "classes": None
}


# ── Background task ───────────────────────────────────────────
def run_build_index():
    global index_status
    try:
        index_status = {
            "status":  "building",
            "message": "Building embedding index...",
            "classes": None
        }
        class_counts = trainer.build_index()
        identifier.reload()
        index_status = {
            "status":  "done",
            "message": "Embedding index built successfully!",
            "classes": len(class_counts)
        }
    except Exception as e:
        index_status = {
            "status":  "error",
            "message": str(e),
            "classes": None
        }


# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Thing Identification System is running! (Embedding Edition)"}


@app.get("/status")
def get_status():
    """Return current index build status."""
    return index_status


@app.get("/classes")
def get_classes():
    """Return all known class names, total count, and per-class image counts."""
    if not identifier.class_counts:
        return {"classes": [], "count": 0, "image_counts": {}}
    return {
        "classes":      sorted(identifier.class_counts.keys()),
        "count":        len(identifier.class_counts),
        "image_counts": identifier.class_counts
    }


@app.post("/train")
def train(background_tasks: BackgroundTasks):
    """
    Build (or rebuild) the embedding index from images in data/.
    Runs in the background. Poll GET /status to check progress.

    When to call:
      - First time after adding all your photos.
      - If you add many new photos to existing classes in bulk.
      - Individual corrections do NOT require calling this — use POST /correct instead.
    """
    if index_status["status"] == "building":
        return JSONResponse(
            status_code=400,
            content={"message": "Index build already in progress."}
        )

    # Validate: at least one class subfolder exists
    if not os.path.exists(DATA_DIR) or not any(
        os.path.isdir(os.path.join(DATA_DIR, d))
        for d in os.listdir(DATA_DIR)
    ):
        return JSONResponse(
            status_code=400,
            content={"message": f"No class subfolders found in '{DATA_DIR}/'. Add photos first."}
        )

    background_tasks.add_task(run_build_index)
    return {"message": "Index build started in background. Poll GET /status to check progress."}


@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    """
    Identify the thing in an uploaded image.

    Request : multipart/form-data, field 'file' = image
    Response:
        {
            "success"    : true,
            "predicted"  : "thing_name",
            "confidence" : 87.3,
            "all_scores" : { "thing_A": 87.3, "thing_B": 61.2, ... }
        }

    Note: confidence is cosine similarity * 100.
          Scores above ~70 are generally reliable.
          Scores below ~50 indicate the image may not match any known class well.
    """
    # Use UUID prefix to prevent temp file collisions under concurrent requests
    temp_path = f"temp_{uuid.uuid4().hex}_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return identifier.identify(temp_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/correct")
async def correct(
    file: UploadFile   = File(...),
    correct_label: str = Form(...)
):
    """
    Correct a wrong identification result.

    Saves the image to data/<correct_label>/ and instantly updates
    the class embedding using the running mean formula.
    No retraining. Response is immediate (< 100ms).

    Also works for adding images to brand-new classes dynamically —
    no need to rebuild the entire index.

    Request : multipart/form-data
        file          - the misidentified image
        correct_label - the correct class name
    Response:
        {
            "message"    : "Correction applied instantly for 'banana'.",
            "saved_to"   : "data/banana/corrected_3.jpg",
            "class_count": 26,
            "note"       : "No retraining needed. The model is already updated."
        }
    """
    if not correct_label.strip():
        return JSONResponse(
            status_code=400,
            content={"message": "correct_label cannot be empty."}
        )

    correct_label = correct_label.strip()

    # Save image to the correct class folder
    class_dir = os.path.join(DATA_DIR, correct_label)
    os.makedirs(class_dir, exist_ok=True)

    # Count only image files to determine next index (ignore .DS_Store etc.)
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
    existing  = len([
        f for f in os.listdir(class_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    ])
    save_path = os.path.join(class_dir, f"corrected_{existing + 1}.jpg")

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Instantly update the embedding — no retraining
    success = identifier.add_correction(save_path, correct_label)

    if not success:
        return JSONResponse(
            status_code=500,
            content={
                "message": "Image saved but embedding update failed. Check server logs."
            }
        )

    return {
        "message":     f"Correction applied instantly for '{correct_label}'.",
        "saved_to":    save_path,
        "class_count": identifier.class_counts.get(correct_label, 1),
        "note":        "No retraining needed. The model is already updated."
    }


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("saved_model", exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Starting Thing Identification System (Embedding Edition)...")
    print("Server: http://127.0.0.1:5000")
    uvicorn.run(app, host="127.0.0.1", port=5000)
