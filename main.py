"""
Thing Identification System - FastAPI Backend
Entry point: starts the API server on http://127.0.0.1:5000

Endpoints:
    GET  /          - Health check
    GET  /status    - Training status
    GET  /classes   - List of known thing classes
    POST /train     - Start training with photos in data/ folder
    POST /identify  - Identify a thing from an uploaded image
    POST /correct   - Correct a wrong result and retrain
"""

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os

from trainer import Trainer
from identifier import Identifier

# ── Configuration ────────────────────────────────────────────
DATA_DIR     = "data"
MODEL_PATH   = "saved_model/model.pth"
CLASSES_FILE = "saved_model/classes.txt"

# ── App & shared instances ───────────────────────────────────
app        = FastAPI(title="Thing Identification System")
trainer    = Trainer(DATA_DIR, MODEL_PATH, CLASSES_FILE)
identifier = Identifier(MODEL_PATH, CLASSES_FILE)

training_status = {
    "status":  "idle",
    "message": "No training started yet.",
    "accuracy": None
}


# ── Background task ──────────────────────────────────────────
def run_training():
    global training_status
    try:
        training_status = {
            "status":  "training",
            "message": "Training in progress...",
            "accuracy": None
        }
        accuracy = trainer.train()
        identifier.reload()
        training_status = {
            "status":  "done",
            "message": "Training completed successfully!",
            "accuracy": accuracy
        }
    except Exception as e:
        training_status = {
            "status":  "error",
            "message": str(e),
            "accuracy": None
        }


# ── Routes ───────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Thing Identification System is running!"}


@app.get("/status")
def get_status():
    """Return current training status."""
    return training_status


@app.get("/classes")
def get_classes():
    """Return list of all trained thing classes."""
    if not os.path.exists(CLASSES_FILE):
        return {"classes": [], "count": 0}
    with open(CLASSES_FILE, "r") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    return {"classes": classes, "count": len(classes)}


@app.post("/train")
def train(background_tasks: BackgroundTasks):
    """
    Start training using images inside the data/ folder.
    Training runs in the background — poll /status to check progress.
    """
    if training_status["status"] == "training":
        return JSONResponse(
            status_code=400,
            content={"message": "Training is already in progress!"}
        )

    # Validate data folder exists and has classes
    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
        return JSONResponse(
            status_code=400,
            content={"message": f"No data found in '{DATA_DIR}/' folder. Please add images first."}
        )

    background_tasks.add_task(run_training)
    return {"message": "Training started in background. Poll GET /status to check progress."}


@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    """
    Identify the thing in an uploaded image.

    Request : multipart/form-data with field 'file' (image)
    Response:
        {
            "success": true,
            "predicted": "thing_name",
            "confidence": 94.5,
            "all_scores": { "thing_A": 94.5, "thing_B": 5.5 }
        }
    """
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = identifier.identify(temp_path)
        return result

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/correct")
async def correct(
    background_tasks: BackgroundTasks,
    file: UploadFile  = File(...),
    correct_label: str = Form(...)
):
    """
    Correct a wrong identification result.
    Saves the image under the correct class and triggers retraining.

    Request : multipart/form-data
        file          - the misidentified image
        correct_label - the correct thing name (must match a folder in data/)
    """
    if not correct_label.strip():
        return JSONResponse(
            status_code=400,
            content={"message": "correct_label cannot be empty."}
        )

    # Save image into the correct class folder
    class_dir = os.path.join(DATA_DIR, correct_label.strip())
    os.makedirs(class_dir, exist_ok=True)

    existing   = len(os.listdir(class_dir))
    save_path  = os.path.join(class_dir, f"corrected_{existing + 1}.jpg")

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Trigger retraining in background
    background_tasks.add_task(run_training)

    return {
        "message":   f"Image saved under '{correct_label}' and retraining started!",
        "saved_to":  save_path,
        "note":      "Poll GET /status to check retraining progress."
    }


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("saved_model", exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Starting Thing Identification System...")
    print(f"Server: http://127.0.0.1:5000")
    uvicorn.run(app, host="127.0.0.1", port=5000)
