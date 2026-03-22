============================================================
  Thing Identification System - Python Backend
  Platform : Windows
  Python   : 3.11
  Mode     : CPU only
============================================================

PROJECT STRUCTURE
-----------------
backend/
  main.py                 - FastAPI server (entry point)
  model.py                - PyTorch MobileNetV2 model
  trainer.py              - Training and retraining logic (auto-scaling)
  identifier.py           - Image identification (inference)
  requirements.txt        - Common Python dependencies
  requirements_torch.txt  - PyTorch CPU dependencies
  start.bat               - Start the server (double-click)
  README.txt              - This file

  data/                   - PUT YOUR TRAINING PHOTOS HERE
    thing_A/              - Photos of Thing A
    thing_B/              - Photos of Thing B
    thing_C/              - Add as many classes as needed

  saved_model/            - Auto-generated after training
    model.pth             - Trained model weights
    classes.txt           - List of class names


SETUP - STEP BY STEP
---------------------

STEP 1 - Install Python 3.11
  Download from: https://www.python.org/downloads/release/python-3119/
  File: python-3.11.9-amd64.exe (Windows 64-bit)
  During install: check "Add Python 3.11 to PATH"

STEP 2 - Prepare your photos
  Place photos per thing inside the data/ folder.
  Each subfolder name becomes the class label.
  Example:
    data/
      apple/
        photo1.jpg
        photo2.jpg
        ...
      banana/
        photo1.jpg
        ...
  Supported formats: .jpg .jpeg .png .bmp
  Minimum recommended: 20 photos per class
  Best results: 100+ photos per class

STEP 3 - Download packages (on internet machine)
  Open Command Prompt in the backend/ folder and run:

  Command 1 - Download common packages:
    pip download -r requirements.txt -d ./offline_packages

  Command 2 - Download PyTorch CPU:
    pip download -r requirements_torch.txt --index-url https://download.pytorch.org/whl/cpu -d ./offline_packages

STEP 4 - Copy to offline machine
  Copy these items to your offline machine:
    backend/               (all project files)
    offline_packages/      (all downloaded .whl files)

STEP 5 - Install packages (on offline machine)
  Open Command Prompt in the backend/ folder and run:

    pip install --no-index --find-links=./offline_packages -r requirements.txt
    pip install --no-index --find-links=./offline_packages -r requirements_torch.txt

STEP 6 - Start the server
  Double-click start.bat
  OR run in Command Prompt:
    python main.py

  Server runs at: http://127.0.0.1:5000


TRAINING - AUTO SCALING
------------------------
trainer.py automatically detects your dataset size and
adjusts all training settings accordingly. You do NOT need
to change any settings manually.

  Scale   | Avg photos/class | Epochs | Batch | Augmentation
  --------|------------------|--------|-------|-------------
  SMALL   | less than 50     |   30   |   8   | Heavy
  MEDIUM  | 50 to 500        |   20   |  32   | Moderate
  LARGE   | 500+             |   10   |  64   | Light

  Notes:
  - SMALL  : Heavy augmentation compensates for fewer photos
  - MEDIUM : Balanced settings for good accuracy
  - LARGE  : Less augmentation needed, data speaks for itself
  - All scales use 80% train / 20% validation split automatically
  - Best model is saved automatically during training
  - Each epoch shows time taken so you can estimate total time

  Example output (LARGE scale):
    [Trainer] Dataset scale : LARGE
    [Trainer] Epochs        : 10
    [Trainer] Batch size    : 64
    [Trainer] Train set     : 8000 images
    [Trainer] Val set       : 2000 images
    [Trainer] Epoch [  1/10] Loss: 0.82 | Train: 88.5% | Val: 85.2% | Time: 45.3s
    [Trainer] Epoch [  2/10] Loss: 0.61 | Train: 91.2% | Val: 89.0% | Time: 44.8s
    ...
    [Trainer] -- Training Complete! Best Accuracy: 95.5% --

  Estimated training time on CPU:
    SMALL  (20 photos/class)   : 1 to 3 minutes
    MEDIUM (100 photos/class)  : 5 to 15 minutes
    LARGE  (1000 photos/class) : 30 to 90 minutes


API ENDPOINTS
-------------

  GET  /           Health check - confirm server is running
  GET  /status     Get current training status
  GET  /classes    Get list of all trained thing classes
  POST /train      Start training with photos in data/ folder
  POST /identify   Identify a thing from an uploaded image
  POST /correct    Correct a wrong result and retrain


USAGE FLOW
----------

1. Add photos to data/<thing_name>/ folders

2. Train the model:
   POST http://127.0.0.1:5000/train

3. Check training status:
   GET  http://127.0.0.1:5000/status
   Response: { "status": "done", "accuracy": 95.0 }

4. Identify an image:
   POST http://127.0.0.1:5000/identify
   Body: multipart/form-data, field "file" = image file
   Response:
     {
       "success"    : true,
       "predicted"  : "apple",
       "confidence" : 94.5,
       "all_scores" : { "apple": 94.5, "banana": 5.5 }
     }

5. Correct a wrong result:
   POST http://127.0.0.1:5000/correct
   Body: multipart/form-data
     file          = the misidentified image
     correct_label = "banana"
   Response:
     {
       "message"  : "Image saved under 'banana' and retraining started!",
       "saved_to" : "data/banana/corrected_1.jpg"
     }

6. Get all known classes:
   GET  http://127.0.0.1:5000/classes
   Response: { "classes": ["apple", "banana"], "count": 2 }


QT C++ INTEGRATION EXAMPLE
---------------------------

  // Train
  QNetworkRequest req(QUrl("http://127.0.0.1:5000/train"));
  manager->post(req, QByteArray());

  // Identify
  QHttpMultiPart *mp = new QHttpMultiPart(QHttpMultiPart::FormDataType);
  QHttpPart imgPart;
  imgPart.setHeader(QNetworkRequest::ContentDispositionHeader,
                    "form-data; name=\"file\"; filename=\"image.jpg\"");
  imgPart.setBodyDevice(imageFile);
  mp->append(imgPart);
  manager->post(QNetworkRequest(QUrl("http://127.0.0.1:5000/identify")), mp);

  // Correct
  // Same as identify but also add form field: correct_label = "thing_name"


TRAINING STATUS VALUES
----------------------
  idle      - No training started yet
  training  - Training is in progress
  done      - Training completed successfully
  error     - Training failed (check server console for details)


NOTES
-----
- All processing is fully offline, no internet required after setup
- trainer.py handles 20, 100, 1000+ photos per class automatically
- Model is saved automatically after each training session
- Each correction adds the image to the dataset and retrains automatically
- MobileNetV2 with transfer learning works well with any dataset size
- Poll GET /status every few seconds to check training progress from Qt
- Training time depends on number of classes, photos, and CPU speed

============================================================
