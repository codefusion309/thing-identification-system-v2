============================================================
  Thing Identification System - Embedding Edition
  Platform : Windows
  Python   : 3.11
  Mode     : CPU only
  Version  : 2.0  (Embedding-based, no retraining on corrections)
============================================================


WHAT CHANGED FROM VERSION 1.0
-------------------------------
Version 1.0 used a trained softmax classifier (MobileNetV2 + custom head).
Every correction triggered a full retrain of all images — taking hours
when you have 1000+ classes.

Version 2.0 uses embedding-based nearest-neighbor classification:

  Feature            v1.0 (Softmax)         v2.0 (Embedding)
  -----------------  ---------------------  ----------------------
  Correction speed   Hours (full retrain)   ~50ms (instant)
  Add new class      Full retrain           Instant (just /correct)
  1000-class support Poor (few-shot softmax) Good (cosine similarity)
  Index build        20-30 epochs per train  One-time extraction
  Index rebuild      POST /train            POST /train (rarely needed)
  Saved files        model.pth, classes.txt  embeddings.pt, counts.json

The external API (endpoints, request/response shapes) is IDENTICAL.
Your Qt C++ client does NOT need to change.


PROJECT STRUCTURE
-----------------
backend/
  main.py                 - FastAPI server (entry point)
  model.py                - MobileNetV2 backbone + embedding extraction
  trainer.py              - One-time embedding index builder
  identifier.py           - Cosine similarity identification + instant correction
  requirements.txt        - Python dependencies
  requirements_torch.txt  - PyTorch CPU dependencies
  start.bat               - Start the server (double-click)
  README.txt              - This file

  data/                   - PUT YOUR TRAINING PHOTOS HERE
    thing_A/              - Photos of Thing A
    thing_B/              - Photos of Thing B
    (1000+ classes fine)  - Add as many as needed

  saved_model/            - Auto-generated after POST /train
    embeddings.pt         - Mean 1280-dim embedding per class
    counts.json           - Image count per class (for running mean)


SETUP - STEP BY STEP
---------------------

STEP 1 - Install Python 3.11
  Download: https://www.python.org/downloads/release/python-3119/
  File    : python-3.11.9-amd64.exe (Windows 64-bit)
  During install: CHECK "Add Python 3.11 to PATH"


STEP 2 - Download Python packages (on internet machine)

  Open Command Prompt in the backend/ folder and run:

  Command 1 - Common packages:
    pip download -r requirements.txt -d ./offline_packages

  Command 2 - PyTorch CPU:
    pip download -r requirements_torch.txt --index-url https://download.pytorch.org/whl/cpu -d ./offline_packages


STEP 3 - Download MobileNetV2 pretrained weights (on internet machine)

  This is the ONLY model file the system needs. It is ~14 MB and must be
  downloaded once before going offline. Run this command:

    python -c "from torchvision import models; models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT); print('MobileNetV2 weights cached successfully.')"

  The weights are saved automatically to:
    Windows : C:\Users\<YourName>\.cache\torch\hub\checkpoints\
    Filename: mobilenet_v2-b0353104.pth  (exact name may vary by torchvision version)

  To verify the file exists:
    dir C:\Users\<YourName>\.cache\torch\hub\checkpoints\


STEP 4 - Copy everything to the offline machine

  Copy ALL of the following to your offline machine:

    backend\               (all project files)
    offline_packages\      (all downloaded .whl files)
    C:\Users\<YourName>\.cache\torch\   (the entire torch cache folder)

  On the offline machine, place the torch cache at:
    C:\Users\<OfflineUserName>\.cache\torch\

  IMPORTANT: The username folder must match. If the offline machine has a
  different Windows username, copy the contents to the correct path manually:
    mkdir C:\Users\<OfflineUserName>\.cache\torch\hub\checkpoints\
    copy mobilenet_v2-b0353104.pth C:\Users\<OfflineUserName>\.cache\torch\hub\checkpoints\


STEP 5 - Install Python 3.11 on the offline machine
  Use the same python-3.11.9-amd64.exe installer.


STEP 6 - Install packages (on offline machine)
  Open Command Prompt in the backend/ folder and run:

    pip install --no-index --find-links=./offline_packages -r requirements.txt
    pip install --no-index --find-links=./offline_packages -r requirements_torch.txt


STEP 7 - Verify MobileNetV2 weights are accessible (on offline machine)
  Run this to confirm no internet is needed:

    python -c "from torchvision import models; m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT); print('OK - loaded from cache, no internet used.')"

  If this prints "OK" without downloading anything, you are fully offline-ready.


STEP 8 - Start the server
  Double-click start.bat
  OR run in Command Prompt:
    python main.py

  Server runs at: http://127.0.0.1:5000


PREPARING YOUR DATA
--------------------
Place photos per class inside the data/ folder.
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
    (add 1000+ classes — no limit)

Supported formats : .jpg  .jpeg  .png  .bmp
Recommended minimum: 10-20 photos per class
More is always better: 50-100 photos per class gives best results


BUILDING THE INDEX (POST /train)
----------------------------------
After placing photos, call POST /train once to build the embedding index.

What happens:
  1. MobileNetV2 processes every image (one forward pass each, no training).
  2. All image vectors for a class are averaged into one mean embedding.
  3. Results saved to saved_model/embeddings.pt and saved_model/counts.json.

Performance (CPU):
  Class count  x  Images/class  =  Total      Est. time
  -----------     -------------     --------   ----------
  10              25                250        ~30 seconds
  100             25                2,500      ~5 minutes
  1000            25                25,000     ~10-20 minutes
  1000            100               100,000    ~40-80 minutes

This is a ONE-TIME operation. Corrections after this are instant.
Only rebuild the index (POST /train again) if you add large batches
of new photos in bulk.


CORRECTIONS - INSTANT, NO RETRAINING
---------------------------------------
When the system identifies something incorrectly:

  POST /correct  with  file=<image>  and  correct_label=<class_name>

What happens internally:
  1. Image saved to data/<correct_label>/corrected_N.jpg
  2. Embedding extracted for the new image (~50ms)
  3. Running mean updated for that class:
       new_mean = (old_mean × n + new_embedding) / (n + 1)
  4. Updated embeddings.pt and counts.json saved to disk

Result: Instant. The next /identify call already benefits from the correction.
No polling /status needed. No waiting.

You can also add a BRAND-NEW class this way — if correct_label does not
exist yet, it is registered automatically with that single image.


CONFIDENCE SCORES
------------------
The system returns cosine similarity * 100 as the confidence score.

  Score range  Meaning
  -----------  --------------------------------------------------
  85 - 100     Very confident match
  70 - 85      Confident match
  50 - 70      Uncertain — image may be ambiguous
  Below 50     Poor match — image may not belong to any known class

Unlike softmax probabilities, cosine similarity scores do NOT sum to 100.
Each class gets an independent score. This is more informative —
a score of 45 for every class tells you the image is genuinely unfamiliar,
whereas softmax would still force one class to "win" at 100%.


API ENDPOINTS
-------------

  GET  /           Health check — confirm server is running
  GET  /status     Get index build status (idle / building / done / error)
  GET  /classes    List all classes + image counts per class
  POST /train      Build embedding index from data/ (run once, or after bulk adds)
  POST /identify   Identify a thing from an uploaded image
  POST /correct    Correct a wrong result (instant, no retraining)


USAGE FLOW
----------

1. Add photos to data/<class_name>/ folders.

2. Build the index (one time):
   POST http://127.0.0.1:5000/train

3. Poll until done:
   GET  http://127.0.0.1:5000/status
   Response: { "status": "done", "classes": 1000 }

4. Identify an image:
   POST http://127.0.0.1:5000/identify
   Body: multipart/form-data, field "file" = image file
   Response:
     {
       "success"    : true,
       "predicted"  : "apple",
       "confidence" : 87.3,
       "all_scores" : { "apple": 87.3, "banana": 61.2, ... }
     }

5. Correct a wrong result (instant):
   POST http://127.0.0.1:5000/correct
   Body: multipart/form-data
     file          = the misidentified image
     correct_label = "banana"
   Response:
     {
       "message"    : "Correction applied instantly for 'banana'.",
       "saved_to"   : "data/banana/corrected_1.jpg",
       "class_count": 26,
       "note"       : "No retraining needed. The model is already updated."
     }

6. Get all known classes:
   GET  http://127.0.0.1:5000/classes
   Response:
     {
       "classes"      : ["apple", "banana", ...],
       "count"        : 1000,
       "image_counts" : { "apple": 25, "banana": 26, ... }
     }


QT C++ INTEGRATION EXAMPLE
---------------------------
(No changes needed from v1.0 — the API is identical)

  // Build index (call once after setup)
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

  // Correct (instant — no need to poll /status afterward)
  // Same as identify but also add form field: correct_label = "banana"


INDEX BUILD STATUS VALUES
--------------------------
  idle      - No index built yet
  building  - Index build in progress (poll every few seconds)
  done      - Index ready, system is fully operational
  error     - Build failed (check server console for details)


NOTES
-----
- All processing is fully offline after initial setup.
- POST /correct response is immediate. No polling needed.
- New classes can be added at any time via POST /correct — no rebuild needed.
- POST /train only needs to be called again if you add photos in bulk
  (e.g. 50+ new images to an existing class, or 10+ new classes at once).
- The backbone (MobileNetV2) is never modified — it stays as the
  original ImageNet pretrained model forever.
- saved_model/embeddings.pt grows slightly with each new class but
  remains small: ~5 KB per class, so 1000 classes ≈ 5 MB total.

============================================================
