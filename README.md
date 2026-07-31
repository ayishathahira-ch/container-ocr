# Container OCR Pipeline

Automatically counts shipping containers passing through a gate camera and extracts each container's ID number from the video.

---

## What it does

- Counts the total number of containers in a video
- Reads and lists each container's ID number (e.g. `TCLU 655534 4561`)
- Saves an annotated image for each container showing the detection and OCR result
- Organises output into separate folders per video

---

## Pipeline Flow

```
Input Video
    ↓
Step 1: extract_frames()          → extracts frames at chosen FPS
    ↓
Step 2: push_to_frame_queue()     → pushes frames into Queue 1
    ↓
Step 3: pick_frame_by_timestamp() → sorts frames chronologically
    ↓
Step 4: run_yolo_detection()      → YOLO detects container regions
                                    Centroid + IoU tracker assigns unique IDs
                                    Crossing line counts each container once
    ↓
Step 5: push_to_detection_queue() → pushes detections into Queue 2
    ↓
Step 6: extract_text_ocr()        → EasyOCR reads container number
    ↓
Step 7: save_output()             → saves annotated image per container
    ↓
Step 8: main()                    → prints final count + container numbers
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Tracking | Custom Centroid + IoU Tracker |
| OCR | EasyOCR |
| Video Processing | OpenCV |
| Model Training | Kaggle GPU (Tesla T4) |
| Language | Python 3.10 |
| Platform | Ubuntu 22.04 |

---

## Dataset

- **Source:** Roboflow Universe — container by LPR
- **Images:** 6,289
- **Class:** `con-region` (container number plate region)
- **Train / Val / Test split:** 87% / 8% / 5%
- **License:** CC BY 4.0

---

## Model Training

Trained on Kaggle with free GPU (Tesla T4):

```
Model      : YOLOv8n
Epochs     : 100
Image size : 640 × 640
Batch size : 16
mAP50      : 99.5%
mAP50-95   : 83.2%
Precision  : 99.6%
Recall     : 99.8%
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/container-ocr.git
cd container-ocr
```

### 2. Create virtual environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add your model and video

```bash
cp /path/to/best.pt .
cp /path/to/your_video.mp4 videos/
```

---

## Usage

```bash
python cont_ocr_pipeline.py \
  --video videos/input.mp4 \
  --model best.pt \
  --output results \
  --fps 5.0 \
  --conf 0.4 \
  --gap 8
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--video` | required | Path to input video |
| `--model` | required | Path to best.pt |
| `--output` | results | Output directory |
| `--fps` | 3.0 | Frames to sample per second |
| `--conf` | 0.5 | YOLO confidence threshold |
| `--gpu` | False | Use GPU for OCR |
| `--lang` | en | OCR language |
| `--gap` | 8 | Min frames between new containers |

---

## Output

### Terminal

```
════════════════════════════════════════════════════════════
  CONTAINER OCR — RESULTS
════════════════════════════════════════════════════════════
  Total containers detected : 58
  Unique container numbers  : 42

  Container   1  →  TCLU6554561
  Container   2  →  MSKU1234567
  Container   3  →  UNKNOWN
  ...

  Output saved to : results/annotated/video_name/
  Total time      : 135.2s
════════════════════════════════════════════════════════════
```

### Files

```
results/
├── frames/
│   └── video_name/        ← extracted frames
└── annotated/
    └── video_name/        ← one composite image per container
                             (full frame with bbox + OCR crop + text)
```

---

## Project Structure

```
container_ocr/
├── cont_ocr_pipeline.py   ← main pipeline script
├── best.pt                ← trained YOLO model (not in repo)
├── requirements.txt       ← Python dependencies
├── README.md              ← this file
├── .gitignore
├── videos/                ← input videos (not in repo)
└── results/               ← output (not in repo)
```

---

## Tuning Tips

| Problem | Fix |
|---|---|
| Too many containers counted | Increase `--gap` to 10 or 12 |
| Too few containers counted | Decrease `--gap` to 6 |
| OCR returning UNKNOWN | Increase `--fps` to 5.0 |
| Detections missing | Decrease `--conf` to 0.3 |
| False detections | Increase `--conf` to 0.6 |

---

## License

MIT License
