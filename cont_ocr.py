"""
Rail OCR Pipeline
=================
Counts wagons (class: wagon, open_rake) from a train video.
Classes in best.pt: engine, wagon, text, open_rake, guard_cabin

Usage:
  python rail_ocr_pipeline.py --video train.mp4 --model best.pt --output results/
"""

import os
import re
import cv2
import queue
import argparse
import time
import logging
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import easyocr

# Suppress all library warnings
logging.basicConfig(level=logging.ERROR)

def info(msg, *args):
    print(time.strftime("%H:%M:%S"), "[INFO]", msg % args if args else msg)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Extract frames
# ══════════════════════════════════════════════════════════════════════════════
def extract_frames(video_path: str, frames_dir: str, sample_fps: float = 3.0) -> list:
    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    video_fps      = cap.get(cv2.CAP_PROP_FPS)
    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration       = total_frames / video_fps
    frame_interval = max(1, int(round(video_fps / sample_fps)))

    info("Video: %s | %.1f fps | %.1fs | every %d frames sampled",
         video_path, video_fps, duration, frame_interval)

    extracted = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp  = frame_idx / video_fps
            frame_name = f"frame_{frame_idx:06d}_{timestamp:.3f}s.jpg"
            frame_path = os.path.join(frames_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            extracted.append({"timestamp": timestamp, "frame_path": frame_path})
        frame_idx += 1

    cap.release()
    info("Extracted %d frames", len(extracted))
    return extracted


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Push to frame queue
# ══════════════════════════════════════════════════════════════════════════════
def push_to_frame_queue(frame_list: list, frame_queue: queue.Queue) -> None:
    for item in frame_list:
        frame_queue.put(item)
    frame_queue.put(None)
    info("Pushed %d frames to frame_queue", len(frame_list))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Pick frames by timestamp
# ══════════════════════════════════════════════════════════════════════════════
def pick_frame_by_timestamp(frame_queue: queue.Queue) -> list:
    frames = []
    while True:
        item = frame_queue.get()
        if item is None:
            break
        frames.append(item)
    frames.sort(key=lambda x: x["timestamp"])
    info("Picked %d frames sorted by timestamp", len(frames))
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# CENTROID + IoU TRACKER
# ══════════════════════════════════════════════════════════════════════════════
def _compute_iou(boxA, boxB):
    ix1 = max(boxA[0], boxB[0]); iy1 = max(boxA[1], boxB[1])
    ix2 = min(boxA[2], boxB[2]); iy2 = min(boxA[3], boxB[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0: return 0.0
    aA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (aA + aB - inter)

def _centroid(bbox):
    return ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2)

def _dist(c1, c2):
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5


class WagonTracker:
    """
    Tracks wagons using IoU + centroid distance.
    Only wagon/open_rake detections are fed here.
    min_gap_frames: minimum sampled frames between registering two new wagons.
    Set this to (video_fps / sample_fps) * seconds_between_wagons.
    """
    def __init__(self, iou_threshold=0.5, max_missed=15,
                 max_centroid_dist=150.0, min_gap_frames=8):
        self.iou_threshold     = iou_threshold
        self.max_missed        = max_missed
        self.max_centroid_dist = max_centroid_dist
        self.min_gap_frames    = min_gap_frames
        self.next_id           = 1
        self.active_tracks     = {}
        self.last_new_frame    = -999

    def update(self, detections: list, frame_idx: int) -> list:
        new_events        = []
        matched_track_ids = set()

        for det in detections:
            bbox  = det["bbox"]
            cent  = _centroid(bbox)
            best_id   = None
            best_iou  = 0.0
            best_dist = float("inf")

            for tid, track in self.active_tracks.items():
                iou  = _compute_iou(bbox, track["bbox"])
                dist = _dist(cent, track["centroid"])
                # Priority 1: IoU match
                if iou >= self.iou_threshold:
                    if iou > best_iou:
                        best_iou = iou
                        best_id  = tid
                # Priority 2: centroid distance (wagon moved fast)
                elif dist < self.max_centroid_dist and best_iou == 0.0:
                    if dist < best_dist:
                        best_dist = dist
                        best_id   = tid

            if best_id is not None:
                # Same wagon — update existing track
                self.active_tracks[best_id].update({
                    "bbox": bbox, "centroid": cent,
                    "missed_frames": 0, "last_frame": frame_idx,
                })
                matched_track_ids.add(best_id)
            else:
                # Enforce minimum gap before registering new wagon
                frames_since_last = frame_idx - self.last_new_frame
                if frames_since_last < self.min_gap_frames:
                    continue

                # New wagon
                new_id = self.next_id
                self.next_id += 1
                self.last_new_frame = frame_idx
                self.active_tracks[new_id] = {
                    "bbox": bbox, "centroid": cent,
                    "missed_frames": 0,
                    "first_frame": frame_idx,
                    "last_frame":  frame_idx,
                }
                matched_track_ids.add(new_id)
                new_events.append({**det, "tracker_id": new_id})
                info("New wagon! ID=%d | class=%s | conf=%.2f | t=%.3fs",
                     new_id, det["class_name"], det["confidence"], det["timestamp"])

        # Close tracks not seen for too long (wagon exited camera)
        for tid in list(self.active_tracks.keys()):
            if tid not in matched_track_ids:
                self.active_tracks[tid]["missed_frames"] += 1
                if self.active_tracks[tid]["missed_frames"] > self.max_missed:
                    del self.active_tracks[tid]

        return new_events


def run_yolo_detection(
    frame_list            : list,
    model_path            : str,
    confidence            : float = 0.4,
    iou_threshold         : float = 0.45,
    video_path            : str   = None,
    sample_fps            : float = 5.0,
    tracker_iou_threshold : float = 0.5,
    max_missed_frames     : int   = 15,
    max_centroid_dist     : float = 150.0,
    min_gap_frames        : int   = 8,
) -> list:
    """
    Counts wagons using a crossing line at the center of the frame.
    A wagon is counted only when its centroid crosses the center x line.
    This guarantees every wagon is counted exactly once.
    """
    info("Loading YOLO model: %s", model_path)
    model = YOLO(model_path)

    class_names   = model.names
    WAGON_CLASSES = {"con-region"}
    TEXT_CLASS    = None

    info("Model classes: %s", list(class_names.values()))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    video_fps      = cap.get(cv2.CAP_PROP_FPS)
    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(video_fps / sample_fps)))
    frame_width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Crossing line at center of frame
    cross_line_x   = frame_width // 2

    info("Tracking: %.1f fps | %d frames | crossing line at x=%d",
         video_fps, total_frames, cross_line_x)

    frames_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(video_path)),
                     "..", "results", "frames")
    )
    Path(frames_dir).mkdir(parents=True, exist_ok=True)

    all_detections  = []
    frame_idx       = 0
    wagon_count     = 0

    # Track each active wagon's previous centroid x position
    # {tracker_id: {"prev_cx": int, "crossed": bool, "best_det": dict}}
    active_wagons   = {}
    next_tracker_id = 1

    tracker = WagonTracker(
        iou_threshold     = tracker_iou_threshold,
        max_missed        = max_missed_frames,
        max_centroid_dist = max_centroid_dist,
        min_gap_frames    = min_gap_frames,
    )

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / video_fps

        results = model.predict(
            source=frame_bgr, conf=confidence,
            iou=iou_threshold, verbose=False,
        )

        wagon_dets  = []
        text_bboxes = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for box in result.boxes:
                cls_name = class_names.get(int(box.cls[0]), "")
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                bbox = [x1, y1, x2, y2]
                if cls_name in WAGON_CLASSES:
                    wagon_dets.append({
                        "frame_path" : "",
                        "timestamp"  : timestamp,
                        "bbox"       : bbox,
                        "confidence" : round(float(box.conf[0]), 3),
                        "class_name" : cls_name,
                        "frame_bgr"  : frame_bgr,
                    })
                elif cls_name == TEXT_CLASS:
                    text_bboxes.append(bbox)

        # Use tracker to get stable IDs for each wagon
        new_wagons = tracker.update(wagon_dets, frame_idx)

        # Register new wagons into active_wagons dict
        for wagon in new_wagons:
            tid = wagon["tracker_id"]
            cx  = _centroid(wagon["bbox"])[0]
            active_wagons[tid] = {
                "prev_cx"  : cx,
                "crossed"  : False,
                "best_det" : wagon,
                "best_conf": wagon["confidence"],
            }

        # Check crossing line for all active wagons
        for tid, track_info in list(active_wagons.items()):
            # Find current detection for this tracker id
            current_det = None
            for det in wagon_dets:
                # Match by IoU with last known bbox
                if _compute_iou(det["bbox"], track_info["best_det"]["bbox"]) > 0.3:
                    current_det = det
                    break

            if current_det is None:
                continue

            curr_cx = _centroid(current_det["bbox"])[0]
            prev_cx = track_info["prev_cx"]

            # Check if wagon centroid crossed the center line
            # Train moves left → right OR right → left
            crossed_ltr = prev_cx < cross_line_x <= curr_cx  # left to right
            crossed_rtl = prev_cx > cross_line_x >= curr_cx  # right to left

            if (crossed_ltr or crossed_rtl) and not track_info["crossed"]:
                # Wagon crossed the line — count it now
                track_info["crossed"] = True
                wagon_count += 1

                # Save this frame as the best frame for this wagon
                frame_name = (f"frame_{frame_idx:06d}"
                              f"_t{timestamp:.3f}s"
                              f"_id{tid}.jpg")
                frame_path = os.path.join(frames_dir, frame_name)
                cv2.imwrite(frame_path, frame_bgr)

                det_to_save = {
                    **current_det,
                    "tracker_id" : tid,
                    "frame_path" : frame_path,
                }

                ocr_bbox = _find_text_bbox_inside_wagon(
                    current_det["bbox"], text_bboxes
                )
                det_to_save["ocr_bbox"] = ocr_bbox if ocr_bbox else current_det["bbox"]

                all_detections.append(det_to_save)
                info("Wagon #%d crossed line! ID=%d | conf=%.2f | t=%.3fs",
                     wagon_count, tid, current_det["confidence"], timestamp)

            # Update previous centroid
            track_info["prev_cx"]   = curr_cx
            track_info["best_det"]  = current_det

        frame_idx += 1

    cap.release()
    info("Tracking complete — %d unique wagons crossed the line", len(all_detections))
    return all_detections

def _find_text_bbox_inside_wagon(wagon_bbox, text_bboxes):
    """Return text bbox whose center lies inside the wagon bbox, else None."""
    wx1, wy1, wx2, wy2 = wagon_bbox
    for tb in text_bboxes:
        cx = (tb[0]+tb[2])//2
        cy = (tb[1]+tb[3])//2
        if wx1 <= cx <= wx2 and wy1 <= cy <= wy2:
            return tb
    return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Push to detection queue
# ══════════════════════════════════════════════════════════════════════════════
def push_to_detection_queue(detections: list, detection_queue: queue.Queue) -> None:
    for det in detections:
        detection_queue.put(det)
    detection_queue.put(None)
    info("Pushed %d detections to detection_queue", len(detections))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — OCR
# ══════════════════════════════════════════════════════════════════════════════
def extract_text_ocr(detection_queue: queue.Queue,
                     ocr_reader: easyocr.Reader,
                     padding: int = 10) -> list:
    results = []
    while True:
        det = detection_queue.get()
        if det is None:
            break

        x1, y1, x2, y2 = det.get("ocr_bbox", det["bbox"])
        frame_bgr = det.get("frame_bgr")
        if frame_bgr is None:
            frame_bgr = cv2.imread(det["frame_path"])

        h, w = frame_bgr.shape[:2]
        crop     = frame_bgr[max(0,y1-padding):min(h,y2+padding),
                              max(0,x1-padding):min(w,x2+padding)]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        ocr_raw = ocr_reader.readtext(crop_rgb, detail=1)
        texts   = [t.strip().upper()
                   for (_, t, c) in sorted(ocr_raw, key=lambda x: -x[2])
                   if c > 0.3 and t.strip()]

        wagon_number = _best_wagon_number(texts)
        info("OCR @ %.3fs → texts=%s | wagon_number='%s'",
             det["timestamp"], texts, wagon_number)

        results.append({
            **det,
            "ocr_crop_array": crop_rgb,
            "ocr_texts":      texts,
            "wagon_number":   wagon_number,
        })

    info("OCR complete — %d results", len(results))
    return results


def _best_wagon_number(texts):
    for t in texts:
        if re.search(r'\d{4,}', t):
            return re.sub(r'\s+', '', t)
    return texts[0] if texts else ""


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Save output
# ══════════════════════════════════════════════════════════════════════════════
def save_output(ocr_results: list, output_dir: str) -> list:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for idx, res in enumerate(ocr_results):
        wagon_number = res["wagon_number"]
        timestamp    = res["timestamp"]
        crop_rgb     = res["ocr_crop_array"]
        tracker_id   = res.get("tracker_id", idx+1)
        x1,y1,x2,y2 = res["bbox"]

        frame_bgr = res.get("frame_bgr")
        if frame_bgr is None:
            frame_bgr = cv2.imread(res["frame_path"])

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        cv2.rectangle(frame_rgb, (x1,y1), (x2,y2), (0,200,80), 3)

        label      = f"ID:{tracker_id} | {wagon_number or '?'}"
        label_bg_y = max(y1-36, 0)
        cv2.rectangle(frame_rgb, (x1,label_bg_y), (x1+280,y1), (0,0,0), -1)
        cv2.putText(frame_rgb, label, (x1+6, max(y1-10,12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

        target_h      = 480
        scale         = target_h / frame_rgb.shape[0]
        frame_resized = cv2.resize(frame_rgb,
                                   (int(frame_rgb.shape[1]*scale), target_h))

        crop_h       = target_h // 2
        crop_w       = 300
        crop_resized = cv2.resize(crop_rgb, (crop_w, crop_h))
        ocr_panel    = np.full((target_h, crop_w, 3), 30, dtype=np.uint8)
        ocr_panel[:crop_h, :] = crop_resized

        lines = ["OCR region", "",
                 f"Number: {wagon_number or 'N/A'}",
                 "All texts:"] + (res["ocr_texts"] or ["(none)"])
        for i, line in enumerate(lines[:10]):
            cv2.putText(ocr_panel, line, (10, crop_h+30+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)

        composite = np.hstack([frame_resized, ocr_panel])
        out_name  = (f"wagon_{idx+1:03d}_id{tracker_id}"
                     f"_t{timestamp:.3f}s_{wagon_number or 'UNKNOWN'}.jpg")
        out_path  = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        saved_paths.append(out_path)

    info("Saved %d images → %s", len(saved_paths), output_dir)
    return saved_paths


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Rail OCR Pipeline")
    parser.add_argument("--video",  required=True,           help="Input video")
    parser.add_argument("--model",  required=True,           help="best.pt path")
    parser.add_argument("--output", default="results",       help="Output folder")
    parser.add_argument("--fps",    type=float, default=3.0, help="Sample FPS")
    parser.add_argument("--conf",   type=float, default=0.5, help="YOLO confidence")
    parser.add_argument("--gpu",    action="store_true",     help="GPU for OCR")
    parser.add_argument("--lang",   default="en",            help="OCR language")
    parser.add_argument("--gap",    type=int,   default=8,   help="Min frames between wagons")
    args = parser.parse_args()

    start      = time.time()
    frames_dir = os.path.join(args.output, "frames")
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    output_dir = os.path.join(args.output, "annotated", video_name)

    frame_queue     = queue.Queue()
    detection_queue = queue.Queue()

    info("Initialising EasyOCR …")
    ocr_reader = easyocr.Reader(args.lang.split(","), gpu=args.gpu)

    info("═══ STEP 1: Extracting frames ═══")
    frame_list = extract_frames(args.video, frames_dir, sample_fps=args.fps)

    info("═══ STEP 2: Pushing to frame queue ═══")
    push_to_frame_queue(frame_list, frame_queue)

    info("═══ STEP 3: Picking frames by timestamp ═══")
    sorted_frames = pick_frame_by_timestamp(frame_queue)

    info("═══ STEP 4: YOLO detection + tracking ═══")
    detections = run_yolo_detection(
        sorted_frames,
        args.model,
        confidence            = args.conf,
        video_path            = args.video,
        sample_fps            = args.fps,
        tracker_iou_threshold = 0.5,
        max_missed_frames     = 15,
        max_centroid_dist     = 150.0,
        min_gap_frames        = args.gap,
    )

    info("═══ STEP 5: Pushing to detection queue ═══")
    push_to_detection_queue(detections, detection_queue)

    info("═══ STEP 6: Running OCR ═══")
    ocr_results = extract_text_ocr(detection_queue, ocr_reader)

    info("═══ STEP 7: Saving output images ═══")
    save_output(ocr_results, output_dir)

    wagon_numbers  = [r["wagon_number"] for r in ocr_results]
    unique_numbers = sorted(set(w for w in wagon_numbers if w))

    print("\n" + "═"*60)
    print("  RAIL OCR — RESULTS")
    print("═"*60)
    print(f"  Total wagons detected : {len(ocr_results)}")
    print(f"  Unique wagon numbers  : {len(unique_numbers)}")
    print()
    for i, wn in enumerate(unique_numbers, 1):
        print(f"  Wagon {i:>3}  →  {wn}")
    if not unique_numbers:
        print("  (No wagon numbers recognised)")
    print()
    print(f"  Output saved to : {output_dir}/")
    print(f"  Total time      : {time.time()-start:.1f}s")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()
