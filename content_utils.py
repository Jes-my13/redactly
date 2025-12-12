# content_utils.py
import os
import cv2
import traceback

# Try to use ultralytics YOLO if available
try:
    from ultralytics import YOLO
    _has_ultralytics = True
except Exception:
    _has_ultralytics = False

# scene classifier from your scene_utils
try:
    from scene_utils import classify_scene
except Exception:
    def classify_scene(path):
        return {"scene": "unknown", "top_predictions": []}

# default model path (expects yolov8n.pt in same folder). Change if needed.
YOLO_MODEL_PATH = "yolov8n.pt"

# Load Haar cascade for face detection fallback
_cascade = None
try:
    _cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
except Exception:
    _cascade = None

# Load YOLO model as a global to avoid re-loading many times
_yolo_model = None
if _has_ultralytics and os.path.exists(YOLO_MODEL_PATH):
    try:
        _yolo_model = YOLO(YOLO_MODEL_PATH)
    except Exception:
        _yolo_model = None

def _detect_with_yolo(image_path):
    """
    Runs YOLO inference and returns a list of detections.
    Each detection: {"label":str, "confidence":float, "box":[x1,y1,x2,y2]}
    """
    detections = []
    try:
        global _yolo_model
        if _yolo_model is None:
            # try to create model even if it wasn't created at import time
            if _has_ultralytics and os.path.exists(YOLO_MODEL_PATH):
                _yolo_model = YOLO(YOLO_MODEL_PATH)
            else:
                return detections

        # run inference
        results = _yolo_model(image_path)  # <--- this creates `results`
        # results is a list-like, take first frame
        r = results[0]
        names = r.names if hasattr(r, "names") else {}
        boxes = getattr(r, "boxes", [])
        for b in boxes:
            try:
                cls_id = int(b.cls.cpu().numpy()) if hasattr(b, "cls") else int(b[5])  # fallback
            except Exception:
                # some versions expose .cls as numpy already
                try:
                    cls_id = int(b.cls)
                except Exception:
                    cls_id = -1
            try:
                conf = float(b.conf.cpu().numpy()) if hasattr(b, "conf") else float(getattr(b, "conf", 0.0))
            except Exception:
                conf = float(getattr(b, "conf", 0.0)) if hasattr(b, "conf") else 0.0
            label = names.get(cls_id, f"class_{cls_id}") if isinstance(names, dict) else str(cls_id)
            try:
                xyxy = b.xyxy.cpu().numpy().tolist() if hasattr(b, "xyxy") else [0,0,0,0]
            except Exception:
                xyxy = [0,0,0,0]
            detections.append({
                "label": label,
                "confidence": conf,
                "box": xyxy
            })
    except Exception as e:
        # return partial detections and include error in payload
        detections.append({"error": f"yolo_inference_failed: {e}"})
    return detections

def _detect_faces_opencv(image_path, scaleFactor=1.1, minNeighbors=5, minSize=(30,30)):
    """
    Simple face detector using OpenCV Haar cascade as fallback.
    Returns list of face boxes [x, y, w, h]
    """
    faces = []
    try:
        img = cv2.imread(image_path)
        if img is None:
            return faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if _cascade is None:
            return faces
        detected = _cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
        for (x, y, w, h) in detected:
            faces.append([int(x), int(y), int(w), int(h)])
    except Exception:
        pass
    return faces

def analyze_content(image_path):
    """
    High-level function used by main_advanced.py
    Returns a dict with keys:
      - objects: list of detected objects (from YOLO) or empty
      - faces: list of face boxes (opencv) + count
      - scene: output from classify_scene(...) (dict)
      - raw_errors: optional list of error messages
    """
    out = {
        "objects": [],
        "faces": {"count": 0, "boxes": []},
        "scene": {"scene": "unknown", "top_predictions": []},
        "raw_errors": []
    }

    try:
        # 1) Object detection (YOLO) if available
        if _has_ultralytics:
            objs = _detect_with_yolo(image_path)
            out["objects"] = objs

        # 2) Face detection (if YOLO produced face labels we could use them; also use cascade)
        # Try to count "face" or "person" from YOLO results first
        face_count = 0
        face_boxes = []
        for obj in out["objects"]:
            if isinstance(obj, dict) and "label" in obj:
                lbl = obj["label"].lower()
                if "face" in lbl or lbl == "person":
                    face_count += 1
                    if "box" in obj:
                        # convert xyxy or similar to (x,y,w,h) if possible
                        xy = obj["box"]
                        if len(xy) == 4:
                            try:
                                x1,y1,x2,y2 = map(int, xy)
                                face_boxes.append([x1, y1, x2-x1, y2-y1])
                            except Exception:
                                pass

        # If YOLO didn't find faces, fallback to OpenCV Haar cascade
        if face_count == 0:
            faces = _detect_faces_opencv(image_path)
            face_boxes = faces
            face_count = len(faces)

        out["faces"]["count"] = face_count
        out["faces"]["boxes"] = face_boxes

        # 3) Scene classification
        try:
            scene_info = classify_scene(image_path)
            # If classify_scene returns a dict like earlier helper, keep it; else put into dict
            if isinstance(scene_info, dict):
                out["scene"] = scene_info
            else:
                out["scene"] = {"scene": str(scene_info)}
        except Exception as e:
            out["raw_errors"].append(f"scene_classify_failed: {e}")

    except Exception as e:
        tb = traceback.format_exc()
        out["raw_errors"].append(f"analyze_content_exception: {e}")
        out["raw_errors"].append(tb)

    return out

# quick test
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) > 1:
        res = analyze_content(sys.argv[1])
        print(json.dumps(res, indent=2))
    else:
        print("usage: python content_utils.py /path/to/image.jpg")
