# scene_utils.py
import os
import json
import torch
from torchvision import models, transforms
from PIL import Image
from collections import Counter

# ---- Config ----
IMAGENET_LABELS_FILE = "imagenet_classes.txt"  # one label per line (0..999)
TOPK = 5

# A mapping from higher-level scene categories -> list of keywords to look for in ImageNet labels.
# Extend this dict to add more scene categories or keywords.
SCENE_KEYWORDS = {
    "wrestling_arena": ["wrestling", "boxing ring", "arena", "stadium", "ring"],
    "medical_context": ["hospital", "clinic", "operating room", "laboratory", "waiting room"],
    "outdoor_scene": ["forest", "beach", "street", "park", "field", "mountain"],
    "indoor_scene": ["kitchen", "living room", "office", "bedroom", "restaurant", "hall"],
    "water_scene": ["ocean", "sea", "lake", "river", "beach"],
    "vehicle_scene": ["car", "truck", "bus", "motorcycle", "airplane", "train", "boat"],
    "animal_scene": ["dog", "cat", "bird", "fish", "horse", "lion", "tiger", "bear"],
    "human_activity": ["soccer", "football", "tennis", "basketball", "wrestling", "dancing", "running"],
    "food_scene": ["pizza", "sandwich", "hotdog", "coffee", "tea", "ice cream", "fruit"],
    # Add more categories and keywords as needed
}

# ---- Model & transforms ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---- Utilities ----
def load_imagenet_labels(file_path=IMAGENET_LABELS_FILE):
    """
    Loads imagenet labels from a file, one label per line (0..999).
    If file doesn't exist, returns a fallback list of 'class_{idx}' strings.
    """
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
            # If file contains JSON mapping like {"0": "tench", ...}
            if labels and labels[0].startswith("{"):
                try:
                    f.seek(0)
                    data = json.load(f)
                    # If mapping { "0": ["n01440764", "tench"], ... } style, handle common formats:
                    if isinstance(data, dict):
                        # try to convert dict to ordered list by numeric keys
                        ordered = [data[str(i)] if str(i) in data else data.get(i) for i in range(1000)]
                        # flatten if necessary
                        labels = [v[1] if isinstance(v, list) and len(v) > 1 else v for v in ordered]
                except Exception:
                    pass
            return labels
    else:
        # fallback: create placeholder labels so indexing won't crash
        return [f"class_{i}" for i in range(1000)]

IMAGENET_LABELS = load_imagenet_labels()

def map_labels_to_scene(predicted_labels):
    """
    Given a list of predicted ImageNet labels (strings), map to scene categories using SCENE_KEYWORDS.
    Returns the best matched scene and scores for all scenes.
    """
    # normalize label strings
    preds = [p.lower() for p in predicted_labels]
    scene_scores = Counter()

    for scene, keywords in SCENE_KEYWORDS.items():
        for kw in keywords:
            kw_low = kw.lower()
            # add score if any predicted label contains the keyword or equals it
            for p in preds:
                if kw_low in p or p in kw_low:
                    scene_scores[scene] += 1

    if scene_scores:
        best_scene, best_score = scene_scores.most_common(1)[0]
        # if best_score is 0 or all zeros, return unknown
        if best_score == 0:
            return "unknown", dict(scene_scores)
        return best_scene, dict(scene_scores)
    else:
        return "unknown", {}

# ---- Main function ----
def classify_scene(image_path):
    """
    Classifies the scene in an image using pre-trained ResNet.
    Returns dict: {
      "top_predictions": [(label, prob), ...],
      "scene": best_scene_label,
      "scene_scores": {scene:score,...}
    }
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return {"error": f"Image not found: {image_path}"}
    except Exception as e:
        return {"error": f"Unable to open image: {e}"}

    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_probs, top_idxs = torch.topk(probs, TOPK)

    # Convert indices to labels
    top_predictions = []
    for p, idx in zip(top_probs.cpu().numpy(), top_idxs.cpu().numpy()):
        label = IMAGENET_LABELS[idx] if idx < len(IMAGENET_LABELS) else f"class_{idx}"
        top_predictions.append((label, float(p)))

    predicted_labels = [lab for lab, _ in top_predictions]
    scene_label, scene_scores = map_labels_to_scene(predicted_labels)

    return {
        "top_predictions": top_predictions,
        "scene": scene_label,
        "scene_scores": scene_scores
    }

# ---- Quick test when run directly ----
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scene_utils.py /path/to/image.jpg")
    else:
        out = classify_scene(sys.argv[1])
        print(json.dumps(out, indent=2))
