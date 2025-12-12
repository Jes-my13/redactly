# ocr_utils_advanced.py

import pytesseract
from PIL import Image

def extract_text_and_regions(image_path):
    """
    Performs OCR and returns text along with bounding box information.
    """
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    ocr_results = []
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        text = data['text'][i].strip()
        if text:
            # Get bounding box coordinates
            box = {
                "left": data['left'][i],
                "top": data['top'][i],
                "width": data['width'][i],
                "height": data['height'][i]
            }
            ocr_results.append({"text": text, "box": box})
            
    return ocr_results