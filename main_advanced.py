# main_advanced.py

from ocr_utils_advanced import extract_text_and_regions
from content_utils import analyze_content
from metadata_utils_advanced import extract_raw_metadata
from llm_utils import validate_and_infer_metadata
from layout_utils import analyze_layout

def process_image_advanced(image_path):
    """
    Processes an image using an advanced pipeline for comprehensive data extraction.
    """
    results = {}
    
    # 1. Multi-layer Metadata Extraction
    raw_metadata = extract_raw_metadata(image_path)
    results["metadata"] = validate_and_infer_metadata(raw_metadata)

    # 2. Deep Learning for Content Extraction (objects, scenes, faces)
    results["content"] = analyze_content(image_path)

    # 3. Enhanced OCR & Layout Analysis
    text_data = extract_text_and_regions(image_path)
    results["ocr"] = text_data
    results["layout"] = analyze_layout(image_path, text_data)

    return results

if __name__ == "__main__":
    img_path = "sample.jpg"
    print(f"--- Processing image with advanced pipeline: {img_path} ---")
    try:
        details = process_image_advanced(img_path)
        print("\n--- Comprehensive Details Extracted ---")
        # For structured output, print as JSON
        import json
        print(json.dumps(details, indent=2))
    except FileNotFoundError:
        print(f"Error: The file '{img_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")