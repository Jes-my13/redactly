# layout_utils.py

def analyze_layout(image_path, ocr_data):
    """
    Analyzes the layout of the image based on OCR regions.
    For this example, it identifies main text blocks.
    """
    if not ocr_data:
        return {"layout_structure": "No text detected."}
    
    # Simple logic to identify the largest text block
    main_text_block = {}
    max_height = 0
    
    # Group text blocks by proximity and size
    text_groups = {}
    for item in ocr_data:
        text = item["text"]
        box = item["box"]
        # In a real system, we'd use more complex spatial clustering.
        # This is a basic simulation of that.
        if box["height"] > max_height and "SHOCKING" in text.upper():
             max_height = box["height"]
             main_text_block = {"text": " ".join([i['text'] for i in ocr_data]), "type": "headline"}
    
    if not main_text_block:
         main_text_block = {"text": " ".join([i['text'] for i in ocr_data]), "type": "caption"}

    return {
        "layout_structure": "Hierarchical text blocks detected.",
        "text_blocks": [main_text_block]
    }