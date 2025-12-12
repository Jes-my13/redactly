# metadata_utils_advanced.py

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def extract_raw_metadata(image_path):
    """
    Extracts all available raw EXIF, GPS, and other metadata from an image.
    """
    metadata = {}
    try:
        with Image.open(image_path) as img:
            info = img._getexif()
            if info:
                for tag, value in info.items():
                    tag_name = TAGS.get(tag, tag)
                    if tag_name == 'GPSInfo':
                        gps_data = {}
                        for t, v in value.items():
                            gps_data[GPSTAGS.get(t, t)] = v
                        metadata[tag_name] = gps_data
                    else:
                        metadata[tag_name] = str(value) # Convert all to string for LLM input
    except (IOError, AttributeError):
        pass  # No metadata or error reading file
    return metadata