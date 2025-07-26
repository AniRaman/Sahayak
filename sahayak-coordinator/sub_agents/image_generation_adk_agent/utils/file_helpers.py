# file_helpers.py (Place this inside the 'utils' folder)

import os
import base64
import uuid
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("file_helpers")

# Directory where images will be saved, relative to the project root.
# This is what the ADK Web server automatically serves.
IMAGES_SAVE_DIR = os.path.join("static", "generated_images")

# Ensure the directory exists on module load (or on first call)
# It's good to ensure it's created once when the server starts.
os.makedirs(IMAGES_SAVE_DIR, exist_ok=True)

from typing import Optional # <--- Make sure this import exists at the top

def save_image_from_base64(image_data: bytes, original_query: str) -> Optional[str]:
    """
    Saves raw image bytes as a PNG file and returns its web-accessible path.

    Args:
        image_data (bytes): The raw image data (not base64 encoded).
        original_query (str): The original user query, used for generating a filename.

    Returns:
        Optional[str]: The web-accessible URL path (e.g., '/static/generated_images/...') on success,
                       or None if saving fails.
    """
    try:
        # Sanitize the original query to create a descriptive filename
        # Remove common command phrases and non-alphanumeric characters
        clean_query = re.sub(r'(?:create|generate|draw|show me a|make a|illustrate|visualize)\s*(?:an?|the)?\s*', '', original_query, flags=re.IGNORECASE).strip()
        clean_query = re.sub(r'[^a-zA-Z0-9\s_.-]', '', clean_query).strip()
        
        filename_base = clean_query.replace(' ', '_').replace('.', '').replace('-', '_')[:60] # Limit length
        if not filename_base:
            filename_base = "generated_image"
        
        filename = f"{filename_base}_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(IMAGES_SAVE_DIR, filename)

        logger.info(f"Attempting to save image to: {os.path.abspath(filepath)}")
        
        with open(filepath, "wb") as f:
            f.write(image_data)

        web_path = f"/static/generated_images/{filename}" # This is the path for the web UI
        logger.info(f"\n--- Image Saved Locally ---")
        logger.info(f"Image successfully saved to: {os.path.abspath(filepath)}")
        logger.info(f"Access via web: {web_path}")
        logger.info("---------------------------\n")
        return web_path
    except Exception as e:
        logger.error(f"❌ Error saving image: {e}")
        return None