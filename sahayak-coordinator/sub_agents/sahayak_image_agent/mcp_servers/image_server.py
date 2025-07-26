# sahayak_image_agent/mcp_servers/image_server.py

import os
import logging
import aiohttp # Added for network requests
import base64 # Added for base64 decoding
import re # Added for filename sanitization
import uuid # Added for unique filenames

# Vertex AI imports for GenerativeModel (Imagen)
from vertexai.preview.generative_models import GenerativeModel
import asyncio # For running async ops if not already in an async context.

logger = logging.getLogger("image_server")

# --- Configuration for Google Imagen API (Generative Language API) ---
# Duplicated here for clarity within this file, but actual API key will come from env.
IMAGE_GENERATION_MODEL = "imagen-3.0-generate-002"
IMAGE_GENERATION_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model_id}:predict?key={api_key}"


async def generate_image_model_call(prompt: str) -> bytes | str:
    """
    Generates an image from a prompt using Google Imagen API.
    Returns the raw image bytes on success, or an error string on failure.
    """
    api_key = os.getenv("GOOGLE_API_KEY") # Ensure .env has GOOGLE_API_KEY
    if not api_key:
        logger.error("GOOGLE_API_KEY not found for Imagen generation in image_server.py.")
        return "Error: GOOGLE_API_KEY not found. Please set it in your .env file."

    api_url = IMAGE_GENERATION_API_URL_TEMPLATE.format(model_id=IMAGE_GENERATION_MODEL, api_key=api_key)

    payload = {
        "instances": {
            "prompt": prompt
        },
        "parameters": {
            "sampleCount": 1
        }
    }

    logger.info(f"📸 [IMAGEN API CALL] Requesting image for: '{prompt}' from Imagen-3.0 (GL API).")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload) as response:
                response.raise_for_status() # Raises an exception for 4xx/5xx responses
                result = await response.json()

                if result.get("predictions") and len(result["predictions"]) > 0 and result["predictions"][0].get("bytesBase64Encoded"):
                    image_base64 = result["predictions"][0]["bytesBase64Encoded"]
                    image_binary_data = base64.b64decode(image_base64)
                    
                    logger.info(f"✅ [IMAGEN API SUCCESS] Raw image data received for '{prompt}'.")
                    return image_binary_data # Return raw bytes
                else:
                    logger.warning(f"Imagen returned no image data for prompt: '{prompt}'. Response: {result}")
                    return "Failed to generate image: No valid response from the model."
    except aiohttp.ClientError as e:
        logger.error(f"❌ [IMAGEN API ERROR] Network/API error calling Imagen: {e}", exc_info=True)
        return f"Network/API error calling Imagen: {e}. Check internet and API key/permissions."
    except Exception as e:
        logger.error(f"❌ [IMAGEN API ERROR] Unexpected error during Imagen call: {e}", exc_info=True)
        return f"Unexpected error during Imagen call: {e}"

# Note: generate_image_from_prompt is now an alias for generate_image_model_call
# This is to match the import name expected by agent.py
async def generate_image_from_prompt(prompt: str) -> bytes | str:
    return await generate_image_model_call(prompt)