# image_agent.py (Place this inside the 'agent' folder)

import os
import logging
import re
import aiohttp # <--- Ensure aiohttp is imported for direct API calls
import base64

# ADK and Google GenAI imports
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.genai import types # For Content, Part, FunctionCall
from google.adk.models import Gemini # For the agent's brain model

# NO MORE VERTEX AI IMPORTS FOR IMAGEN HERE!
# We will use direct HTTP calls as per your working CLI example.

# Import the utility function from the 'utils' package
from utils.file_helpers import save_image_from_base64

# Setup basic logging for this agent
logger = logging.getLogger("image-agent")

# --- Configuration for Image Generation Model (Direct Generative Language API) ---
IMAGE_API_MODEL = "imagen-3.0-generate-002"
IMAGE_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model_id}:predict?key={api_key}"


# --- Tool Function for Image Generation ---
@FunctionTool
async def generate_image(topic: str) -> dict:
    """
    Generates an educational image (diagram, graph, etc.) based on the given topic.
    Uses Imagen-3.0 via Generative Language API (GOOGLE_API_KEY) directly.
    """
    api_key = os.getenv("GOOGLE_API_KEY") # This is the key for Generative Language API
    if not api_key:
        logger.error("GOOGLE_API_KEY not found for image generation tool.")
        return {"error": "Authentication Error: GOOGLE_API_KEY not found. Please ensure it's set correctly in your .env file."}

    api_url = IMAGE_API_URL_TEMPLATE.format(model_id=IMAGE_API_MODEL, api_key=api_key)

    payload = {
        "instances": {
            "prompt": topic
        },
        "parameters": {
            "sampleCount": 1
        }
    }

    logger.info(f"📸 [TOOL] Requesting image for: '{topic}' from Imagen-3.0 (GL API).")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload) as response:
                response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
                result = await response.json()

                if result.get("predictions") and len(result["predictions"]) > 0 and result["predictions"][0].get("bytesBase64Encoded"):
                    image_base64 = result["predictions"][0]["bytesBase64Encoded"]
                    image_data = base64.b64decode(image_base64)
                    saved_web_path = save_image_from_base64(image_data, topic)
                    
                    if saved_web_path:
                        logger.info(f"✅ Image generated and saved. Web path: {saved_web_path}")
                        return {"image_url": saved_web_path, "message": f"🖼️ Image for '{topic}' generated successfully."}
                    else:
                        return {"error": f"Image data received but failed to save to disk for '{topic}'."}
                else:
                    error_message = f"Could not generate image for topic '{topic}'. Unexpected API response structure: {result}"
                    logger.error(f"Image generation failed: {error_message}")
                    return {"error": error_message}
    except aiohttp.ClientError as e:
        error_message = f"Error connecting to image generation API for '{topic}': {e}. Please check your network and API key permissions."
        logger.error(f"Network/API Error: {error_message}")
        return {"error": error_message}
    except Exception as e:
        error_message = f"An unexpected error occurred during image generation for '{topic}': {e}"
        logger.error(f"Unexpected Tool Error: {error_message}")
        return {"error": error_message}

# --- Define the LlmAgent for Image Generation ---
image_generator_agent = LlmAgent(
    name="image_generation_agent",
    model=Gemini(model="gemini-1.5-flash-latest"), # This uses GOOGLE_API_KEY
    instruction=(
        "You are an educational image generation assistant. "
        "Your primary role is to create relevant educational images (like diagrams, graphs, charts, or illustrations) "
        "based on the user's text description by calling the 'generate_image' tool.\n"
        "Be specific when calling the tool; extract the clearest possible topic for the image. "
        "Examples of what you can generate: 'water cycle diagram', 'human heart anatomy', 'graph of quadratic equation', 'animal cell structure', 'photosynthesis process', 'solar system planets'.\n"
        "When the 'generate_image' tool returns a successful response with an 'image_url' and 'message', combine them. "
        "First, output the 'message' to confirm generation, then immediately follow with a markdown link to the image, like this: [Generated Image Link](image_url). "
        "If the tool returns an 'error' message, you MUST inform the user about that specific error. "
        "For example, 'I'm sorry, I couldn't generate the image because: [error message]. Please check your configuration and try again.'\n"
        "Do NOT try to describe or generate the image content in text yourself, or output base64 data. "
        "If a request cannot be visualized (e.g., abstract concepts like 'love', inappropriate content, or something very complex that can't be a single image), politely decline "
        "by saying 'I cannot generate an image for that request because...' and explain why."
    ),
    tools=[
        FunctionTool(func=generate_image)
    ],
)