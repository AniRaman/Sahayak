# sahayak_image_agent/agent.py

import os
import base64
import re
import uuid
import logging
import aiohttp # Necessary for making HTTP requests to Google Imagen API
from typing import Optional # Necessary for type hints

# Load environment variables (crucial for API keys). This should load from .env
from dotenv import load_dotenv
load_dotenv()

# Google ADK and Generative AI imports
from google.adk.agents import LlmAgent, InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import FunctionTool
from google.adk.models import LlmRequest, LlmResponse # Correct types for ADK callbacks
from google.genai import types as genai_types # Alias for google.genai.types (for Content, FunctionCall, etc.)
from google.adk.models import Gemini # For the agent's core LLM (gemini-1.5-flash-latest)

# Import the actual image generation function from your mcp_servers.image_server
from sahayak_image_agent.mcp_servers.image_server import generate_image_model_call

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sahayak-image-agent")

# --- Configuration for Google Imagen API (Generative Language API) ---
IMAGE_GENERATION_MODEL = "imagen-3.0-generate-002"
IMAGE_GENERATION_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model_id}:predict?key={api_key}"

# --- Directory for saving generated images ---
# This path is relative to the *root of the agent app* (sahayak_image_agent/).
# ADK web serves 'static/' relative to the app root.
STATIC_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "generated_images")

# Ensure directory exists at startup
os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)
logger.info(f"Image save directory ensured: {STATIC_IMAGES_DIR}")


# --- Helper Function: Save Image Data to File ---
# This function is now internal to this agent.py file
def _save_image_data_to_file(image_data_bytes: bytes, prompt_text: str) -> Optional[str]:
    """
    Saves raw image bytes to a file in the agent's static directory.
    Returns the local file path relative to the STATIC_IMAGES_DIR, or None on failure.
    """
    try:
        # Sanitize prompt for filename
        filename_base = re.sub(r'[^a-zA-Z0-9\s_]', '', prompt_text).strip()
        filename_base = filename_base.replace(' ', '_')[:50]
        if not filename_base:
            filename_base = "generated_image"
        
        filename = f"{filename_base}_{uuid.uuid4().hex[:8]}.png"
        
        # Use the already correctly defined STATIC_IMAGES_DIR
        filepath = os.path.join(STATIC_IMAGES_DIR, filename)

        # Ensure directory exists if it wasn't created at startup for some reason
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(image_data_bytes)
        
        logger.info(f"✅ [IMAGE SAVED] Image saved to: {filepath}")
        # Return the local file path (not web URL here)
        return filepath
    except Exception as e:
        logger.error(f"❌ [SAVE ERROR] Error saving image to file '{filepath}': {e}")
        return None


# --- Tool Function for the ADK Agent ---
@FunctionTool
async def generate_image_tool(topic: str) -> str: # Tool returns a simple string message
    """
    Generates an image based on the given topic and saves it locally.
    Returns a confirmation message or an error message.
    """
    logger.info(f"📸 [TOOL CALLED] generate_image_tool received topic: '{topic}'")
    
    # Call the actual image generation function from mcp_servers.image_server
    # This now returns raw bytes or an error string.
    image_bytes_or_error = await generate_image_model_call(topic) 
    
    if isinstance(image_bytes_or_error, bytes):
        # Image generation was successful, now save the bytes locally
        local_file_path = _save_image_data_to_file(image_bytes_or_error, topic)
        
        if local_file_path:
            # Agent's response will just be text confirming download, no image or URL in chat.
            return f"Image for '{topic}' successfully generated and saved to your system at: {local_file_path}"
        else:
            logger.error(f"Image generation succeeded but file saving failed for '{topic}'.")
            return "❌ Image generation succeeded but failed to save to disk."
    else:
        # If generate_image_model_call returned an error string or None
        logger.error(f"Image generation failed from model: {image_bytes_or_error}")
        return image_bytes_or_error or "❌ Image generation failed for an unknown reason."


# --- Callback for before_model_callback ---
async def _force_image_tool_call_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    Intercepts user messages to explicitly trigger the image generation tool based on keywords.
    """
    logger.info("\n--- Running _force_image_tool_call_callback ---")
    
    # Access user content through the invocation context from callback_context
    ctx: InvocationContext = callback_context._invocation_context
    user_message_content = ctx.user_content

    user_message_text = ""
    # Ensure robust extraction of text from user_message_content.parts
    if user_message_content and user_message_content.parts:
        for part in user_message_content.parts:
            if hasattr(part, 'text') and isinstance(part.text, str): 
                 user_message_text += part.text
            elif isinstance(part, dict) and 'text' in part and isinstance(part['text'], str):
                user_message_text += part['text']

    logger.info(f"User Message captured by callback: '{user_message_text}'")

    if not user_message_text:
        logger.info("Callback: No user message text found. Allowing LLM to process.")
        return None

    user_message_lower = user_message_text.lower()

    # Keywords to trigger image generation
    image_trigger_phrases = [
        "create image of", "generate image of", "draw image of",
        "show me an image of", "make an image of", "create a picture of",
        "generate a picture of", "draw a picture of", "show me a picture of",
        "make a picture of", "create a diagram of", "generate a diagram of",
        "draw a diagram of", "show me a diagram of", "make a diagram of",
        "generate graph of", "draw graph of", "show me graph of",
        "create illustration of", "generate illustration of",
        "image of", "picture of", "diagram of", "graph of", "illustration of",
        "visualize", "illustrate"
    ]

    topic = None
    for phrase in image_trigger_phrases:
        if phrase in user_message_lower:
            match = re.search(r'{} (.*)'.format(re.escape(phrase)), user_message_lower)
            if match:
                topic = match.group(1).strip()
                break
    
    if not topic and any(keyword in user_message_lower for keyword in ["image", "picture", "diagram", "graph", "illustration"]):
        topic = user_message_text.strip()

    if topic:
        topic = re.sub(r'\s*(?:image|picture|diagram|graph|illustration|of|a|an|etc)\s*$', '', topic, flags=re.IGNORECASE).strip()
        
        if not topic:
            logger.info("Callback: Image keyword found, but topic is empty after cleaning. Allowing LLM to process.")
            return None

        logger.info(f"Callback: Image generation keyword/intent detected. Extracted topic: '{topic}'. Forcing tool call.")
        
        # Return an LlmResponse with a FunctionCall part
        function_call = genai_types.FunctionCall(
            name="generate_image_tool", # Must match the tool name defined above
            args={"topic": topic}
        )
        return LlmResponse( # LlmResponse from google.adk.models
            content=genai_types.Content( # Content from google.genai.types
                parts=[genai_types.Part(function_call=function_call)], # Part from google.genai.types
                role="model"
            )
        )
    else:
        logger.info("Callback: No explicit image generation intent detected. Allowing LLM to process the request normally.")
        return None


# --- Define the ADK LlmAgent ---
root_agent = LlmAgent( # Name kept as 'sahayak_image_agent' as per request
    name="sahayak_image_agent", # Must match agent.yaml name in the sahayak folder
    instruction="You generate images based on user prompts using the tool. Always return a text message about the image generation result.",
    model="gemini-1.5-flash-latest",
    tools=[
        generate_image_tool # Correctly pass the decorated function
    ],
    before_model_callback=_force_image_tool_call_callback # Use the defined async callback function
)



# root_agent = Agent(
#     name="sahayak_worksheet_agent",
#     model="gemini-2.0-flash",
#     description="Worksheet agent that creates structured assignments from uploaded content only.",
#     instruction=(
#         "Use only the uploaded file or image to extract topics.\n"
#         "Generate an assignment with Easy, Medium, and Hard sections (unless specified otherwise).\n"
#         "Default: 5 MCQs + 3 Descriptive in each section.\n"
#         "Respect any user-specified language or pattern.\n"
#         "Never hallucinate."
#     ),
#     tools=[
#         extract_content_from_upload,
#         generate_questions_with_language,
#     ]
# )