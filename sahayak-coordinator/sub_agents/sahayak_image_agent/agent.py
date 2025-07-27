# # sahayak_image_agent/agent.py

# import os
# import base64
# import re
# import uuid
# import logging
# import aiohttp # Necessary for making HTTP requests to Google Imagen API
# from typing import Optional # Necessary for type hints

# # Load environment variables (crucial for API keys). This should load from .env
# from dotenv import load_dotenv
# load_dotenv()

# # Google ADK and Generative AI imports
# from google.adk.agents import LlmAgent, InvocationContext
# from google.adk.agents.callback_context import CallbackContext
# from google.adk.tools import FunctionTool
# from google.adk.models import LlmRequest, LlmResponse # Correct types for ADK callbacks
# from google.genai import types as genai_types # Alias for google.genai.types (for Content, FunctionCall, etc.)
# from google.adk.models import Gemini # For the agent's core LLM (gemini-1.5-flash-latest)

# # Import the actual image generation function from your mcp_servers.image_server
# from sahayak_image_agent.mcp_servers.image_server import generate_image_model_call

# # --- Configure Logging ---
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("sahayak-image-agent")

# # --- Configuration for Google Imagen API (Generative Language API) ---
# IMAGE_GENERATION_MODEL = "imagen-3.0-generate-002"
# IMAGE_GENERATION_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model_id}:predict?key={api_key}"

# # --- Directory for saving generated images ---
# # This path is relative to the *root of the agent app* (sahayak_image_agent/).
# # ADK web serves 'static/' relative to the app root.
# STATIC_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "generated_images")

# # Ensure directory exists at startup
# os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)
# logger.info(f"Image save directory ensured: {STATIC_IMAGES_DIR}")


# # --- Helper Function: Save Image Data to File ---
# # This function is now internal to this agent.py file
# def _save_image_data_to_file(image_data_bytes: bytes, prompt_text: str) -> Optional[str]:
#     """
#     Saves raw image bytes to a file in the agent's static directory.
#     Returns the local file path relative to the STATIC_IMAGES_DIR, or None on failure.
#     """
#     try:
#         # Sanitize prompt for filename
#         filename_base = re.sub(r'[^a-zA-Z0-9\s_]', '', prompt_text).strip()
#         filename_base = filename_base.replace(' ', '_')[:50]
#         if not filename_base:
#             filename_base = "generated_image"
        
#         filename = f"{filename_base}_{uuid.uuid4().hex[:8]}.png"
        
#         # Use the already correctly defined STATIC_IMAGES_DIR
#         filepath = os.path.join(STATIC_IMAGES_DIR, filename)

#         # Ensure directory exists if it wasn't created at startup for some reason
#         os.makedirs(os.path.dirname(filepath), exist_ok=True)
#         with open(filepath, "wb") as f:
#             f.write(image_data_bytes)
        
#         logger.info(f"✅ [IMAGE SAVED] Image saved to: {filepath}")
#         # Return the local file path (not web URL here)
#         return filepath
#     except Exception as e:
#         logger.error(f"❌ [SAVE ERROR] Error saving image to file '{filepath}': {e}")
#         return None


# # --- Tool Function for the ADK Agent ---
# @FunctionTool
# async def generate_image_tool(topic: str) -> str: # Tool returns a simple string message
#     """
#     Generates an image based on the given topic and saves it locally.
#     Returns a confirmation message or an error message.
#     """
#     logger.info(f"📸 [TOOL CALLED] generate_image_tool received topic: '{topic}'")
    
#     # Call the actual image generation function from mcp_servers.image_server
#     # This now returns raw bytes or an error string.
#     image_bytes_or_error = await generate_image_model_call(topic) 
    
#     if isinstance(image_bytes_or_error, bytes):
#         # Image generation was successful, now save the bytes locally
#         local_file_path = _save_image_data_to_file(image_bytes_or_error, topic)
        
#         if local_file_path:
#             # Agent's response will just be text confirming download, no image or URL in chat.
#             return f"Image for '{topic}' successfully generated and saved to your system at: {local_file_path}"
#         else:
#             logger.error(f"Image generation succeeded but file saving failed for '{topic}'.")
#             return "❌ Image generation succeeded but failed to save to disk."
#     else:
#         # If generate_image_model_call returned an error string or None
#         logger.error(f"Image generation failed from model: {image_bytes_or_error}")
#         return image_bytes_or_error or "❌ Image generation failed for an unknown reason."


# # --- Callback for before_model_callback ---
# async def _force_image_tool_call_callback(
#     callback_context: CallbackContext, llm_request: LlmRequest
# ) -> Optional[LlmResponse]:
#     """
#     Intercepts user messages to explicitly trigger the image generation tool based on keywords.
#     """
#     logger.info("\n--- Running _force_image_tool_call_callback ---")
    
#     # Access user content through the invocation context from callback_context
#     ctx: InvocationContext = callback_context._invocation_context
#     user_message_content = ctx.user_content

#     user_message_text = ""
#     # Ensure robust extraction of text from user_message_content.parts
#     if user_message_content and user_message_content.parts:
#         for part in user_message_content.parts:
#             if hasattr(part, 'text') and isinstance(part.text, str): 
#                  user_message_text += part.text
#             elif isinstance(part, dict) and 'text' in part and isinstance(part['text'], str):
#                 user_message_text += part['text']

#     logger.info(f"User Message captured by callback: '{user_message_text}'")

#     if not user_message_text:
#         logger.info("Callback: No user message text found. Allowing LLM to process.")
#         return None

#     user_message_lower = user_message_text.lower()

#     # Keywords to trigger image generation
#     image_trigger_phrases = [
#         "create image of", "generate image of", "draw image of",
#         "show me an image of", "make an image of", "create a picture of",
#         "generate a picture of", "draw a picture of", "show me a picture of",
#         "make a picture of", "create a diagram of", "generate a diagram of",
#         "draw a diagram of", "show me a diagram of", "make a diagram of",
#         "generate graph of", "draw graph of", "show me graph of",
#         "create illustration of", "generate illustration of",
#         "image of", "picture of", "diagram of", "graph of", "illustration of",
#         "visualize", "illustrate"
#     ]

#     topic = None
#     for phrase in image_trigger_phrases:
#         if phrase in user_message_lower:
#             match = re.search(r'{} (.*)'.format(re.escape(phrase)), user_message_lower)
#             if match:
#                 topic = match.group(1).strip()
#                 break
    
#     if not topic and any(keyword in user_message_lower for keyword in ["image", "picture", "diagram", "graph", "illustration"]):
#         topic = user_message_text.strip()

#     if topic:
#         topic = re.sub(r'\s*(?:image|picture|diagram|graph|illustration|of|a|an|etc)\s*$', '', topic, flags=re.IGNORECASE).strip()
        
#         if not topic:
#             logger.info("Callback: Image keyword found, but topic is empty after cleaning. Allowing LLM to process.")
#             return None

#         logger.info(f"Callback: Image generation keyword/intent detected. Extracted topic: '{topic}'. Forcing tool call.")
        
#         # Return an LlmResponse with a FunctionCall part
#         function_call = genai_types.FunctionCall(
#             name="generate_image_tool", # Must match the tool name defined above
#             args={"topic": topic}
#         )
#         return LlmResponse( # LlmResponse from google.adk.models
#             content=genai_types.Content( # Content from google.genai.types
#                 parts=[genai_types.Part(function_call=function_call)], # Part from google.genai.types
#                 role="model"
#             )
#         )
#     else:
#         logger.info("Callback: No explicit image generation intent detected. Allowing LLM to process the request normally.")
#         return None


# # --- Define the ADK LlmAgent ---
# root_agent = LlmAgent( # Name kept as 'sahayak_image_agent' as per request
#     name="sahayak_image_agent", # Must match agent.yaml name in the sahayak folder
#     instruction="You generate images based on user prompts using the tool. Always return a text message about the image generation result.",
#     model="gemini-1.5-flash-latest",
#     tools=[
#         generate_image_tool # Correctly pass the decorated function
#     ],
#     before_model_callback=_force_image_tool_call_callback # Use the defined async callback function
# )



# sahayak_image_agent/agent.py

# import os
# import base64
# import re
# import uuid
# import logging
# import aiohttp
# from typing import Optional

# # Load environment variables
# from dotenv import load_dotenv
# load_dotenv()

# # Google ADK and Generative AI imports
# from google.adk.agents import LlmAgent, InvocationContext
# from google.adk.agents.callback_context import CallbackContext # This import is no longer needed without the callback
# from google.adk.tools import FunctionTool
# from google.adk.models import LlmRequest, LlmResponse # These imports are no longer needed without the callback
# from google.genai import types as genai_types
# from google.adk.models import Gemini

# # Import the actual image generation function from your mcp_servers.image_server
# from .mcp_servers.image_server import generate_image_model_call

# # --- Configure Logging ---
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("sahayak-image-agent")

# # --- Configuration for Google Imagen API (Generative Language API) ---
# IMAGE_GENERATION_MODEL = "imagen-3.0-generate-002"
# IMAGE_GENERATION_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model_id}:predict?key={api_key}"

# # --- Directory for saving generated images ---
# PROJECT_ROOT_FOR_STATIC = os.getcwd() 
# STATIC_IMAGES_DIR = os.path.join(PROJECT_ROOT_FOR_STATIC, "sahayak_image_agent", "static", "generated_images")

# os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)
# logger.info(f"Image save directory ensured: {STATIC_IMAGES_DIR}")


# # --- Core Image Generation Logic (using Google Imagen API) ---
# async def generate_image_from_prompt(prompt: str) -> Optional[str]: 
#     api_key = os.getenv("GEMINI_API_KEY")
#     if not api_key:
#         logger.error("GOOGLE_API_KEY not found for Imagen generation. Please set it in your .env file.")
#         return None

#     api_url = IMAGE_GENERATION_API_URL_TEMPLATE.format(model_id=IMAGE_GENERATION_MODEL, api_key=api_key)

#     payload = {
#         "instances": {
#             "prompt": prompt
#         },
#         "parameters": {
#             "sampleCount": 1
#         }
#     }

#     logger.info(f"📸 [IMAGEN CALL] Requesting image for: '{prompt}' from Imagen-3.0 (GL API).")
#     try:
#         async with aiohttp.ClientSession() as session:
#             async with session.post(api_url, json=payload) as response:
#                 response.raise_for_status()
#                 result = await response.json()

#                 if result.get("predictions") and len(result["predictions"]) > 0 and result["predictions"][0].get("bytesBase64Encoded"):
#                     image_base64 = result["predictions"][0]["bytesBase64Encoded"]
#                     image_binary_data = base64.b64decode(image_base64)
                    
#                     filename_base = re.sub(r'[^a-zA-Z0-9\s_]', '', prompt).strip()
#                     filename_base = filename_base.replace(' ', '_')[:50]
#                     if not filename_base:
#                         filename_base = "generated_image"
                    
#                     filename = f"{filename_base}_{uuid.uuid4().hex[:8]}.png"
#                     local_file_path = os.path.join(STATIC_IMAGES_DIR, filename)

#                     os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
#                     with open(local_file_path, "wb") as f:
#                         f.write(image_binary_data)
                    
#                     logger.info(f"✅ [IMAGEN SUCCESS] Image generated and saved to: {local_file_path}")
#                     return f"/static/generated_images/{filename}" 
#                 else:
#                     logger.warning(f"Imagen returned no image data for prompt: '{prompt}'. Response: {result}")
#                     return None
#     except aiohttp.ClientError as e:
#         logger.error(f"❌ [IMAGEN ERROR] Network/API error calling Imagen: {e}", exc_info=True)
#         return None
#     except Exception as e:
#         logger.error(f"❌ [IMAGEN ERROR] Unexpected error during Imagen call: {e}", exc_info=True)
#         return None


# # --- Tool Function for the ADK Agent ---
# @FunctionTool
# async def generate_image_tool(topic: str) -> str:
#     """
#     Generates an image based on the given topic and saves it locally.
#     Returns a confirmation message or an error message.
#     """
#     logger.info(f"📸 [TOOL CALLED] generate_image_tool received topic: '{topic}'")
    
#     local_image_path = await generate_image_from_prompt(topic)
    
#     if local_image_path:
#         return f"Image for '{topic}' successfully generated and saved to your system at: {local_image_path}"
#     else:
#         return "❌ Image generation failed. Please check the logs for details and ensure API key/permissions are correct."


# # --- Define the ADK LlmAgent ---
# root_agent = LlmAgent( # Name kept as 'sahayak_image_agent' as per request
#     name="sahayak_image_agent", # Must match agent.yaml name in the sahayak folder
#     instruction=(
#         "You are an image generation assistant. Your primary task is to generate images "
#         "based on user descriptions. "
#         "**When the user requests an image, ALWAYS call the 'generate_image_tool' with the user's explicit topic.** " # STRONGER INSTRUCTION
#         "**Prioritize calling the tool over responding with text if an image request is clear.** "
#         "For example, if the user says 'create an image of a red car', you should call the tool "
#         "with 'red car' as the topic. If they say 'show me a diagram of the human heart', call it with 'human heart diagram'. "
#         "After successfully calling the tool, respond to the user with the confirmation message provided by the tool. " # NEW: Explicitly tell it to respond AFTER tool call
#         "If the tool reports a failure, inform the user about the error. " # Handle tool errors.
#         "Decline requests that are inappropriate or cannot be clearly visualized (e.g., abstract concepts). "
#         "Do NOT try to describe or generate the image content in text yourself. "
#         "Your responses should be concise and focused on the image generation process."
#     ),
#     model="gemini-1.5-flash-latest",
#     tools=[
#         generate_image_tool # Correctly pass the decorated function
#     ],
#     # REMOVED: before_model_callback=_force_image_tool_call_callback # <--- THIS LINE IS REMOVED
# )



# sahayak_image_agent/agent.py

import os
import base64
import re
import uuid
import logging
import aiohttp
from typing import Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Google ADK and Generative AI imports
from google.adk.agents import LlmAgent # Removed InvocationContext, CallbackContext as they are not needed without the callback
from google.adk.tools import FunctionTool
# Removed LlmRequest, LlmResponse as they are no longer needed without the callback
from google.genai import types as genai_types
from google.adk.models import Gemini

# Import the actual image generation function from your mcp_servers.image_server
from .mcp_servers.image_server import generate_image_model_call

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sahayak-image-agent")

# --- Configuration for Google Imagen API (Generative Language API) ---
IMAGE_GENERATION_MODEL = "imagen-3.0-generate-002"
IMAGE_GENERATION_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model_id}:predict?key={api_key}"

# --- Directory for saving generated images ---
# This path should be relative to the *root of the agent app* (sahayak_image_agent/).
# ADK web serves 'static/' relative to the app root.
STATIC_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "generated_images")

os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)
logger.info(f"Image save directory ensured: {STATIC_IMAGES_DIR}")


# --- Core Image Generation Logic (using Google Imagen API) ---
async def generate_image_from_prompt(prompt: str) -> Optional[str]: 
    """
    Generates an image based on the given prompt text using Google Imagen API.
    Saves the image locally and returns its web-accessible path (e.g., /static/...).
    Returns None on failure.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found for Imagen generation. Please set it in your .env file.")
        return None

    api_url = IMAGE_GENERATION_API_URL_TEMPLATE.format(model_id=IMAGE_GENERATION_MODEL, api_key=api_key)

    payload = {
        "instances": {
            "prompt": prompt
        },
        "parameters": {
            "sampleCount": 1
        }
    }

    logger.info(f"📸 [IMAGEN CALL] Requesting image for: '{prompt}' from Imagen-3.0 (GL API).")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload) as response:
                response.raise_for_status()
                result = await response.json()

                if result.get("predictions") and len(result["predictions"]) > 0 and result["predictions"][0].get("bytesBase64Encoded"):
                    image_base64 = result["predictions"][0]["bytesBase64Encoded"]
                    image_binary_data = base64.b64decode(image_base64)
                    
                    filename_base = re.sub(r'[^a-zA-Z0-9\s_]', '', prompt).strip()
                    filename_base = filename_base.replace(' ', '_')[:50]
                    if not filename_base:
                        filename_base = "generated_image"
                    
                    filename = f"{filename_base}_{uuid.uuid4().hex[:8]}.png"
                    local_file_path = os.path.join(STATIC_IMAGES_DIR, filename)

                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    with open(local_file_path, "wb") as f:
                        f.write(image_binary_data)
                    
                    logger.info(f"✅ [IMAGEN SUCCESS] Image generated and saved to: {local_file_path}")
                    return f"/static/generated_images/{filename}" 
                else:
                    logger.warning(f"Imagen returned no image data for prompt: '{prompt}'. Response: {result}")
                    return None
    except aiohttp.ClientError as e:
        logger.error(f"❌ [IMAGEN ERROR] Network/API error calling Imagen: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"❌ [IMAGEN ERROR] Unexpected error during Imagen call: {e}", exc_info=True)
        return None


# --- Tool Function for the ADK Agent ---
@FunctionTool
async def generate_image_tool(topic: str) -> str: # Tool returns a Markdown string for display
    """
    Generates an image based on the given topic and saves it locally.
    Returns a Markdown string for display in the chat.
    """
    logger.info(f"📸 [TOOL CALLED] generate_image_tool received topic: '{topic}'")
    
    web_accessible_image_url = await generate_image_from_prompt(topic) 
    
    if web_accessible_image_url:
        # Return the complete Markdown string directly from the tool
        return (
            f"🖼️ Here is your image for '{topic}':\n\n"
            f"![Generated Image]({web_accessible_image_url})\n\n"
            f"You can download it [here]({web_accessible_image_url})"
        )
    else:
        # If generate_image_from_prompt returned None, indicate failure
        return "❌ Image generation failed. Please check the logs for details and ensure API key/permissions are correct."


# --- Define the ADK LlmAgent ---
root_agent = LlmAgent( # Name kept as 'sahayak_image_agent' as per request
    name="sahayak_image_agent", # Must match agent.yaml name in the sahayak folder
    instruction=(
        "You are an image generation assistant. Your primary task is to generate images "
        "based on user descriptions. "
        "**When the user requests an image, ALWAYS call the 'generate_image_tool' with the user's explicit topic.** " # STRONGER INSTRUCTION
        "**Prioritize calling the tool over responding with text if an image request is clear.** "
        "For example, if the user says 'create an image of a red car', you should call the tool "
        "with 'red car' as the topic. If they say 'show me a diagram of the human heart', call it with 'human heart diagram'. "
        "After the 'generate_image_tool' is executed, you will receive its direct output. "
        "You MUST respond to the user with the exact string returned by the tool, as it is already pre-formatted Markdown. " # CRITICAL: LLM instruction for output
        "If the tool returns an error message, you MUST inform the user about that specific error. "
        "For example, 'I'm sorry, I couldn't generate the image because: [error message]. Please check your configuration and try again.' "
        "Decline requests that are inappropriate or cannot be clearly visualized (e.g., abstract concepts). "
        "Do NOT try to describe or generate the image content in text yourself. "
        "Your responses should be concise and focused on the image generation process."
    ),
    model="gemini-1.5-flash-latest",
    tools=[
        generate_image_tool # Correctly pass the decorated function
    ],
    # REMOVED: before_model_callback=_force_image_tool_call_callback # <--- THIS LINE IS REMOVED
)