# # server.py (Place this in the root 'image_generation_adk_agent' folder)

# import os
# import asyncio
# import logging
# from dotenv import load_dotenv

# # ADK Server and Session Service
# from mcp.server import NotificationOptions, Server
# from google.adk.sessions import DatabaseSessionService

# # Vertex AI initialization (REQUIRED if your image generation tool uses Vertex AI models like Imagen-2)
# from vertexai import init as vertexai_init

# # Import your agent definition from the 'agent' package
# # Note: When running this file directly, it will load the agent.
# # The ADK Web server itself will then discover it.
# from agent.image_agent import image_generator_agent

# # --- Configure Logging ---
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("adk-server-startup")

# # --- Load Environment Variables ---
# load_dotenv()

# # --- Initialize Vertex AI (for Imagen-2 or other Vertex AI models) ---
# # This must be done BEFORE any Vertex AI models are loaded or used.
# project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
# if project_id:
#     try:
#         vertexai_init(project=project_id, location="us-central1") # Use a region where Imagen is available
#         logger.info(f"Vertex AI initialized for project: {project_id}, location: us-central1")
#     except Exception as e:
#         logger.error(f"Failed to initialize Vertex AI: {e}. "
#                      "Ensure GOOGLE_CLOUD_PROJECT_ID is correct and your service account "
#                      "has 'Vertex AI User' and 'Service Usage Consumer' roles. "
#                      "Also, check if GOOGLE_APPLICATION_CREDENTIALS points to a valid JSON key.")
# else:
#     logger.error("GOOGLE_CLOUD_PROJECT_ID not found in environment. Vertex AI will not be initialized. "
#                  "Image generation via Vertex AI will fail.")

# # --- Database Session Service ---
# db_url = os.getenv("DB_URL", "sqlite:///adk_image_agent_sessions.db")
# session_service = DatabaseSessionService(db_url=db_url)
# logger.info(f"Using database for sessions: {db_url}")

# # --- Initialize the ADK Server ---
# # Register your agent(s) explicitly here for the server to pick them up.
# adk_server = Server(
#     name="ImageGenerationAgent", # This is the name that will appear in the ADK Web UI
#     session_service=session_service,
#     agents=[image_generator_agent], # List your agent(s) here
# )

# # --- Main function to run the ADK server ---
# async def main():
#     logger.info("Starting Google ADK Web Server for Image Generation Agent...")
#     await adk_server.serve()

# if __name__ == "__main__":
#     asyncio.run(main()) # Run the main async function


# server.py (Place this in the root 'image_generation_adk_agent' folder)

import os
import asyncio
import logging
from dotenv import load_dotenv

# ADK Server and Session Service
from google.adk.server import Server
from google.adk.sessions import DatabaseSessionService

# Import your agent definition from the 'agent' package
from agent.image_agent import image_generator_agent

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("adk-server-startup")

# --- Load Environment Variables ---
load_dotenv()

# --- NO VERTEX AI INITIALIZATION BLOCK HERE (as per last iteration) ---
# Assuming your agent and its tools exclusively use Generative Language API with GOOGLE_API_KEY.

# --- Database Session Service (still needed for session management, but handled differently by Server) ---
db_url = os.getenv("DB_URL", "sqlite:///adk_image_agent_sessions.db")
session_service = DatabaseService(db_url=db_url) # Using DatabaseService, not DatabaseSessionService directly in Server
logger.info(f"Using database for sessions: {db_url}")


# --- Initialize the ADK Server ---
# REMOVE 'session_service=session_service' from here
adk_server = Server(
    name="ImageGenerationAgent", # This is the name that will appear in the ADK Web UI
    agents=[image_generator_agent], # List your agent(s) here
    # The ADK Server automatically serves content from a 'static' folder at the root.
)

# --- Expose the FastAPI app for Uvicorn ---
app = adk_server.app # This is the FastAPI app instance that Uvicorn will run