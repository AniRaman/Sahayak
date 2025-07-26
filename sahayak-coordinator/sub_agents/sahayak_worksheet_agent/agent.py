"""
Sahayak Worksheet Agent - Student Material Generator

This agent transforms educational content into student-ready materials:
1. Processes images using Google Vision API to extract text
2. Analyzes textbook/lesson content 
3. Generates presentation slides (PPT files)
4. Creates differentiated worksheets (PDF files) at easy/medium/hard levels
5. Saves files to Google Cloud Storage
6. Stores metadata in Firestore for easy retrieval
7. Provides download links to users

Can be used standalone by users or called by contentAgent via sequential pipeline.
"""
import logging
from google.adk.agents import Agent
from .mcp_servers.worksheet_server import extract_content_from_upload, generate_educational_materials

logging.basicConfig(level=logging.INFO)

def detect_language_from_query(query: str) -> str:
    query = query.lower()
    if "telugu" in query: return "Telugu"
    if "hindi" in query: return "Hindi"
    if "tamil" in query: return "Tamil"
    if "kannada" in query: return "Kannada"
    if "french" in query: return "French"
    if "german" in query: return "German"
    if "spanish" in query: return "Spanish"
    return "English"

async def generate_educational_materials_with_language(arguments: dict) -> str:
    content = arguments.get("content", "")
    subject = arguments.get("subject", "")
    grade_level = arguments.get("grade_level", "")
    curriculum = arguments.get("curriculum", "CBSE")
    output_format = arguments.get("output_format", "both")
    language = detect_language_from_query(content)
    
    return await generate_educational_materials(
        content, subject, grade_level, curriculum, output_format, language
    )

root_agent = Agent(
    name="sahayak_worksheet_agent",
    model="gemini-2.0-flash",
    description="Worksheet agent that creates educational materials from uploaded content.",
    instruction=(
        "Always start by asking please provide the file with content, subject, grade_level, curriculum, output_format, language.\n"
        "Use uploaded files or provided content to generate educational materials.\n"
        "First call extract_content_from_upload to extract text from uploaded files.\n"
        "Then call generate_educational_materials_with_language to create comprehensive materials.\n"
        "Generate presentation slides and differentiated worksheets (Easy, Medium, Hard levels).\n"
        "Always extract content from files before generating materials."
    ),
    tools=[
        extract_content_from_upload,
        generate_educational_materials_with_language,
    ]
)

if __name__ == "__main__":
    print(f"📝 Agent {root_agent.name} initialized successfully.")
    for i, tool in enumerate(getattr(root_agent, 'tools', []), 1):
        print(f"  {i}. {tool.__name__}")