
import logging
from mcp.server import Server
from mcp.types import Resource, Tool, TextContent
from typing import List
from PIL import Image
import pytesseract
from io import BytesIO
import pdfplumber
import base64
import os
from fpdf import FPDF
from urllib.parse import quote
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))

logger = logging.getLogger("worksheet-server")
logging.basicConfig(level=logging.INFO)

STATIC_FOLDER = "static"
os.makedirs(STATIC_FOLDER, exist_ok=True)

def save_questions_as_pdf(title: str, content: str, filename: str) -> str:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    for line in content.strip().split('\n'):
        pdf.multi_cell(0, 10, line)
    path = os.path.join(STATIC_FOLDER, filename)
    pdf.output(path)
    return path

def detect_custom_pattern(content: str) -> str:
    content = content.lower()
    if "only mcq" in content or "only multiple choice" in content:
        return "mcq_only"
    elif "only descriptive" in content:
        return "descriptive_only"
    elif "only hard" in content:
        return "hard_only"
    elif "only easy" in content:
        return "easy_only"
    elif "only medium" in content:
        return "medium_only"
    return "default"

async def extract_content_from_upload(file_content: str, file_type: str) -> str:
    try:
        if file_type == "application/pdf":
            pdf_bytes = base64.b64decode(file_content)
            with BytesIO(pdf_bytes) as f:
                with pdfplumber.open(f) as pdf:
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        elif file_type.startswith("image/"):
            image_bytes = base64.b64decode(file_content)
            image = Image.open(BytesIO(image_bytes))
            text = pytesseract.image_to_string(image)
        else:
            text = file_content
            print("status : extracting content success","result :",text)
        return text.strip() or "No text found in the uploaded file."
    except Exception as e:
        logger.error(f"Error extracting content: {e}")
        return "Error: Unable to extract content."

async def generate_educational_materials(
    content: str,
    subject: str = "",
    grade_level: str = "",
    curriculum: str = "",
    output_format: str = "both",
    language: str = "English"
) -> str:
    """Generate comprehensive educational materials using the full pipeline from agent.py"""
    try:
        if not content.strip() or content.startswith("Error"):
            return "❌ No valid educational content provided."

        # Import the full function from the implementation file
        import sys
        import os
        print("[Inside generate_educational_materials]")
        # Add the parent directory to the path to import from agent_full_implementation
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)
        
        from agent_full_implementation import generate_educational_materials as full_generate_materials
        
        # Call the comprehensive generation function
        result = await full_generate_materials(
            content=content,
            subject=subject,
            grade_level=grade_level,
            curriculum=curriculum,
            output_format=output_format,
            save_to_storage=True
        )
        
        if result.get("status") == "success":
            # Format the response for the user
            response_parts = []
            response_parts.append(f"📚 Educational materials generated successfully!")
            
            # Add content analysis info
            content_analysis = result.get("content_analysis", {})
            if content_analysis:
                topic = content_analysis.get("topic", "Educational Content")
                subject = content_analysis.get("subject", subject)
                response_parts.append(f"📖 **Topic:** {topic}")
                response_parts.append(f"🎯 **Subject:** {subject}")
            
            # Add download links
            download_links = result.get("download_links", {})
            if download_links:
                response_parts.append("\n📥 **Download Links:**")
                
                for file_key, link_info in download_links.items():
                    filename = link_info.get("filename", "")
                    download_url = link_info.get("download_url", "")
                    file_type = link_info.get("file_type", "")
                    
                    # Debug the URL to see if bucket name is being escaped
                    print(f"[URL_DEBUG] Original URL: {download_url}")
                    
                    # Clean any escaped underscores in the URL
                    clean_url = download_url.replace("agentic\\_storage", "agentic_storage").replace("%5C_", "_")
                    print(f"[URL_DEBUG] Cleaned URL: {clean_url}")
                    
                    if file_type == "presentation":
                        response_parts.append(f"📊 **{filename}** - [Download Slides]({clean_url})")
                    elif file_type == "worksheet":
                        difficulty = filename.split("_")[-2] if "_" in filename else "worksheet"
                        response_parts.append(f"📄 **{filename} ({difficulty.title()} Level)** - [Download PDF]({clean_url})")
            
            # Add metadata
            metadata = result.get("metadata", {})
            files_generated = metadata.get("files_generated", [])
            if files_generated:
                response_parts.append(f"\n✅ Generated {len(files_generated)} files: {', '.join(files_generated)}")
            
            return "\n".join(response_parts)
        else:
            error_message = result.get("error_message", "Unknown error occurred")
            return f"❌ Error generating materials: {error_message}"
            
    except Exception as e:
        logger.error(f"Educational materials generation failed: {e}")
        return f"❌ Error: Failed to generate educational materials: {str(e)}"

class WorksheetServer:
    def __init__(self):
        self.server = Server("worksheet-server")
        self._setup_handlers()

    def _setup_handlers(self):
        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            return [Resource(uri="worksheet://upload", name="Worksheet Upload", description="Upload a file/image", mimeType="application/octet-stream")]

        @self.server.list_tools()
        async def list_tools(self) -> List[Tool]:
            return [
                Tool(
                    name="extract_content_from_upload",
                    description="Extracts text from uploaded file or image",
                    inputSchema={"type": "object", "properties": {
                        "file_content": {"type": "string"},
                        "file_type": {"type": "string"}
                    }, "required": ["file_content", "file_type"]}
                ),
                Tool(
                    name="generate_educational_materials",
                    description="Generates comprehensive educational materials including slides and worksheets.",
                    inputSchema={"type": "object", "properties": {
                        "content": {"type": "string"},
                        "subject": {"type": "string", "default": ""},
                        "grade_level": {"type": "string", "default": ""},
                        "curriculum": {"type": "string", "default": ""},
                        "output_format": {"type": "string", "default": "both"},
                        "language": {"type": "string", "default": "English"}
                    }, "required": ["content"]}
                )
            ]

        @self.server.call_tool()
        async def call_tool(self, name: str, arguments: dict) -> List[TextContent]:
            if name == "extract_content_from_upload":
                result = await extract_content_from_upload(arguments["file_content"], arguments["file_type"])
                return [TextContent(type="text", text=result)]
            elif name == "generate_educational_materials":
                result = await generate_educational_materials(
                    content=arguments["content"],
                    subject=arguments.get("subject", ""),
                    grade_level=arguments.get("grade_level", ""),
                    curriculum=arguments.get("curriculum", "CBSE"),
                    output_format=arguments.get("output_format", "both"),
                    language=arguments.get("language", "English")
                )
                return [TextContent(type="text", text=result)]
            else:
                raise ValueError(f"Unknown tool: {name}")

if __name__ == "__main__":
    server = WorksheetServer()
    import asyncio
    asyncio.run(server.server.serve())
