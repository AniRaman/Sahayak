
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
        return text.strip() or "No text found in the uploaded file."
    except Exception as e:
        logger.error(f"Error extracting content: {e}")
        return "Error: Unable to extract content."

async def generate_questions_by_difficulty(content: str, language: str = "English") -> str:
    try:
        if not content.strip() or content.startswith("Error"):
            return "❌ No valid educational content provided."

        model = genai.GenerativeModel("gemini-2.0-flash")
        pattern = detect_custom_pattern(content)

        if pattern == "mcq_only":
            pattern_instruction = "Generate 10 multiple choice questions (MCQs)."
        elif pattern == "descriptive_only":
            pattern_instruction = "Generate 10 descriptive (open-ended) questions."
        elif pattern == "hard_only":
            pattern_instruction = "Generate 5 hard level MCQs and 3 hard descriptive questions."
        elif pattern == "easy_only":
            pattern_instruction = "Generate 5 easy level MCQs and 3 easy descriptive questions."
        elif pattern == "medium_only":
            pattern_instruction = "Generate 5 medium level MCQs and 3 medium descriptive questions."
        else:
            pattern_instruction = (
                "For each of the 3 difficulty levels (Easy, Medium, Hard), in the specified language:\n"
                "- Generate 5 MCQs\n- Generate 3 Descriptive Questions\n"
                "Label each section clearly."
            )

        prompt = (
    f"You are an academic assistant.\n"
    f"Using the following content, generate an assignment in {language}.\n\n"
    f"The assignment must be presented in plain text format.\n"
    f"DO NOT use JSON or any structured data format like XML or YAML.\n"
    f"Instead, format like a printed assignment paper with sections and questions.\n\n"
    f"{pattern_instruction}\n\n"
    f"CONTENT:\n\"\"\"\n{content[:4000]}\n\"\"\""
)


        response = model.generate_content(prompt)
        output = response.text.strip()

        filename = f"worksheet_assignment_{language.lower()}.pdf"
        save_questions_as_pdf("Assignment Worksheet", output, filename)

        url = f"http://localhost:8000/static/{quote(filename)}"
        return f"📘 Assignment generated successfully in {language}.\n\n📄 [Download PDF Assignment]({url})\n\n{output}"

    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return "Error: Failed to generate assignment questions."

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
                    name="generate_questions_by_difficulty",
                    description="Generates assignment questions and a downloadable PDF.",
                    inputSchema={"type": "object", "properties": {
                        "content": {"type": "string"},
                        "language": {"type": "string", "default": "English"}
                    }, "required": ["content"]}
                )
            ]

        @self.server.call_tool()
        async def call_tool(self, name: str, arguments: dict) -> List[TextContent]:
            if name == "extract_content_from_upload":
                result = await extract_content_from_upload(arguments["file_content"], arguments["file_type"])
                return [TextContent(type="text", text=result)]
            elif name == "generate_questions_by_difficulty":
                result = await generate_questions_by_difficulty(arguments["content"], arguments.get("language", "English"))
                return [TextContent(type="text", text=result)]
            else:
                raise ValueError(f"Unknown tool: {name}")

if __name__ == "__main__":
    server = WorksheetServer()
    import asyncio
    asyncio.run(server.server.serve())
