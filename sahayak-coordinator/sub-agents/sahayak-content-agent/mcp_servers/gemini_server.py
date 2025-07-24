#!/usr/bin/env python3
"""
Gemini AI MCP Server for Content Generation
Provides AI-powered content generation capabilities via MCP protocol
"""

import asyncio
import json
import logging
from typing import Any, Sequence
import os

import google.generativeai as genai
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini-server")

class GeminiServer:
    def __init__(self):
        self.server = Server("gemini-content-server")
        self.model = None
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_resources()
        async def handle_list_resources() -> list[Resource]:
            """List available Gemini AI resources"""
            return [
                Resource(
                    uri="gemini://models",
                    name="Available Gemini Models",
                    description="List of available Gemini AI models",
                    mimeType="application/json",
                ),
                Resource(
                    uri="gemini://templates/education",
                    name="Educational Content Templates",
                    description="Pre-built templates for educational content generation",
                    mimeType="application/json",
                )
            ]

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read Gemini AI resources"""
            if uri == "gemini://models":
                models = [
                    "gemini-1.5-pro",
                    "gemini-1.5-flash", 
                    "gemini-pro"
                ]
                return json.dumps({"available_models": models})
                
            elif uri == "gemini://templates/education":
                templates = {
                    "lesson_plan": {
                        "structure": ["objectives", "prerequisites", "content", "activities", "assessment"],
                        "prompt_template": "Create a comprehensive lesson plan for {topic} targeting {audience}"
                    },
                    "explanation": {
                        "structure": ["introduction", "explanation", "examples", "summary"],
                        "prompt_template": "Explain {concept} clearly for {difficulty_level} learners"
                    },
                    "study_guide": {
                        "structure": ["key_concepts", "important_facts", "practice_questions", "resources"],
                        "prompt_template": "Create a study guide for {subject} covering {topics}"
                    }
                }
                return json.dumps(templates)
                
            else:
                raise ValueError(f"Unknown resource: {uri}")

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available Gemini tools"""
            return [
                Tool(
                    name="generate_educational_content",
                    description="Generate educational content using Gemini AI",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Content generation prompt"
                            },
                            "content_type": {
                                "type": "string", 
                                "enum": ["lesson_plan", "explanation", "study_guide", "lecture_notes", "concept_breakdown"],
                                "description": "Type of educational content to generate"
                            },
                            "subject": {
                                "type": "string",
                                "description": "Subject area (e.g., Mathematics, Science, History)"
                            },
                            "difficulty": {
                                "type": "string",
                                "enum": ["beginner", "intermediate", "advanced"],
                                "description": "Difficulty level for the content"
                            },
                            "language": {
                                "type": "string",
                                "default": "english",
                                "description": "Language for content generation"
                            },
                            "curriculum_standard": {
                                "type": "string",
                                "description": "Educational curriculum standard to align with"
                            }
                        },
                        "required": ["prompt", "content_type"]
                    },
                ),
                Tool(
                    name="analyze_content_quality",
                    description="Analyze and provide feedback on educational content quality",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Educational content to analyze"
                            },
                            "criteria": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Quality criteria to evaluate (clarity, accuracy, engagement, etc.)"
                            }
                        },
                        "required": ["content"]
                    },
                ),
                Tool(
                    name="suggest_improvements",
                    description="Suggest improvements for educational content",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Educational content to improve"
                            },
                            "target_audience": {
                                "type": "string",
                                "description": "Target audience for the content"
                            },
                            "improvement_focus": {
                                "type": "string",
                                "enum": ["clarity", "engagement", "depth", "accessibility", "structure"],
                                "description": "Specific area to focus improvements on"
                            }
                        },
                        "required": ["content"]
                    },
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent | EmbeddedResource]:
            """Handle tool calls"""
            if name == "generate_educational_content":
                return await self._generate_content(arguments)
            elif name == "analyze_content_quality":
                return await self._analyze_quality(arguments)
            elif name == "suggest_improvements":
                return await self._suggest_improvements(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _generate_content(self, args: dict) -> list[TextContent]:
        """Generate educational content using Gemini"""
        try:
            prompt = args["prompt"]
            content_type = args["content_type"]
            subject = args.get("subject", "General")
            difficulty = args.get("difficulty", "intermediate")
            language = args.get("language", "english")
            curriculum = args.get("curriculum_standard", "")
            
            # Build comprehensive prompt
            system_prompt = f"""
            You are an expert educational content creator. Generate high-quality {content_type} content.
            
            Subject: {subject}
            Difficulty Level: {difficulty}
            Language: {language}
            {f"Curriculum Standard: {curriculum}" if curriculum else ""}
            
            Requirements:
            1. Content should be pedagogically sound and age-appropriate
            2. Use clear, engaging language suitable for the difficulty level
            3. Include practical examples and real-world applications
            4. Structure content logically with proper headings
            5. Ensure accuracy and current information
            
            Content Request: {prompt}
            """
            
            # Generate content using Gemini
            response = await self.model.generate_content_async(system_prompt)
            generated_text = response.text
            
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "content": generated_text,
                        "metadata": {
                            "content_type": content_type,
                            "subject": subject,
                            "difficulty": difficulty,
                            "language": language,
                            "word_count": len(generated_text.split()),
                            "model_used": "gemini-1.5-pro"
                        }
                    }, indent=2)
                )
            ]
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "error": str(e)
                    })
                )
            ]
    
    async def _analyze_quality(self, args: dict) -> list[TextContent]:
        """Analyze content quality using Gemini"""
        try:
            content = args["content"]
            criteria = args.get("criteria", ["clarity", "accuracy", "engagement", "structure"])
            
            analysis_prompt = f"""
            Analyze this educational content for quality based on these criteria: {', '.join(criteria)}
            
            Content to analyze:
            {content}
            
            For each criterion, provide:
            1. Score (1-10)
            2. Specific feedback
            3. Areas for improvement
            
            Provide overall assessment with actionable recommendations.
            """
            
            response = await self.model.generate_content_async(analysis_prompt)
            analysis_result = response.text
            
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "analysis": analysis_result,
                        "criteria_evaluated": criteria
                    }, indent=2)
                )
            ]
            
        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "error": str(e)
                    })
                )
            ]
    
    async def _suggest_improvements(self, args: dict) -> list[TextContent]:
        """Suggest content improvements using Gemini"""
        try:
            content = args["content"]
            target_audience = args.get("target_audience", "students")
            focus = args.get("improvement_focus", "overall")
            
            improvement_prompt = f"""
            Suggest specific improvements for this educational content.
            
            Target Audience: {target_audience}
            Improvement Focus: {focus}
            
            Current content:
            {content}
            
            Provide:
            1. Specific improvement suggestions
            2. Rewritten sections if needed
            3. Additional elements to consider
            4. Overall enhancement strategy
            """
            
            response = await self.model.generate_content_async(improvement_prompt)
            suggestions = response.text
            
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "suggestions": suggestions,
                        "focus_area": focus
                    }, indent=2)
                )
            ]
            
        except Exception as e:
            logger.error(f"Improvement suggestions failed: {e}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "error": str(e)
                    })
                )
            ]

    async def run(self):
        """Initialize and run the Gemini MCP server"""
        # Initialize Gemini AI
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable not set")
            raise ValueError("Gemini API key required")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-pro")
        
        logger.info("Gemini MCP Server initialized successfully")
        
        # Run server
        async with self.server.request_context():
            await self.server.run()

async def main():
    """Main entry point"""
    server = GeminiServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())