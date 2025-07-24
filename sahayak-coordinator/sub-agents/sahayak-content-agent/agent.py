"""
Sahayak Content Agent - Educational Content Generation

This agent specializes in:
1. Generating instructional text (lectures, explanations, examples)
2. Creating lesson plans and educational outlines
3. Handling direct teacher queries and prompts
4. Integrating with external tools via MCP (web search, knowledge bases)
5. Supporting multilingual content generation
6. Aligning content with curriculum standards
7. Triggering other agents when needed (ExamPaperAgent, etc.)
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import re

# Google ADK imports - proper implementation
from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

# Google Cloud imports
from google.cloud import firestore
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentType(Enum):
    LESSON_PLAN = "lesson_plan" 
    EXPLANATION = "explanation"
    LECTURE_NOTES = "lecture_notes"
    CONCEPT_BREAKDOWN = "concept_breakdown"
    EXAMPLE_SET = "example_set"
    STUDY_GUIDE = "study_guide"

class DifficultyLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class Language(Enum):
    ENGLISH = "en"
    HINDI = "hi"
    SPANISH = "es"
    FRENCH = "fr"

@dataclass
class ContentRequest:
    """Represents a content generation request"""
    prompt: str
    content_type: ContentType
    subject: Optional[str] = None
    difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    language: Language = Language.ENGLISH
    curriculum_standard: Optional[str] = None
    target_audience: Optional[str] = None
    additional_context: Dict[str, Any] = None

@dataclass
class GeneratedContent:
    """Represents generated educational content"""
    title: str
    content: str
    content_type: ContentType
    metadata: Dict[str, Any]
    requires_followup: bool = False
    suggested_agents: List[str] = None

class SahayakContentAgent(Agent):
    """
    Content Agent extending ADK Agent class
    Generates high-quality educational content using Gemini AI and MCP toolsets
    """
    
    def __init__(self):
        # Initialize parent ADK Agent
        super().__init__(
            name="sahayak-content-agent",
            description="Educational content generation specialist with curriculum alignment capabilities"
        )
        
        self.agent_id = os.getenv("CONTENT_AGENT_ID", "sahayak-content-agent")
        self.agent_name = os.getenv("AGENT_NAME", "ContentAgent")
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        
        # Initialize Google services
        self._init_google_services()
        
        # Initialize MCP toolsets for external tools
        self._init_mcp_toolsets()
        
        # Content generation templates
        self._init_content_templates()
        
        # Add tool interfaces for coordinator to call
        self.add_tool_interfaces()
        
        logger.info(f"Sahayak Content Agent {self.agent_id} initialized")

    def _init_google_services(self):
        """Initialize Google Cloud services"""
        try:
            # Configure Gemini AI
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.gemini_model = genai.GenerativeModel(
                os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
            )
            
            # Initialize Firestore for knowledge base
            self.firestore_client = firestore.Client(project=self.project_id)
            
            logger.info("Google services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google services: {e}")
            raise

    def _init_mcp_toolsets(self):
        """Initialize MCP toolsets for external tool communication"""
        try:
            # Initialize MCP toolset for Gemini server
            self.gemini_mcp = MCPToolset(
                server_params=StdioServerParameters(
                    command="python",
                    args=[
                        os.path.join(os.path.dirname(__file__), 
                                   "..", "mcp_servers", "gemini_server.py")
                    ],
                    env={
                        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
                        "GOOGLE_CLOUD_PROJECT_ID": self.project_id
                    }
                )
            )
            
            # Initialize MCP toolset for Firestore server
            self.firestore_mcp = MCPToolset(
                server_params=StdioServerParameters(
                    command="python",
                    args=[
                        os.path.join(os.path.dirname(__file__), 
                                   "..", "mcp_servers", "firestore_server.py")
                    ],
                    env={
                        "GOOGLE_CLOUD_PROJECT_ID": self.project_id
                    }
                )
            )
            
            # Initialize MCP toolset for Web Search server
            self.web_search_mcp = MCPToolset(
                server_params=StdioServerParameters(
                    command="python",
                    args=[
                        os.path.join(os.path.dirname(__file__), 
                                   "..", "mcp_servers", "web_search_server.py")
                    ],
                    env={
                        "GOOGLE_SEARCH_API_KEY": os.getenv("GOOGLE_SEARCH_API_KEY"),
                        "GOOGLE_SEARCH_ENGINE_ID": os.getenv("GOOGLE_SEARCH_ENGINE_ID")
                    }
                )
            )
            
            # Initialize MCP toolset for Curriculum server
            self.curriculum_mcp = MCPToolset(
                server_params=StdioServerParameters(
                    command="python",
                    args=[
                        os.path.join(os.path.dirname(__file__), 
                                   "..", "mcp_servers", "curriculum_server.py")
                    ],
                    env={}
                )
            )
            
            # Store all toolsets for easy access
            self.mcp_toolsets = {
                "gemini": self.gemini_mcp,
                "firestore": self.firestore_mcp,
                "web_search": self.web_search_mcp,
                "curriculum": self.curriculum_mcp
            }
            
            logger.info("MCP toolsets initialized successfully")
            
        except Exception as e:
            logger.error(f"MCP toolsets initialization failed: {e}")
            self.mcp_toolsets = {}

    def _init_content_templates(self):
        """Initialize content generation templates"""
        self.content_templates = {
            ContentType.LESSON_PLAN: {
                "structure": [
                    "Learning Objectives",
                    "Prerequisites", 
                    "Main Content",
                    "Activities & Examples",
                    "Assessment Ideas",
                    "Extension Activities"
                ],
                "prompt_template": """
                Create a comprehensive lesson plan for: {topic}
                
                Subject: {subject}
                Difficulty Level: {difficulty}
                Target Audience: {audience}
                Curriculum Standard: {standard}
                
                Structure the response with these sections:
                1. Learning Objectives (3-5 clear, measurable objectives)
                2. Prerequisites (what students should know beforehand)
                3. Main Content (detailed explanation of concepts)
                4. Activities & Examples (hands-on activities and real-world examples)
                5. Assessment Ideas (how to evaluate student understanding)
                6. Extension Activities (for advanced learners)
                
                Make it engaging, practical, and age-appropriate.
                Language: {language}
                """
            },
            
            ContentType.EXPLANATION: {
                "structure": [
                    "Concept Introduction",
                    "Detailed Explanation", 
                    "Examples",
                    "Common Misconceptions",
                    "Summary"
                ],
                "prompt_template": """
                Provide a clear, comprehensive explanation of: {topic}
                
                Subject: {subject}
                Difficulty Level: {difficulty}
                Context: {context}
                
                Structure your explanation with:
                1. Concept Introduction (what is this concept and why is it important)
                2. Detailed Explanation (break down the concept step by step)
                3. Examples (provide 2-3 relevant, relatable examples)
                4. Common Misconceptions (address typical student confusions)
                5. Summary (key takeaways in bullet points)
                
                Use analogies and metaphors where helpful.
                Language: {language}
                """
            },
            
            ContentType.CONCEPT_BREAKDOWN: {
                "structure": [
                    "Core Concepts",
                    "Relationships",
                    "Applications", 
                    "Practice Points"
                ],
                "prompt_template": """
                Break down this complex topic into digestible concepts: {topic}
                
                Subject: {subject}
                Level: {difficulty}
                
                Provide:
                1. Core Concepts (identify 3-5 main ideas)
                2. Relationships (how concepts connect to each other)
                3. Applications (real-world uses and relevance)
                4. Practice Points (what students should practice to master this)
                
                Use hierarchical structure and clear connections.
                Language: {language}
                """
            }
        }

    async def analyze_content_request(self, prompt: str, metadata: Dict = None) -> ContentRequest:
        """
        Analyze incoming prompt to determine content generation approach
        
        Args:
            prompt: Teacher's prompt/question
            metadata: Additional context from coordinator
            
        Returns:
            ContentRequest object with structured requirements
        """
        try:
            # Use Gemini to analyze the request
            analysis_prompt = f"""
            Analyze this educational content request and determine:
            1. What type of content is needed
            2. Subject matter
            3. Appropriate difficulty level
            4. Target audience
            5. Whether additional tools/resources are needed
            
            Request: {prompt}
            Context: {json.dumps(metadata or {})}
            
            Respond in JSON format:
            {{
                "content_type": "lesson_plan|explanation|lecture_notes|concept_breakdown|example_set|study_guide",
                "subject": "subject name",
                "difficulty": "beginner|intermediate|advanced", 
                "target_audience": "description of intended learners",
                "requires_web_search": true/false,
                "requires_knowledge_base": true/false,
                "estimated_length": "short|medium|long",
                "should_trigger_exam_agent": true/false,
                "key_concepts": ["concept1", "concept2", "concept3"],
                "curriculum_alignment": "standard name or null"
            }}
            """
            
            response = await self.gemini_model.generate_content_async(analysis_prompt)
            analysis = json.loads(response.text)
            
            # Create ContentRequest
            content_request = ContentRequest(
                prompt=prompt,
                content_type=ContentType(analysis.get("content_type", "explanation")),
                subject=analysis.get("subject"),
                difficulty=DifficultyLevel(analysis.get("difficulty", "intermediate")),
                curriculum_standard=analysis.get("curriculum_alignment"),
                target_audience=analysis.get("target_audience"),
                additional_context={
                    **(metadata or {}),
                    **analysis
                }
            )
            
            logger.info(f"Content request analyzed: {content_request.content_type.value}")
            return content_request
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Analysis parsing failed, using defaults: {e}")
            return ContentRequest(
                prompt=prompt,
                content_type=ContentType.EXPLANATION,
                additional_context=metadata or {}
            )

    async def generate_content(self, content_request: ContentRequest) -> GeneratedContent:
        """
        Generate educational content based on the request
        
        Args:
            content_request: Structured content generation request
            
        Returns:
            GeneratedContent with the generated educational material
        """
        try:
            # Get appropriate template
            template_info = self.content_templates.get(
                content_request.content_type,
                self.content_templates[ContentType.EXPLANATION]
            )
            
            # Gather additional context if needed
            additional_context = ""
            if content_request.additional_context.get("requires_web_search"):
                additional_context += await self._web_search_context(content_request)
            
            if content_request.additional_context.get("requires_knowledge_base"):
                additional_context += await self._knowledge_base_context(content_request)
            
            # Format the generation prompt
            generation_prompt = template_info["prompt_template"].format(
                topic=content_request.prompt,
                subject=content_request.subject or "General",
                difficulty=content_request.difficulty.value,
                audience=content_request.target_audience or "Students",
                standard=content_request.curriculum_standard or "Not specified",
                language=content_request.language.value,
                context=additional_context
            )
            
            # Add curriculum-specific guidance
            if content_request.curriculum_standard:
                generation_prompt += f"\n\nIMPORTANT: Align content with {content_request.curriculum_standard} standards and requirements."
            
            # Add multilingual support
            if content_request.language != Language.ENGLISH:
                generation_prompt += f"\n\nGenerate all content in {content_request.language.value} language, maintaining educational quality and cultural relevance."
            
            # Generate content using Gemini MCP toolset
            gemini_toolset = self.mcp_toolsets.get("gemini")
            if gemini_toolset:
                gemini_result = await gemini_toolset.call_tool(
                    "generate_educational_content",
                    {
                        "prompt": generation_prompt,
                        "content_type": content_request.content_type.value,
                        "subject": content_request.subject or "General",
                        "difficulty": content_request.difficulty.value,
                        "language": content_request.language.value,
                        "curriculum_standard": content_request.curriculum_standard or ""
                    }
                )
                
                if gemini_result.get("status") == "success":
                    generated_text = gemini_result.get("content", "")
                else:
                    # Fallback to direct Gemini call
                    response = await self.gemini_model.generate_content_async(generation_prompt)
                    generated_text = response.text
            else:
                # Fallback to direct Gemini call
                response = await self.gemini_model.generate_content_async(generation_prompt)
                generated_text = response.text
            
            # Extract title from content
            title = self._extract_title(generated_text, content_request.prompt)
            
            # Determine if followup agents are needed
            requires_followup = content_request.additional_context.get("should_trigger_exam_agent", False)
            suggested_agents = []
            
            if requires_followup:
                suggested_agents.append("exam_agent")
            
            if content_request.additional_context.get("requires_visual", False):
                suggested_agents.append("art_agent")
            
            # Create GeneratedContent
            generated_content = GeneratedContent(
                title=title,
                content=generated_text,
                content_type=content_request.content_type,
                metadata={
                    "subject": content_request.subject,
                    "difficulty": content_request.difficulty.value,
                    "language": content_request.language.value,
                    "curriculum_standard": content_request.curriculum_standard,
                    "word_count": len(generated_text.split()),
                    "generation_timestamp": asyncio.get_event_loop().time()
                },
                requires_followup=requires_followup,
                suggested_agents=suggested_agents
            )
            
            logger.info(f"Content generated successfully: {title[:50]}...")
            return generated_content
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            raise

    async def _web_search_context(self, content_request: ContentRequest) -> str:
        """Gather additional context via web search MCP toolset"""
        try:
            web_search_toolset = self.mcp_toolsets.get("web_search")
            if not web_search_toolset:
                logger.warning("Web search toolset not available")
                return f"\n\nAdditional Context: Latest information about {content_request.subject}\n"
            
            # Search for educational content related to the topic
            search_result = await web_search_toolset.call_tool(
                "search_educational_content",
                {
                    "query": content_request.prompt,
                    "subject": content_request.subject,
                    "content_type": "articles",
                    "trusted_only": True,
                    "max_results": 5
                }
            )
            
            if search_result.get("status") == "success":
                results = search_result.get("results", [])
                context = "\n\nWeb Search Context:\n"
                for result in results[:3]:  # Top 3 results
                    context += f"- {result.get('title', '')}: {result.get('snippet', '')}\n"
                return context
            else:
                logger.warning(f"Web search failed: {search_result.get('error', 'Unknown error')}")
                return f"\n\nAdditional Context: Latest information about {content_request.subject}\n"
                
        except Exception as e:
            logger.error(f"Web search context gathering failed: {e}")
            return f"\n\nAdditional Context: Latest information about {content_request.subject}\n"

    async def _knowledge_base_context(self, content_request: ContentRequest) -> str:
        """Gather context from knowledge base via Firestore MCP toolset"""
        try:
            firestore_toolset = self.mcp_toolsets.get("firestore")
            if not firestore_toolset:
                logger.warning("Firestore toolset not available")
                return ""
            
            # Search for relevant educational resources
            search_result = await firestore_toolset.call_tool(
                "search_educational_resources",
                {
                    "query": content_request.prompt,
                    "subject": content_request.subject,
                    "difficulty_level": content_request.difficulty.value,
                    "limit": 3
                }
            )
            
            if search_result.get("status") == "success":
                results = search_result.get("results", [])
                context = "\n\nKnowledge Base Context:\n"
                for result in results:
                    context += f"- {result.get('title', 'Resource')}: {result.get('summary', 'No summary')}\n"
                return context
            else:
                logger.warning(f"Knowledge base search failed: {search_result.get('error', 'Unknown error')}")
                return ""
                
        except Exception as e:
            logger.error(f"Knowledge base context gathering failed: {e}")
            return ""

    def _extract_title(self, content: str, prompt: str) -> str:
        """Extract or generate appropriate title for the content"""
        # Try to find title in first line
        lines = content.split('\n')
        first_line = lines[0].strip()
        
        # Check if first line looks like a title
        if len(first_line) < 100 and not first_line.endswith('.'):
            return first_line.replace('#', '').strip()
        
        # Generate title from prompt
        words = prompt.split()[:8]  # First 8 words
        title = ' '.join(words)
        
        if len(title) > 60:
            title = title[:60] + "..."
            
        return title

    async def enhance_content_with_tools(self, content: GeneratedContent) -> GeneratedContent:
        """
        Enhance generated content with additional tools and resources
        
        Args:
            content: Generated content to enhance
            
        Returns:
            Enhanced GeneratedContent
        """
        try:
            enhanced_content = content.content
            
            # Add relevant diagrams/images suggestions
            if content.metadata.get("subject") in ["Science", "Mathematics", "Biology", "Physics", "Chemistry"]:
                image_suggestions = await self._suggest_visuals(content)
                if image_suggestions:
                    enhanced_content += f"\n\n## Suggested Visuals:\n{image_suggestions}"
            
            # Add interactive elements suggestions
            interactive_suggestions = await self._suggest_activities(content)
            if interactive_suggestions:
                enhanced_content += f"\n\n## Interactive Activities:\n{interactive_suggestions}"
            
            # Update content
            content.content = enhanced_content
            content.metadata["enhanced"] = True
            content.metadata["enhancement_timestamp"] = asyncio.get_event_loop().time()
            
            logger.info("Content enhanced with additional tools")
            return content
            
        except Exception as e:
            logger.error(f"Content enhancement failed: {e}")
            return content

    async def _suggest_visuals(self, content: GeneratedContent) -> str:
        """Suggest appropriate visuals for the content"""
        try:
            visual_prompt = f"""
            Based on this educational content about {content.metadata.get('subject')}, 
            suggest 3-5 specific visual aids that would enhance learning:
            
            Content Preview: {content.content[:500]}...
            
            For each visual, specify:
            1. Type (diagram, chart, illustration, etc.)
            2. Description of what it should show
            3. Educational purpose
            
            Format as a bulleted list.
            """
            
            response = await self.gemini_model.generate_content_async(visual_prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Visual suggestion failed: {e}")
            return ""

    async def _suggest_activities(self, content: GeneratedContent) -> str:
        """Suggest interactive activities for the content"""
        try:
            activity_prompt = f"""
            Create 3-4 interactive learning activities for this content:
            
            Subject: {content.metadata.get('subject')}
            Difficulty: {content.metadata.get('difficulty')}
            Content Preview: {content.content[:500]}...
            
            Activities should be:
            1. Hands-on and engaging
            2. Appropriate for the difficulty level
            3. Reinforce key concepts
            4. Suitable for classroom or individual use
            
            Format each activity with: Title, Description, Materials needed, Time required
            """
            
            response = await self.gemini_model.generate_content_async(activity_prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Activity suggestion failed: {e}")
            return ""

    def add_tool_interfaces(self):
        """Add tool interfaces that the coordinator can call via MCP"""
        # Register the main content generation tool
        self.add_tool(
            name="generate_educational_content",
            description="Generate educational content including lesson plans, explanations, and study materials",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Content generation prompt from teacher"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional metadata including subject, difficulty, curriculum standard"
                    }
                },
                "required": ["prompt"]
            },
            handler=self.process_request
        )
        
        # Register content analysis tool
        self.add_tool(
            name="analyze_content_request",
            description="Analyze teacher input to determine content generation approach",
            parameters={
                "type": "object", 
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Teacher's prompt/question to analyze"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional context from coordinator"
                    }
                },
                "required": ["prompt"]
            },
            handler=self.analyze_content_request
        )

    async def process_request(self, prompt: str, metadata: Dict = None) -> Dict[str, Any]:
        """
        Main processing method for content generation requests
        
        Args:
            prompt: Teacher's prompt/request
            metadata: Additional context from coordinator
            
        Returns:
            Structured response with generated content
        """
        try:
            # Step 1: Analyze the request
            content_request = await self.analyze_content_request(prompt, metadata)
            
            # Step 2: Generate content
            generated_content = await self.generate_content(content_request)
            
            # Step 3: Enhance with tools if enabled
            if os.getenv("ENABLE_CONTENT_ENHANCEMENT", "true").lower() == "true":
                generated_content = await self.enhance_content_with_tools(generated_content)
            
            # Step 4: Format response
            response = {
                "status": "success",
                "agent": self.agent_name,
                "content": {
                    "title": generated_content.title,
                    "body": generated_content.content,
                    "type": generated_content.content_type.value,
                    "metadata": generated_content.metadata
                },
                "requires_followup": generated_content.requires_followup,
                "suggested_agents": generated_content.suggested_agents or [],
                "tools_used": self._get_used_tools()
            }
            
            logger.info(f"Content request processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Content processing failed: {e}")
            return {
                "status": "error",
                "agent": self.agent_name,
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }

    def _get_used_tools(self) -> List[str]:
        """Get list of MCP toolsets used during content generation"""
        used_tools = []
        
        # Check which MCP toolsets are available and potentially used
        for tool_name, toolset in self.mcp_toolsets.items():
            if toolset:
                used_tools.append(f"mcp_{tool_name}")
        
        # Always include direct Gemini and Firestore as fallbacks
        used_tools.extend(["gemini_ai_direct", "firestore_direct"])
        
        return used_tools

async def main():
    """Main entry point for running Content Agent as MCP server"""
    try:
        # Initialize and run the content agent
        agent = SahayakContentAgent()
        
        # Start the agent's MCP server
        await agent.run()
        
    except Exception as e:
        logger.error(f"Content Agent startup failed: {e}")
        raise

# Global instance for ADK Agent integration
sahayak_content_agent = SahayakContentAgent()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())