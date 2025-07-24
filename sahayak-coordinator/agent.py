"""
Sahayak Coordinator Agent - Central Educational AI Orchestrator

This agent serves as the central coordinator that:
1. Receives teacher input (text/image/PDF) 
2. Routes requests to appropriate specialized agents
3. Manages workflow and agent permissions
4. Aggregates outputs from multiple agents
5. Handles personalization and curriculum alignment
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Google ADK imports - proper implementation
from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

# Google Cloud imports
from google.cloud import firestore
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InputType(Enum):
    TEXT_CHAT = "text_chat"
    IMAGE_PDF = "image_pdf"
    UNKNOWN = "unknown"

class AgentType(Enum):
    CONTENT_AGENT = "content_agent"
    WORKSHEET_AGENT = "worksheet_agent" 
    ART_AGENT = "art_agent"
    QA_AGENT = "qa_agent"
    EXAM_AGENT = "exam_agent"

@dataclass
class TeacherInput:
    """Represents input from a teacher"""
    content: str
    input_type: InputType
    metadata: Dict[str, Any]
    teacher_id: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None

@dataclass
class AgentTask:
    """Represents a task to be executed by an agent"""
    agent_type: AgentType
    task_data: Dict[str, Any]
    priority: int = 1
    dependencies: List[str] = None

class SahayakCoordinatorAgent(Agent):
    """
    Central coordinator extending ADK Agent class
    Manages multi-agent educational content pipeline using MCP toolsets
    """
    
    def __init__(self):
        # Initialize parent ADK Agent
        super().__init__(
            name="sahayak-coordinator",
            description="Central orchestrator for educational content generation pipeline"
        )
        
        self.agent_id = os.getenv("COORDINATOR_AGENT_ID", "sahayak-coordinator")
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        
        # Initialize Google services
        self._init_google_services()
        
        # Initialize MCP toolsets for sub-agents
        self._init_mcp_toolsets()
        
        # Agent registry and sessions
        self.registered_agents = {}
        self.active_sessions = {}
        
        logger.info(f"Sahayak Coordinator Agent {self.agent_id} initialized")

    def _init_google_services(self):
        """Initialize Google Cloud services"""
        try:
            # Configure Gemini AI
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.gemini_model = genai.GenerativeModel(
                os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
            )
            
            # Initialize Firestore for persistent memory
            self.firestore_client = firestore.Client(project=self.project_id)
            
            logger.info("Google services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google services: {e}")
            raise

    def _init_mcp_toolsets(self):
        """Initialize MCP toolsets for sub-agent communication"""
        try:
            # Initialize MCP toolset for content agent
            self.content_agent_mcp = MCPToolset(
                server_params=StdioServerParameters(
                    command="python",
                    args=[
                        os.path.join(os.path.dirname(__file__), 
                                   "sub-agents", "sahayak-content-agent", "agent.py")
                    ],
                    env={"CONTENT_AGENT_ID": "sahayak-content-agent"}
                )
            )
            
            # Initialize additional MCP toolsets for other agents as needed
            # (placeholder for future agents)
            self.agent_toolsets = {
                "content_agent": self.content_agent_mcp
            }
            
            logger.info("MCP toolsets initialized successfully")
            
        except Exception as e:
            logger.error(f"MCP toolsets initialization failed: {e}")
            self.agent_toolsets = {}

    async def analyze_input(self, raw_input: str, metadata: Dict = None) -> TeacherInput:
        """
        Analyze teacher input to determine type and extract content
        
        Args:
            raw_input: Raw input from teacher (text, file path, etc.)
            metadata: Additional metadata about the input
            
        Returns:
            TeacherInput object with processed information
        """
        try:
            # Use Gemini to analyze input type and intent
            analysis_prompt = f"""
            Analyze this teacher input and determine:
            1. Input type (text_chat or image_pdf)
            2. Educational intent (lesson planning, worksheet creation, Q&A, etc.)
            3. Subject matter if identifiable
            4. Complexity level (beginner, intermediate, advanced)
            
            Input: {raw_input}
            
            Respond in JSON format:
            {{
                "input_type": "text_chat|image_pdf",
                "intent": "lesson_generation|worksheet_creation|qa|assessment",
                "subject": "subject name or null",
                "complexity": "beginner|intermediate|advanced",
                "requires_visual": true/false,
                "curriculum_standard": "detected standard or null"
            }}
            """
            
            response = await self.gemini_model.generate_content_async(analysis_prompt)
            analysis = json.loads(response.text)
            
            # Determine input type
            input_type = InputType.TEXT_CHAT
            if analysis.get("input_type") == "image_pdf":
                input_type = InputType.IMAGE_PDF
                
            # Create TeacherInput object
            teacher_input = TeacherInput(
                content=raw_input,
                input_type=input_type,
                metadata={
                    **(metadata or {}),
                    **analysis
                }
            )
            
            logger.info(f"Input analyzed: {input_type.value}, intent: {analysis.get('intent')}")
            return teacher_input
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse Gemini analysis, using fallback")
            return TeacherInput(
                content=raw_input,
                input_type=InputType.TEXT_CHAT,
                metadata=metadata or {}
            )
        except Exception as e:
            logger.error(f"Input analysis failed: {e}")
            raise

    def route_to_agents(self, teacher_input: TeacherInput) -> List[AgentTask]:
        """
        Route teacher input to appropriate agents based on type and intent
        
        Args:
            teacher_input: Analyzed teacher input
            
        Returns:
            List of agent tasks to execute
        """
        tasks = []
        metadata = teacher_input.metadata
        
        if teacher_input.input_type == InputType.IMAGE_PDF:
            # Image/PDF input -> WorksheetAgent (after OCR)
            tasks.append(AgentTask(
                agent_type=AgentType.WORKSHEET_AGENT,
                task_data={
                    "input_content": teacher_input.content,
                    "requires_ocr": True,
                    "generate_slides": True,
                    "generate_worksheets": True,
                    "difficulty_levels": ["easy", "medium", "hard"]
                },
                priority=1
            ))
            
            # Also trigger ArtAgent for visuals
            tasks.append(AgentTask(
                agent_type=AgentType.ART_AGENT,
                task_data={
                    "context": "worksheet_visuals",
                    "subject": metadata.get("subject"),
                    "style": "educational_diagram"
                },
                priority=2,
                dependencies=["worksheet_agent"]
            ))
            
        elif teacher_input.input_type == InputType.TEXT_CHAT:
            # Text input -> ContentAgent
            tasks.append(AgentTask(
                agent_type=AgentType.CONTENT_AGENT,
                task_data={
                    "prompt": teacher_input.content,
                    "intent": metadata.get("intent", "lesson_generation"),
                    "subject": metadata.get("subject"),
                    "complexity": metadata.get("complexity", "intermediate"),
                    "curriculum_standard": metadata.get("curriculum_standard")
                },
                priority=1
            ))
            
            # Conditionally trigger ExamAgent for deep learning topics
            if metadata.get("intent") in ["lesson_generation", "concept_explanation"]:
                tasks.append(AgentTask(
                    agent_type=AgentType.EXAM_AGENT,
                    task_data={
                        "context": "practice_questions",
                        "subject": metadata.get("subject"),
                        "complexity": metadata.get("complexity")
                    },
                    priority=2,
                    dependencies=["content_agent"]
                ))
        
        logger.info(f"Routed to {len(tasks)} agents")
        return tasks

    async def execute_agent_tasks(self, tasks: List[AgentTask]) -> Dict[str, Any]:
        """
        Execute agent tasks in proper order with dependency management
        
        Args:
            tasks: List of agent tasks to execute
            
        Returns:
            Dictionary of results from each agent
        """
        results = {}
        
        # Sort tasks by priority
        tasks.sort(key=lambda x: x.priority)
        
        for task in tasks:
            try:
                # Check dependencies
                if task.dependencies:
                    for dep in task.dependencies:
                        if dep not in results:
                            logger.warning(f"Dependency {dep} not satisfied for {task.agent_type}")
                            continue
                
                # Execute task based on agent type
                if task.agent_type == AgentType.CONTENT_AGENT:
                    result = await self._call_content_agent(task.task_data)
                    results["content_agent"] = result
                    
                elif task.agent_type == AgentType.WORKSHEET_AGENT:
                    result = await self._call_worksheet_agent(task.task_data)
                    results["worksheet_agent"] = result
                    
                elif task.agent_type == AgentType.ART_AGENT:
                    result = await self._call_art_agent(task.task_data)
                    results["art_agent"] = result
                    
                elif task.agent_type == AgentType.EXAM_AGENT:
                    result = await self._call_exam_agent(task.task_data)
                    results["exam_agent"] = result
                    
                elif task.agent_type == AgentType.QA_AGENT:
                    result = await self._call_qa_agent(task.task_data)
                    results["qa_agent"] = result
                    
                logger.info(f"Task completed: {task.agent_type.value}")
                
            except Exception as e:
                logger.error(f"Task execution failed for {task.agent_type}: {e}")
                results[task.agent_type.value] = {"error": str(e)}
        
        return results

    async def _call_content_agent(self, task_data: Dict) -> Dict[str, Any]:
        """Call ContentAgent via ADK MCP toolset"""
        try:
            content_agent_toolset = self.agent_toolsets.get("content_agent")
            if not content_agent_toolset:
                raise ValueError("Content agent MCP toolset not available")
            
            # Call the content agent's generate_educational_content tool via MCP
            result = await content_agent_toolset.call_tool(
                "generate_educational_content",
                {
                    "prompt": task_data.get("prompt", ""),
                    "metadata": {
                        "intent": task_data.get("intent", "lesson_generation"),
                        "subject": task_data.get("subject"),
                        "complexity": task_data.get("complexity", "intermediate"),
                        "curriculum_standard": task_data.get("curriculum_standard")
                    }
                }
            )
            
            logger.info("ContentAgent called successfully via MCP")
            return result
            
        except Exception as e:
            logger.error(f"ContentAgent MCP call failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "agent": "content_agent"
            }

    async def _call_worksheet_agent(self, task_data: Dict) -> Dict[str, Any]:
        """Call WorksheetAgent via ADK MCP toolset"""
        try:
            # Placeholder implementation - will be implemented when worksheet agent is created
            logger.info("Calling WorksheetAgent (not yet implemented)")
            return {
                "status": "success", 
                "slides": "Generated slides (placeholder)",
                "worksheets": "Generated worksheets (placeholder)",
                "agent": "worksheet_agent",
                "note": "Agent not yet implemented with ADK"
            }
        except Exception as e:
            logger.error(f"WorksheetAgent call failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "agent": "worksheet_agent"
            }

    async def _call_art_agent(self, task_data: Dict) -> Dict[str, Any]:
        """Call ArtAgent via ADK MCP toolset"""
        try:
            # Placeholder implementation - will be implemented when art agent is created
            logger.info("Calling ArtAgent (not yet implemented)")
            return {
                "status": "success",
                "images": ["image1.png", "image2.png"],
                "agent": "art_agent",
                "note": "Agent not yet implemented with ADK"
            }
        except Exception as e:
            logger.error(f"ArtAgent call failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "agent": "art_agent"
            }

    async def _call_exam_agent(self, task_data: Dict) -> Dict[str, Any]:
        """Call ExamPaperAgent via ADK MCP toolset"""
        try:
            # Placeholder implementation - will be implemented when exam agent is created
            logger.info("Calling ExamPaperAgent (not yet implemented)")
            return {
                "status": "success",
                "questions": "Generated exam questions (placeholder)", 
                "agent": "exam_agent",
                "note": "Agent not yet implemented with ADK"
            }
        except Exception as e:
            logger.error(f"ExamPaperAgent call failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "agent": "exam_agent"
            }

    async def _call_qa_agent(self, task_data: Dict) -> Dict[str, Any]:
        """Call QA-Agent via ADK MCP toolset"""
        try:
            # Placeholder implementation - will be implemented when QA agent is created
            logger.info("Calling QA-Agent (not yet implemented)")
            return {
                "status": "success",
                "answer": "Generated answer (placeholder)",
                "agent": "qa_agent",
                "note": "Agent not yet implemented with ADK"
            }
        except Exception as e:
            logger.error(f"QA-Agent call failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "agent": "qa_agent"
            }

    async def orchestrate(self, raw_input: str, teacher_id: str = None) -> Dict[str, Any]:
        """
        Main orchestration method - coordinates entire pipeline
        
        Args:
            raw_input: Raw input from teacher
            teacher_id: Optional teacher identifier for personalization
            
        Returns:
            Complete orchestrated response with all agent outputs
        """
        try:
            # Step 1: Analyze input
            teacher_input = await self.analyze_input(raw_input)
            teacher_input.teacher_id = teacher_id
            
            # Step 2: Load teacher preferences if available
            if teacher_id:
                preferences = await self._load_teacher_preferences(teacher_id)
                teacher_input.preferences = preferences
            
            # Step 3: Route to appropriate agents
            agent_tasks = self.route_to_agents(teacher_input)
            
            # Step 4: Execute agent tasks
            agent_results = await self.execute_agent_tasks(agent_tasks)
            
            # Step 5: Aggregate and format results
            final_output = await self._aggregate_results(agent_results, teacher_input)
            
            # Step 6: Store interaction for learning
            if teacher_id:
                await self._store_interaction(teacher_id, teacher_input, final_output)
            
            logger.info("Orchestration completed successfully")
            return final_output
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }

    async def _load_teacher_preferences(self, teacher_id: str) -> Dict[str, Any]:
        """Load teacher preferences from Firestore"""
        try:
            doc_ref = self.firestore_client.collection("teachers").document(teacher_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict().get("preferences", {})
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Failed to load teacher preferences: {e}")
            return {}

    async def _aggregate_results(self, agent_results: Dict[str, Any], teacher_input: TeacherInput) -> Dict[str, Any]:
        """Aggregate results from all agents into final output"""
        
        final_output = {
            "status": "success",
            "input_analysis": {
                "type": teacher_input.input_type.value,
                "metadata": teacher_input.metadata
            },
            "generated_content": {},
            "files": [],
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Process each agent result
        for agent_name, result in agent_results.items():
            if result.get("status") == "success":
                final_output["generated_content"][agent_name] = result
                
                # Collect any generated files
                if "files" in result:
                    final_output["files"].extend(result["files"])
            else:
                final_output["generated_content"][agent_name] = {
                    "error": result.get("error", "Unknown error")
                }
        
        return final_output

    async def _store_interaction(self, teacher_id: str, teacher_input: TeacherInput, output: Dict[str, Any]):
        """Store interaction in Firestore for learning and personalization"""
        try:
            interaction_data = {
                "teacher_id": teacher_id,
                "input": {
                    "content": teacher_input.content,
                    "type": teacher_input.input_type.value,
                    "metadata": teacher_input.metadata
                },
                "output": output,
                "timestamp": firestore.SERVER_TIMESTAMP
            }
            
            self.firestore_client.collection("interactions").add(interaction_data)
            logger.info(f"Interaction stored for teacher {teacher_id}")
            
        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")

# Global instance for ADK Agent integration
sahayak_coordinator = SahayakCoordinatorAgent()