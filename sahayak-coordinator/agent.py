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

# Google Cloud imports
from google.cloud import firestore
import google.generativeai as genai

# Import sub-agent functions
from sub_agents.sahayak_content_agent.agent import (
    generate_educational_content,
    search_web_for_education,
    align_with_curriculum,
    create_assessment_questions
)

from sub_agents.sahayak_retrieval_agent.agent import (
    search_educational_content,  # Primary unified search function
    query_file_storage
)

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
        # Initialize ADK Agent with proper structure
        super().__init__(
            name="sahayak_coordinator",
            model="gemini-1.5-pro",
            description="Central orchestrator for educational content generation pipeline",
            instruction=(
                "You are the central coordinator for an advanced educational AI system called Sahayak. "
                "Your primary responsibility is to orchestrate multiple specialized educational agents "
                "to provide comprehensive educational content generation services.\n\n"
                "Your core capabilities include:\n\n"
                "1. INTELLIGENT INPUT ANALYSIS:\n"
                "   - Analyze teacher inputs (text, images, PDFs) to understand educational intent\n"
                "   - Determine subject matter, difficulty level, and curriculum requirements\n"
                "   - Identify the most appropriate specialized agents for each task\n\n"
                "2. MULTI-AGENT ORCHESTRATION:\n"
                "   - Route requests to specialized sub-agents (Content Agent, Worksheet Agent, etc.)\n"
                "   - Manage task dependencies and execution priorities\n"
                "   - Coordinate parallel agent execution for efficiency\n\n"
                "3. EDUCATIONAL WORKFLOW MANAGEMENT:\n"
                "   - Handle complex educational content generation workflows\n"
                "   - Ensure curriculum alignment across all generated materials\n"
                "   - Manage quality control and content integration\n\n"
                "4. RESPONSE AGGREGATION:\n"
                "   - Combine outputs from multiple agents into coherent deliverables\n"
                "   - Format results appropriately for teacher consumption\n"
                "   - Provide comprehensive educational packages\n\n"
                "Always prioritize educational quality, curriculum alignment, and teacher needs. "
                "Ensure all generated content is pedagogically sound and appropriate for the target audience."
            ),
            tools=[]  # Will be populated with MCP toolsets for sub-agents
        )
        
        # Initialize Google services
        self._init_google_services()
        
        logger.info(f"Sahayak Coordinator Agent {self.name} initialized")

    def _init_google_services(self):
        """Initialize Google Cloud services"""
        try:
            # Configure Gemini AI
            if os.getenv("GEMINI_API_KEY"):
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                logger.info("Gemini AI configured successfully")
            
            logger.info("Google services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google services: {e}")
            # Don't raise to allow testing without full cloud setup


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
            
            # Get Gemini model dynamically
            gemini_model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-pro"))
            response = await gemini_model.generate_content_async(analysis_prompt)
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
        """Call ContentAgent via direct function calls"""
        try:
            logger.info("📞 Calling ContentAgent with direct function calls...")
            
            # Determine content type from metadata
            intent = task_data.get("intent", "explanation")
            content_type = "lesson_plan" if intent == "lesson_generation" else "explanation"
            
            # Generate educational content using the parameters
            content_result = generate_educational_content(
                prompt=task_data.get("prompt", ""),
                subject=task_data.get("subject", "General"),
                difficulty=task_data.get("complexity", "intermediate"),
                content_type=content_type,
                curriculum_standard=task_data.get("curriculum_standard", ""),
                language="english"
            )
            
            # Search knowledge base for additional context using unified search
            knowledge_result = search_educational_content(
                query=task_data.get("prompt", ""),
                content_types=["lessons", "articles", "explanations"],
                subject=task_data.get("subject", ""),
                difficulty=task_data.get("complexity", "intermediate"),
                limit=3
            )
            
            # Check curriculum alignment if standards are specified
            alignment_result = {}
            if task_data.get("curriculum_standard"):
                alignment_result = align_with_curriculum(
                    content=content_result.get("content", {}).get("body", ""),
                    curriculum=task_data.get("curriculum_standard"),
                    subject=task_data.get("subject", ""),
                    grade_level=""
                )
            
            # Aggregate all results
            final_result = {
                "status": "success",
                "agent": "content_agent",
                "primary_content": content_result,
                "knowledge_base_results": knowledge_result,
                "curriculum_alignment": alignment_result,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            logger.info("✅ ContentAgent completed successfully with direct function calls")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ ContentAgent direct call failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "agent": "content_agent"
            }

    async def _call_worksheet_agent(self, task_data: Dict) -> Dict[str, Any]:
        """Call WorksheetAgent via ADK MCP toolset"""
        try:
            logger.info("Calling WorksheetAgent with real implementation")
            
            # Import the actual WorksheetAgent function
            from sub_agents.sahayak_worksheet_agent.agent import generate_educational_materials
            
            # Extract parameters from task_data
            content = task_data.get("content", "")
            file_info = task_data.get("file_info", {})
            subject = task_data.get("subject", "General")
            grade_level = task_data.get("grade_level", "Middle School")
            curriculum = task_data.get("curriculum", "General")
            output_format = task_data.get("output_format", "both")
            
            # Prepare parameters for WorksheetAgent
            agent_params = {
                "content": content,
                "subject": subject,
                "grade_level": grade_level,
                "curriculum": curriculum,
                "output_format": output_format,
                "save_to_storage": True
            }
            
            # Add file parameters if available
            if file_info:
                if "artifact_filename" in file_info:
                    agent_params["artifact_filename"] = file_info["artifact_filename"]
                if "pdf_base64" in file_info:
                    agent_params["pdf_base64"] = file_info["pdf_base64"]
                if "image_base64" in file_info:
                    agent_params["image_base64"] = file_info["image_base64"]
                if "word_base64" in file_info:
                    agent_params["word_base64"] = file_info["word_base64"]
            
            logger.info(f"Calling WorksheetAgent with params: content={len(content)} chars, subject={subject}, grade_level={grade_level}")
            
            # Call the actual WorksheetAgent function
            result = generate_educational_materials(**agent_params)
            
            # Add agent identifier to result
            result["agent"] = "worksheet_agent"
            
            logger.info(f"WorksheetAgent returned: {result.get('status')}")
            return result
            
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
            # Get Firestore client dynamically
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
            if not project_id:
                return {}
            firestore_client = firestore.Client(project=project_id)
            doc_ref = firestore_client.collection("teachers").document(teacher_id)
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
            
            # Get Firestore client dynamically
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
            if project_id:
                firestore_client = firestore.Client(project=project_id)
                firestore_client.collection("interactions").add(interaction_data)
            logger.info(f"Interaction stored for teacher {teacher_id}")
            
        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")


# Convenience functions for direct teacher interaction with the coordinator
def create_lesson_plan(
    topic: str,
    subject: str = "General",
    grade_level: str = "intermediate",
    curriculum: str = "",
    duration: int = 45
) -> dict:
    """
    Create a comprehensive lesson plan for teachers.
    
    Args:
        topic (str): The topic or concept to create a lesson plan for
        subject (str): Subject area (e.g., Mathematics, Science, History)
        grade_level (str): Difficulty/grade level (beginner, intermediate, advanced)
        curriculum (str): Curriculum standard (CBSE, IB, Common Core, etc.)
        duration (int): Lesson duration in minutes
        
    Returns:
        dict: Status and generated lesson plan
    """
    logger.info(f"📚 create_lesson_plan called for topic: '{topic}', subject: {subject}, grade: {grade_level}")
    
    try:
        # Use the imported content agent functions directly
        prompt = f"Create a detailed {duration}-minute lesson plan on '{topic}' for {grade_level} level {subject} students."
        
        # Generate the lesson plan content
        result = generate_educational_content(
            prompt=prompt,
            subject=subject,
            difficulty=grade_level,
            content_type="lesson_plan",
            curriculum_standard=curriculum,
            language="english"
        )
        
        if result.get("status") == "success":
            logger.info(f"✅ Lesson plan created successfully for {topic}")
            return {
                "status": "success",
                "lesson_plan": result.get("content", {}),
                "topic": topic,
                "subject": subject,
                "grade_level": grade_level,
                "duration_minutes": duration,
                "curriculum": curriculum
            }
        else:
            return result
            
    except Exception as e:
        logger.error(f"❌ Lesson plan creation failed: {e}")
        return {
            "status": "error",
            "error_message": f"Failed to create lesson plan: {str(e)}"
        }


def explain_concept(
    concept: str,
    subject: str = "General",
    difficulty: str = "intermediate",
    include_examples: bool = True
) -> dict:
    """
    Generate a clear explanation of an educational concept.
    
    Args:
        concept (str): The concept to explain
        subject (str): Subject area
        difficulty (str): Difficulty level (beginner, intermediate, advanced)
        include_examples (bool): Whether to include practical examples
        
    Returns:
        dict: Status and concept explanation
    """
    logger.info(f"💡 explain_concept called for: '{concept}', subject: {subject}, difficulty: {difficulty}")
    
    try:
        prompt = f"Explain the concept of '{concept}' in {subject}"
        if include_examples:
            prompt += " with practical examples and real-world applications"
        prompt += f" for {difficulty} level students."
        
        result = generate_educational_content(
            prompt=prompt,
            subject=subject,
            difficulty=difficulty,
            content_type="explanation",
            curriculum_standard="",
            language="english"
        )
        
        if result.get("status") == "success":
            logger.info(f"✅ Concept explanation generated successfully for {concept}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Concept explanation failed: {e}")
        return {
            "status": "error",
            "error_message": f"Failed to explain concept: {str(e)}"
        }


def search_educational_resources(
    query: str,
    subject: str = "",
    difficulty: str = "intermediate",
    resource_type: str = "all",
    include_documents: bool = True
) -> dict:
    """
    Comprehensive search for educational resources using specialized retrieval agent.
    
    Args:
        query (str): Search query for educational content
        subject (str): Subject area filter
        difficulty (str): Difficulty level filter
        resource_type (str): Type of resource (articles, videos, lessons, all)
        include_documents (bool): Whether to search for documents/files
        
    Returns:
        dict: Status and comprehensive search results
    """
    print(f"[RESOURCE_SEARCH] Comprehensive search for '{query}' in {subject} {difficulty}")
    
    try:
        # Use the unified search function for comprehensive database search
        unified_results = search_educational_content(
            query=query,
            content_types=["all"] if resource_type == "all" else [resource_type],
            subject=subject,
            difficulty=difficulty,
            limit=15  # Higher limit since we're doing unified search
        )
        
        # Use content agent for web search (still separate as it searches external sources)
        web_results = search_web_for_education(
            query=query,
            subject=subject,
            content_type=resource_type,
            max_results=5
        )
        
        # Calculate total results from unified search
        total_unified = len(unified_results.get("results", []))
        total_web = len(web_results.get("results", []))
        
        # Extract categorized results from unified search
        categorized = unified_results.get("categorized_results", {})
        
        combined_results = {
            "status": "success",
            "query": query,
            "search_sources": {
                "unified_database_search": unified_results,
                "web_search": web_results,
                "documents": {"status": "included_in_unified_search"}
            },
            "summary": {
                "total_results": total_unified + total_web,
                "database_results": total_unified,
                "web_results": total_web,
                "database_breakdown": {
                    "educational_resources": len(categorized.get("educational_resources", [])),
                    "worksheets": len(categorized.get("worksheets", [])),
                    "documents": len(categorized.get("educational_documents", []))
                },
                "search_parameters": {
                    "query": query,
                    "subject": subject,
                    "difficulty": difficulty,
                    "resource_type": resource_type,
                    "include_documents": include_documents
                }
            }
        }
        
        print(f"[RESOURCE_SUCCESS] Found {combined_results['summary']['total_results']} total resources")
        print(f"  - Database (Unified): {total_unified}")
        print(f"    - Educational Resources: {len(categorized.get('educational_resources', []))}")
        print(f"    - Worksheets: {len(categorized.get('worksheets', []))}")
        print(f"    - Documents: {len(categorized.get('educational_documents', []))}")
        print(f"  - Web Results: {total_web}")
        
        return combined_results
        
    except Exception as e:
        print(f"[RESOURCE_ERROR] Educational resource search failed: {e}")
        return {
            "status": "error",
            "error_message": f"Resource search failed: {str(e)}"
        }


def retrieve_educational_worksheets(
    subject: str,
    grade_level: str = "",
    difficulty: str = "intermediate", 
    curriculum: str = "",
    worksheet_type: str = "practice"
) -> dict:
    """
    Retrieve educational worksheets using the specialized retrieval agent.
    
    Args:
        subject (str): Subject area (Mathematics, Science, English, etc.)
        grade_level (str): Grade or class level 
        difficulty (str): Difficulty level (easy, medium, hard)
        curriculum (str): Curriculum standard (CBSE, IB, Common Core, Cambridge)
        worksheet_type (str): Type of worksheet (practice, assessment, homework, quiz)
        
    Returns:
        dict: Status and retrieved worksheets
    """
    print(f"[WORKSHEET_RETRIEVAL] Retrieving {subject} {grade_level} {difficulty} worksheets")
    
    try:
        # Use unified search for worksheets specifically
        worksheet_results = search_educational_content(
            query=f"{subject} worksheets",
            content_types=["worksheets"],
            subject=subject,
            difficulty=difficulty,
            grade_level=grade_level,
            curriculum=curriculum,
            limit=10
        )
        
        if worksheet_results.get("status") == "success":
            # Extract worksheets from unified search results
            all_results = worksheet_results.get("results", [])
            worksheets = [result for result in all_results if result.get("source_collection") == "worksheets"]
            metadata = worksheet_results.get("search_metadata", {})
            
            print(f"[WORKSHEET_SUCCESS] Retrieved {len(worksheets)} worksheets")
            
            # Enhance results with additional context
            enhanced_result = {
                "status": "success",
                "worksheets": worksheets,
                "retrieval_summary": {
                    "total_worksheets": len(worksheets),
                    "subject": subject,
                    "grade_level": grade_level,
                    "difficulty": difficulty,
                    "curriculum": curriculum,
                    "worksheet_type": worksheet_type,
                    "filters_applied": metadata.get("filters_applied", []),
                    "distributions": {
                        "difficulty": metadata.get("difficulty_distribution", {}),
                        "type": metadata.get("type_distribution", {})
                    }
                },
                "recommendations": [
                    f"Found {len(worksheets)} {difficulty} level {subject} worksheets",
                    f"Worksheets are suitable for {grade_level} students" if grade_level else "Grade level not specified",
                    f"Curriculum alignment: {curriculum}" if curriculum else "No specific curriculum filtering applied"
                ]
            }
            
            return enhanced_result
        else:
            return worksheet_results
            
    except Exception as e:
        print(f"[WORKSHEET_ERROR] Worksheet retrieval failed: {e}")
        return {
            "status": "error",
            "error_message": f"Worksheet retrieval failed: {str(e)}"
        }


def check_curriculum_alignment(
    content: str,
    curriculum: str,
    subject: str,
    grade: str = ""
) -> dict:
    """
    Check how well content aligns with curriculum standards.
    
    Args:
        content (str): Educational content to analyze
        curriculum (str): Curriculum standard (CBSE, IB, Common Core, Cambridge)
        subject (str): Subject area
        grade (str): Grade level
        
    Returns:
        dict: Status and alignment analysis
    """
    logger.info(f"📊 check_curriculum_alignment called for {curriculum} {subject} Grade {grade}")
    
    try:
        result = align_with_curriculum(
            content=content,
            curriculum=curriculum,
            subject=subject,
            grade_level=grade
        )
        
        if result.get("status") == "success":
            alignment_score = result.get("alignment_score", 0)
            logger.info(f"✅ Curriculum alignment check completed: {alignment_score:.2%} alignment")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Curriculum alignment check failed: {e}")
        return {
            "status": "error",
            "error_message": f"Alignment check failed: {str(e)}"
        }


# Create the coordinator as a proper ADK Agent with teacher-facing tools
coordinator_agent = Agent(
    name="sahayak_coordinator",
    model="gemini-2.0-flash",
    description="Central coordinator for comprehensive educational content generation and curriculum management",
    instruction=(
        "You are the Sahayak Coordinator, an advanced educational AI system designed to help teachers "
        "create high-quality educational content, lesson plans, and instructional materials. "
        "You coordinate multiple specialized educational agents to provide comprehensive solutions.\\n\\n"
        "Your core capabilities include:\\n\\n"
        "1. LESSON PLANNING: Create detailed, curriculum-aligned lesson plans for any subject and grade level\\n"
        "2. CONCEPT EXPLANATION: Generate clear, engaging explanations of complex educational concepts\\n"
        "3. RESOURCE DISCOVERY: Search and find relevant educational resources and materials\\n"
        "4. CURRICULUM ALIGNMENT: Check and ensure content aligns with educational standards\\n\\n"
        "You support multiple curriculum standards including CBSE, IB, Common Core, and Cambridge. "
        "Always prioritize pedagogical soundness, age-appropriateness, and educational quality in all generated content.\\n\\n"
        
        "IMPORTANT - File Delivery Policy:\\n"
        "When users request educational files (worksheets, documents, presentations), you CAN and SHOULD provide them with direct access. "
        "Use the search tools to find files, then provide download links from the search results. "
        "Present download links clearly using the format: '📄 [TITLE] - Click to download: [URL]'. "
        "Never say you cannot provide files - your goal is to make educational resources easily accessible."
    ),
    tools=[
        create_lesson_plan,
        explain_concept,
        search_educational_resources,
        check_curriculum_alignment
    ]
)

# Keep the original coordinator for backwards compatibility
root_agent = coordinator_agent

if __name__ == "__main__":
    # For testing purposes only
    print(f"Agent {root_agent.name} initialized successfully")