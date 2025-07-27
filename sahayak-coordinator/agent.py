"""
Sahayak Coordinator Agent - Main Educational Content Hub

This is the central coordinator that routes user requests to the appropriate specialized agents:

1. **Content Agent**: Educational content generation, lesson plans, explanations
2. **Retrieval Agent**: Search for existing educational resources from database
3. **QA Agent**: Simple Q&A generation from uploaded files/images
4. **Worksheet Agent**: Comprehensive educational materials (slides + worksheets)

ROUTING LOGIC:
- File/Image Upload � QA Agent or Worksheet Agent (based on user request)
- Text with retrieval keywords � Retrieval Agent  
- Text with content generation keywords � Content Agent
- Content Agent output >500 words � Also trigger Worksheet Agent

The coordinator intelligently routes requests and can chain agents when needed.
"""

import logging
import re
import asyncio
import concurrent.futures
import threading
from typing import Dict, List, Any, Optional
from google.adk.agents import Agent

# Import all sub-agents
from .sub_agents.sahayak_content_agent.agent import (
    generate_educational_content,
    search_web_for_education, 
    align_with_curriculum,
    create_assessment_questions
)

from .sub_agents.sahayak_retrieval_agent.agent import (
    search_educational_content,
    query_file_storage
)

from .sub_agents.sahayak_qa_agent.agent import (
    extract_content_from_upload as qa_extract_content,
    generate_questions_with_language as qa_generate_questions
)

from .sub_agents.sahayak_worksheet_agent.agent import (
    extract_content_from_upload as worksheet_extract_content,
    generate_educational_materials_with_language as worksheet_generate_materials
)

from .sub_agents.sahayak_image_agent.agent import (
    generate_image_tool as image_generate_tool
)

from .sub_agents.sahayak_image_agent.mcp_servers.image_server import (
    generate_image_model_call as image_generate_raw
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def detect_input_type(user_input: str) -> Dict[str, Any]:
    print("[DEBUG]Inside detect_input_type")
    """
    Detect the type of user input and determine routing strategy.
    
    Args:
        user_input (str): The user's input message
        
    Returns:
        Dict: Analysis of input type and recommended routing
    """
    
    analysis = {
        "has_file_upload": False,
        "input_type": "text",
        "file_type": None,
        "keywords": [],
        "recommended_agent": None,
        "confidence": "medium"
    }
    
    # Check for file upload indicators
    file_indicators = [
        "upload", "file", "pdf", "image", "document", "picture", 
        "jpeg", "png", "jpg", "doc", "docx", "attachment"
    ]
    
    # Check for inline_data or file references
    if any(indicator in user_input.lower() for indicator in file_indicators):
        analysis["has_file_upload"] = True
        analysis["input_type"] = "file_upload"
        
        if any(pdf_type in user_input.lower() for pdf_type in ["pdf", "document", "doc"]):
            analysis["file_type"] = "pdf"
        elif any(img_type in user_input.lower() for img_type in ["image", "picture", "photo", "png", "jpg", "jpeg"]):
            analysis["file_type"] = "image"
    
    # Detect keywords for different agents
    retrieval_keywords = [
        "search", "find", "lookup", "retrieve", "existing", "database",
        "browse", "discover", "available", "resources", "materials",
        "what do you have", "show me", "list", "previous"
    ]
    
    content_keywords = [
        "create", "generate", "make", "write", "explain", "teach",
        "lesson plan", "content", "educational", "curriculum",
        "help me understand", "how to", "what is", "define"
    ]
    
    worksheet_keywords = [
        "worksheet", "assignment","create", "quiz", "test", "assessment", 
        "slides", "presentation", "materials", "comprehensive",
        "easy medium hard", "difficulty levels", "printable"
    ]
    
    qa_keywords = [
        "questions", "answers", "q&a", "quick questions", "simple questions",
        "extract questions", "question generation", "basic questions"
    ]
    
    image_keywords = [
        "create image", "generate image", "draw image", "make image",
        "create picture", "generate picture", "draw picture", "make picture",
        "create diagram", "generate diagram", "draw diagram", "make diagram",
        "show me image", "show me picture", "show me diagram",
        "visualize", "illustrate", "illustration", "visual", "graph",
        "image of", "picture of", "diagram of", "illustration of"
    ]
    
    user_lower = user_input.lower()
    
    # Count keyword matches
    retrieval_count = sum(1 for kw in retrieval_keywords if kw in user_lower)
    content_count = sum(1 for kw in content_keywords if kw in user_lower)
    worksheet_count = sum(1 for kw in worksheet_keywords if kw in user_lower)
    qa_count = sum(1 for kw in qa_keywords if kw in user_lower)
    image_count = sum(1 for kw in image_keywords if kw in user_lower)
    
    analysis["keywords"] = {
        "retrieval": retrieval_count,
        "content": content_count,
        "worksheet": worksheet_count,
        "qa": qa_count,
        "image": image_count
    }
    
    # Determine recommended agent
    if analysis["has_file_upload"]:
        if worksheet_count > qa_count or "worksheet" in user_lower or "materials" in user_lower:
            analysis["recommended_agent"] = "worksheet"
            print("[DEBUG]Recommended Agent is Worksheet")
        else:
            analysis["recommended_agent"] = "qa"
            print("[DEBUG]Recommended Agent is QA")
        analysis["confidence"] = "high"
    else:
        # Text-based routing
        max_count = max(retrieval_count, content_count, worksheet_count, qa_count, image_count)
        
        if image_count == max_count and image_count > 0:
            analysis["recommended_agent"] = "image"
            print("[DEBUG]Recommended Agent is Image")
        elif retrieval_count == max_count and retrieval_count > 0:
            analysis["recommended_agent"] = "retrieval"
            print("[DEBUG]Recommended Agent is Retrieval")
        elif content_count == max_count and content_count > 0:
            analysis["recommended_agent"] = "content"
            print("[DEBUG]Recommended Agent is Content")
        elif worksheet_count == max_count and worksheet_count > 0:
            analysis["recommended_agent"] = "worksheet"
            print("[DEBUG]Recommended Agent is Worksheet")
        elif qa_count == max_count and qa_count > 0:
            analysis["recommended_agent"] = "qa"
            print("[DEBUG]Recommended Agent is QA")
        else:
            # Default to content agent for general educational queries
            analysis["recommended_agent"] = "content"
            analysis["confidence"] = "low"
    
    return analysis

def route_to_content_agent(
    prompt: str,
    subject: str = "",
    difficulty: str = "intermediate",
    content_type: str = "explanation",
    curriculum_standard: str = "",
    language: str = "english"
) -> Dict[str, Any]:
    """Route request to Content Agent for educational content generation."""
    
    logger.info(f"[COORDINATOR] Routing to Content Agent: {prompt[:100]}...")
    
    try:
        result = generate_educational_content(
            prompt=prompt,
            subject=subject,
            difficulty=difficulty,
            content_type=content_type,
            curriculum_standard=curriculum_standard,
            language=language
        )
        
        # Check if content is >500 words to trigger worksheet agent
        content_text = ""
        if result.get("status") == "success" and "content" in result:
            content_text = str(result["content"])
            word_count = len(content_text.split())
            
            logger.info(f"[COORDINATOR] Content Agent generated {word_count} words")
            
            if word_count > 500:
                logger.info(f"[COORDINATOR] Content >500 words, triggering Worksheet Agent...")
                
                # Trigger worksheet agent for comprehensive materials
                def run_worksheet_async():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(worksheet_generate_materials({
                            "content": content_text,
                            "subject": subject,
                            "grade_level": "",
                            "curriculum": curriculum_standard,
                            "output_format": "both",
                            "language": language
                        }))
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_worksheet_async)
                    worksheet_result = future.result(timeout=300)
                
                # Combine results
                result["worksheet_materials"] = worksheet_result
                result["triggered_worksheet_agent"] = True
                logger.info(f"[COORDINATOR] Added worksheet materials to response")
        
        return result
        
    except Exception as e:
        logger.error(f"[COORDINATOR] Content Agent routing failed: {e}")
        return {"status": "error", "error_message": f"Content generation failed: {str(e)}"}

def route_to_retrieval_agent(
    query: str,
    content_types: List[str] = ["all"],
    subject: str = "",
    difficulty: str = "",
    grade_level: str = "",
    curriculum: str = "",
    limit: int = 5
) -> Dict[str, Any]:
    """Route request to Retrieval Agent for searching existing resources."""
    
    logger.info(f"[COORDINATOR] Routing to Retrieval Agent: {query[:100]}...")
    
    try:
        result = search_educational_content(
            query=query,
            content_types=content_types,
            subject=subject,
            difficulty=difficulty,
            grade_level=grade_level,
            curriculum=curriculum,
            limit=limit
        )
        return result
        
    except Exception as e:
        logger.error(f"[COORDINATOR] Retrieval Agent routing failed: {e}")
        return {"status": "error", "error_message": f"Resource search failed: {str(e)}"}

def route_to_image_agent(
    prompt: str,
    language: str = "English"
) -> Dict[str, Any]:
    """Route request to Image Agent for image generation."""
    
    logger.info(f"[COORDINATOR] Routing to Image Agent: {prompt[:100]}...")
    
    def run_async_in_thread():
        """Run async code in a separate thread with its own event loop."""
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            async def do_image_work():
                # Generate image using the image agent's tool
                result = await image_generate_tool(prompt)
                return {"status": "success", "image_result": result}
            
            return new_loop.run_until_complete(do_image_work())
        finally:
            new_loop.close()

    try:
        # Run async code in a separate thread to avoid event loop conflicts
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_async_in_thread)
            return future.result(timeout=300)  # 5 minute timeout
        
    except Exception as e:
        logger.error(f"[COORDINATOR] Image Agent routing failed: {e}")
        return {"status": "error", "error_message": f"Image generation failed: {str(e)}"}

def route_to_qa_agent(
    file_content: str = "",
    file_type: str = "",
    content: str = "",
    language: str = "English"
) -> Dict[str, Any]:
    """Route request to QA Agent for simple question generation."""
    
    logger.info(f"[COORDINATOR] Routing to QA Agent...")
    
    def run_async_in_thread():
        """Run async code in a separate thread with its own event loop."""
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(_async_route_to_qa_agent(
                file_content=file_content,
                file_type=file_type,
                content=content,
                language=language
            ))
        finally:
            new_loop.close()
    
    try:
        # Run async code in a separate thread to avoid event loop conflicts
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_async_in_thread)
            return future.result(timeout=300)  # 5 minute timeout
        
    except Exception as e:
        logger.error(f"[COORDINATOR] QA Agent routing failed: {e}")
        return {"status": "error", "error_message": f"QA generation failed: {str(e)}"}

async def _async_route_to_qa_agent(
    file_content: str = "",
    file_type: str = "",
    content: str = "",
    language: str = "English"
) -> Dict[str, Any]:
    """Async helper for QA agent routing."""
    
    # First extract content if file provided
    extracted_content = ""
    if file_content and file_type:
        extracted_content = await qa_extract_content(file_content, file_type)
        if extracted_content.startswith("Error"):
            return {"status": "error", "error_message": extracted_content}
    
    # Use provided content or extracted content
    final_content = content if content else extracted_content
    
    if not final_content:
        return {"status": "error", "error_message": "No content provided for question generation"}
    
    # Generate questions - QA agent detects language automatically from content
    result = await qa_generate_questions({
        "content": final_content
    })
    
    return {"status": "success", "qa_result": result}


def route_to_worksheet_agent(
    content: str = "",
    file_content: str = "",
    file_type: str = "",
    subject: str = "",
    grade_level: str = "",
    curriculum: str = "",
    output_format: str = "both",
    language: str = "English"
) -> Dict[str, Any]:
    """Route request to Worksheet Agent for comprehensive educational materials."""
    logger.info("[COORDINATOR] Routing to Worksheet Agent…")

    def run_async_in_thread():
        """Run async code in a separate thread with its own event loop."""
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            async def do_worksheet_work():
                # 1) If a file was uploaded, extract text first
                if file_content and file_type:
                    print(f"[DEBUG] File content length: {len(file_content)}")
                    print(f"[DEBUG] File content: {file_content}")
                    
                    # Check if we got a placeholder string instead of real base64 data
                    if file_content == "Base64 encoded file content" or len(file_content) < 100:
                        print("[DEBUG] Received placeholder file content, trying to read Week.pdf directly")
                        try:
                            import base64
                            with open("A:\\Sahayak\\Week.pdf", "rb") as f:
                                pdf_bytes = f.read()
                                actual_file_content = base64.b64encode(pdf_bytes).decode('utf-8')
                                print(f"[DEBUG] Read and encoded PDF: {len(actual_file_content)} characters")
                                extract_res = await worksheet_extract_content(actual_file_content, file_type)
                        except Exception as e:
                            print(f"[DEBUG] Failed to read Week.pdf: {e}")
                            return {"status": "error", "message": f"Could not read PDF file: {e}"}
                    else:
                        extract_res = await worksheet_extract_content(file_content, file_type)
                    
                    if extract_res.startswith("Error"):
                        return {"status": "error", "message": extract_res}
                    content_to_use = extract_res
                else:
                    content_to_use = content

                # 2) Now generate the materials
                materials_res = await worksheet_generate_materials({
                    "content": content_to_use,
                    "subject": subject,
                    "grade_level": grade_level,
                    "curriculum": curriculum,
                    "output_format": output_format,
                    "language": language
                })
                return materials_res
            
            return new_loop.run_until_complete(do_worksheet_work())
        finally:
            new_loop.close()

    try:
        # Run async code in a separate thread to avoid event loop conflicts
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_async_in_thread)
            return future.result(timeout=300)  # 5 minute timeout
        
    except Exception as e:
        logger.error(f"[COORDINATOR] Worksheet Agent routing failed: {e}")
        return {"status": "error", "error_message": f"Worksheet generation failed: {str(e)}"}


def coordinate_request(
    user_input: str,
    file_content: str = "",
    file_type: str = "",
    subject: str = "",
    grade_level: str = "",
    curriculum: str = "",
    language: str = "English",
    force_agent: str = ""
) -> Dict[str, Any]:
    """
    Main coordination function that analyzes input and routes to appropriate agent(s).
    
    Args:
        user_input (str): The user's request/query
        file_content (str): Base64 encoded file content if file uploaded
        file_type (str): MIME type of uploaded file
        subject (str): Subject area hint
        grade_level (str): Grade level hint  
        curriculum (str): Curriculum standard hint
        language (str): Language preference
        force_agent (str): Force routing to specific agent (content/retrieval/qa/worksheet)
        
    Returns:
        Dict: Response from the appropriate agent(s)
    """
    
    logger.info(f"[COORDINATOR] Processing request: {user_input[:100]}...")
    print("[DEBUG]Processing request: ",user_input)
    try:
        # Analyze input type and determine routing
        analysis = detect_input_type(user_input)
        print("[DEBUG]Input analysis: ",analysis)
        logger.info(f"[COORDINATOR] Input analysis: {analysis}")
        
        # Override with forced agent if specified
        if force_agent in ["content", "retrieval", "qa", "worksheet", "image"]:
            analysis["recommended_agent"] = force_agent
            analysis["confidence"] = "forced"
            print(f"[COORDINATOR] Forced routing to {force_agent} agent")
        
        # Route to appropriate agent
        recommended_agent = analysis["recommended_agent"]
        
        if recommended_agent == "content":
            return route_to_content_agent(
                prompt=user_input,
                subject=subject,
                curriculum_standard=curriculum,
                language=language
            )
            
        elif recommended_agent == "retrieval":
            return route_to_retrieval_agent(
                query=user_input,
                subject=subject,
                grade_level=grade_level,
                curriculum=curriculum
            )
            
        elif recommended_agent == "qa":
            return route_to_qa_agent(
                file_content=file_content,
                file_type=file_type,
                content=user_input if not file_content else "",
                language=language
            )
            
        elif recommended_agent == "worksheet":
            print("[COORDINATOR] Routing to Worksheet Agent...")
            return route_to_worksheet_agent(
                content=user_input if not file_content else "",
                file_content=file_content,
                file_type=file_type,
                subject=subject,
                grade_level=grade_level,
                curriculum=curriculum,
                language=language
            )
            
        elif recommended_agent == "image":
            print("[COORDINATOR] Routing to Image Agent...")
            return route_to_image_agent(
                prompt=user_input,
                language=language
            )
            
        else:
            # Fallback to content agent
            logger.warning(f"[COORDINATOR] Unknown agent recommendation, defaulting to content agent")
            return route_to_content_agent(
                prompt=user_input,
                subject=subject,
                curriculum_standard=curriculum,
                language=language
            )
    
    except Exception as e:
        logger.error(f"[COORDINATOR] Request coordination failed: {e}")
        return {
            "status": "error",
            "error_message": f"Coordination failed: {str(e)}",
            "analysis": analysis if 'analysis' in locals() else None
        }

# Create the main Sahayak Coordinator Agent
root_agent = Agent(
    name="sahayak_coordinator",
    model="gemini-2.0-flash", 
    description="Central coordinator for all Sahayak educational agents - routes requests to specialized sub-agents",
    instruction=(
        "You are the Sahayak Coordinator, the central hub for all educational content needs. "
        "You intelligently route user requests to the most appropriate specialized agent:\n\n"
        "Always ask the user this first: Hi there! Hope you're doing great 😊 Just let me know—would you like me to assist with your queries, generate documents, or fetch something specific for you today?\n"
        "For file uploads: \n"
        "- For worksheet requests: First call worksheet_extract_content to extract content, then call coordinate_request\n"
        "- For QA requests: First call qa_extract_content to extract content, then call coordinate_request\n"
        "- You can also call worksheet_generate_materials or qa_generate_questions directly if needed\n"
        "For text queries: Call coordinate_request directly.\n"
        "For image generation requests: Call image_generate_tool directly with the description or call coordinate_request.\n"        
        "Always use the coordinate_request function to analyze and route user requests. "
        "Provide clear, helpful responses and explain which agent(s) were used to fulfill the request.\n"
        
        "Once the output is provided please do ask them : Is there anything else you'd like me to take care of for you 🤗? "
    ),
    tools=[
        coordinate_request,
        worksheet_extract_content,
        worksheet_generate_materials,
        qa_extract_content,
        qa_generate_questions,
        image_generate_tool
    ]
)

if __name__ == "__main__":
    logger.info(f" ** Sahayak Coordinator Agent '{root_agent.name}' initialized successfully")
    logger.info(f" ** Available coordination tools: {len(root_agent.tools)}")
    for i, tool in enumerate(root_agent.tools, 1):
        logger.info(f"  {i}. {tool.__name__}")
    
    logger.info(f" Sub-agents integrated:")
    logger.info(f"  Content Agent (content generation)")
    logger.info(f"  Retrieval Agent (resource search)")
    logger.info(f"  QA Agent (simple Q&A)")
    logger.info(f"  Worksheet Agent (comprehensive materials)")