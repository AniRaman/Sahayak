"""
Simplified Sahayak Coordinator for Cloud Run deployment
This version doesn't depend on Google ADK and uses direct function calls
"""

import logging
import re
import asyncio
import concurrent.futures
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def detect_input_type(user_input: str) -> Dict[str, Any]:
    """
    Detect the type of user input and determine routing strategy.
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
        "worksheet", "assignment", "quiz", "test", "assessment", 
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
        else:
            analysis["recommended_agent"] = "qa"
        analysis["confidence"] = "high"
    else:
        # Text-based routing
        max_count = max(retrieval_count, content_count, worksheet_count, qa_count, image_count)
        
        if image_count == max_count and image_count > 0:
            analysis["recommended_agent"] = "image"
        elif retrieval_count == max_count and retrieval_count > 0:
            analysis["recommended_agent"] = "retrieval"
        elif content_count == max_count and content_count > 0:
            analysis["recommended_agent"] = "content"
        elif worksheet_count == max_count and worksheet_count > 0:
            analysis["recommended_agent"] = "worksheet"
        elif qa_count == max_count and qa_count > 0:
            analysis["recommended_agent"] = "qa"
        else:
            # Default to content agent for general educational queries
            analysis["recommended_agent"] = "content"
            analysis["confidence"] = "low"
    
    return analysis

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
    Simplified coordination function that analyzes input and returns routing recommendations.
    """
    
    logger.info(f"[COORDINATOR] Processing request: {user_input[:100]}...")
    
    try:
        # Analyze input type and determine routing
        analysis = detect_input_type(user_input)
        logger.info(f"[COORDINATOR] Input analysis: {analysis}")
        
        # Override with forced agent if specified
        if force_agent in ["content", "retrieval", "qa", "worksheet", "image"]:
            analysis["recommended_agent"] = force_agent
            analysis["confidence"] = "forced"
            logger.info(f"[COORDINATOR] Forced routing to {force_agent} agent")
        
        # For this simplified version, return the analysis and routing recommendation
        return {
            "status": "success",
            "analysis": analysis,
            "recommended_agent": analysis["recommended_agent"],
            "message": f"Request analyzed. Recommended agent: {analysis['recommended_agent']}",
            "routing_confidence": analysis["confidence"],
            "keywords_detected": analysis["keywords"],
            "input_type": analysis["input_type"],
            "file_type": analysis.get("file_type"),
            "user_input": user_input,
            "parameters": {
                "subject": subject,
                "grade_level": grade_level, 
                "curriculum": curriculum,
                "language": language
            }
        }
    
    except Exception as e:
        logger.error(f"[COORDINATOR] Request coordination failed: {e}")
        return {
            "status": "error",
            "error_message": f"Coordination failed: {str(e)}",
            "analysis": None
        }