"""
Simplified pipeline for testing - uses direct function calls instead of ADK SequentialAgent
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def analyze_content(content: str, additional_context: Dict = None) -> Dict[str, Any]:
    """Step 1: Analyze educational content"""
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        prompt = f"""
        Analyze this educational content and extract key components:
        
        Content: {content}
        Context: {additional_context or {}}
        
        Extract and return JSON with:
        {{
          "subject": "subject name",
          "topic": "specific topic",
          "grade_level": "grade level",
          "difficulty": "beginner/intermediate/advanced",
          "key_concepts": ["concept1", "concept2", ...],
          "learning_objectives": ["objective1", "objective2", ...],
          "vocabulary": [{{"term": "term", "definition": "definition"}}],
          "examples": ["example1", "example2"],
          "content_summary": "brief summary"
        }}
        """
        
        response = model.generate_content(prompt)
        
        # Try to parse JSON response
        try:
            result = json.loads(response.text)
            print("status : analyzing content success","result :",result)
            return {"status": "success", "analysis": result}
        except json.JSONDecodeError:
            # If JSON parsing fails, return raw response
            print("status : analyzing content error","result :",response.text)
            return {"status": "success", "analysis": {"raw_response": response.text}}
            
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def generate_slides(content_analysis: Dict) -> Dict[str, Any]:
    """Step 2: Generate slide structure"""
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        analysis = content_analysis.get("analysis", {})
        
        prompt = f"""
        Create a slide presentation structure based on this content analysis:
        {json.dumps(analysis, indent=2)}
        
        Generate JSON with slide structure:
        {{
          "presentation_title": "title",
          "total_slides": 8,
          "slides": [
            {{
              "slide_number": 1,
              "slide_type": "title",
              "title": "Topic Name",
              "subtitle": "Grade Level",
              "content": []
            }},
            {{
              "slide_number": 2,
              "slide_type": "objectives",
              "title": "Learning Objectives",
              "content": ["objective1", "objective2"]
            }}
          ]
        }}
        
        Create 6-8 slides total including title, objectives, key concepts, examples, vocabulary, and summary.
        """
        
        response = model.generate_content(prompt)
        
        try:
            result = json.loads(response.text)
            print("status : generating slides success","result :",result)
            return {"status": "success", "slides": result}
        except json.JSONDecodeError:
            print("status : generating slides error","result :",response.text)
            return {"status": "success", "slides": {"raw_response": response.text}}
            
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def generate_worksheets(content_analysis: Dict, slide_structure: Dict) -> Dict[str, Any]:
    """Step 3: Generate differentiated worksheets"""
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        analysis = content_analysis.get("analysis", {})
        slides = slide_structure.get("slides", {})
        
        prompt = f"""
        Create three differentiated worksheets based on:
        Content Analysis: {json.dumps(analysis, indent=2)}
        Slide Structure: {json.dumps(slides, indent=2)}
        
        Generate JSON with worksheets:
        {{
          "worksheet_title": "Topic Worksheets",
          "subject": "subject",
          "topic": "topic",
          "grade_level": "grade",
          "worksheets": {{
            "easy": {{
              "title": "Foundational Practice",
              "instructions": "Complete the following exercises",
              "estimated_time": "20 minutes",
              "questions": [
                {{
                  "question_number": 1,
                  "question_type": "multiple_choice",
                  "question": "Question text?",
                  "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
                  "correct_answer": "A",
                  "points": 2
                }}
              ]
            }},
            "medium": {{
              "title": "Application Practice",
              "instructions": "Solve these problems showing your work",
              "estimated_time": "30 minutes",
              "questions": [
                {{
                  "question_number": 1,
                  "question_type": "short_answer",
                  "question": "Explain the concept...",
                  "points": 5
                }}
              ]
            }},
            "hard": {{
              "title": "Advanced Analysis",
              "instructions": "Complete these challenging problems",
              "estimated_time": "40 minutes",
              "questions": [
                {{
                  "question_number": 1,
                  "question_type": "extended_response",
                  "question": "Analyze and evaluate...",
                  "points": 10
                }}
              ]
            }}
          }}
        }}
        
        Create 8-10 questions for easy, 6-8 for medium, 4-6 for hard level.
        """
        
        response = model.generate_content(prompt)
        
        try:
            result = json.loads(response.text)
            print("status : generating worksheets success","result :",result)
            return {"status": "success", "worksheets": result}
        except json.JSONDecodeError:
            print("status : generating worksheets error","result :",response.text)
            return {"status": "success", "worksheets": {"raw_response": response.text}}
            
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def process_educational_content_simple(content: str, additional_context: Dict = None) -> Dict[str, Any]:
    """
    Simple sequential processing of educational content
    """
    
    try:
        logger.info("Starting simple educational content processing")
        
        # Step 1: Analyze content
        analysis_result = analyze_content(content, additional_context)
        print("analysis_result :",analysis_result)
        if analysis_result.get("status") != "success":
            return analysis_result
        
        # Step 2: Generate slides
        slides_result = generate_slides(analysis_result)
        if slides_result.get("status") != "success":
            return slides_result
        
        # Step 3: Generate worksheets
        worksheets_result = generate_worksheets(analysis_result, slides_result)
        print("worksheets_result :",worksheets_result)
        if worksheets_result.get("status") != "success":
            return worksheets_result
        
        # Combine results
        final_result = {
            "status": "success",
            "content_analysis": analysis_result.get("analysis", {}),
            "slide_structure": slides_result.get("slides", {}),
            "worksheets": worksheets_result.get("worksheets", {}),
            "processing_metadata": {
                "timestamp": datetime.now().isoformat(),
                "pipeline_steps": ["content_analysis", "slide_generation", "worksheet_generation"],
                "input_content_length": len(content),
                "additional_context": additional_context
            }
        }
        print("final_result :",final_result)
        
        logger.info("Simple educational content processing completed")
        return final_result
        
    except Exception as e:
        logger.error(f"Simple pipeline processing failed: {e}")
        return {
            "status": "error",
            "error_message": str(e),
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Test the simple pipeline
    test_content = """
    Photosynthesis is the process by which green plants use sunlight to make their own food.
    Plants capture light energy using chlorophyll and convert carbon dioxide and water into glucose and oxygen.
    This process is essential for life on Earth as it produces the oxygen we breathe.
    """
    
    result = process_educational_content_simple(
        content=test_content,
        additional_context={"subject": "Biology", "grade_level": "Grade 6"}
    )
    
    print("Simple Pipeline Test:")
    print(f"Status: {result.get('status')}")
    if result.get('status') == 'success':
        print("✅ Content analysis completed")
        print("✅ Slide structure generated")
        print("✅ Worksheets created")
    else:
        print(f"❌ Error: {result.get('error_message')}")