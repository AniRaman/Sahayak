"""
Sahayak Content Agent - Educational Content Generation

This agent specializes in:
1. Generating instructional text (lectures, explanations, examples)
2. Creating lesson plans and educational outlines
3. Handling direct teacher queries and prompts
4. Integrating with external tools for enhanced content
5. Supporting multilingual content generation
6. Aligning content with curriculum standards
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from google.cloud import firestore

# Google ADK imports
from google.adk.agents import Agent

# Configure logging to show in terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configure Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "your-api-key"))


def generate_educational_content(
    prompt: str,
    subject: str = "General",
    difficulty: str = "intermediate", 
    content_type: str = "explanation",
    curriculum_standard: str = "",
    language: str = "english"
) -> dict:
    """
    Generate educational content using Gemini AI.
    
    Args:
        prompt (str): The educational content request/topic
        subject (str): Subject area (e.g., Mathematics, Science, History)
        difficulty (str): Difficulty level (beginner, intermediate, advanced)
        content_type (str): Type of content (lesson_plan, explanation, study_guide, etc.)
        curriculum_standard (str): Curriculum to align with (CBSE, IB, Common Core, etc.)
        language (str): Language for content generation
        
    Returns:
        dict: Status and generated educational content
    """
    logger.info(f"🎓 generate_educational_content called with prompt: '{prompt[:50]}...', subject: {subject}, difficulty: {difficulty}")
    
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Create comprehensive educational prompt
        system_prompt = f"""
        You are an expert educational content creator. Generate high-quality {content_type} content.
        
        Topic: {prompt}
        Subject: {subject}
        Difficulty Level: {difficulty}
        Language: {language}
        {f"Curriculum Standard: {curriculum_standard}" if curriculum_standard else ""}
        
        Requirements:
        1. Content should be pedagogically sound and age-appropriate
        2. Use clear, engaging language suitable for the difficulty level
        3. Include practical examples and real-world applications
        4. Structure content logically with proper headings
        5. Ensure accuracy and current information
        
        Generate comprehensive educational content that helps teachers deliver effective instruction.
        """
        
        logger.info(f"🤖 Calling Gemini model for content generation...")
        response = model.generate_content(system_prompt)
        generated_text = response.text
        
        # Extract title from content
        lines = generated_text.split('\n')
        title = lines[0].strip() if lines else prompt[:50]
        
        result = {
            "status": "success",
            "content": {
                "title": title,
                "body": generated_text,
                "type": content_type,
                "subject": subject,
                "difficulty": difficulty,
                "language": language,
                "curriculum_standard": curriculum_standard,
                "word_count": len(generated_text.split())
            }
        }
        
        logger.info(f"✅ Content generated successfully: {len(generated_text)} characters")
        return result
        
    except Exception as e:
        logger.error(f"❌ Content generation failed: {e}")
        return {
            "status": "error",
            "error_message": f"Failed to generate educational content: {str(e)}"
        }


# search_knowledge_base function moved to sahayak_retrieval_agent
# Import from retrieval agent when needed


def search_web_for_education(
    query: str,
    subject: str = "",
    content_type: str = "articles",
    max_results: int = 5
) -> dict:
    """
    Search the web for educational content and resources.
    
    Args:
        query (str): Search query for educational content
        subject (str): Subject area to focus search
        content_type (str): Type of content to search for
        max_results (int): Maximum number of results
        
    Returns:
        dict: Status and web search results
    """
    print(f"[WEB_SEARCH] search_web_for_education called with query: '{query}', subject: {subject}, type: {content_type}")
    
    try:
        # Check if API keys are configured
        api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        
        if not api_key or not engine_id:
            print("WARNING: Google Search API not configured, returning mock results")
            # Return mock educational results
            mock_results = [
                {
                    "title": f"Educational Resource on {query}",
                    "url": "https://example-edu.com/resource1",
                    "snippet": f"Comprehensive educational material about {query} suitable for {subject} students.",
                    "source": "example-edu.com"
                },
                {
                    "title": f"Teaching Guide: {query}",
                    "url": "https://teaching-portal.com/guide",
                    "snippet": f"Step-by-step teaching guide for {query} with practical examples and activities.",
                    "source": "teaching-portal.com"
                }
            ]
            
            return {
                "status": "success",
                "results": mock_results,
                "query": query,
                "note": "Mock results - API not configured"
            }
        
        # Actual Google Custom Search API implementation
        print(f"[API_CALL] Performing real web search using Google Custom Search API...")
        
        try:
            from googleapiclient.discovery import build
            from googleapiclient.errors import HttpError
        except ImportError:
            print(f"[ERROR] googleapiclient not installed. Install with: pip install google-api-python-client")
            return {
                "status": "error",
                "error_message": "Google API client library not installed"
            }
        
        try:
            # Build the Custom Search service
            service = build("customsearch", "v1", developerKey=api_key)
            
            # Enhance query with educational context
            enhanced_query = f"{query}"
            if subject:
                enhanced_query += f" {subject}"
            
            # Add educational site preferences and filters
            educational_sites = "site:edu OR site:org OR site:medium.com OR site:researchgate.net OR site:sciencedirect.com OR site:khanacademy.org OR site:coursera.org OR site:edx.org"
            educational_keywords = "education OR learning OR teaching OR curriculum OR lesson OR study"
            
            # Combine query with educational focus
            final_query = f"{enhanced_query} ({educational_keywords})"
            
            print(f"[QUERY] Enhanced search query: '{final_query}'")
            print(f"[SEARCH] Searching for {content_type} content with max {max_results} results...")
            
            # Execute the search
            search_result = service.cse().list(
                q=final_query,
                cx=engine_id,
                num=min(max_results, 10),  # Google API limits to 10 per request
                safe='active',  # Enable safe search for educational content
                gl='us',  # Geographic location
                hl='en',  # Language
                # Optional: Add date restrictions for recent content
                # dateRestrict='m1'  # Last month
            ).execute()
            
            # Parse results
            results = []
            items = search_result.get('items', [])
            
            print(f"[PROCESSING] Processing {len(items)} search results...")
            
            for i, item in enumerate(items[:max_results]):
                # Extract domain from URL for source
                from urllib.parse import urlparse
                parsed_url = urlparse(item['link'])
                source_domain = parsed_url.netloc.replace('www.', '')
                
                # Determine educational quality score based on domain and content
                educational_score = 0
                edu_domains = ['.edu', '.org', 'khanacademy', 'coursera', 'edx', 'mit', 'stanford', 'harvard', 'medium', 'researchgate', 'sciencedirect']
                for edu_domain in edu_domains:
                    if edu_domain in source_domain.lower():
                        educational_score += 1
                        break
                
                result_item = {
                    "id": i + 1,
                    "title": item.get('title', 'No title'),
                    "url": item.get('link', ''),
                    "snippet": item.get('snippet', 'No description available'),
                    "source": source_domain,
                    "educational_score": educational_score,
                    "content_type": content_type,
                    "search_rank": i + 1
                }
                
                # Add additional metadata if available
                if 'pagemap' in item:
                    pagemap = item['pagemap']
                    if 'metatags' in pagemap and pagemap['metatags']:
                        metatag = pagemap['metatags'][0]
                        result_item['author'] = metatag.get('author', '')
                        result_item['description'] = metatag.get('description', result_item['snippet'])
                
                results.append(result_item)
                print(f"  [RESULT] {i+1}. {item.get('title', 'No title')[:50]}... (from {source_domain})")
            
            # Sort results by educational quality score and search rank
            results.sort(key=lambda x: (x['educational_score'], -x['search_rank']), reverse=True)
            
            search_info = search_result.get('searchInformation', {})
            total_results = search_info.get('totalResults', '0')
            search_time = search_info.get('searchTime', 0)
            
            print(f"[SUCCESS] Found {len(results)} educational web resources")
            print(f"[STATS] Total available: {total_results}, Search time: {search_time}s")
            
            return {
                "status": "success",
                "results": results,
                "query": query,
                "enhanced_query": final_query,
                "total_available": total_results,
                "search_time": search_time,
                "results_count": len(results),
                "educational_focus": True
            }
            
        except HttpError as e:
            print(f"[API_ERROR] Google Custom Search API error: {e}")
            error_details = str(e)
            if "quotaExceeded" in error_details:
                print(f"[QUOTA] API quota exceeded, falling back to mock results")
                # Return mock results as fallback
                return {
                    "status": "success",
                    "results": mock_results,
                    "query": query,
                    "note": "API quota exceeded - showing mock results"
                }
            else:
                return {
                    "status": "error",
                    "error_message": f"Google Search API error: {str(e)}"
                }
                
        except Exception as e:
            print(f"[ERROR] Unexpected error in web search: {e}")
            return {
                "status": "error",
                "error_message": f"Web search failed: {str(e)}"
            }
        
    except Exception as e:
        print(f"[ERROR] Web search failed: {e}")
        return {
            "status": "error",
            "error_message": f"Web search failed: {str(e)}"
        }


def align_with_curriculum(
    content: str,
    curriculum: str,
    subject: str,
    grade_level: str = ""
) -> dict:
    """
    Analyze content alignment with specific curriculum standards using parallel AI + Web search with multi-dimensional scoring.
    
    Args:
        content (str): Educational content to analyze
        curriculum (str): Curriculum standard (CBSE, IB, Common Core, Cambridge)
        subject (str): Subject area
        grade_level (str): Grade or year level
        
    Returns:
        dict: Status and comprehensive curriculum alignment analysis
    """
    print(f"[CURRICULUM_ALIGN] Starting alignment analysis for {curriculum} {subject} {grade_level}")
    
    try:
        import asyncio
        import concurrent.futures
        import re
        from datetime import datetime
        
        # Step 1: Parallel Data Collection
        def get_ai_curriculum_standards():
            """Extract curriculum standards using Gemini AI"""
            try:
                print(f"[AI_QUERY] Querying Gemini for {curriculum} {subject} {grade_level} standards...")
                
                model = genai.GenerativeModel("gemini-2.0-flash")
                
                ai_prompt = f"""
                You are an expert curriculum analyst. Provide detailed, accurate curriculum standards for:
                
                Curriculum: {curriculum}
                Subject: {subject}
                Grade/Level: {grade_level}
                
                Please provide:
                1. Specific learning outcomes/objectives
                2. Key topics that must be covered
                3. Skills students should develop
                4. Assessment criteria if applicable
                
                Format your response as a structured JSON with these fields:
                {{
                    "learning_outcomes": ["outcome1", "outcome2", ...],
                    "key_topics": ["topic1", "topic2", ...],
                    "skills": ["skill1", "skill2", ...],
                    "assessment_criteria": ["criteria1", "criteria2", ...],
                    "confidence_level": "high|medium|low",
                    "source_basis": "training_data_description"
                }}
                
                Be honest about your confidence level. If you're not certain, mark confidence as "low".
                """
                
                response = model.generate_content(ai_prompt)
                raw_response = response.text
                print(f"[AI_RAW_RESPONSE] Length: {len(raw_response)} chars")
                print(f"[AI_RAW_RESPONSE] First 100 chars: {raw_response[:100]}")
                print(f"[AI_RAW_RESPONSE] Last 100 chars: {raw_response[-100:]}")
                
                try:
                    import json
                    import re
                    
                    # Method 1: Try direct parsing first
                    try:
                        ai_data = json.loads(raw_response)
                        print(f"[JSON_SUCCESS] Direct parsing worked")
                    except json.JSONDecodeError as e:
                        print(f"[JSON_DIRECT_FAIL] {e}")
                        
                        # Method 2: Extract JSON from markdown code blocks
                        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
                        json_match = re.search(json_pattern, raw_response, re.DOTALL)
                        
                        if json_match:
                            json_text = json_match.group(1)
                            print(f"[JSON_MARKDOWN] Found JSON in code blocks")
                            ai_data = json.loads(json_text)
                        else:
                            # Method 3: Look for JSON object boundaries
                            # Find first { and last } to extract JSON
                            start_idx = raw_response.find('{')
                            end_idx = raw_response.rfind('}')
                            
                            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                                json_text = raw_response[start_idx:end_idx+1]
                                print(f"[JSON_EXTRACT] Extracted JSON from position {start_idx} to {end_idx}")
                                print(f"[JSON_EXTRACT] Extracted text: {json_text[:200]}...")
                                ai_data = json.loads(json_text)
                            else:
                                raise json.JSONDecodeError("No valid JSON found", raw_response, 0)
                    
                    ai_confidence = ai_data.get("confidence_level", "low")
                    
                    print(f"[AI_RESULT] Successfully parsed JSON with confidence: {ai_confidence}")
                    print(f"[AI_RESULT] Found {len(ai_data.get('learning_outcomes', []))} learning outcomes")
                    print(f"[AI_RESULT] Found {len(ai_data.get('key_topics', []))} key topics")
                    
                    return {
                        "status": "success",
                        "data": ai_data,
                        "source": "gemini_ai",
                        "confidence": ai_confidence,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                except json.JSONDecodeError as e:
                    # Final fallback parsing if JSON extraction fails
                    print(f"[AI_FALLBACK] All JSON parsing methods failed: {e}")
                    print(f"[AI_FALLBACK] Response type: {type(raw_response)}")
                    print(f"[AI_FALLBACK] Response encoding issues? {repr(raw_response[:200])}")
                    
                    return {
                        "status": "partial",
                        "data": {
                            "raw_response": raw_response,
                            "confidence_level": "medium",
                            "parsing_error": str(e)
                        },
                        "source": "gemini_ai",
                        "confidence": "medium", 
                        "timestamp": datetime.now().isoformat()
                    }
                    
            except Exception as e:
                print(f"[AI_ERROR] Gemini query failed: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "source": "gemini_ai"
                }
        
        def get_web_curriculum_standards():
            """Search for official curriculum documents on the web"""
            try:
                print(f"[WEB_QUERY] Searching web for {curriculum} {subject} {grade_level} curriculum documents...")
                
                # Construct targeted search query for curriculum documents
                search_queries = [
                    f"{curriculum} {subject} {grade_level} curriculum standards filetype:pdf",
                    f"{curriculum} {subject} {grade_level} learning outcomes official",
                    f'"{curriculum}" "{subject}" "{grade_level}" syllabus site:gov OR site:edu OR site:org'
                ]
                
                best_results = []
                
                for query in search_queries:
                    try:
                        web_results = search_web_for_education(
                            query=query,
                            subject=subject,
                            content_type="curriculum",
                            max_results=3
                        )
                        
                        if web_results.get("status") == "success":
                            for result in web_results.get("results", []):
                                # Calculate source authority score
                                domain = result.get("source", "").lower()
                                authority_score = calculate_source_authority(domain, curriculum)
                                
                                result["authority_score"] = authority_score
                                best_results.append(result)
                            
                            if best_results:
                                break  # Got good results, stop searching
                                
                    except Exception as e:
                        print(f"[WEB_ERROR] Search query failed: {e}")
                        continue
                
                # Sort by authority score and take top results
                best_results.sort(key=lambda x: x.get("authority_score", 0), reverse=True)
                top_results = best_results[:3]
                
                print(f"[WEB_RESULT] Found {len(top_results)} curriculum documents")
                return {
                    "status": "success" if top_results else "no_results",
                    "data": top_results,
                    "source": "web_search",
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                print(f"[WEB_ERROR] Web search failed: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "source": "web_search"
                }
        
        def calculate_source_authority(domain: str, curriculum: str) -> float:
            """Calculate authority score for a domain based on curriculum"""
            print(f"[AUTHORITY] Calculating authority for domain: {domain}")
            
            # Official government and education authority domains
            official_domains = {
                "CBSE": ["cbse.gov.in", "cbse.nic.in", "ncert.nic.in", "mhrd.gov.in", "education.gov.in"],
                "IB": ["ibo.org", "ibdiploma.org"],
                "Common Core": ["corestandards.org", "achievethecore.org", "ed.gov"],
                "Cambridge": ["cambridgeinternational.org", "cie.org.uk"]
            }
            
            # Educational organization domains
            edu_orgs = ["khanacademy.org", "edutopia.org", "teachertoolkit.org", "scholastic.com"]
            
            # Government and official domains
            gov_indicators = [".gov", ".edu", ".org"]
            
            domain_lower = domain.lower()
            
            # Check for curriculum-specific official sources
            if curriculum in official_domains:
                for official_domain in official_domains[curriculum]:
                    if official_domain in domain_lower:
                        return 1.0  # Maximum authority
            
            # Check for educational organizations
            for edu_org in edu_orgs:
                if edu_org in domain_lower:
                    return 0.8
            
            # Check for government/educational domains
            for indicator in gov_indicators:
                if indicator in domain_lower:
                    return 0.7
            
            # Default for other domains
            return 0.4
        
        def calculate_content_completeness(data: dict, source_type: str) -> float:
            """Calculate how complete the curriculum information is"""
            if source_type == "ai":
                # For AI results
                if isinstance(data, dict):
                    completeness = 0.0
                    if data.get("learning_outcomes"):
                        completeness += 0.3
                    if data.get("key_topics"):
                        completeness += 0.3
                    if data.get("skills"):
                        completeness += 0.2
                    if data.get("assessment_criteria"):
                        completeness += 0.2
                    return completeness
                else:
                    return 0.3  # Partial for raw text
            
            elif source_type == "web":
                # For web results - assess based on snippet content richness
                if not data:
                    return 0.0
                
                snippet_length = len(str(data.get("snippet", "")))
                if snippet_length > 200:
                    return 1.0
                elif snippet_length > 100:
                    return 0.6
                else:
                    return 0.3
            
            return 0.0
        
        def calculate_recency_score(timestamp_str: str) -> float:
            """Calculate recency score based on current date"""
            try:
                current_year = datetime.now().year
                
                # For web results, try to extract year from various sources
                if timestamp_str:
                    # Look for 4-digit years in the string
                    years = re.findall(r'20[0-9]{2}', str(timestamp_str))
                    if years:
                        year = int(years[-1])  # Take the most recent year found
                        age = current_year - year
                        
                        if age == 0:
                            return 1.0  # Current year
                        elif age == 1:
                            return 0.9  # Last year
                        elif age <= 3:
                            return 0.7  # Within 3 years
                        elif age <= 5:
                            return 0.5  # Within 5 years
                        else:
                            return 0.3  # Older than 5 years
                
                return 0.6  # Default if no date found
                
            except:
                return 0.6  # Default fallback
        
        # Step 2: Execute parallel data collection
        print(f"[PARALLEL] Starting parallel AI and Web queries...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            ai_future = executor.submit(get_ai_curriculum_standards)
            web_future = executor.submit(get_web_curriculum_standards)
            
            ai_result = ai_future.result(timeout=30)  # 30 second timeout
            web_result = web_future.result(timeout=30)  # 30 second timeout
        
        print(f"[PARALLEL] Completed parallel queries - AI: {ai_result.get('status')}, Web: {web_result.get('status')}")
        
        # Step 3: Multi-dimensional scoring
        ai_scores = {}
        web_scores = {}
        
        if ai_result.get("status") in ["success", "partial"]:
            ai_data = ai_result.get("data", {})
            ai_scores = {
                "source_authority": 0.6,  # AI has medium authority
                "content_completeness": calculate_content_completeness(ai_data, "ai"),
                "recency": 0.8,  # AI training data is relatively recent
                "confidence": 0.9 if ai_result.get("confidence") == "high" else 0.7 if ai_result.get("confidence") == "medium" else 0.4
            }
            ai_scores["overall"] = sum(ai_scores.values()) / len(ai_scores)
        
        if web_result.get("status") == "success" and web_result.get("data"):
            best_web_result = web_result["data"][0]  # Top result
            web_scores = {
                "source_authority": best_web_result.get("authority_score", 0.4),
                "content_completeness": calculate_content_completeness(best_web_result, "web"),
                "recency": calculate_recency_score(best_web_result.get("snippet", "")),
                "confidence": 0.8  # Web documents generally reliable
            }
            web_scores["overall"] = sum(web_scores.values()) / len(web_scores)
        
        print(f"[SCORING] AI overall score: {ai_scores.get('overall', 0):.2f}, Web overall score: {web_scores.get('overall', 0):.2f}")
        
        # Step 4: Smart Content alignment analysis using AI semantic understanding
        def analyze_content_alignment(content: str, standards_data: dict, source_type: str) -> dict:
            """Analyze how well content aligns with curriculum standards using AI semantic analysis"""
            print(f"[ALIGNMENT_ANALYSIS] Starting {source_type} semantic analysis...")
            
            alignment_details = {
                "aligned_topics": [],
                "missing_topics": [],
                "coverage_score": 0.0,
                "detailed_analysis": {},
                "confidence": "low"
            }
            
            if source_type == "ai" and isinstance(standards_data, dict):
                try:
                    # Use AI for semantic curriculum alignment analysis
                    model = genai.GenerativeModel("gemini-2.0-flash")
                    
                    learning_outcomes = standards_data.get('learning_outcomes', [])[:15]  # Limit to avoid token limits
                    key_topics = standards_data.get('key_topics', [])
                    skills = standards_data.get('skills', [])[:10]
                    
                    alignment_prompt = f"""
                    You are an expert educational curriculum analyst. Analyze how well the given educational content aligns with the specified curriculum standards.
                    
                    CONTENT TO ANALYZE:
                    \"\"\"{content[:2000]}\"\"\"
                    
                    CURRICULUM STANDARDS TO CHECK AGAINST:
                    
                    Learning Outcomes:
                    {chr(10).join(f"- {outcome}" for outcome in learning_outcomes)}
                    
                    Key Topics:
                    {chr(10).join(f"- {topic}" for topic in key_topics)}
                    
                    Skills:
                    {chr(10).join(f"- {skill}" for skill in skills)}
                    
                    Analyze the content and provide your assessment as JSON:
                    {{
                        "overall_alignment_score": 0.75,  // 0.0 to 1.0 based on how well content matches curriculum
                        "topic_coverage": {{
                            "fully_covered": ["topic1", "topic2"],  // Topics thoroughly addressed
                            "partially_covered": ["topic3", "topic4"],  // Topics mentioned but not detailed
                            "not_covered": ["topic5", "topic6"]  // Required topics missing
                        }},
                        "learning_outcome_alignment": {{
                            "addressed_outcomes": ["outcome1", "outcome2"],  // Learning outcomes the content addresses
                            "missed_outcomes": ["outcome3", "outcome4"]  // Critical outcomes not addressed
                        }},
                        "skill_development": {{
                            "developed_skills": ["skill1", "skill2"],  // Skills this content helps develop
                            "underdeveloped_skills": ["skill3", "skill4"]  // Skills needing more attention
                        }},
                        "content_quality": {{
                            "depth_score": 0.8,  // How thoroughly topics are covered (0-1)
                            "clarity_score": 0.9,  // How clearly concepts are explained (0-1)
                            "engagement_score": 0.7,  // How engaging the content is (0-1)
                            "age_appropriateness": 0.85  // How suitable for grade level (0-1)
                        }},
                        "recommendations": [
                            "Add more examples of quadratic equation applications",
                            "Include practice problems for weak areas"
                        ],
                        "analysis_confidence": "high",  // high|medium|low - how confident you are in this analysis
                        "rationale": "The content covers fundamental concepts well but lacks advanced applications..."
                    }}
                    
                    Be thorough in your analysis. Consider:
                    1. Does the content teach what the curriculum requires?
                    2. Are explanations appropriate for the grade level?
                    3. Does it develop the required skills?
                    4. What's missing that should be added?
                    5. Is the depth appropriate for curriculum requirements?
                    
                    IMPORTANT: Be honest about your confidence level. If the curriculum standards are unclear or you're uncertain, mark confidence as "low".
                    """
                    
                    print(f"[AI_SEMANTIC] Querying Gemini for semantic alignment analysis...")
                    response = model.generate_content(alignment_prompt)
                    raw_response = response.text
                    
                    # Parse AI alignment response with the same robust parsing as before
                    try:
                        import json
                        import re
                        
                        # Try multiple parsing methods
                        ai_alignment_data = None
                        
                        try:
                            ai_alignment_data = json.loads(raw_response)
                            print(f"[SEMANTIC_SUCCESS] Direct JSON parsing worked")
                        except json.JSONDecodeError:
                            # Try extracting from markdown
                            json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
                            json_match = re.search(json_pattern, raw_response, re.DOTALL)
                            
                            if json_match:
                                json_text = json_match.group(1)
                                ai_alignment_data = json.loads(json_text)
                                print(f"[SEMANTIC_MARKDOWN] Extracted from markdown blocks")
                            else:
                                # Extract JSON boundaries
                                start_idx = raw_response.find('{')
                                end_idx = raw_response.rfind('}')
                                
                                if start_idx != -1 and end_idx != -1:
                                    json_text = raw_response[start_idx:end_idx+1]
                                    ai_alignment_data = json.loads(json_text)
                                    print(f"[SEMANTIC_EXTRACT] Extracted JSON boundaries")
                        
                        if ai_alignment_data:
                            # Process the semantic analysis results
                            overall_score = ai_alignment_data.get("overall_alignment_score", 0.0)
                            topic_coverage = ai_alignment_data.get("topic_coverage", {})
                            outcome_alignment = ai_alignment_data.get("learning_outcome_alignment", {})
                            content_quality = ai_alignment_data.get("content_quality", {})
                            
                            alignment_details = {
                                "coverage_score": float(overall_score),
                                "aligned_topics": topic_coverage.get("fully_covered", []) + topic_coverage.get("partially_covered", []),
                                "missing_topics": topic_coverage.get("not_covered", []),
                                "confidence": ai_alignment_data.get("analysis_confidence", "medium"),
                                "detailed_analysis": {
                                    "alignment_method": "ai_semantic_analysis",
                                    "topic_coverage": topic_coverage,
                                    "learning_outcomes": outcome_alignment,
                                    "skill_development": ai_alignment_data.get("skill_development", {}),
                                    "content_quality_scores": content_quality,
                                    "recommendations": ai_alignment_data.get("recommendations", []),
                                    "rationale": ai_alignment_data.get("rationale", ""),
                                    "total_topics_analyzed": len(key_topics),
                                    "total_outcomes_analyzed": len(learning_outcomes)
                                }
                            }
                            
                            print(f"[SEMANTIC_RESULT] Overall alignment: {overall_score:.2%}, Confidence: {alignment_details['confidence']}")
                            print(f"[SEMANTIC_TOPICS] Covered: {len(topic_coverage.get('fully_covered', []))}, Missing: {len(topic_coverage.get('not_covered', []))}")
                            
                        else:
                            raise json.JSONDecodeError("No valid JSON extracted", raw_response, 0)
                            
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        print(f"[SEMANTIC_FALLBACK] AI analysis parsing failed: {e}")
                        # Fallback to basic text analysis
                        alignment_details = {
                            "coverage_score": 0.3,  # Conservative estimate
                            "aligned_topics": ["Content analysis unavailable"],
                            "missing_topics": ["Unable to determine specific gaps"],
                            "confidence": "low",
                            "detailed_analysis": {
                                "alignment_method": "fallback_text_analysis",
                                "error": str(e),
                                "raw_response_length": len(raw_response),
                                "note": "AI semantic analysis failed, using conservative estimates"
                            }
                        }
                        
                except Exception as e:
                    print(f"[SEMANTIC_ERROR] AI semantic analysis failed: {e}")
                    alignment_details = {
                        "coverage_score": 0.0,
                        "aligned_topics": [],
                        "missing_topics": ["Analysis failed"],
                        "confidence": "low",
                        "detailed_analysis": {
                            "alignment_method": "error_fallback",
                            "error": str(e)
                        }
                    }
            
            elif source_type == "web":
                try:
                    # Smart web document analysis using AI
                    document_title = standards_data.get("title", "")
                    document_snippet = standards_data.get("snippet", "")
                    document_url = standards_data.get("url", "")
                    
                    # Only do web analysis if we have meaningful document data
                    if len(document_snippet) > 50:
                        model = genai.GenerativeModel("gemini-2.0-flash")
                        
                        web_alignment_prompt = f"""
                        Analyze how well the educational content aligns with the curriculum document found online.
                        
                        CONTENT TO ANALYZE:
                        \"\"\"{content[:1500]}\"\"\"
                        
                        CURRICULUM DOCUMENT REFERENCE:
                        Title: {document_title}
                        URL: {document_url}
                        Description: {document_snippet}
                        
                        Based on the document description, assess alignment as JSON:
                        {{
                            "document_relevance": 0.8,  // How relevant is this document to the content (0-1)
                            "alignment_score": 0.6,  // How well content matches document expectations (0-1)
                            "confidence": "medium",  // high|medium|low
                            "aligned_aspects": ["aspect1", "aspect2"],
                            "potential_gaps": ["gap1", "gap2"],
                            "document_authority": "high",  // high|medium|low based on source
                            "analysis_note": "Content partially aligns with curriculum document..."
                        }}
                        
                        Consider:
                        1. Is this document actually relevant to curriculum analysis?
                        2. Does the content match what the document suggests should be taught?
                        3. How authoritative is this source?
                        """
                        
                        print(f"[WEB_SEMANTIC] Analyzing alignment with web document...")
                        response = model.generate_content(web_alignment_prompt)
                        
                        try:
                            # Parse web alignment response
                            import json
                            import re
                            
                            web_data = None
                            raw_response = response.text
                            
                            try:
                                web_data = json.loads(raw_response)
                            except json.JSONDecodeError:
                                json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
                                json_match = re.search(json_pattern, raw_response, re.DOTALL)
                                if json_match:
                                    web_data = json.loads(json_match.group(1))
                                else:
                                    start_idx = raw_response.find('{')
                                    end_idx = raw_response.rfind('}')
                                    if start_idx != -1 and end_idx != -1:
                                        web_data = json.loads(raw_response[start_idx:end_idx+1])
                            
                            if web_data:
                                document_relevance = web_data.get("document_relevance", 0.5)
                                
                                # Only use web analysis if document is relevant
                                if document_relevance > 0.6:
                                    alignment_details = {
                                        "coverage_score": float(web_data.get("alignment_score", 0.0)),
                                        "aligned_topics": web_data.get("aligned_aspects", []),
                                        "missing_topics": web_data.get("potential_gaps", []),
                                        "confidence": web_data.get("confidence", "low"),
                                        "detailed_analysis": {
                                            "alignment_method": "web_document_semantic_analysis",
                                            "document_relevance": document_relevance,
                                            "document_authority": web_data.get("document_authority", "medium"),
                                            "document_title": document_title,
                                            "document_source": standards_data.get("source", ""),
                                            "analysis_note": web_data.get("analysis_note", ""),
                                            "authority_score": standards_data.get("authority_score", 0.5)
                                        }
                                    }
                                    print(f"[WEB_SEMANTIC_SUCCESS] Document relevance: {document_relevance:.2f}, Alignment: {alignment_details['coverage_score']:.2%}")
                                else:
                                    print(f"[WEB_SEMANTIC_SKIP] Document relevance too low: {document_relevance:.2f}")
                                    alignment_details["coverage_score"] = 0.0
                                    alignment_details["detailed_analysis"]["skip_reason"] = "Document not sufficiently relevant"
                            
                        except Exception as e:
                            print(f"[WEB_SEMANTIC_ERROR] Web document analysis failed: {e}")
                            alignment_details["coverage_score"] = 0.0
                            alignment_details["detailed_analysis"]["error"] = str(e)
                    
                    else:
                        print(f"[WEB_SKIP] Document snippet too short for meaningful analysis")
                        alignment_details["coverage_score"] = 0.0
                        alignment_details["detailed_analysis"]["skip_reason"] = "Insufficient document content"
                        
                except Exception as e:
                    print(f"[WEB_ERROR] Web document analysis failed: {e}")
                    alignment_details = {
                        "coverage_score": 0.0,
                        "aligned_topics": [],
                        "missing_topics": [],
                        "confidence": "low",
                        "detailed_analysis": {
                            "alignment_method": "web_error_fallback",
                            "error": str(e)
                        }
                    }
            
            return alignment_details
        
        # Perform alignment analysis for both sources
        ai_alignment = {}
        web_alignment = {}
        
        if ai_result.get("status") in ["success", "partial"]:
            ai_alignment = analyze_content_alignment(content, ai_result.get("data", {}), "ai")
            print(f"[AI_ALIGNMENT] Coverage score: {ai_alignment.get('coverage_score', 0):.2f}")
        
        if web_result.get("status") == "success" and web_result.get("data"):
            web_alignment = analyze_content_alignment(content, web_result["data"][0], "web")
            print(f"[WEB_ALIGNMENT] Coverage score: {web_alignment.get('coverage_score', 0):.2f}")
        
        # Step 5: Intelligent fusion of results
        final_result = {
            "status": "success",
            "curriculum": curriculum,
            "subject": subject,
            "grade_level": grade_level,
            "analysis_timestamp": datetime.now().isoformat(),
            "sources_used": [],
            "fusion_strategy": "",
            "final_alignment_score": 0.0,
            "recommendations": [],
            "detailed_breakdown": {
                "ai_analysis": {},
                "web_analysis": {},
                "scoring_details": {}
            }
        }
        
        # Determine fusion strategy based on source quality
        if ai_scores.get("overall", 0) > 0.7 and web_scores.get("overall", 0) > 0.7:
            # Both sources are high quality - merge results
            final_result["fusion_strategy"] = "high_quality_merge"
            final_result["sources_used"] = ["ai", "web"]
            
            # Weighted average based on overall scores
            ai_weight = ai_scores["overall"] / (ai_scores["overall"] + web_scores["overall"])
            web_weight = web_scores["overall"] / (ai_scores["overall"] + web_scores["overall"])
            
            final_result["final_alignment_score"] = (
                ai_alignment.get("coverage_score", 0) * ai_weight +
                web_alignment.get("coverage_score", 0) * web_weight
            )
            
            # Merge recommendations
            final_result["recommendations"] = [
                f"Content shows {final_result['final_alignment_score']:.1%} alignment with {curriculum} standards",
                "Analysis based on both AI knowledge and official curriculum documents",
                "High confidence in results due to multiple source validation"
            ]
            
        elif ai_scores.get("overall", 0) > web_scores.get("overall", 0):
            # AI source is better
            final_result["fusion_strategy"] = "ai_primary"
            final_result["sources_used"] = ["ai"]
            final_result["final_alignment_score"] = ai_alignment.get("coverage_score", 0)
            
            final_result["recommendations"] = [
                f"Content shows {final_result['final_alignment_score']:.1%} alignment based on AI analysis",
                "Analysis based on AI knowledge of curriculum standards",
                f"AI confidence level: {ai_result.get('confidence', 'medium')}"
            ]
            
        elif web_scores.get("overall", 0) > 0:
            # Web source is better
            final_result["fusion_strategy"] = "web_primary"
            final_result["sources_used"] = ["web"]
            final_result["final_alignment_score"] = web_alignment.get("coverage_score", 0)
            
            web_doc = web_result["data"][0]
            final_result["recommendations"] = [
                f"Content shows {final_result['final_alignment_score']:.1%} alignment based on official documents",
                f"Reference document: {web_doc.get('title', 'Official curriculum document')}",
                f"Source authority: {web_doc.get('authority_score', 0):.1f}/1.0"
            ]
            
        else:
            # Both sources failed or low quality
            final_result["fusion_strategy"] = "fallback"
            final_result["sources_used"] = []
            final_result["final_alignment_score"] = 0.5  # Default neutral score
            final_result["recommendations"] = [
                "Unable to retrieve reliable curriculum standards for comparison",
                "Content analysis based on general educational principles",
                "Consider manual verification with official curriculum documents"
            ]
        
        # Add detailed breakdown
        final_result["detailed_breakdown"] = {
            "ai_analysis": {
                "scores": ai_scores,
                "alignment": ai_alignment,
                "raw_result": ai_result
            },
            "web_analysis": {
                "scores": web_scores,
                "alignment": web_alignment,
                "raw_result": web_result
            },
            "scoring_details": {
                "ai_overall": ai_scores.get("overall", 0),
                "web_overall": web_scores.get("overall", 0),
                "fusion_rationale": f"Selected {final_result['fusion_strategy']} strategy based on source quality comparison"
            }
        }
        
        print(f"[FUSION_COMPLETE] Final alignment score: {final_result['final_alignment_score']:.2%} using {final_result['fusion_strategy']} strategy")
        
        return final_result
        
    except Exception as e:
        print(f"[ERROR] Curriculum alignment failed: {e}")
        return {
            "status": "error",
            "error_message": f"Curriculum alignment analysis failed: {str(e)}",
            "curriculum": curriculum,
            "subject": subject,
            "grade_level": grade_level
        }


def create_assessment_questions(
    topic: str,
    subject: str = "General",
    difficulty: str = "intermediate",
    question_count: int = 5,
    question_types: str = "mixed"
) -> dict:
    """
    Create assessment questions and quizzes for educational topics.
    
    Args:
        topic (str): The topic to create questions about
        subject (str): Subject area (e.g., Mathematics, Science, History)
        difficulty (str): Difficulty level (beginner, intermediate, advanced)
        question_count (int): Number of questions to generate
        question_types (str): Types of questions (mcq, short_answer, essay, mixed)
        
    Returns:
        dict: Status and generated assessment questions
    """
    logger.info(f"📝 create_assessment_questions called for topic: '{topic}', count: {question_count}, type: {question_types}")
    
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Create assessment prompt
        system_prompt = f"""
        Create {question_count} assessment questions about '{topic}' in {subject}.
        
        Requirements:
        - Difficulty level: {difficulty}
        - Question types: {question_types}
        - Include answer keys for all questions
        - Vary question formats (multiple choice, short answer, etc.)
        - Ensure questions test understanding, not just memorization
        - Include questions that require application of concepts
        
        Format as JSON:
        {{
            "questions": [
                {{
                    "id": 1,
                    "type": "multiple_choice",
                    "question": "Question text",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer": "A",
                    "explanation": "Why this is correct"
                }}
            ]
        }}
        """
        
        logger.info(f"🤖 Generating assessment questions...")
        response = model.generate_content(system_prompt)
        
        # Try to parse JSON response
        try:
            import json
            questions_data = json.loads(response.text)
        except:
            # Fallback to simple format
            questions_data = {
                "questions": [
                    {
                        "id": 1,
                        "type": "short_answer",
                        "question": f"Explain the key concepts of {topic}",
                        "correct_answer": "Generated answer based on content",
                        "explanation": "Assessment of understanding"
                    }
                ]
            }
        
        result = {
            "status": "success",
            "assessment": {
                "topic": topic,
                "subject": subject,
                "difficulty": difficulty,
                "question_count": len(questions_data.get("questions", [])),
                "questions": questions_data.get("questions", [])
            }
        }
        
        logger.info(f"✅ Created {len(questions_data.get('questions', []))} assessment questions")
        return result
        
    except Exception as e:
        logger.error(f"❌ Assessment creation failed: {e}")
        return {
            "status": "error",
            "error_message": f"Failed to create assessment: {str(e)}"
        }


# Create the ADK Agent with all educational tools
root_agent = Agent(
    name="sahayak_content_agent",
    model="gemini-2.0-flash",
    description="Educational content generation specialist with curriculum alignment capabilities",
    instruction=(
        "You are an expert educational content creator specializing in generating high-quality "
        "instructional materials for teachers and students. Your expertise includes:\n\n"
        "1. Creating comprehensive lesson plans aligned with curriculum standards (CBSE, IB, Common Core, Cambridge)\n"
        "2. Generating clear explanations of complex concepts for different grade levels\n"
        "3. Developing engaging educational activities and assessments\n"
        "4. Supporting multilingual content generation (English, Hindi, Spanish, French)\n"
        "5. Aligning content with specific difficulty levels (beginner, intermediate, advanced)\n\n"
        "Always ensure your content is:\n"
        "- Pedagogically sound and age-appropriate\n"
        "- Engaging and interactive\n"
        "- Aligned with specified curriculum standards\n"
        "- Clear and well-structured\n"
        "- Supported by relevant examples and real-world applications\n\n"
        "When generating content, consider the target audience, subject matter, difficulty level, "
        "and curriculum requirements. Use your available tools to search for additional context, "
        "align with curriculum standards, and create comprehensive educational materials."
    ),
    tools=[
        generate_educational_content,
        search_web_for_education,
        align_with_curriculum,
        create_assessment_questions
    ]
)

if __name__ == "__main__":
    # For testing purposes
    print(f"🎓 Agent {root_agent.name} initialized successfully with {len(root_agent.tools)} tools")
    for i, tool in enumerate(root_agent.tools, 1):
        print(f"  {i}. {tool.__name__}")