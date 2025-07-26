"""
Sahayak Retrieval Agent - Database & Storage Retrieval Specialist

This agent specializes in:
1. Unified intelligent search across all educational content types
2. Querying file storage systems (Google Cloud Storage)
3. Advanced metadata-based searches and filtering
4. Resource discovery and recommendation
5. Educational material indexing and categorization
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from google.cloud import firestore
from google.cloud import storage

# Load environment variables
from dotenv import load_dotenv
# Load from current directory first, then parent directories
load_dotenv(override=True)

# Google ADK imports
from google.adk.agents import Agent

# Configure logging to show in terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def search_educational_content(
    query: str,
    content_types: List[str] = ["all"],
    subject: str = "",
    difficulty: str = "",
    grade_level: str = "",
    curriculum: str = "",
    file_format: str = "",
    limit: int = 20
) -> dict:
    """
    UNIFIED educational content search function.
    
    Replaces search_knowledge_base, retrieve_worksheets, and find_documents
    with a single intelligent search across all educational collections.
    
    Args:
        query (str): Search query for educational content
        content_types (list): Types of content to search ["lessons", "worksheets", "documents", "all"]
        subject (str): Subject area filter (Mathematics, Science, etc.)
        difficulty (str): Difficulty level (beginner/easy, intermediate/medium, advanced/hard)
        grade_level (str): Grade level filter (Grade 6, Grade 7, etc.)
        curriculum (str): Curriculum standard (CBSE, IB, Cambridge, etc.)
        file_format (str): File format filter (pdf, pptx, docx, etc.)
        limit (int): Maximum total results to return
        
    Returns:
        dict: Unified search results from all relevant collections
    """
    print(f"[UNIFIED_SEARCH] Searching for '{query}' across content types: {content_types}")
    
    try:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        database_id = os.getenv("FIRESTORE_DATABASE_ID", "(default)")
        
        if not project_id:
            return {
                "status": "error",
                "error_message": "Google Cloud Project ID not configured"
            }
        
        # Initialize Firestore client
        db = firestore.Client(project=project_id, database=database_id)
        print(f"[DATABASE] Connected for unified search to '{database_id}' in project {project_id}")
        
        # Map content types to collections
        collection_mapping = _get_collection_mapping(content_types)
        print(f"[MAPPING] Will search collections: {list(collection_mapping.keys())}")
        
        # Search each relevant collection
        all_results = []
        search_stats = {}
        
        for collection_name, search_config in collection_mapping.items():
            try:
                print(f"[COLLECTION] Searching {collection_name}...")
                
                # Build collection-specific query
                collection_results = _search_collection(
                    db, collection_name, query, subject, difficulty, 
                    grade_level, curriculum, file_format, search_config, limit
                )
                
                # Add collection source to each result
                for result in collection_results:
                    result["source_collection"] = collection_name
                    result["content_category"] = search_config["category"]
                
                all_results.extend(collection_results)
                search_stats[collection_name] = len(collection_results)
                
                print(f"[RESULTS] Found {len(collection_results)} results in {collection_name}")
                
            except Exception as e:
                error_msg = str(e)
                print(f"[ERROR] Failed to search {collection_name}: {e}")
                
                # Check if this is a Firestore index error
                if "requires an index" in error_msg or "400" in error_msg:
                    print(f"[FALLBACK] Firestore index missing for {collection_name}, trying Cloud Storage...")
                    
                    # Attempt fallback to Cloud Storage
                    fallback_results = _fallback_to_storage_search(
                        collection_name, query, subject, difficulty, grade_level, 
                        curriculum, file_format, search_config, limit
                    )
                    
                    if fallback_results:
                        print(f"[FALLBACK_SUCCESS] Found {len(fallback_results)} results from Cloud Storage for {collection_name}")
                        # Add collection source to each result
                        for result in fallback_results:
                            result["source_collection"] = collection_name
                            result["content_category"] = search_config["category"]
                            result["search_method"] = "cloud_storage_fallback"
                        
                        all_results.extend(fallback_results)
                        search_stats[collection_name] = f"storage_fallback: {len(fallback_results)}"
                    else:
                        search_stats[collection_name] = f"error: {error_msg}"
                else:
                    search_stats[collection_name] = f"error: {error_msg}"
                continue
        
        # Merge, rank, and limit results
        final_results = _merge_and_rank_results(all_results, query, limit)
        
        # Categorize results for easy consumption
        categorized_results = _categorize_results(final_results)
        
        print(f"[UNIFIED_SUCCESS] Found {len(final_results)} total results across all collections")
        
        # Check if any results came from storage fallback
        fallback_used = any(result.get("search_method") == "cloud_storage_fallback" for result in final_results)
        fallback_count = sum(1 for result in final_results if result.get("search_method") == "cloud_storage_fallback")
        
        return {
            "status": "success",
            "results": final_results,
            "categorized_results": categorized_results,
            "search_metadata": {
                "query": query,
                "content_types_requested": content_types,
                "collections_searched": list(collection_mapping.keys()),
                "collection_results": search_stats,
                "total_results": len(final_results),
                "fallback_used": fallback_used,
                "fallback_results_count": fallback_count,
                "search_methods": {
                    "firestore": len(final_results) - fallback_count,
                    "cloud_storage_fallback": fallback_count
                },
                "filters_applied": {
                    "subject": subject if subject else None,
                    "difficulty": difficulty if difficulty else None,
                    "grade_level": grade_level if grade_level else None,
                    "curriculum": curriculum if curriculum else None,
                    "file_format": file_format if file_format else None
                },
                "search_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Unified search failed: {e}")
        return {
            "status": "error",
            "error_message": f"Unified search failed: {str(e)}"
        }


def _get_collection_mapping(content_types: List[str]) -> Dict[str, Any]:
    """
    Map user-friendly content types to Firestore collections with search configuration.
    
    Returns:
        dict: Mapping of collection_name -> search_config
    """
    collections = {}
    
    # Educational Resources Collection
    if "all" in content_types or any(t in content_types for t in ["lessons", "articles", "explanations", "resources"]):
        collections["educational_resources"] = {
            "category": "educational_resource",
            "difficulty_field": "difficulty_level",
            "type_field": "resource_type",
            "curriculum_field": "curriculum_alignment",
            "sort_field": "created_date"
        }
    
    # Worksheets Collection  
    if "all" in content_types or "worksheets" in content_types:
        collections["worksheets"] = {
            "category": "worksheet",
            "difficulty_field": "difficulty", 
            "type_field": "worksheet_type",
            "curriculum_field": "curriculum",
            "sort_field": "rating"
        }
    
    # Educational Documents Collection
    if "all" in content_types or any(t in content_types for t in ["documents", "presentations", "files", "pdfs"]):
        collections["educational_documents"] = {
            "category": "document",
            "difficulty_field": None,  # Documents don't have difficulty
            "type_field": "document_type", 
            "curriculum_field": "curriculum",
            "sort_field": "upload_date"
        }
    
    return collections


def _search_collection(
    db, collection_name: str, query: str, subject: str, difficulty: str,
    grade_level: str, curriculum: str, file_format: str, config: Dict[str, Any], limit: int
) -> List[Dict[str, Any]]:
    """
    Search a specific Firestore collection with normalized field mapping.
    
    Returns:
        list: Search results from the collection
    """
    collection_ref = db.collection(collection_name)
    
    # Apply filters based on collection configuration
    filters_applied = []
    
    # Subject filter (common across all collections)
    if subject:
        collection_ref = collection_ref.where("subject", "==", subject)
        filters_applied.append(f"subject={subject}")
    
    # Difficulty filter (with field name normalization)
    if difficulty and config["difficulty_field"]:
        # Normalize difficulty values
        normalized_difficulty = _normalize_difficulty(difficulty, collection_name)
        if normalized_difficulty:
            collection_ref = collection_ref.where(config["difficulty_field"], "==", normalized_difficulty)
            filters_applied.append(f"difficulty={normalized_difficulty}")
    
    # Grade level filter (worksheets and some resources)
    if grade_level:
        if collection_name == "worksheets":
            collection_ref = collection_ref.where("grade_level", "==", grade_level)
            filters_applied.append(f"grade_level={grade_level}")
        elif collection_name in ["educational_resources", "educational_documents"]:
            collection_ref = collection_ref.where("grade_levels", "array_contains", grade_level)
            filters_applied.append(f"grade_levels_contains={grade_level}")
    
    # Curriculum filter (with field name normalization)
    if curriculum and config["curriculum_field"]:
        if collection_name == "worksheets":
            collection_ref = collection_ref.where(config["curriculum_field"], "==", curriculum)
        else:
            collection_ref = collection_ref.where(config["curriculum_field"], "array_contains", curriculum)
        filters_applied.append(f"curriculum={curriculum}")
    
    # File format filter (mainly for documents)
    if file_format and collection_name == "educational_documents":
        collection_ref = collection_ref.where("file_format", "==", file_format)
        filters_applied.append(f"file_format={file_format}")
    
    # Order and limit
    if config["sort_field"]:
        # Use descending order for dates and ratings
        direction = firestore.Query.DESCENDING if config["sort_field"] in ["created_date", "upload_date", "rating"] else firestore.Query.ASCENDING
        collection_ref = collection_ref.order_by(config["sort_field"], direction=direction)
    
    collection_ref = collection_ref.limit(limit)
    
    # Execute query
    docs = collection_ref.stream()
    
    results = []
    query_lower = query.lower()
    
    for doc in docs:
        try:
            data = doc.to_dict()
            if not data:
                continue
            
            # Calculate relevance score
            relevance_score = _calculate_relevance_score(data, query_lower, collection_name)
            
            # Only include results with some relevance
            if relevance_score > 0:
                # Normalize result format across collections
                result = _normalize_result_format(doc.id, data, collection_name, relevance_score)
                results.append(result)
                
        except Exception as e:
            print(f"[ERROR] Failed to process document {doc.id}: {e}")
            continue
    
    # Sort by relevance score
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return results


def _normalize_difficulty(difficulty: str, collection_name: str) -> str:
    """
    Normalize difficulty values across different collections.
    
    User Input -> Educational Resources -> Worksheets
    easy/beginner -> beginner -> easy
    medium/intermediate -> intermediate -> medium  
    hard/advanced -> advanced -> hard
    """
    difficulty_lower = difficulty.lower()
    
    if collection_name == "educational_resources":
        if difficulty_lower in ["easy", "beginner"]:
            return "beginner"
        elif difficulty_lower in ["medium", "intermediate"]:
            return "intermediate"
        elif difficulty_lower in ["hard", "advanced"]:
            return "advanced"
    
    elif collection_name == "worksheets":
        if difficulty_lower in ["easy", "beginner"]:
            return "easy"
        elif difficulty_lower in ["medium", "intermediate"]:
            return "medium"
        elif difficulty_lower in ["hard", "advanced"]:
            return "hard"
    
    return difficulty  # Return as-is if no mapping needed


def _calculate_relevance_score(data: dict, query_lower: str, collection_name: str) -> float:
    """
    Calculate relevance score for search results across different collections.
    """
    score = 0.0
    
    # Get searchable fields based on collection
    if collection_name == "educational_resources":
        title = data.get("title", "").lower()
        summary = data.get("summary", "").lower()
        tags = " ".join(data.get("tags", [])).lower()
        content_preview = data.get("content_preview", "").lower()
        
        searchable_fields = [
            (title, 3.0),        # Title matches are most important
            (summary, 2.0),      # Summary matches are valuable
            (tags, 2.0),         # Tag matches indicate relevance
            (content_preview, 1.0) # Content matches are useful
        ]
        
    elif collection_name == "worksheets":
        title = data.get("title", "").lower()
        description = data.get("description", "").lower()
        topics = " ".join(data.get("topics_covered", [])).lower()
        objectives = " ".join(data.get("learning_objectives", [])).lower()
        
        searchable_fields = [
            (title, 3.0),
            (description, 2.0),
            (topics, 2.5),       # Topics are very important for worksheets
            (objectives, 1.5)
        ]
        
    elif collection_name == "educational_documents":
        title = data.get("title", "").lower()
        description = data.get("description", "").lower()
        tags = " ".join(data.get("tags", [])).lower()
        
        searchable_fields = [
            (title, 3.0),
            (description, 2.0),
            (tags, 2.0)
        ]
    
    else:
        return 0.0
    
    # Calculate score based on exact and partial matches
    for field_text, weight in searchable_fields:
        if query_lower in field_text:
            score += weight
        
        # Keyword matching for broader search
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 3 and word in field_text:
                score += weight * 0.3
    
    return score


def _normalize_result_format(doc_id: str, data: dict, collection_name: str, relevance_score: float) -> dict:
    """
    Normalize result format across different collections for consistent API.
    """
    # Base structure common to all results
    normalized = {
        "id": doc_id,
        "title": data.get("title", "Untitled"),
        "description": data.get("summary") or data.get("description", "No description available"),
        "subject": data.get("subject", "General"),
        "relevance_score": relevance_score,
        "source_collection": collection_name
    }
    
    # Collection-specific fields
    if collection_name == "educational_resources":
        normalized.update({
            "resource_type": data.get("resource_type", "unknown"),
            "difficulty_level": data.get("difficulty_level", "intermediate"),
            "tags": data.get("tags", []),
            "author": data.get("author", "Unknown"),
            "grade_levels": data.get("grade_levels", []),
            "curriculum_alignment": data.get("curriculum_alignment", []),
            "created_date": data.get("created_date"),
            "rating": data.get("rating", 0),
            "download_count": data.get("download_count", 0)
        })
        
    elif collection_name == "worksheets":
        normalized.update({
            "worksheet_type": data.get("worksheet_type", "practice"),
            "difficulty": data.get("difficulty", "medium"),
            "grade_level": data.get("grade_level", ""),
            "curriculum": data.get("curriculum", ""),
            "topics_covered": data.get("topics_covered", []),
            "page_count": data.get("page_count", 1),
            "answer_key_available": data.get("answer_key_available", False),
            "estimated_time": data.get("estimated_time", "30 minutes"),
            "created_by": data.get("created_by", "Unknown"),
            "rating": data.get("rating", 0)
        })
        
    elif collection_name == "educational_documents":
        normalized.update({
            "document_type": data.get("document_type", "unknown"),
            "file_format": data.get("file_format", ""),
            "file_size_bytes": data.get("file_size_bytes", 0),
            "file_size_display": _format_file_size(data.get("file_size_bytes", 0)),
            "grade_levels": data.get("grade_levels", []),
            "curriculum": data.get("curriculum", []),
            "author": data.get("author", "Unknown"),
            "upload_date": data.get("upload_date"),
            "download_count": data.get("download_count", 0),
            "views": data.get("views", 0)
        })
    
    # File URL (common to all collections)
    normalized["file_url"] = data.get("file_url", "")
    normalized["download_url"] = data.get("download_url", data.get("file_url", ""))
    
    return normalized


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes > 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    elif size_bytes > 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes} bytes"


def _merge_and_rank_results(all_results: List[Dict[str, Any]], query: str, limit: int) -> List[Dict[str, Any]]:
    """
    Merge results from all collections and apply final ranking.
    """
    # Remove duplicates based on file_url if any
    seen_urls = set()
    unique_results = []
    
    for result in all_results:
        file_url = result.get("file_url", "")
        if file_url and file_url not in seen_urls:
            seen_urls.add(file_url)
            unique_results.append(result)
        elif not file_url:  # Include results without file URLs
            unique_results.append(result)
    
    # Sort by relevance score (highest first)
    unique_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Apply final limit
    return unique_results[:limit]


def _categorize_results(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Categorize unified results for easier consumption.
    """
    categorized = {
        "educational_resources": [],
        "worksheets": [],
        "educational_documents": []
    }
    
    for result in results:
        collection = result.get("source_collection")
        if collection in categorized:
            categorized[collection].append(result)
    
    return categorized


def _fallback_to_storage_search(
    collection_name: str, query: str, subject: str, difficulty: str,
    grade_level: str, curriculum: str, file_format: str, config: Dict[str, Any], limit: int
) -> List[Dict[str, Any]]:
    """
    Fallback to Cloud Storage when Firestore indexes are missing.
    
    Maps collection-based search to storage path patterns and performs
    intelligent file filtering based on search parameters.
    """
    try:
        # Map collection to storage paths
        storage_paths = _map_collection_to_storage_paths(collection_name, subject, grade_level, curriculum)
        
        # Map parameters to file patterns
        file_patterns = _build_storage_file_patterns(collection_name, file_format, subject, query)
        
        all_storage_results = []
        
        # Search each relevant storage path
        for storage_path in storage_paths:
            for pattern in file_patterns:
                print(f"[STORAGE_SEARCH] Searching path: {storage_path}, pattern: {pattern}")
                
                storage_result = query_file_storage(
                    storage_path=storage_path,
                    file_pattern=pattern,
                    include_metadata=True,
                    sort_by="modified_date",
                    limit=50  # Get more files to rank and filter down to top 3-5
                )
                
                if storage_result.get("status") == "success":
                    storage_files = storage_result.get("files", [])
                    
                    # Transform storage files to unified format
                    transformed_results = _transform_storage_files_to_search_results(
                        storage_files, collection_name, query, subject, difficulty, 
                        grade_level, curriculum, config
                    )
                    
                    all_storage_results.extend(transformed_results)
        
        # Remove duplicates and apply relevance filtering - return only top 3-5 files
        filtered_results = _filter_and_rank_storage_results(all_storage_results, query, min(5, limit))
        
        print(f"[STORAGE_FALLBACK] Processed {len(all_storage_results)} files, returning {len(filtered_results)} relevant results")
        
        return filtered_results
        
    except Exception as e:
        print(f"[FALLBACK_ERROR] Storage fallback failed: {e}")
        return []


def _map_collection_to_storage_paths(collection_name: str, subject: str, grade_level: str, curriculum: str) -> List[str]:
    """
    Map Firestore collection names to likely Cloud Storage folder structures.
    """
    base_paths = []
    
    if collection_name == "educational_resources":
        # Common folder structures for educational resources
        base_paths = [
            "educational_resources",
            "lessons",
            "articles", 
            "explanations",
            "resources"
        ]
        
        # Add subject-specific paths
        if subject:
            subject_normalized = subject.lower().replace(" ", "_")
            base_paths.extend([
                f"educational_resources/{subject_normalized}",
                f"lessons/{subject_normalized}",
                f"subjects/{subject_normalized}"
            ])
            
    elif collection_name == "worksheets":
        # ALWAYS include base paths - search everything first
        base_paths = [
            "worksheets",           # Base folder - MOST IMPORTANT
            "practice", 
            "exercises",
            "assignments",
            ""                      # Root level search too
        ]
        
        # Add subject-specific paths (but keep base paths)
        if subject:
            subject_normalized = subject.lower().replace(" ", "_")
            base_paths.extend([
                f"worksheets/{subject_normalized}",
                f"practice/{subject_normalized}",
                f"exercises/{subject_normalized}",
                f"assignments/{subject_normalized}"
            ])
            
        # Add grade-specific paths (but keep base paths)  
        if grade_level:
            grade_normalized = grade_level.lower().replace(" ", "_")
            base_paths.extend([
                f"worksheets/{grade_normalized}",
                f"practice/{grade_normalized}",
                f"exercises/{grade_normalized}",
                f"assignments/{grade_normalized}"
            ])
            
        # Add combined subject+grade paths
        if subject and grade_level:
            subject_norm = subject.lower().replace(" ", "_")
            grade_norm = grade_level.lower().replace(" ", "_")
            base_paths.extend([
                f"worksheets/{subject_norm}/{grade_norm}",
                f"worksheets/{grade_norm}/{subject_norm}",
                f"practice/{subject_norm}/{grade_norm}",
                f"exercises/{subject_norm}/{grade_norm}"
            ])
            
    elif collection_name == "educational_documents":
        base_paths = [
            "documents",
            "presentations", 
            "pdfs",
            "materials",
            "files"
        ]
        
        if subject:
            subject_normalized = subject.lower().replace(" ", "_")
            base_paths.extend([
                f"documents/{subject_normalized}",
                f"materials/{subject_normalized}"
            ])
    
    # Remove duplicates and return
    return list(set(base_paths))


def _build_storage_file_patterns(collection_name: str, file_format: str, subject: str, query: str = "") -> List[str]:
    """
    Simple approach: Just get ALL files in the folders.
    No keyword filtering - let user see everything.
    """
    patterns = []
    
    # If specific file format requested, prioritize it
    if file_format:
        patterns.append(f"*.{file_format}")
    
    # Otherwise, get ALL common educational file formats
    patterns.extend(["*.pdf", "*.pptx", "*.docx", "*.doc", "*.ppt", "*.txt", "*.xlsx"])
    
    # CATCH-ALL pattern to find ANY file
    patterns.append("*")
    
    # Remove duplicates
    return list(set(patterns))


def _transform_storage_files_to_search_results(
    storage_files: List[Dict[str, Any]], collection_name: str, query: str,
    subject: str, difficulty: str, grade_level: str, curriculum: str, config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Transform Cloud Storage file results to match unified search result format.
    """
    transformed_results = []
    query_lower = query.lower()
    
    for file_info in storage_files:
        try:
            file_name = file_info.get("name", "")
            file_path = file_info.get("full_path", file_name)
            
            # Calculate relevance score based on filename and path
            relevance_score = _calculate_storage_file_relevance(
                file_name, file_path, query_lower, subject, difficulty, grade_level
            )
            
            # Only include files with some relevance
            if relevance_score > 0:
                # Create normalized result matching Firestore format
                result = {
                    "id": f"storage_{file_path.replace('/', '_')}",
                    "title": _extract_title_from_filename(file_name),
                    "description": f"Educational file: {file_name}",
                    "subject": _extract_subject_from_path(file_path, subject),
                    "relevance_score": relevance_score,
                    "file_url": file_info.get("download_url", file_info.get("public_url", "")),
                    "download_url": file_info.get("download_url", file_info.get("public_url", "")),
                    "source_collection": collection_name,
                    "search_method": "cloud_storage_fallback"
                }
                
                # Add collection-specific fields
                if collection_name == "educational_resources":
                    result.update({
                        "resource_type": _infer_resource_type_from_filename(file_name),
                        "difficulty_level": difficulty if difficulty else "intermediate",
                        "tags": _extract_tags_from_filename(file_name),
                        "author": "Unknown",
                        "grade_levels": [grade_level] if grade_level else [],
                        "curriculum_alignment": [curriculum] if curriculum else [],
                        "created_date": file_info.get("created"),
                        "rating": 3.0,  # Default rating for storage files
                        "download_count": 0
                    })
                    
                elif collection_name == "worksheets":
                    result.update({
                        "worksheet_type": _infer_worksheet_type_from_filename(file_name),
                        "difficulty": difficulty if difficulty else "medium",
                        "grade_level": grade_level if grade_level else "",
                        "curriculum": curriculum if curriculum else "",
                        "topics_covered": _extract_topics_from_filename(file_name),
                        "page_count": 1,
                        "answer_key_available": "answer" in file_name.lower() or "solution" in file_name.lower(),
                        "estimated_time": "30 minutes",
                        "created_by": "Unknown",
                        "rating": 3.0
                    })
                    
                elif collection_name == "educational_documents":
                    result.update({
                        "document_type": _infer_document_type_from_filename(file_name),
                        "file_format": file_info.get("content_type", "").split("/")[-1],
                        "file_size_bytes": file_info.get("size_bytes", 0),
                        "file_size_display": file_info.get("size_display", ""),
                        "grade_levels": [grade_level] if grade_level else [],
                        "curriculum": [curriculum] if curriculum else [],
                        "author": "Unknown",
                        "upload_date": file_info.get("created"),
                        "download_count": 0,
                        "views": 0
                    })
                
                transformed_results.append(result)
                
        except Exception as e:
            print(f"[TRANSFORM_ERROR] Failed to transform file {file_info.get('name', 'unknown')}: {e}")
            continue
    
    return transformed_results


def _calculate_storage_file_relevance(
    file_name: str, file_path: str, query_lower: str, subject: str, difficulty: str, grade_level: str
) -> float:
    """
    Calculate relevance score prioritizing educational file quality over keyword matching.
    """
    score = 5.0  # Base score for all educational files
    file_name_lower = file_name.lower()
    file_path_lower = file_path.lower()
    
    # Strong bonus for premium educational file formats
    if file_name_lower.endswith('.pdf'):
        score += 3.0  # PDFs are often high-quality educational content
    elif file_name_lower.endswith(('.pptx', '.ppt')):
        score += 2.5  # Presentations are valuable
    elif file_name_lower.endswith(('.docx', '.doc')):
        score += 2.0  # Documents are useful
    
    # Bonus for being in the main worksheets folder (more organized)
    if file_path_lower.startswith('worksheets/') and file_path_lower.count('/') == 1:
        score += 2.0
    
    # Query relevance (but not essential)
    if query_lower:
        query_words = [w for w in query_lower.split() if len(w) > 3]
        for word in query_words:
            if word in file_name_lower:
                score += 1.0
            if word in file_path_lower:
                score += 0.5
    
    # Subject relevance (if provided)
    if subject and subject.lower() in file_name_lower:
        score += 1.5
    
    # Grade level relevance (if provided) 
    if grade_level:
        grade_variations = [grade_level.lower(), grade_level.replace(' ', ''), grade_level.replace('grade ', '')]
        for variation in grade_variations:
            if variation in file_name_lower:
                score += 1.0
                break
    
    # Penalty for very generic names
    generic_terms = ['untitled', 'document', 'new', 'copy', 'temp']
    if any(term in file_name_lower for term in generic_terms):
        score -= 1.0
    
    return max(0.1, score)  # Minimum score to avoid zero


def _filter_and_rank_storage_results(results: List[Dict[str, Any]], query: str, limit: int) -> List[Dict[str, Any]]:
    """
    Filter and rank storage results by relevance. Return only top 3-5 best files.
    """
    if not results:
        return []
    
    # Remove duplicates based on file_url
    seen_urls = set()
    unique_results = []
    
    for result in results:
        file_url = result.get("file_url", "")
        if file_url and file_url not in seen_urls:
            seen_urls.add(file_url)
            unique_results.append(result)
        elif not file_url:
            unique_results.append(result)
    
    # Sort by relevance score (highest first)
    unique_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    # Return only top results (3-5 files max)
    top_results = unique_results[:limit]
    
    print(f"[RANKING] Processed {len(unique_results)} unique files, returning top {len(top_results)} results")
    for i, result in enumerate(top_results):
        print(f"  {i+1}. {result.get('title', 'Unknown')} (score: {result.get('relevance_score', 0):.1f})")
    
    return top_results


# Helper functions for file analysis
def _extract_title_from_filename(filename: str) -> str:
    """Extract a readable title from filename."""
    title = filename.split("/")[-1]  # Get just the filename
    title = title.rsplit(".", 1)[0]  # Remove extension
    title = title.replace("_", " ").replace("-", " ")  # Replace underscores and hyphens
    return title.title()


def _extract_subject_from_path(file_path: str, provided_subject: str) -> str:
    """Extract subject from file path or use provided subject."""
    if provided_subject:
        return provided_subject
    
    path_lower = file_path.lower()
    subjects = ["mathematics", "math", "science", "physics", "chemistry", "biology", 
                "english", "history", "geography", "computer", "programming"]
    
    for subject in subjects:
        if subject in path_lower:
            return subject.title()
    
    return "General"


def _infer_resource_type_from_filename(filename: str) -> str:
    """Infer resource type from filename."""
    filename_lower = filename.lower()
    if any(word in filename_lower for word in ["lesson", "tutorial"]):
        return "lesson"
    elif any(word in filename_lower for word in ["article", "reading"]):
        return "article"
    elif any(word in filename_lower for word in ["explanation", "guide"]):
        return "explanation"
    else:
        return "resource"


def _infer_worksheet_type_from_filename(filename: str) -> str:
    """Infer worksheet type from filename."""
    filename_lower = filename.lower()
    if "practice" in filename_lower:
        return "practice"
    elif any(word in filename_lower for word in ["test", "quiz", "exam"]):
        return "assessment"
    elif "homework" in filename_lower:
        return "homework"
    else:
        return "practice"


def _infer_document_type_from_filename(filename: str) -> str:
    """Infer document type from filename."""
    filename_lower = filename.lower()
    if any(word in filename_lower for word in ["presentation", "slides", "ppt"]):
        return "presentation"
    elif "manual" in filename_lower:
        return "manual"
    elif any(word in filename_lower for word in ["reference", "guide"]):
        return "reference"
    else:
        return "document"


def _extract_tags_from_filename(filename: str) -> List[str]:
    """Extract potential tags from filename."""
    filename_lower = filename.lower()
    tags = []
    
    # Common educational tags
    tag_patterns = ["algebra", "geometry", "calculus", "trigonometry", "statistics",
                   "physics", "chemistry", "biology", "grammar", "vocabulary"]
    
    for pattern in tag_patterns:
        if pattern in filename_lower:
            tags.append(pattern.title())
    
    return tags


def _extract_topics_from_filename(filename: str) -> List[str]:
    """Extract potential topics from filename."""
    return _extract_tags_from_filename(filename)  # Same logic for now


def query_file_storage(
    storage_path: str = "",
    file_pattern: str = "*",
    bucket_name: str = "",
    include_metadata: bool = True,
    sort_by: str = "modified_date",
    limit: int = 20
) -> Dict[str, Any]:
    """
    Query cloud file storage (Google Cloud Storage) for educational materials.
    
    Args:
        storage_path (str): Path within the storage bucket to search
        file_pattern (str): File pattern to match (*.pdf, *.pptx, etc.)
        bucket_name (str): Specific bucket name (uses default if empty)
        include_metadata (bool): Whether to include detailed file metadata
        sort_by (str): Sort criteria (name, size, modified_date, created_date)
        limit (int): Maximum number of files to return
        
    Returns:
        dict: Status and file storage query results
    """
    print(f"[STORAGE_QUERY] Querying cloud storage: path='{storage_path}', pattern='{file_pattern}'")
    
    try:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        default_bucket = os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET", "sahayak-educational-content")
        
        if not project_id:
            return {
                "status": "error",
                "error_message": "Google Cloud Project ID not configured"
            }
        
        # Initialize Cloud Storage client
        storage_client = storage.Client(project=project_id)
        
        # Use specified bucket or default
        bucket_name = bucket_name or default_bucket
        
        try:
            bucket = storage_client.bucket(bucket_name)
            print(f"[STORAGE] Connected to bucket: {bucket_name}")
        except Exception as e:
            print(f"[STORAGE_ERROR] Cannot access bucket {bucket_name}: {e}")
            # Return mock data for testing
            return {
                "status": "success",
                "files": [
                    {
                        "name": "sample_worksheet_math_grade6.pdf",
                        "size_bytes": 245760,
                        "size_display": "240 KB", 
                        "content_type": "application/pdf",
                        "storage_path": "worksheets/mathematics/grade6/",
                        "public_url": "https://storage.googleapis.com/example/sample.pdf",
                        "created": "2024-01-15T10:30:00Z",
                        "modified": "2024-01-15T10:30:00Z",
                        "metadata": {
                            "subject": "Mathematics",
                            "grade": "6",
                            "type": "worksheet"
                        }
                    }
                ],
                "storage_metadata": {
                    "bucket": bucket_name,
                    "path_searched": storage_path,
                    "pattern": file_pattern,
                    "total_found": 1,
                    "note": "Mock data - storage not accessible"
                }
            }
        
        # Build prefix for path-based searching
        prefix = storage_path.rstrip('/') + '/' if storage_path else ""
        
        # List blobs with prefix
        blobs = bucket.list_blobs(prefix=prefix)
        
        files = []
        import fnmatch
        
        for blob in blobs:
            try:
                # Check if file matches pattern
                if file_pattern != "*" and not fnmatch.fnmatch(blob.name, file_pattern):
                    continue
                
                # Get file size display
                if blob.size > 1024 * 1024:
                    size_display = f"{blob.size / (1024 * 1024):.1f} MB"
                elif blob.size > 1024:
                    size_display = f"{blob.size / 1024:.1f} KB"
                else:
                    size_display = f"{blob.size} bytes"
                
                file_item = {
                    "name": blob.name.split('/')[-1],  # Just filename
                    "full_path": blob.name,
                    "size_bytes": blob.size,
                    "size_display": size_display,
                    "content_type": blob.content_type,
                    "storage_path": '/'.join(blob.name.split('/')[:-1]) + '/',
                    "public_url": blob.public_url if hasattr(blob, 'public_url_set') and blob.public_url_set else "",
                    "download_url": blob.generate_signed_url(expiration=timedelta(hours=1)) if blob.exists() else "",
                    "created": blob.time_created.isoformat() if blob.time_created else "",
                    "modified": blob.updated.isoformat() if blob.updated else "",
                    "etag": blob.etag,
                    "generation": blob.generation
                }
                
                # Include metadata if requested
                if include_metadata and blob.metadata:
                    file_item["metadata"] = blob.metadata
                
                files.append(file_item)
                
                # Stop if we've reached the limit
                if len(files) >= limit:
                    break
                    
            except Exception as e:
                print(f"[ERROR] Failed to process blob {blob.name}: {e}")
                continue
        
        # Sort files based on criteria
        if sort_by == "name":
            files.sort(key=lambda x: x["name"].lower())
        elif sort_by == "size":
            files.sort(key=lambda x: x["size_bytes"], reverse=True)
        elif sort_by == "created_date":
            files.sort(key=lambda x: x["created"], reverse=True)
        elif sort_by == "modified_date":
            files.sort(key=lambda x: x["modified"], reverse=True)
        
        print(f"[STORAGE_SUCCESS] Found {len(files)} files in cloud storage")
        
        # Calculate statistics
        type_stats = {}
        size_stats = {"small": 0, "medium": 0, "large": 0}
        
        for file_item in files:
            content_type = file_item["content_type"]
            size_bytes = file_item["size_bytes"]
            
            # Categorize by content type
            if "pdf" in content_type:
                type_stats["PDF"] = type_stats.get("PDF", 0) + 1
            elif "image" in content_type:
                type_stats["Image"] = type_stats.get("Image", 0) + 1
            elif "document" in content_type or "officedocument" in content_type:
                type_stats["Document"] = type_stats.get("Document", 0) + 1
            elif "presentation" in content_type:
                type_stats["Presentation"] = type_stats.get("Presentation", 0) + 1
            else:
                type_stats["Other"] = type_stats.get("Other", 0) + 1
            
            # Categorize by size
            if size_bytes < 1024 * 1024:  # < 1MB
                size_stats["small"] += 1
            elif size_bytes < 10 * 1024 * 1024:  # < 10MB
                size_stats["medium"] += 1
            else:
                size_stats["large"] += 1
        
        return {
            "status": "success",
            "files": files,
            "storage_metadata": {
                "bucket": bucket_name,
                "path_searched": storage_path,
                "pattern": file_pattern,
                "sort_by": sort_by,
                "total_found": len(files),
                "type_distribution": type_stats,
                "size_distribution": size_stats,
                "query_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        print(f"[ERROR] File storage query failed: {e}")
        return {
            "status": "error",
            "error_message": f"File storage query failed: {str(e)}"
        }




# Create the ADK Retrieval Agent with unified search architecture
root_agent = Agent(
    name="sahayak_retrieval_agent",
    model="gemini-2.0-flash",
    description="Unified database and storage retrieval specialist for educational materials",
    instruction=(
        "You are the Sahayak Retrieval Agent, a specialized database and storage expert focused on "
        "finding, retrieving, and organizing educational materials from various data sources. "
        "Your expertise includes comprehensive search capabilities across knowledge bases, "
        "document repositories, and cloud storage systems.\\n\\n"
        
        "Your core capabilities include:\\n\\n"
        
        "1. UNIFIED EDUCATIONAL SEARCH (PRIMARY):\\n"
        "   - Single intelligent search across all educational content types\\n"
        "   - Search lessons, worksheets, documents simultaneously or selectively\\n"
        "   - Advanced multi-collection filtering and relevance ranking\\n"
        "   - Unified result format with categorized outputs\\n"
        "   - Smart content type mapping and field normalization\\n\\n"
        
        "2. CLOUD STORAGE QUERIES:\\n"
        "   - Query Google Cloud Storage for educational materials\\n"
        "   - Search by file patterns, paths, and metadata\\n"
        "   - Generate secure download URLs and access controls\\n"
        "   - Organize files by type, size, and modification date\\n\\n"
        
        
        "Always prioritize search relevance, result accuracy, and user efficiency. "
        "Provide comprehensive metadata to help teachers make informed decisions about "
        "educational resources. Ensure all retrieved materials are properly categorized "
        "and include sufficient information for effective educational use."
    ),
    tools=[
        search_educational_content,  # Primary unified search function
        query_file_storage          # Direct cloud storage access
    ]
)

if __name__ == "__main__":
    # For testing purposes
    print(f"[AGENT_INIT] Retrieval Agent '{root_agent.name}' initialized successfully")
    print(f"[AGENT_TOOLS] Available tools: {len(root_agent.tools)}")
    for i, tool in enumerate(root_agent.tools, 1):
        print(f"  {i}. {tool.__name__}")