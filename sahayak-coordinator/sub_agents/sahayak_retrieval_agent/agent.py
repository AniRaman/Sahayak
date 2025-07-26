"""
Sahayak Retrieval Agent - Database & Storage Retrieval Specialist

Clean, optimized implementation focusing on:
1. Unified intelligent search across all educational content types
2. Cloud Storage file queries with intelligent fallback
3. Top 3-5 results filtering to avoid overwhelming users
"""

import os
import json
import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
from google.cloud import firestore
from google.cloud import storage
from google.oauth2 import service_account
from dotenv import load_dotenv
from google.adk.agents import Agent

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _get_authenticated_storage_client():
    """
    Create an authenticated Google Cloud Storage client with service account credentials.
    Supports multiple authentication methods for flexibility.
    """
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT_ID environment variable not set")
    
    # Method 1: Sahayak-specific service account JSON file path
    sahayak_service_account_path = os.getenv("SAHAYAK_SERVICE_ACCOUNT_CREDENTIALS")
    if sahayak_service_account_path and os.path.exists(sahayak_service_account_path):
        print(f"[AUTH] Using Sahayak service account file: {sahayak_service_account_path}")
        credentials = service_account.Credentials.from_service_account_file(sahayak_service_account_path)
        return storage.Client(project=project_id, credentials=credentials)
    
    # Method 2: Sahayak service account JSON content from environment variable
    service_account_json = os.getenv("SAHAYAK_SERVICE_ACCOUNT_JSON")
    if service_account_json:
        print(f"[AUTH] Using Sahayak service account JSON from environment variable")
        try:
            service_account_info = json.loads(service_account_json)
            credentials = service_account.Credentials.from_service_account_info(service_account_info)
            return storage.Client(project=project_id, credentials=credentials)
        except json.JSONDecodeError as e:
            print(f"[AUTH_ERROR] Invalid JSON in SAHAYAK_SERVICE_ACCOUNT_JSON: {e}")
    
    # Method 3: Default credentials (for local development or Compute Engine)
    print(f"[AUTH] Using default application credentials")
    try:
        return storage.Client(project=project_id)
    except Exception as e:
        print(f"[AUTH_ERROR] Failed to create storage client with default credentials: {e}")
        raise


def search_educational_content(
    query: str,
    content_types: List[str] = ["all"],
    subject: str = "",
    difficulty: str = "",
    grade_level: str = "",
    curriculum: str = "",
    file_format: str = "",
    limit: int = 5  # Default to 5 to avoid overwhelming users
) -> dict:
    """
    UNIFIED educational content search with intelligent fallback to Cloud Storage.
    Returns only top 3-5 results to ensure user-friendly experience.
    """
    print(f"[UNIFIED_SEARCH] Searching for '{query}' across content types: {content_types}")
    
    try:
        # Get collections to search
        collections = _get_collection_mapping(content_types)["collections"]
        
        # Connect to Firestore
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        database_id = os.getenv("FIRESTORE_DATABASE_ID", "agenticdb")
        
        if not project_id:
            return {"status": "error", "error_message": "Google Cloud Project ID not configured"}
        
        db = firestore.Client(project=project_id, database=database_id)
        print(f"[DATABASE] Connected to '{database_id}' in project {project_id}")
        
        all_results = []
        query_lower = query.lower()
        
        # Search each collection with fallback
        for collection_name in collections:
            try:
                results = _search_collection(db, collection_name, query_lower, subject, difficulty, grade_level, curriculum, file_format)
                all_results.extend(results)
                print(f"[RESULTS] Found {len(results)} results in {collection_name}")
                
                # If Firestore returns empty results, immediately try Cloud Storage fallback
                if len(results) == 0:
                    print(f"[FALLBACK] No Firestore results for {collection_name}, trying Cloud Storage...")
                    fallback_results = _fallback_to_storage_search(collection_name, query_lower, subject, difficulty, grade_level, curriculum, file_format, limit)
                    all_results.extend(fallback_results)
                
            except Exception as e:
                error_msg = str(e)
                if "requires an index" in error_msg or "400" in error_msg:
                    print(f"[FALLBACK] Firestore index missing for {collection_name}, trying Cloud Storage...")
                    fallback_results = _fallback_to_storage_search(collection_name, query_lower, subject, difficulty, grade_level, curriculum, file_format, limit)
                    all_results.extend(fallback_results)
                else:
                    print(f"[ERROR] Failed to search {collection_name}: {e}")
        
        # Filter and rank results - return only top 3-5
        final_results = _merge_and_rank_results(all_results, query, min(5, limit))
        categorized = _categorize_results(final_results)
        
        print(f"[UNIFIED_SUCCESS] Found {len(final_results)} total results across all collections")
        
        return {
            "status": "success",
            "results": final_results,
            "categorized_results": categorized,
            "search_metadata": {
                "query": query,
                "collections_searched": collections,
                "total_results": len(final_results),
                "filters_applied": {
                    "content_types": content_types,
                    "subject": subject,
                    "difficulty": difficulty,
                    "grade_level": grade_level,
                    "curriculum": curriculum,
                    "file_format": file_format
                }
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Unified search failed: {e}")
        return {"status": "error", "error_message": str(e)}


def _get_collection_mapping(content_types: List[str]) -> Dict[str, Any]:
    """Map content types to Firestore collections"""
    collections = []
    
    if "all" in content_types:
        collections = ["educational_resources", "worksheets", "educational_documents"]
    else:
        if any(t in content_types for t in ["lessons", "articles", "explanations", "tutorials"]):
            collections.append("educational_resources")
        # Fix: Map both 'worksheet' and 'worksheets' to the worksheets collection
        if any(t in content_types for t in ["worksheet", "worksheets"]):
            collections.append("worksheets")
        if any(t in content_types for t in ["documents", "presentations", "pdfs", "manuals"]):
            collections.append("educational_documents")
    
    return {"collections": list(set(collections))}


def _search_collection(db, collection_name: str, query_lower: str, subject: str, difficulty: str, grade_level: str, curriculum: str, file_format: str) -> List[Dict]:
    """Search a single Firestore collection"""
    print(f"[COLLECTION] Searching {collection_name}...")
    
    collection_ref = db.collection(collection_name)
    query_ref = collection_ref
    
    # Apply filters
    if subject:
        query_ref = query_ref.where("subject", "==", subject)
    if difficulty:
        normalized_difficulty = _normalize_difficulty(difficulty, collection_name)
        difficulty_field = "difficulty_level" if collection_name == "educational_resources" else "difficulty"
        query_ref = query_ref.where(difficulty_field, "==", normalized_difficulty)
    if grade_level:
        if collection_name == "worksheets":
            query_ref = query_ref.where("grade_level", "==", grade_level)
        elif collection_name == "educational_resources":
            query_ref = query_ref.where("grade_levels", "array_contains", grade_level)
    
    # Execute query and process results
    docs = query_ref.limit(20).stream()
    results = []
    
    for doc in docs:
        try:
            data = doc.to_dict()
            if not data:
                continue
            
            relevance_score = _calculate_relevance_score(data, query_lower, collection_name)
            if relevance_score > 0:
                result = _normalize_result_format(doc.id, data, collection_name, relevance_score)
                results.append(result)
                
        except Exception as e:
            print(f"[ERROR] Failed to process document {doc.id}: {e}")
            continue
    
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results


def _normalize_difficulty(difficulty: str, collection_name: str) -> str:
    """Normalize difficulty levels across collections"""
    difficulty_lower = difficulty.lower()
    
    if collection_name in ["educational_resources", "educational_documents"]:
        mapping = {"easy": "beginner", "medium": "intermediate", "hard": "advanced"}
        return mapping.get(difficulty_lower, difficulty)
    elif collection_name == "worksheets":
        mapping = {"beginner": "easy", "intermediate": "medium", "advanced": "hard"}
        return mapping.get(difficulty_lower, difficulty)
    
    return difficulty


def _calculate_relevance_score(data: dict, query_lower: str, collection_name: str) -> float:
    """Calculate relevance score for search results"""
    score = 0.0
    
    # Text content scoring
    searchable_fields = ["title", "summary", "description", "topics_covered", "tags"]
    query_words = [w for w in query_lower.split() if len(w) > 2]
    
    for field in searchable_fields:
        field_value = data.get(field, "")
        if isinstance(field_value, str):
            field_lower = field_value.lower()
            weight = 2.0 if field == "title" else 1.0
            
            # Exact phrase match
            if query_lower in field_lower:
                score += weight * 2.0
            
            # Individual word matches
            for word in query_words:
                if word in field_lower:
                    score += weight * 0.3
    
    return score


def _normalize_result_format(doc_id: str, data: dict, collection_name: str, relevance_score: float) -> dict:
    """Normalize result format across collections"""
    return {
        "id": doc_id,
        "title": data.get("title", "Untitled"),
        "description": data.get("summary") or data.get("description", "No description available"),
        "subject": data.get("subject", "General"),
        "relevance_score": relevance_score,
        "source_collection": collection_name,
        "file_url": data.get("file_url", ""),
        "download_url": data.get("download_url", ""),
        "difficulty": data.get("difficulty_level") or data.get("difficulty", ""),
        "grade_level": data.get("grade_level", ""),
        "curriculum": data.get("curriculum", ""),
        "resource_type": data.get("resource_type") or data.get("worksheet_type") or data.get("document_type", "")
    }


def _merge_and_rank_results(all_results: List[Dict[str, Any]], query: str, limit: int) -> List[Dict[str, Any]]:
    """Merge and rank results, removing duplicates and returning top results only"""
    if not all_results:
        return []
    
    # Remove duplicates based on file_url
    seen_urls = set()
    unique_results = []
    
    for result in all_results:
        file_url = result.get("file_url", "")
        if file_url and file_url not in seen_urls:
            seen_urls.add(file_url)
            unique_results.append(result)
        elif not file_url:
            unique_results.append(result)
    
    # Sort by relevance score
    unique_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Return only top results (3-5 max to avoid overwhelming users)
    return unique_results[:limit]


def _categorize_results(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize results by source collection"""
    categories = {
        "educational_resources": [],
        "worksheets": [],
        "educational_documents": []
    }
    
    for result in results:
        collection = result.get("source_collection", "")
        if collection in categories:
            categories[collection].append(result)
    
    return categories


def _fallback_to_storage_search(collection_name: str, query_lower: str, subject: str, difficulty: str, grade_level: str, curriculum: str, file_format: str, limit: int) -> List[Dict[str, Any]]:
    """Intelligent fallback to Cloud Storage when Firestore indexes are missing"""
    try:
        # Map collection to storage paths
        storage_paths = _map_collection_to_storage_paths(collection_name, subject, grade_level, curriculum)
        
        # Get file patterns
        patterns = _build_storage_file_patterns(collection_name, file_format, subject, query_lower)
        
        all_storage_results = []
        
        # Search each path/pattern combination - but limit to avoid too many queries
        for path in storage_paths[:2]:  # Limit to 2 paths
            for pattern in patterns[:3]:  # Limit to 3 patterns
                print(f"[STORAGE_SEARCH] Searching path: {path}, pattern: {pattern}")
                storage_result = query_file_storage(path, pattern, limit=10)
                
                if storage_result.get("status") == "success":
                    files = storage_result.get("files", [])
                    transformed = _transform_storage_files_to_search_results(files, collection_name, subject, difficulty, grade_level, curriculum, query_lower)
                    all_storage_results.extend(transformed)
                    
                    # If we found some files, we can break early to speed up the search
                    if len(transformed) > 0:
                        print(f"[STORAGE_EARLY_EXIT] Found {len(transformed)} files, stopping search for this collection")
                        break
            
            # If we found files in this path, no need to search more paths
            if len(all_storage_results) > 0:
                break
        
        # Filter and rank storage results - return only top 3-5
        filtered_results = _filter_and_rank_storage_results(all_storage_results, query_lower, min(5, limit))
        
        print(f"[STORAGE_FALLBACK] Processed {len(all_storage_results)} files, returning {len(filtered_results)} relevant results")
        return filtered_results
        
    except Exception as e:
        print(f"[FALLBACK_ERROR] Storage fallback failed: {e}")
        return []


def _map_collection_to_storage_paths(collection_name: str, subject: str, grade_level: str, curriculum: str) -> List[str]:
    """Map collection names to likely Cloud Storage paths"""
    if collection_name == "worksheets":
        return ["worksheets", "practice", "exercises", "assignments", ""]
    elif collection_name == "educational_resources":
        return ["lessons", "articles", "explanations", "resources", ""]
    elif collection_name == "educational_documents":
        return ["documents", "presentations", "files", "pdfs", "materials", ""]
    return [""]


def _build_storage_file_patterns(collection_name: str, file_format: str, subject: str, query: str = "") -> List[str]:
    """Build file patterns for storage search - simplified approach"""
    patterns = []
    
    # File format patterns
    if file_format:
        patterns.append(f"*.{file_format}")
    else:
        patterns.extend(["*.pdf", "*.pptx", "*.docx", "*.doc", "*.ppt", "*.txt", "*.xlsx"])
    
    patterns.append("*")  # Catch-all pattern
    return patterns


def _transform_storage_files_to_search_results(files: List[Dict], collection_name: str, subject: str, difficulty: str, grade_level: str, curriculum: str, query_lower: str) -> List[Dict[str, Any]]:
    """Transform storage files to unified search result format"""
    results = []
    
    for file_info in files:
        try:
            file_name = file_info.get("name", "")
            file_path = file_info.get("full_path", file_name)
            
            # Calculate relevance score
            relevance_score = _calculate_storage_file_relevance(file_name, file_path, query_lower, subject, difficulty, grade_level)
            
            if relevance_score > 0:
                result = {
                    "id": f"storage_{file_path.replace('/', '_')}",
                    "title": _extract_title_from_filename(file_name),
                    "description": f"Educational file: {file_name}",
                    "subject": _extract_subject_from_path(file_path, subject),
                    "relevance_score": relevance_score,
                    "file_url": file_info.get("download_url", file_info.get("public_url", "")),
                    "download_url": file_info.get("download_url", file_info.get("public_url", "")),
                    "source_collection": collection_name,
                    "search_method": "cloud_storage_fallback",
                    "difficulty": difficulty if difficulty else "intermediate",
                    "grade_level": grade_level if grade_level else "",
                    "curriculum": curriculum if curriculum else "",
                    "resource_type": _infer_resource_type_from_filename(file_name)
                }
                results.append(result)
                
        except Exception as e:
            print(f"[ERROR] Failed to transform file {file_info.get('name', 'unknown')}: {e}")
            continue
    
    return results


def _calculate_storage_file_relevance(file_name: str, file_path: str, query_lower: str, subject: str, difficulty: str, grade_level: str) -> float:
    """Calculate relevance score prioritizing file quality over keyword matching"""
    score = 5.0  # Base score
    file_name_lower = file_name.lower()
    file_path_lower = file_path.lower()
    
    # Premium file format bonus
    if file_name_lower.endswith('.pdf'):
        score += 3.0
    elif file_name_lower.endswith(('.pptx', '.ppt')):
        score += 2.5
    elif file_name_lower.endswith(('.docx', '.doc')):
        score += 2.0
    
    # Organized folder bonus
    if file_path_lower.startswith('worksheets/') and file_path_lower.count('/') == 1:
        score += 2.0
    
    # Query relevance (secondary)
    if query_lower:
        query_words = [w for w in query_lower.split() if len(w) > 3]
        for word in query_words:
            if word in file_name_lower:
                score += 1.0
            if word in file_path_lower:
                score += 0.5
    
    # Subject relevance
    if subject and subject.lower() in file_name_lower:
        score += 1.5
    
    # Grade level relevance
    if grade_level:
        grade_variations = [grade_level.lower(), grade_level.replace(' ', ''), grade_level.replace('grade ', '')]
        for variation in grade_variations:
            if variation in file_name_lower:
                score += 1.0
                break
    
    # Penalty for generic names
    generic_terms = ['untitled', 'document', 'new', 'copy', 'temp']
    if any(term in file_name_lower for term in generic_terms):
        score -= 1.0
    
    return max(0.1, score)


def _filter_and_rank_storage_results(results: List[Dict[str, Any]], query: str, limit: int) -> List[Dict[str, Any]]:
    """Filter and rank storage results - return only top 3-5 files"""
    if not results:
        return []
    
    # Remove duplicates
    seen_urls = set()
    unique_results = []
    
    for result in results:
        file_url = result.get("file_url", "")
        if file_url and file_url not in seen_urls:
            seen_urls.add(file_url)
            unique_results.append(result)
        elif not file_url:
            unique_results.append(result)
    
    # Sort by relevance score
    unique_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    # Return only top results
    top_results = unique_results[:limit]
    
    print(f"[RANKING] Processed {len(unique_results)} unique files, returning top {len(top_results)} results")
    for i, result in enumerate(top_results):
        print(f"  {i+1}. {result.get('title', 'Unknown')} (score: {result.get('relevance_score', 0):.1f})")
    
    return top_results


# Helper functions for file analysis
def _extract_title_from_filename(filename: str) -> str:
    """Extract readable title from filename"""
    title = filename.rsplit('.', 1)[0]  # Remove extension
    title = title.replace('_', ' ').replace('-', ' ')  # Replace separators
    return ' '.join(word.capitalize() for word in title.split())


def _extract_subject_from_path(file_path: str, provided_subject: str) -> str:
    """Extract subject from file path"""
    if provided_subject:
        return provided_subject
    
    path_lower = file_path.lower()
    subjects = {
        'math': 'Mathematics', 'algebra': 'Mathematics', 'geometry': 'Mathematics',
        'science': 'Science', 'physics': 'Physics', 'chemistry': 'Chemistry', 'biology': 'Biology',
        'english': 'English', 'history': 'History', 'geography': 'Geography'
    }
    
    for key, value in subjects.items():
        if key in path_lower:
            return value
    
    return "General"


def _infer_resource_type_from_filename(filename: str) -> str:
    """Infer resource type from filename"""
    filename_lower = filename.lower()
    if any(word in filename_lower for word in ["lesson", "tutorial"]):
        return "lesson"
    elif any(word in filename_lower for word in ["worksheet", "practice"]):
        return "worksheet"
    elif any(word in filename_lower for word in ["presentation", "slides"]):
        return "presentation"
    return "document"


def query_file_storage(
    storage_path: str = "",
    file_pattern: str = "*",
    bucket_name: str = "",
    limit: int = 20
) -> dict:
    """
    Query Google Cloud Storage for educational files.
    Simplified version focusing on core functionality.
    """
    print(f"[STORAGE_QUERY] Querying cloud storage: path='{storage_path}', pattern='{file_pattern}'")
    
    try:
        default_bucket = os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET", "agentic_storage")
        bucket_name = bucket_name or default_bucket
        
        # Use authenticated storage client for signed URL generation
        storage_client = _get_authenticated_storage_client()
        
        try:
            bucket = storage_client.bucket(bucket_name)
            print(f"[STORAGE] Connected to bucket: {bucket_name}")
        except Exception as e:
            print(f"[STORAGE_ERROR] Cannot access bucket {bucket_name}: {e}")
            return {"status": "error", "error_message": f"Cannot access bucket: {e}"}
        
        # List blobs
        prefix = storage_path.rstrip('/') + '/' if storage_path else ""
        blobs = bucket.list_blobs(prefix=prefix)
        
        files = []
        import fnmatch
        
        for blob in blobs:
            try:
                if file_pattern != "*" and not fnmatch.fnmatch(blob.name, file_pattern):
                    continue
                
                # Calculate file size display
                if blob.size > 1024 * 1024:
                    size_display = f"{blob.size / (1024 * 1024):.1f} MB"
                elif blob.size > 1024:
                    size_display = f"{blob.size / 1024:.1f} KB"
                else:
                    size_display = f"{blob.size} bytes"
                
                # Generate secure signed URL for file access
                download_url = ""
                public_url = ""
                
                try:
                    # Generate signed URL with proper authentication (24-hour expiration)
                    download_url = blob.generate_signed_url(
                        version="v4",
                        expiration=timedelta(hours=24),
                        method="GET"
                    )
                    print(f"[SIGNED_URL] Generated secure download URL for {blob.name}")
                except Exception as e:
                    print(f"[SIGNED_URL_ERROR] Failed to generate signed URL for {blob.name}: {e}")
                    
                    # Fallback: Try to use public URL format (will work if bucket becomes public)
                    try:
                        public_url = blob.public_url
                        download_url = public_url
                        print(f"[FALLBACK_URL] Using public URL format for {blob.name}")
                    except:
                        # Final fallback: construct standard URL
                        download_url = f"https://storage.googleapis.com/{bucket_name}/{blob.name}"
                        print(f"[FALLBACK_URL] Using standard URL format for {blob.name}")
                
                file_item = {
                    "name": blob.name.split('/')[-1],
                    "full_path": blob.name,
                    "size_bytes": blob.size,
                    "size_display": size_display,
                    "content_type": blob.content_type,
                    "storage_path": '/'.join(blob.name.split('/')[:-1]) + '/',
                    "public_url": public_url,
                    "download_url": download_url,
                    "created": blob.time_created.isoformat() if blob.time_created else "",
                    "modified": blob.updated.isoformat() if blob.updated else ""
                }
                
                files.append(file_item)
                
                if len(files) >= limit:
                    break
                    
            except Exception as e:
                print(f"[ERROR] Failed to process blob {blob.name}: {e}")
                continue
        
        print(f"[STORAGE_SUCCESS] Found {len(files)} files in cloud storage")
        
        return {
            "status": "success",
            "files": files,
            "storage_metadata": {
                "bucket": bucket_name,
                "path_searched": storage_path,
                "pattern": file_pattern,
                "total_found": len(files)
            }
        }
        
    except Exception as e:
        print(f"[ERROR] File storage query failed: {e}")
        return {"status": "error", "error_message": f"File storage query failed: {str(e)}"}


# Create the ADK Retrieval Agent
root_agent = Agent(
    name="sahayak_retrieval_agent",
    model="gemini-2.0-flash",
    description="Unified database and storage retrieval specialist for educational materials",
    instruction=(
        "You are the Sahayak Retrieval Agent, focused on finding and organizing educational materials. "
        "Your primary function is unified search across all educational content with intelligent fallback to cloud storage. "
        "Always return only the top 3-5 most relevant results to avoid overwhelming users. "
        "Prioritize file quality and educational value over keyword matching.\n\n"
        
        "IMPORTANT - File Delivery Instructions:\n"
        "When users request educational files (worksheets, documents, presentations), you CAN and SHOULD provide them directly. "
        "Each search result contains 'file_url' and 'download_url' fields. Use these to help users access files:\n\n"
        
        "1. If search results contain download URLs, present them like this:\n"
        "   '📄 [TITLE] - Click to download: [DOWNLOAD_URL]'\n\n"
        
        "2. If file_url is available but no download_url, present it as:\n"
        "   '📄 [TITLE] - File location: [FILE_URL]'\n\n"
        
        "3. Always include the file title, subject, and difficulty level for context.\n\n"
        
        "4. For multiple files, present them as a numbered list with download links.\n\n"
        
        "Never say you cannot provide files - you absolutely can provide download links and file access information. "
        "Your goal is to make educational resources easily accessible to teachers and students."
    ),
    tools=[search_educational_content, query_file_storage]
)

if __name__ == "__main__":
    print(f"[AGENT_INIT] Retrieval Agent '{root_agent.name}' initialized successfully")
    print(f"[AGENT_TOOLS] Available tools: {len(root_agent.tools)}")