"""
Sahayak Retrieval Agent - Database & Storage Retrieval Specialist

This agent specializes in:
1. Searching educational knowledge bases and databases
2. Retrieving worksheets, PDFs, and educational documents
3. Querying file storage systems (Google Cloud Storage, local storage)
4. Advanced metadata-based searches and filtering
5. Resource discovery and recommendation
6. Educational material indexing and categorization
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from google.cloud import firestore
from google.cloud import storage

# Google ADK imports
from google.adk.agents import Agent

# Configure logging to show in terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def search_knowledge_base(
    query: str,
    subject: str = "",
    difficulty_level: str = "intermediate",
    limit: int = 10,
    resource_type: str = "all",
    date_range: int = 0
) -> dict:
    """
    Search the educational knowledge base for relevant resources with advanced filtering.
    
    Args:
        query (str): Search query for educational resources
        subject (str): Subject area filter (Mathematics, Science, History, etc.)
        difficulty_level (str): Difficulty level filter (beginner, intermediate, advanced)
        limit (int): Maximum number of results to return
        resource_type (str): Type of resource (worksheet, lesson_plan, assessment, all)
        date_range (int): Days back to search (0 = all time, 30 = last 30 days)
        
    Returns:
        dict: Status and comprehensive search results from knowledge base
    """
    print(f"[KNOWLEDGE_SEARCH] Searching for '{query}' in {subject} {difficulty_level} resources")
    
    try:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        if not project_id:
            print("[WARNING] No Google Cloud Project ID configured")
            return {
                "status": "error",
                "error_message": "Google Cloud Project ID not configured"
            }
        
        # Initialize Firestore client
        db = firestore.Client(project=project_id)
        print(f"[DATABASE] Connected to Firestore database")
        
        # Build comprehensive query
        collection_ref = db.collection("educational_resources")
        
        # Apply filters based on parameters
        filters = []
        
        if subject:
            collection_ref = collection_ref.where("subject", "==", subject)
            filters.append(f"subject={subject}")
        
        if difficulty_level and difficulty_level != "all":
            collection_ref = collection_ref.where("difficulty_level", "==", difficulty_level)
            filters.append(f"difficulty={difficulty_level}")
        
        if resource_type and resource_type != "all":
            collection_ref = collection_ref.where("resource_type", "==", resource_type)
            filters.append(f"type={resource_type}")
        
        # Date range filtering
        if date_range > 0:
            cutoff_date = datetime.now() - timedelta(days=date_range)
            collection_ref = collection_ref.where("created_date", ">=", cutoff_date)
            filters.append(f"last_{date_range}_days")
        
        # Order by relevance score if available, otherwise by creation date
        collection_ref = collection_ref.order_by("created_date", direction=firestore.Query.DESCENDING)
        
        # Execute query
        docs = collection_ref.limit(limit * 2).stream()  # Get more than needed for filtering
        
        results = []
        query_lower = query.lower()
        
        print(f"[FILTERING] Applied filters: {', '.join(filters) if filters else 'none'}")
        
        for doc in docs:
            try:
                data = doc.to_dict()
                if not data:
                    continue
                
                # Calculate relevance score based on text matching
                relevance_score = 0
                title = data.get("title", "").lower()
                summary = data.get("summary", "").lower()
                tags = " ".join(data.get("tags", [])).lower()
                content_preview = data.get("content_preview", "").lower()
                
                # Multi-field text matching with different weights
                if query_lower in title:
                    relevance_score += 3  # Title matches are most important
                if query_lower in summary:
                    relevance_score += 2  # Summary matches are valuable
                if query_lower in tags:
                    relevance_score += 2  # Tag matches indicate relevance
                if query_lower in content_preview:
                    relevance_score += 1  # Content matches are useful
                
                # Keyword matching for broader search
                query_words = query_lower.split()
                for word in query_words:
                    if len(word) > 3:  # Skip short words
                        if word in title: relevance_score += 1
                        if word in summary: relevance_score += 0.5
                        if word in tags: relevance_score += 0.5
                
                # Only include results with some relevance
                if relevance_score > 0:
                    result_item = {
                        "id": doc.id,
                        "title": data.get("title", "Untitled Resource"),
                        "summary": data.get("summary", "No description available"),
                        "subject": data.get("subject", "General"),
                        "difficulty_level": data.get("difficulty_level", "intermediate"),
                        "resource_type": data.get("resource_type", "unknown"),
                        "tags": data.get("tags", []),
                        "file_url": data.get("file_url", ""),
                        "file_type": data.get("file_type", ""),
                        "file_size": data.get("file_size", 0),
                        "created_date": data.get("created_date"),
                        "author": data.get("author", "Unknown"),
                        "grade_levels": data.get("grade_levels", []),
                        "curriculum_alignment": data.get("curriculum_alignment", []),
                        "download_count": data.get("download_count", 0),
                        "rating": data.get("rating", 0),
                        "relevance_score": relevance_score
                    }
                    results.append(result_item)
                    
            except Exception as e:
                print(f"[ERROR] Failed to process document {doc.id}: {e}")
                continue
        
        # Sort by relevance score (highest first) and limit results
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        final_results = results[:limit]
        
        print(f"[RESULTS] Found {len(final_results)} relevant resources (from {len(results)} candidates)")
        
        # Calculate search statistics
        subject_breakdown = {}
        type_breakdown = {}
        
        for result in final_results:
            subj = result["subject"]
            rtype = result["resource_type"]
            
            subject_breakdown[subj] = subject_breakdown.get(subj, 0) + 1
            type_breakdown[rtype] = type_breakdown.get(rtype, 0) + 1
        
        return {
            "status": "success",
            "results": final_results,
            "search_metadata": {
                "query": query,
                "filters_applied": filters,
                "total_found": len(final_results),
                "total_candidates": len(results),
                "subject_breakdown": subject_breakdown,
                "type_breakdown": type_breakdown,
                "search_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Knowledge base search failed: {e}")
        return {
            "status": "error", 
            "error_message": f"Knowledge base search failed: {str(e)}"
        }


def retrieve_worksheets(
    subject: str,
    grade_level: str = "",
    difficulty: str = "intermediate",
    worksheet_type: str = "practice",
    curriculum: str = "",
    limit: int = 8
) -> dict:
    """
    Retrieve worksheets specifically from the database with detailed filtering.
    
    Args:
        subject (str): Subject area (Mathematics, Science, English, etc.)
        grade_level (str): Grade or class level (Grade 6, Class 10, etc.)
        difficulty (str): Difficulty level (easy, medium, hard)
        worksheet_type (str): Type of worksheet (practice, assessment, homework, quiz)
        curriculum (str): Curriculum standard (CBSE, IB, Common Core, Cambridge)
        limit (int): Maximum number of worksheets to retrieve
        
    Returns:
        dict: Status and retrieved worksheet details
    """
    print(f"[WORKSHEET_RETRIEVAL] Searching for {subject} {grade_level} {difficulty} {worksheet_type} worksheets")
    
    try:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        if not project_id:
            return {
                "status": "error",
                "error_message": "Google Cloud Project ID not configured"
            }
        
        db = firestore.Client(project=project_id)
        print(f"[DATABASE] Connected for worksheet retrieval")
        
        # Build specific worksheet query
        worksheets_ref = db.collection("worksheets")
        
        # Apply mandatory filters
        if subject:
            worksheets_ref = worksheets_ref.where("subject", "==", subject)
        
        # Apply optional filters
        filters_applied = [f"subject={subject}"]
        
        if grade_level:
            worksheets_ref = worksheets_ref.where("grade_level", "==", grade_level)
            filters_applied.append(f"grade={grade_level}")
        
        if difficulty != "all":
            worksheets_ref = worksheets_ref.where("difficulty", "==", difficulty)
            filters_applied.append(f"difficulty={difficulty}")
        
        if worksheet_type != "all":
            worksheets_ref = worksheets_ref.where("worksheet_type", "==", worksheet_type)
            filters_applied.append(f"type={worksheet_type}")
        
        if curriculum:
            worksheets_ref = worksheets_ref.where("curriculum", "==", curriculum)
            filters_applied.append(f"curriculum={curriculum}")
        
        # Order by popularity (download count) or rating
        worksheets_ref = worksheets_ref.order_by("rating", direction=firestore.Query.DESCENDING)
        worksheets_ref = worksheets_ref.limit(limit)
        
        # Execute query
        docs = worksheets_ref.stream()
        
        worksheets = []
        for doc in docs:
            try:
                data = doc.to_dict()
                if data:
                    worksheet_item = {
                        "worksheet_id": doc.id,
                        "title": data.get("title", "Untitled Worksheet"),
                        "description": data.get("description", ""),
                        "subject": data.get("subject", subject),
                        "grade_level": data.get("grade_level", grade_level),
                        "difficulty": data.get("difficulty", difficulty),
                        "worksheet_type": data.get("worksheet_type", worksheet_type),
                        "curriculum": data.get("curriculum", ""),
                        "topics_covered": data.get("topics_covered", []),
                        "learning_objectives": data.get("learning_objectives", []),
                        "file_url": data.get("file_url", ""),
                        "file_format": data.get("file_format", "pdf"),
                        "page_count": data.get("page_count", 1),
                        "answer_key_available": data.get("answer_key_available", False),
                        "answer_key_url": data.get("answer_key_url", ""),
                        "created_by": data.get("created_by", "Unknown"),
                        "created_date": data.get("created_date"),
                        "last_updated": data.get("last_updated"),
                        "download_count": data.get("download_count", 0),
                        "rating": data.get("rating", 0.0),
                        "rating_count": data.get("rating_count", 0),
                        "tags": data.get("tags", []),
                        "estimated_time": data.get("estimated_time", "30 minutes"),
                        "prerequisites": data.get("prerequisites", [])
                    }
                    worksheets.append(worksheet_item)
                    
            except Exception as e:
                print(f"[ERROR] Failed to process worksheet {doc.id}: {e}")
                continue
        
        print(f"[WORKSHEET_SUCCESS] Retrieved {len(worksheets)} worksheets")
        
        # Calculate statistics
        difficulty_stats = {}
        type_stats = {}
        
        for ws in worksheets:
            diff = ws["difficulty"]
            wtype = ws["worksheet_type"]
            
            difficulty_stats[diff] = difficulty_stats.get(diff, 0) + 1
            type_stats[wtype] = type_stats.get(wtype, 0) + 1
        
        return {
            "status": "success",
            "worksheets": worksheets,
            "retrieval_metadata": {
                "filters_applied": filters_applied,
                "total_retrieved": len(worksheets),
                "difficulty_distribution": difficulty_stats,
                "type_distribution": type_stats,
                "retrieval_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Worksheet retrieval failed: {e}")
        return {
            "status": "error",
            "error_message": f"Worksheet retrieval failed: {str(e)}"
        }


def find_documents(
    search_term: str,
    document_type: str = "all",
    subject: str = "",
    file_format: str = "all",
    min_size_kb: int = 0,
    max_size_mb: int = 50,
    limit: int = 12
) -> dict:
    """
    Find and retrieve documents (PDFs, presentations, etc.) from storage systems.
    
    Args:
        search_term (str): Search term for document titles and content
        document_type (str): Type of document (pdf, presentation, document, image, all)
        subject (str): Subject area filter
        file_format (str): Specific file format (pdf, pptx, docx, jpg, all)
        min_size_kb (int): Minimum file size in KB
        max_size_mb (int): Maximum file size in MB
        limit (int): Maximum number of documents to find
        
    Returns:
        dict: Status and found document details
    """
    print(f"[DOCUMENT_SEARCH] Searching for '{search_term}' documents of type {document_type}")
    
    try:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        if not project_id:
            return {
                "status": "error",
                "error_message": "Google Cloud Project ID not configured"
            }
        
        db = firestore.Client(project=project_id)
        print(f"[DATABASE] Connected for document search")
        
        # Build document search query
        documents_ref = db.collection("educational_documents")
        
        # Apply filters
        filters_applied = []
        
        if subject:
            documents_ref = documents_ref.where("subject", "==", subject)
            filters_applied.append(f"subject={subject}")
        
        if document_type != "all":
            documents_ref = documents_ref.where("document_type", "==", document_type)
            filters_applied.append(f"type={document_type}")
        
        if file_format != "all":
            documents_ref = documents_ref.where("file_format", "==", file_format)
            filters_applied.append(f"format={file_format}")
        
        # Size filtering will be done post-query since Firestore has limitations
        min_size_bytes = min_size_kb * 1024
        max_size_bytes = max_size_mb * 1024 * 1024
        
        # Order by relevance or recency
        documents_ref = documents_ref.order_by("upload_date", direction=firestore.Query.DESCENDING)
        documents_ref = documents_ref.limit(limit * 2)  # Get more for post-filtering
        
        # Execute query
        docs = documents_ref.stream()
        
        documents = []
        search_term_lower = search_term.lower()
        
        for doc in docs:
            try:
                data = doc.to_dict()
                if not data:
                    continue
                
                # Check file size constraints
                file_size = data.get("file_size_bytes", 0)
                if file_size < min_size_bytes or file_size > max_size_bytes:
                    continue
                
                # Calculate relevance for search term
                relevance_score = 0
                title = data.get("title", "").lower()
                description = data.get("description", "").lower()
                tags = " ".join(data.get("tags", [])).lower()
                
                if search_term_lower in title:
                    relevance_score += 3
                if search_term_lower in description:
                    relevance_score += 2
                if search_term_lower in tags:
                    relevance_score += 1
                
                # Include partial matches
                search_words = search_term_lower.split()
                for word in search_words:
                    if len(word) > 3:
                        if word in title: relevance_score += 1
                        if word in description: relevance_score += 0.5
                
                # Only include documents with some relevance
                if relevance_score > 0:
                    # Convert file size to readable format
                    if file_size > 1024 * 1024:
                        size_display = f"{file_size / (1024 * 1024):.1f} MB"
                    elif file_size > 1024:
                        size_display = f"{file_size / 1024:.1f} KB"
                    else:
                        size_display = f"{file_size} bytes"
                    
                    document_item = {
                        "document_id": doc.id,
                        "title": data.get("title", "Untitled Document"),
                        "description": data.get("description", ""),
                        "document_type": data.get("document_type", "unknown"),
                        "file_format": data.get("file_format", ""),
                        "file_url": data.get("file_url", ""),
                        "download_url": data.get("download_url", ""),
                        "preview_url": data.get("preview_url", ""),
                        "file_size_bytes": file_size,
                        "file_size_display": size_display,
                        "subject": data.get("subject", ""),
                        "grade_levels": data.get("grade_levels", []),
                        "curriculum": data.get("curriculum", []),
                        "tags": data.get("tags", []),
                        "author": data.get("author", "Unknown"),
                        "upload_date": data.get("upload_date"),
                        "last_modified": data.get("last_modified"),
                        "download_count": data.get("download_count", 0),
                        "views": data.get("views", 0),
                        "is_public": data.get("is_public", True),
                        "language": data.get("language", "English"),
                        "relevance_score": relevance_score
                    }
                    documents.append(document_item)
                    
            except Exception as e:
                print(f"[ERROR] Failed to process document {doc.id}: {e}")
                continue
        
        # Sort by relevance and limit results
        documents.sort(key=lambda x: x["relevance_score"], reverse=True)
        final_documents = documents[:limit]
        
        print(f"[DOCUMENT_SUCCESS] Found {len(final_documents)} relevant documents")
        
        # Calculate statistics
        format_stats = {}
        type_stats = {}
        size_stats = {"small": 0, "medium": 0, "large": 0}
        
        for doc in final_documents:
            fmt = doc["file_format"]
            dtype = doc["document_type"]
            size_bytes = doc["file_size_bytes"]
            
            format_stats[fmt] = format_stats.get(fmt, 0) + 1
            type_stats[dtype] = type_stats.get(dtype, 0) + 1
            
            if size_bytes < 1024 * 1024:  # < 1MB
                size_stats["small"] += 1
            elif size_bytes < 10 * 1024 * 1024:  # < 10MB
                size_stats["medium"] += 1
            else:
                size_stats["large"] += 1
        
        return {
            "status": "success",
            "documents": final_documents,
            "search_metadata": {
                "search_term": search_term,
                "filters_applied": filters_applied,
                "total_found": len(final_documents),
                "format_distribution": format_stats,
                "type_distribution": type_stats,
                "size_distribution": size_stats,
                "search_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Document search failed: {e}")
        return {
            "status": "error",
            "error_message": f"Document search failed: {str(e)}"
        }


def query_file_storage(
    storage_path: str = "",
    file_pattern: str = "*",
    bucket_name: str = "",
    include_metadata: bool = True,
    sort_by: str = "modified_date",
    limit: int = 20
) -> dict:
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
                    "public_url": blob.public_url if blob.public_url_set else "",
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


def search_by_metadata(
    metadata_filters: Dict[str, Any],
    collection_name: str = "educational_resources",
    search_mode: str = "AND",
    include_content: bool = False,
    limit: int = 15
) -> dict:
    """
    Advanced search using metadata tags, categories, and custom attributes.
    
    Args:
        metadata_filters (dict): Dictionary of metadata key-value pairs to filter by
        collection_name (str): Firestore collection to search in
        search_mode (str): How to combine filters ("AND" or "OR")
        include_content (bool): Whether to include full content in results
        limit (int): Maximum number of results to return
        
    Returns:
        dict: Status and metadata-based search results
    """
    print(f"[METADATA_SEARCH] Searching {collection_name} with {len(metadata_filters)} filters ({search_mode} mode)")
    
    try:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        if not project_id:
            return {
                "status": "error",
                "error_message": "Google Cloud Project ID not configured"
            }
        
        db = firestore.Client(project=project_id)
        print(f"[DATABASE] Connected for metadata search")
        
        # Start with base collection
        collection_ref = db.collection(collection_name)
        
        # Apply metadata filters
        filters_applied = []
        
        if search_mode == "AND":
            # Apply all filters (intersection)
            for key, value in metadata_filters.items():
                if isinstance(value, list):
                    # Array contains search
                    collection_ref = collection_ref.where(key, "array_contains_any", value)
                    filters_applied.append(f"{key} contains any of {value}")
                else:
                    # Exact match
                    collection_ref = collection_ref.where(key, "==", value)
                    filters_applied.append(f"{key}={value}")
        
        else:  # OR mode is more complex with Firestore limitations
            print("[WARNING] OR mode has limited support due to Firestore constraints")
            # For OR mode, we'll execute multiple queries and merge results
            # This is a simplified implementation
            if metadata_filters:
                first_key, first_value = next(iter(metadata_filters.items()))
                if isinstance(first_value, list):
                    collection_ref = collection_ref.where(first_key, "array_contains_any", first_value)
                else:
                    collection_ref = collection_ref.where(first_key, "==", first_value)
                filters_applied.append(f"{first_key}={first_value}")
        
        # Order and limit
        collection_ref = collection_ref.order_by("created_date", direction=firestore.Query.DESCENDING)
        collection_ref = collection_ref.limit(limit)
        
        # Execute query
        docs = collection_ref.stream()
        
        results = []
        for doc in docs:
            try:
                data = doc.to_dict()
                if not data:
                    continue
                
                # Additional filtering for OR mode or complex conditions
                if search_mode == "OR" and len(metadata_filters) > 1:
                    # Check if document matches any of the remaining filters
                    match_found = False
                    for key, value in list(metadata_filters.items())[1:]:  # Skip first one already applied
                        doc_value = data.get(key)
                        if doc_value:
                            if isinstance(value, list) and isinstance(doc_value, list):
                                if any(v in doc_value for v in value):
                                    match_found = True
                                    break
                            elif doc_value == value:
                                match_found = True
                                break
                    
                    if not match_found:
                        continue
                
                result_item = {
                    "id": doc.id,
                    "title": data.get("title", "Untitled"),
                    "resource_type": data.get("resource_type", "unknown"),
                    "subject": data.get("subject", ""),
                    "difficulty_level": data.get("difficulty_level", ""),
                    "grade_levels": data.get("grade_levels", []),
                    "curriculum_alignment": data.get("curriculum_alignment", []),
                    "tags": data.get("tags", []),
                    "metadata": {
                        key: data.get(key) for key in metadata_filters.keys() 
                        if key in data
                    },
                    "author": data.get("author", "Unknown"),
                    "created_date": data.get("created_date"),
                    "rating": data.get("rating", 0),
                    "download_count": data.get("download_count", 0)
                }
                
                # Include full content if requested
                if include_content:
                    result_item["content"] = data.get("content", "")
                    result_item["summary"] = data.get("summary", "")
                    result_item["file_url"] = data.get("file_url", "")
                
                results.append(result_item)
                
            except Exception as e:
                print(f"[ERROR] Failed to process document {doc.id}: {e}")
                continue
        
        print(f"[METADATA_SUCCESS] Found {len(results)} resources matching metadata criteria")
        
        # Analyze metadata patterns in results
        metadata_analysis = {}
        for key in metadata_filters.keys():
            values = []
            for result in results:
                value = result.get("metadata", {}).get(key)
                if value:
                    if isinstance(value, list):
                        values.extend(value)
                    else:
                        values.append(value)
            
            if values:
                # Count unique values
                from collections import Counter
                metadata_analysis[key] = dict(Counter(values))
        
        return {
            "status": "success",
            "results": results,
            "search_metadata": {
                "collection_searched": collection_name,
                "filters_applied": filters_applied,
                "search_mode": search_mode,
                "total_found": len(results),
                "metadata_patterns": metadata_analysis,
                "search_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Metadata search failed: {e}")
        return {
            "status": "error",
            "error_message": f"Metadata search failed: {str(e)}"
        }


# Create the ADK Retrieval Agent with specialized database and storage tools
root_agent = Agent(
    name="sahayak_retrieval_agent",
    model="gemini-2.0-flash",
    description="Database and storage retrieval specialist for educational materials and resources",
    instruction=(
        "You are the Sahayak Retrieval Agent, a specialized database and storage expert focused on "
        "finding, retrieving, and organizing educational materials from various data sources. "
        "Your expertise includes comprehensive search capabilities across knowledge bases, "
        "document repositories, and cloud storage systems.\\n\\n"
        
        "Your core capabilities include:\\n\\n"
        
        "1. KNOWLEDGE BASE SEARCH:\\n"
        "   - Search educational resources in Firestore databases\\n"
        "   - Apply advanced filtering by subject, difficulty, resource type\\n"
        "   - Calculate relevance scores and rank results intelligently\\n"
        "   - Provide comprehensive search metadata and statistics\\n\\n"
        
        "2. WORKSHEET RETRIEVAL:\\n"
        "   - Find worksheets by subject, grade level, and curriculum\\n"
        "   - Filter by difficulty, worksheet type, and learning objectives\\n"
        "   - Retrieve detailed worksheet metadata including answer keys\\n"
        "   - Provide usage statistics and popularity rankings\\n\\n"
        
        "3. DOCUMENT DISCOVERY:\\n"
        "   - Search for PDFs, presentations, and educational documents\\n"
        "   - Filter by file type, size, and content relevance\\n"
        "   - Provide download links and preview capabilities\\n"
        "   - Analyze document metadata and accessibility\\n\\n"
        
        "4. CLOUD STORAGE QUERIES:\\n"
        "   - Query Google Cloud Storage for educational materials\\n"
        "   - Search by file patterns, paths, and metadata\\n"
        "   - Generate secure download URLs and access controls\\n"
        "   - Organize files by type, size, and modification date\\n\\n"
        
        "5. ADVANCED METADATA SEARCH:\\n"
        "   - Perform complex searches using custom metadata attributes\\n"
        "   - Support AND/OR logic for multi-criteria filtering\\n"
        "   - Analyze metadata patterns and provide insights\\n"
        "   - Enable precision targeting of specific resource types\\n\\n"
        
        "Always prioritize search relevance, result accuracy, and user efficiency. "
        "Provide comprehensive metadata to help teachers make informed decisions about "
        "educational resources. Ensure all retrieved materials are properly categorized "
        "and include sufficient information for effective educational use."
    ),
    tools=[
        search_knowledge_base,
        retrieve_worksheets,  
        find_documents,
        query_file_storage,
        search_by_metadata
    ]
)

if __name__ == "__main__":
    # For testing purposes
    print(f"[AGENT_INIT] Retrieval Agent '{root_agent.name}' initialized successfully")
    print(f"[AGENT_TOOLS] Available tools: {len(root_agent.tools)}")
    for i, tool in enumerate(root_agent.tools, 1):
        print(f"  {i}. {tool.__name__}")