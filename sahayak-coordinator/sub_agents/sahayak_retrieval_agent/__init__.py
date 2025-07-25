"""
Sahayak Retrieval Agent

Database and storage retrieval specialist that handles finding and retrieving 
educational materials from various storage systems including Firestore, 
Cloud Storage, and local databases.

Key Functions:
- search_knowledge_base: Search educational resources in Firestore database
- retrieve_worksheets: Find worksheets by subject, grade, and difficulty
- find_documents: Search for PDFs, presentations, and documents
- query_file_storage: Search cloud storage for educational materials
- search_by_metadata: Advanced search using tags, categories, and filters
"""

from .agent import (
    search_knowledge_base,
    retrieve_worksheets,
    find_documents,
    query_file_storage,
    search_by_metadata,
    root_agent
)

__version__ = "1.0.0"
__agent_name__ = "Sahayak Retrieval Agent"
__specialization__ = "Database & Storage Retrieval Specialist"