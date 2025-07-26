"""
Sahayak Retrieval Agent

Unified database and storage retrieval specialist for educational materials.
Provides intelligent search across all educational content types with a clean,
simplified API.

Core Functions:
- search_educational_content: UNIFIED search across all educational content types
- query_file_storage: Direct cloud storage access for raw file operations

Features:
- Single intelligent search replacing multiple redundant functions
- Smart content type mapping (lessons, worksheets, documents, all)
- Cross-collection relevance ranking and result normalization
- Advanced filtering by subject, difficulty, grade, curriculum, file format
- Direct cloud storage browsing for unindexed files
"""

from .agent import (
    # Core unified functions
    search_educational_content,  # Primary unified search
    query_file_storage,         # Cloud storage queries
    # Agent instance
    root_agent
)

__version__ = "2.0.0"
__agent_name__ = "Sahayak Retrieval Agent"
__specialization__ = "Unified Educational Content Search & Cloud Storage Access"