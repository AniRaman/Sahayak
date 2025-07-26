"""
Sahayak Worksheet Agent

Worksheet agent for generating questions from uploaded files or images.
"""

from .agent import (
    root_agent,
    extract_content_from_upload,
    generate_questions_by_difficulty,
)

__version__ = "1.0.0"
__author__ = "Sahayak Educational AI System"