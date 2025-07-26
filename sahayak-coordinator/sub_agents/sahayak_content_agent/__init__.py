"""
Sahayak Content Agent

Educational content generation specialist with curriculum alignment capabilities.
This agent specializes in creating high-quality instructional materials for teachers and students.

Key Functions:
- generate_educational_content: Main content generation using Gemini AI
- search_knowledge_base: Search educational resources in Firestore
- search_web_for_education: Web search for educational content
- align_with_curriculum: Check curriculum standards alignment
- analyze_content_request: Analyze teacher input and requirements
"""

from .agent import (
    generate_educational_content,
    search_web_for_education,
    align_with_curriculum,
    create_assessment_questions,
    root_agent
)

__version__ = "1.0.0"
__agent_name__ = "Sahayak Content Agent"
__specialization__ = "Educational Content Generation & Curriculum Alignment"