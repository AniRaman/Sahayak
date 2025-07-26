"""
Sahayak Worksheet Agent

Educational material generator that transforms content into:
- Presentation slides (PPT files)
- Differentiated worksheets (PDF files) at easy/medium/hard levels
- Cloud storage integration with secure download links
- Firestore metadata management for easy retrieval
"""

from .agent import (
    generate_educational_materials,
    root_agent
)

__version__ = "1.0.0"
__agent_name__ = "Sahayak Worksheet Agent"
__specialization__ = "Educational Material Generation - Slides & Differentiated Worksheets"