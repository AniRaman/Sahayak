"""
Sahayak Coordinator Agent Package
Central orchestrator for the educational content generation pipeline
"""

__version__ = "1.0.0"
__description__ = "Sahayak Educational AI - Coordinator Agent"

from .agent import (
    coordinate_request,
    root_agent
)

__all__ = ["root_agent"]