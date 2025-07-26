#!/usr/bin/env python3
"""
Firestore MCP Server for Knowledge Base Management
Provides educational knowledge base storage and retrieval via MCP protocol
"""

import asyncio
import json
import logging
from typing import Any, Sequence
import os
from datetime import datetime

from google.cloud import firestore
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("firestore-server")

class FirestoreServer:
    def __init__(self):
        self.server = Server("firestore-knowledge-server")
        self.db = None
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_resources()
        async def handle_list_resources() -> list[Resource]:
            """List available Firestore resources"""
            return [
                Resource(
                    uri="firestore://collections",
                    name="Firestore Collections",
                    description="Available Firestore collections for educational content",
                    mimeType="application/json",
                ),
                Resource(
                    uri="firestore://educational_resources",
                    name="Educational Resources",
                    description="Curated educational materials and resources",
                    mimeType="application/json",
                ),
                Resource(
                    uri="firestore://curriculum_standards",
                    name="Curriculum Standards Database",
                    description="Educational curriculum standards and alignment data",
                    mimeType="application/json",
                )
            ]

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read Firestore resources"""
            if uri == "firestore://collections":
                collections = [
                    "educational_resources",
                    "curriculum_standards", 
                    "lesson_plans",
                    "assessment_templates",
                    "learning_objectives",
                    "subject_taxonomies"
                ]
                return json.dumps({"collections": collections})
                
            elif uri == "firestore://educational_resources":
                # Get sample educational resources
                docs = self.db.collection("educational_resources").limit(10).stream()
                resources = []
                for doc in docs:
                    data = doc.to_dict()
                    resources.append({
                        "id": doc.id,
                        "title": data.get("title", ""),
                        "subject": data.get("subject", ""),
                        "type": data.get("resource_type", ""),
                        "summary": data.get("summary", "")
                    })
                return json.dumps({"resources": resources})
                
            elif uri == "firestore://curriculum_standards":
                # Get curriculum standards
                docs = self.db.collection("curriculum_standards").limit(5).stream()
                standards = []
                for doc in docs:
                    data = doc.to_dict()
                    standards.append({
                        "id": doc.id,
                        "name": data.get("name", ""),
                        "description": data.get("description", ""),
                        "subjects": data.get("subjects", [])
                    })
                return json.dumps({"standards": standards})
                
            else:
                raise ValueError(f"Unknown resource: {uri}")

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available Firestore tools"""
            return [
                Tool(
                    name="search_educational_resources",
                    description="Search for educational resources in the knowledge base",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for educational resources"
                            },
                            "subject": {
                                "type": "string",
                                "description": "Filter by subject area"
                            },
                            "resource_type": {
                                "type": "string",
                                "enum": ["lesson_plan", "assessment", "activity", "reference", "multimedia"],
                                "description": "Type of educational resource"
                            },
                            "difficulty_level": {
                                "type": "string",
                                "enum": ["beginner", "intermediate", "advanced"],
                                "description": "Difficulty level filter"
                            },
                            "limit": {
                                "type": "integer",
                                "default": 10,
                                "description": "Maximum number of results to return"
                            }
                        },
                        "required": ["query"]
                    },
                ),
                Tool(
                    name="get_curriculum_alignment",
                    description="Get curriculum standard alignment information",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "standard": {
                                "type": "string",
                                "description": "Curriculum standard name (e.g., CBSE, IB, Common Core)"
                            },
                            "subject": {
                                "type": "string",
                                "description": "Subject area"
                            },
                            "grade_level": {
                                "type": "string",
                                "description": "Grade or level"
                            },
                            "topic": {
                                "type": "string",
                                "description": "Specific topic or concept to align"
                            }
                        },
                        "required": ["standard", "subject"]
                    },
                ),
                Tool(
                    name="store_generated_content",
                    description="Store generated educational content in the knowledge base",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Content title"
                            },
                            "content": {
                                "type": "string",
                                "description": "Educational content to store"
                            },
                            "subject": {
                                "type": "string",
                                "description": "Subject area"
                            },
                            "content_type": {
                                "type": "string",
                                "description": "Type of content"
                            },
                            "difficulty": {
                                "type": "string",
                                "description": "Difficulty level"
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Content tags for categorization"
                            },
                            "curriculum_alignment": {
                                "type": "string",
                                "description": "Curriculum standard alignment"
                            }
                        },
                        "required": ["title", "content", "subject", "content_type"]
                    },
                ),
                Tool(
                    name="get_related_resources",
                    description="Find resources related to a specific topic or concept",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Topic or concept to find related resources for"
                            },
                            "subject": {
                                "type": "string",
                                "description": "Subject area"
                            },
                            "relation_type": {
                                "type": "string",
                                "enum": ["prerequisite", "followup", "related", "extension"],
                                "description": "Type of relationship to find"
                            }
                        },
                        "required": ["topic"]
                    },
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent | EmbeddedResource]:
            """Handle tool calls"""
            if name == "search_educational_resources":
                return await self._search_resources(arguments)
            elif name == "get_curriculum_alignment":
                return await self._get_curriculum_alignment(arguments)
            elif name == "store_generated_content":
                return await self._store_content(arguments)
            elif name == "get_related_resources":
                return await self._get_related_resources(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _search_resources(self, args: dict) -> list[TextContent]:
        """Search for educational resources"""
        try:
            query = args["query"]
            subject = args.get("subject")
            resource_type = args.get("resource_type")
            difficulty = args.get("difficulty_level")
            limit = args.get("limit", 10)
            
            # Build Firestore query
            collection_ref = self.db.collection("educational_resources")
            
            # Apply filters
            if subject:
                collection_ref = collection_ref.where("subject", "==", subject)
            if resource_type:
                collection_ref = collection_ref.where("resource_type", "==", resource_type)
            if difficulty:
                collection_ref = collection_ref.where("difficulty_level", "==", difficulty)
            
            # For text search, we'll use a simple array_contains approach
            # In production, consider using Algolia or Elasticsearch for full-text search
            docs = collection_ref.limit(limit).stream()
            
            results = []
            for doc in docs:
                data = doc.to_dict()
                # Simple text matching
                if query.lower() in data.get("title", "").lower() or \
                   query.lower() in data.get("summary", "").lower() or \
                   query.lower() in str(data.get("tags", [])).lower():
                    results.append({
                        "id": doc.id,
                        "title": data.get("title", ""),
                        "summary": data.get("summary", ""),
                        "subject": data.get("subject", ""),
                        "resource_type": data.get("resource_type", ""),
                        "difficulty_level": data.get("difficulty_level", ""),
                        "tags": data.get("tags", []),
                        "content_preview": data.get("content", "")[:200] + "..." if data.get("content") else "",
                        "curriculum_alignment": data.get("curriculum_alignment", ""),
                        "created_date": data.get("created_date", "").isoformat() if data.get("created_date") else ""
                    })
            
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "query": query,
                        "filters": {
                            "subject": subject,
                            "resource_type": resource_type,
                            "difficulty_level": difficulty
                        },
                        "results_count": len(results),
                        "results": results
                    }, indent=2)
                )
            ]
            
        except Exception as e:
            logger.error(f"Resource search failed: {e}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "error": str(e)
                    })
                )
            ]

    async def _get_curriculum_alignment(self, args: dict) -> list[TextContent]:
        """Get curriculum alignment information"""
        try:
            standard = args["standard"]
            subject = args["subject"]
            grade_level = args.get("grade_level")
            topic = args.get("topic")
            
            # Query curriculum standards
            query = self.db.collection("curriculum_standards").where("name", "==", standard)
            docs = query.stream()
            
            alignment_info = {}
            for doc in docs:
                data = doc.to_dict()
                if subject in data.get("subjects", {}):
                    subject_data = data["subjects"][subject]
                    
                    if grade_level and grade_level in subject_data.get("grade_levels", {}):
                        grade_data = subject_data["grade_levels"][grade_level]
                        alignment_info = {
                            "standard": standard,
                            "subject": subject,
                            "grade_level": grade_level,
                            "learning_objectives": grade_data.get("learning_objectives", []),
                            "key_topics": grade_data.get("key_topics", []),
                            "assessment_criteria": grade_data.get("assessment_criteria", [])
                        }
                        
                        if topic:
                            # Find specific topic alignment
                            topic_alignments = []
                            for obj in grade_data.get("learning_objectives", []):
                                if topic.lower() in obj.get("description", "").lower():
                                    topic_alignments.append(obj)
                            alignment_info["topic_specific"] = topic_alignments
                    else:
                        # General subject alignment
                        alignment_info = {
                            "standard": standard,
                            "subject": subject,
                            "overview": subject_data.get("overview", ""),
                            "core_concepts": subject_data.get("core_concepts", []),
                            "skills_framework": subject_data.get("skills_framework", {})
                        }
            
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "alignment": alignment_info
                    }, indent=2)
                )
            ]
            
        except Exception as e:
            logger.error(f"Curriculum alignment retrieval failed: {e}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "error": str(e)
                    })
                )
            ]

    async def _store_content(self, args: dict) -> list[TextContent]:
        """Store generated content in knowledge base"""
        try:
            content_data = {
                "title": args["title"],
                "content": args["content"],
                "subject": args["subject"],
                "content_type": args["content_type"],
                "difficulty_level": args["difficulty"],
                "tags": args.get("tags", []),
                "curriculum_alignment": args.get("curriculum_alignment", ""),
                "created_date": datetime.utcnow(),
                "source": "ai_generated",
                "status": "active"
            }
            
            # Add to Firestore
            doc_ref = self.db.collection("educational_resources").add(content_data)
            doc_id = doc_ref[1].id
            
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "message": "Content stored successfully",
                        "document_id": doc_id,
                        "stored_content": {
                            "title": args["title"],
                            "subject": args["subject"],
                            "content_type": args["content_type"]
                        }
                    }, indent=2)
                )
            ]
            
        except Exception as e:
            logger.error(f"Content storage failed: {e}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "error": str(e)
                    })
                )
            ]

    async def _get_related_resources(self, args: dict) -> list[TextContent]:
        """Find related educational resources"""
        try:
            topic = args["topic"]
            subject = args.get("subject")
            relation_type = args.get("relation_type", "related")
            
            # Query for related resources
            collection_ref = self.db.collection("educational_resources")
            
            if subject:
                collection_ref = collection_ref.where("subject", "==", subject)
            
            docs = collection_ref.limit(20).stream()
            
            related_resources = []
            for doc in docs:
                data = doc.to_dict()
                
                # Simple relatedness check based on tags and content
                is_related = False
                if topic.lower() in data.get("title", "").lower():
                    is_related = True
                elif any(topic.lower() in tag.lower() for tag in data.get("tags", [])):
                    is_related = True
                elif topic.lower() in data.get("summary", "").lower():
                    is_related = True
                
                if is_related:
                    related_resources.append({
                        "id": doc.id,
                        "title": data.get("title", ""),
                        "summary": data.get("summary", ""),
                        "subject": data.get("subject", ""),
                        "content_type": data.get("content_type", ""),
                        "difficulty_level": data.get("difficulty_level", ""),
                        "relationship_score": self._calculate_relatedness(topic, data)
                    })
            
            # Sort by relationship score
            related_resources.sort(key=lambda x: x["relationship_score"], reverse=True)
            related_resources = related_resources[:10]  # Top 10 most related
            
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "topic": topic,
                        "relation_type": relation_type,
                        "related_resources": related_resources
                    }, indent=2)
                )
            ]
            
        except Exception as e:
            logger.error(f"Related resources search failed: {e}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "error": str(e)
                    })
                )
            ]

    def _calculate_relatedness(self, topic: str, resource_data: dict) -> float:
        """Calculate relatedness score between topic and resource"""
        score = 0.0
        topic_lower = topic.lower()
        
        # Title match (highest weight)
        if topic_lower in resource_data.get("title", "").lower():
            score += 3.0
        
        # Tags match
        for tag in resource_data.get("tags", []):
            if topic_lower in tag.lower():
                score += 2.0
        
        # Summary match
        if topic_lower in resource_data.get("summary", "").lower():
            score += 1.0
        
        # Content preview match
        if topic_lower in resource_data.get("content", "")[:500].lower():
            score += 1.5
        
        return score

    async def run(self):
        """Initialize and run the Firestore MCP server"""
        # Initialize Firestore
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        if not project_id:
            logger.error("GOOGLE_CLOUD_PROJECT_ID environment variable not set")
            raise ValueError("Google Cloud Project ID required")
            
        self.db = firestore.Client(project=project_id)
        
        logger.info("Firestore MCP Server initialized successfully")
        
        # Run server
        async with self.server.request_context():
            await self.server.run()

async def main():
    """Main entry point"""
    server = FirestoreServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())