#!/usr/bin/env python3
"""
Web Search MCP Server for Educational Content Enhancement
Provides web search capabilities to enhance educational content with current information
"""

import asyncio
import json
import logging
from typing import Any, Sequence
import os
import aiohttp
from datetime import datetime, timedelta

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
logger = logging.getLogger("web-search-server")

class WebSearchServer:
    def __init__(self):
        self.server = Server("web-search-server")
        self.search_api_key = None
        self.search_engine_id = None
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_resources()
        async def handle_list_resources() -> list[Resource]:
            """List available web search resources"""
            return [
                Resource(
                    uri="websearch://educational-sites",
                    name="Educational Websites",
                    description="Curated list of trusted educational websites for content verification",
                    mimeType="application/json",
                ),
                Resource(
                    uri="websearch://academic-sources",
                    name="Academic Sources",
                    description="Academic journals and research sources",
                    mimeType="application/json",
                ),
                Resource(
                    uri="websearch://curriculum-resources",
                    name="Curriculum Resources",
                    description="Official curriculum and educational standard resources",
                    mimeType="application/json",
                )
            ]

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read web search resources"""
            if uri == "websearch://educational-sites":
                educational_sites = {
                    "trusted_domains": [
                        "khanacademy.org",
                        "coursera.org", 
                        "edx.org",
                        "britannica.com",
                        "nationalgeographic.com",
                        "smithsonian.edu",
                        "nasa.gov",
                        "nist.gov",
                        "education.gov",
                        "scholar.google.com"
                    ],
                    "subject_specific": {
                        "mathematics": ["mathworld.wolfram.com", "brilliant.org", "artofproblemsolving.com"],
                        "science": ["nasa.gov", "noaa.gov", "nih.gov", "nsf.gov"],
                        "history": ["loc.gov", "archives.gov", "britannica.com"],
                        "literature": ["gutenberg.org", "poetryfoundation.org", "mla.org"]
                    }
                }
                return json.dumps(educational_sites)
                
            elif uri == "websearch://academic-sources":
                academic_sources = {
                    "databases": [
                        "Google Scholar",
                        "JSTOR",
                        "PubMed",
                        "IEEE Xplore",
                        "SpringerLink"
                    ],
                    "open_access": [
                        "arxiv.org",
                        "biorxiv.org",
                        "plos.org",
                        "doaj.org"
                    ]
                }
                return json.dumps(academic_sources)
                
            elif uri == "websearch://curriculum-resources":
                curriculum_resources = {
                    "international": {
                        "IB": ["ibo.org"],
                        "Cambridge": ["cambridgeinternational.org"],
                        "AP": ["collegeboard.org"]
                    },
                    "national": {
                        "US": ["corestandards.org", "nextgenscience.org"],
                        "UK": ["gov.uk/government/organisations/department-for-education"],
                        "India": ["ncert.nic.in", "cbse.gov.in"]
                    }
                }
                return json.dumps(curriculum_resources)
                
            else:
                raise ValueError(f"Unknown resource: {uri}")

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available web search tools"""
            return [
                Tool(
                    name="search_educational_content",
                    description="Search the web for educational content and resources",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for educational content"
                            },
                            "subject": {
                                "type": "string",
                                "description": "Subject area to focus search on"
                            },
                            "content_type": {
                                "type": "string",
                                "enum": ["articles", "videos", "interactive", "research", "lesson_plans", "assessments"],
                                "description": "Type of content to search for"
                            },
                            "trusted_only": {
                                "type": "boolean",
                                "default": True,
                                "description": "Limit search to trusted educational domains"
                            },
                            "date_range": {
                                "type": "string",
                                "enum": ["any", "past_year", "past_month", "past_week"],
                                "default": "any",
                                "description": "Time range for content freshness"
                            },
                            "max_results": {
                                "type": "integer",
                                "default": 10,
                                "maximum": 20,
                                "description": "Maximum number of search results"
                            }
                        },
                        "required": ["query"]
                    },
                ),
                Tool(
                    name="verify_educational_facts",
                    description="Verify educational facts and information using web sources",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "facts": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of facts to verify"
                            },
                            "subject": {
                                "type": "string",
                                "description": "Subject area for context"
                            },
                            "sources_required": {
                                "type": "integer",
                                "default": 3,
                                "description": "Minimum number of sources for verification"
                            }
                        },
                        "required": ["facts"]
                    },
                ),
                Tool(
                    name="find_current_examples",
                    description="Find current, real-world examples for educational concepts",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "concept": {
                                "type": "string",
                                "description": "Educational concept to find examples for"
                            },
                            "subject": {
                                "type": "string",
                                "description": "Subject area"
                            },
                            "example_type": {
                                "type": "string",
                                "enum": ["case_study", "news_event", "application", "research_finding"],
                                "description": "Type of examples to find"
                            },
                            "recency": {
                                "type": "string",
                                "enum": ["very_recent", "past_year", "past_two_years"],
                                "default": "past_year",
                                "description": "How recent the examples should be"
                            }
                        },
                        "required": ["concept"]
                    },
                ),
                Tool(
                    name="get_curriculum_updates",
                    description="Search for recent updates to curriculum standards and educational policies",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "curriculum": {
                                "type": "string",
                                "description": "Curriculum standard name (e.g., CBSE, IB, Common Core)"
                            },
                            "subject": {
                                "type": "string",
                                "description": "Subject area"
                            },
                            "region": {
                                "type": "string",
                                "description": "Geographic region or country"
                            }
                        },
                        "required": ["curriculum"]
                    },
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent | EmbeddedResource]:
            """Handle tool calls"""
            if name == "search_educational_content":
                return await self._search_educational_content(arguments)
            elif name == "verify_educational_facts":
                return await self._verify_facts(arguments)
            elif name == "find_current_examples":
                return await self._find_current_examples(arguments)
            elif name == "get_curriculum_updates":
                return await self._get_curriculum_updates(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _search_educational_content(self, args: dict) -> list[TextContent]:
        """Search for educational content on the web"""
        try:
            query = args["query"]
            subject = args.get("subject", "")
            content_type = args.get("content_type", "articles")
            trusted_only = args.get("trusted_only", True)
            date_range = args.get("date_range", "any")
            max_results = args.get("max_results", 10)
            
            # Enhance query with subject and content type
            enhanced_query = f"{query} {subject} {content_type}".strip()
            
            # Add site restrictions for trusted domains if enabled
            site_restriction = ""
            if trusted_only:
                trusted_domains = [
                    "khanacademy.org", "coursera.org", "edx.org", "britannica.com",
                    "nationalgeographic.com", "smithsonian.edu", "nasa.gov",
                    "education.gov", "scholar.google.com"
                ]
                site_restriction = " OR ".join([f"site:{domain}" for domain in trusted_domains])
                enhanced_query = f"{enhanced_query} ({site_restriction})"
            
            # Perform search using Google Custom Search API
            search_results = await self._perform_google_search(
                enhanced_query, max_results, date_range
            )
            
            # Filter and enhance results
            filtered_results = []
            for result in search_results:
                # Extract relevant information
                filtered_result = {
                    "title": result.get("title", ""),
                    "url": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "source": self._extract_domain(result.get("link", "")),
                    "educational_relevance": self._assess_educational_relevance(result, subject),
                    "content_type_match": content_type in result.get("snippet", "").lower(),
                    "last_modified": result.get("lastModified", ""),
                    "page_map": result.get("pagemap", {})
                }
                
                # Add additional metadata if available
                if "pagemap" in result:
                    pagemap = result["pagemap"]
                    if "metatags" in pagemap and pagemap["metatags"]:
                        metatag = pagemap["metatags"][0]
                        filtered_result["description"] = metatag.get("description", "")
                        filtered_result["keywords"] = metatag.get("keywords", "")
                
                filtered_results.append(filtered_result)
            
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "query": query,
                        "enhanced_query": enhanced_query,
                        "filters": {
                            "subject": subject,
                            "content_type": content_type,
                            "trusted_only": trusted_only,
                            "date_range": date_range
                        },
                        "results_count": len(filtered_results),
                        "results": filtered_results
                    }, indent=2)
                )
            ]
            
        except Exception as e:
            logger.error(f"Educational content search failed: {e}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "error": str(e)
                    })
                )
            ]

    async def _verify_facts(self, args: dict) -> list[TextContent]:
        """Verify educational facts using web sources"""
        try:
            facts = args["facts"]
            subject = args.get("subject", "")
            sources_required = args.get("sources_required", 3)
            
            verification_results = []
            
            for fact in facts:
                # Search for verification of each fact
                search_query = f'"{fact}" verification source {subject}'
                search_results = await self._perform_google_search(search_query, sources_required * 2)
                
                # Analyze results for fact verification
                supporting_sources = []
                contradicting_sources = []
                
                for result in search_results:
                    snippet = result.get("snippet", "").lower()
                    title = result.get("title", "").lower()
                    
                    # Simple fact checking logic (in production, use more sophisticated NLP)
                    if any(word in snippet or word in title for word in fact.lower().split()):
                        if any(neg in snippet for neg in ["not", "false", "incorrect", "myth"]):
                            contradicting_sources.append({
                                "url": result.get("link", ""),
                                "title": result.get("title", ""),
                                "snippet": result.get("snippet", ""),
                                "source": self._extract_domain(result.get("link", ""))
                            })
                        else:
                            supporting_sources.append({
                                "url": result.get("link", ""),
                                "title": result.get("title", ""),
                                "snippet": result.get("snippet", ""),
                                "source": self._extract_domain(result.get("link", ""))
                            })
                
                # Determine verification status
                verification_status = "verified" if len(supporting_sources) >= sources_required else \
                                    "contradicted" if len(contradicting_sources) > len(supporting_sources) else \
                                    "insufficient_evidence"
                
                verification_results.append({
                    "fact": fact,
                    "status": verification_status,
                    "supporting_sources": supporting_sources[:sources_required],
                    "contradicting_sources": contradicting_sources,
                    "confidence": min(len(supporting_sources) / sources_required, 1.0)
                })
            
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "subject": subject,
                        "sources_required": sources_required,
                        "verification_results": verification_results
                    }, indent=2)
                )
            ]
            
        except Exception as e:
            logger.error(f"Fact verification failed: {e}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "error": str(e)
                    })
                )
            ]

    async def _find_current_examples(self, args: dict) -> list[TextContent]:
        """Find current real-world examples for educational concepts"""
        try:
            concept = args["concept"]
            subject = args.get("subject", "")
            example_type = args.get("example_type", "application")
            recency = args.get("recency", "past_year")
            
            # Build search query for current examples
            time_filter = {
                "very_recent": "past month",
                "past_year": "past year", 
                "past_two_years": "past 2 years"
            }.get(recency, "past year")
            
            search_query = f'{concept} {example_type} {subject} {time_filter} current example'
            
            search_results = await self._perform_google_search(search_query, 15, recency)
            
            # Filter and categorize examples
            examples = []
            for result in search_results:
                example = {
                    "title": result.get("title", ""),
                    "url": result.get("link", ""),
                    "description": result.get("snippet", ""),
                    "source": self._extract_domain(result.get("link", "")),
                    "relevance_score": self._calculate_example_relevance(result, concept, subject),
                    "example_type": self._classify_example_type(result),
                    "date_published": result.get("lastModified", "")
                }
                
                # Only include high-relevance examples
                if example["relevance_score"] > 0.3:
                    examples.append(example)
            
            # Sort by relevance
            examples.sort(key=lambda x: x["relevance_score"], reverse=True)
            examples = examples[:10]  # Top 10 examples
            
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "concept": concept,
                        "subject": subject,
                        "example_type": example_type,
                        "recency": recency,
                        "examples_found": len(examples),
                        "examples": examples
                    }, indent=2)
                )
            ]
            
        except Exception as e:
            logger.error(f"Current examples search failed: {e}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "error": str(e)
                    })
                )
            ]

    async def _get_curriculum_updates(self, args: dict) -> list[TextContent]:
        """Search for curriculum updates and policy changes"""
        try:
            curriculum = args["curriculum"]
            subject = args.get("subject", "")
            region = args.get("region", "")
            
            # Build search query for curriculum updates
            search_query = f'{curriculum} {subject} {region} update change policy 2024 2025'
            
            search_results = await self._perform_google_search(search_query, 10, "past_year")
            
            # Filter for official and relevant sources
            official_domains = [
                "gov.in", "cbse.gov.in", "ncert.nic.in",  # India
                "gov.uk", "gov.us", "education.gov",      # Official gov sites
                "ibo.org", "collegeboard.org",            # International bodies
                "cambridgeinternational.org"              # Cambridge
            ]
            
            updates = []
            for result in search_results:
                domain = self._extract_domain(result.get("link", ""))
                is_official = any(official in domain for official in official_domains)
                
                update = {
                    "title": result.get("title", ""),
                    "url": result.get("link", ""),
                    "summary": result.get("snippet", ""),
                    "source": domain,
                    "is_official_source": is_official,
                    "update_type": self._classify_update_type(result),
                    "date": result.get("lastModified", ""),
                    "relevance": self._calculate_curriculum_relevance(result, curriculum, subject)
                }
                
                updates.append(update)
            
            # Sort by official status and relevance
            updates.sort(key=lambda x: (x["is_official_source"], x["relevance"]), reverse=True)
            
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "curriculum": curriculum,
                        "subject": subject,
                        "region": region,
                        "updates_found": len(updates),
                        "updates": updates
                    }, indent=2)
                )
            ]
            
        except Exception as e:
            logger.error(f"Curriculum updates search failed: {e}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "error": str(e)
                    })
                )
            ]

    async def _perform_google_search(self, query: str, num_results: int = 10, date_range: str = "any") -> list:
        """Perform Google Custom Search API call"""
        if not self.search_api_key or not self.search_engine_id:
            # Fallback to mock results for development
            return self._get_mock_search_results(query, num_results)
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.search_api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": min(num_results, 10)  # Google API limit
        }
        
        # Add date range filter
        if date_range != "any":
            date_restrict = {
                "past_week": "w1",
                "past_month": "m1", 
                "past_year": "y1"
            }.get(date_range)
            if date_restrict:
                params["dateRestrict"] = date_restrict
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("items", [])
                else:
                    logger.error(f"Google Search API error: {response.status}")
                    return []

    def _get_mock_search_results(self, query: str, num_results: int) -> list:
        """Generate mock search results for development"""
        mock_results = []
        for i in range(min(num_results, 5)):
            mock_results.append({
                "title": f"Educational Resource {i+1} for: {query}",
                "link": f"https://example-edu-site{i+1}.com/resource",
                "snippet": f"This is a comprehensive educational resource about {query}. It provides detailed explanations and examples suitable for students and teachers.",
                "lastModified": datetime.now().isoformat()
            })
        return mock_results

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        if not url:
            return ""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return url

    def _assess_educational_relevance(self, result: dict, subject: str) -> float:
        """Assess educational relevance of search result"""
        score = 0.0
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        
        # Subject relevance
        if subject and subject.lower() in title or subject.lower() in snippet:
            score += 0.3
        
        # Educational keywords
        edu_keywords = ["education", "learn", "teach", "student", "curriculum", "lesson", "study"]
        for keyword in edu_keywords:
            if keyword in title or keyword in snippet:
                score += 0.1
        
        return min(score, 1.0)

    def _calculate_example_relevance(self, result: dict, concept: str, subject: str) -> float:
        """Calculate relevance score for examples"""
        score = 0.0
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        
        # Concept match
        if concept.lower() in title:
            score += 0.4
        if concept.lower() in snippet:
            score += 0.2
        
        # Subject match
        if subject and subject.lower() in title or subject.lower() in snippet:
            score += 0.2
        
        # Example indicators
        example_words = ["example", "case", "application", "real", "current", "recent"]
        for word in example_words:
            if word in title or word in snippet:
                score += 0.1
        
        return min(score, 1.0)

    def _classify_example_type(self, result: dict) -> str:
        """Classify the type of example from search result"""
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        
        if "case study" in title or "case study" in snippet:
            return "case_study"
        elif "news" in title or "breaking" in snippet:
            return "news_event"
        elif "research" in title or "study" in snippet:
            return "research_finding"
        else:
            return "application"

    def _calculate_curriculum_relevance(self, result: dict, curriculum: str, subject: str) -> float:
        """Calculate relevance for curriculum updates"""
        score = 0.0
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        
        # Curriculum name match
        if curriculum.lower() in title:
            score += 0.5
        if curriculum.lower() in snippet:
            score += 0.3
        
        # Subject match
        if subject and (subject.lower() in title or subject.lower() in snippet):
            score += 0.2
        
        return min(score, 1.0)

    def _classify_update_type(self, result: dict) -> str:
        """Classify the type of curriculum update"""
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        
        if "new" in title or "launch" in snippet:
            return "new_policy"
        elif "change" in title or "revision" in snippet:
            return "revision"
        elif "update" in title or "amendment" in snippet:
            return "update"
        else:
            return "general"

    async def run(self):
        """Initialize and run the Web Search MCP server"""
        # Initialize search API credentials
        self.search_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        self.search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        
        if not self.search_api_key or not self.search_engine_id:
            logger.warning("Google Search API credentials not provided, using mock results")
        
        logger.info("Web Search MCP Server initialized successfully")
        
        # Run server
        async with self.server.request_context():
            await self.server.run()

async def main():
    """Main entry point"""
    server = WebSearchServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())