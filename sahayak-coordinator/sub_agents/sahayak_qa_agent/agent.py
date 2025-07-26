

############################################################
# import logging
# from google.adk.agents import Agent

# # Import worksheet tools
# from .mcp_servers.worksheet_server import extract_content_from_upload, generate_questions_by_difficulty

# logging.basicConfig(level=logging.INFO)

# root_agent = Agent(
#     name="sahayak_worksheet_agent",
#     model="gemini-2.0-flash",
#     description="Worksheet agent that creates structured assignments using only the uploaded file or image.",
#     instruction=(
#         "You are a worksheet assignment agent.\n"
#         "ONLY use the uploaded content (file or image) to extract concepts.\n\n"
#         "Generate a structured assignment containing:\n"
#         "- 3 sections: Easy, Medium, and Hard\n"
#         "- Each section must have:\n"
#         "  - 5 multiple choice questions (clearly marked A/B/C/D options)\n"
#         "  - 3 descriptive questions (open-ended)\n\n"
#         "Format strictly like an assignment paper with proper section titles.\n"
#         "Label all parts clearly.\n"
#         "DO NOT hallucinate or use any external knowledge.\n"
#         "If content is insufficient, state clearly which part could not be generated."
#     ),
#     tools=[
#         extract_content_from_upload,
#         generate_questions_by_difficulty,
#     ]
# )

# if __name__ == "__main__":
#     print(f"📝 Agent {root_agent.name} initialized successfully.")
#     for i, tool in enumerate(getattr(root_agent, 'tools', []), 1):
#         print(f"  {i}. {tool.__name__}")

####################################################################

# import logging
# from google.adk.agents import Agent
# from .mcp_servers.worksheet_server import extract_content_from_upload, generate_questions_by_difficulty

# logging.basicConfig(level=logging.INFO)

# def detect_language_from_query(query: str) -> str:
#     query = query.lower()
#     if "telugu" in query:
#         return "Telugu"
#     elif "hindi" in query:
#         return "Hindi"
#     elif "tamil" in query:
#         return "Tamil"
#     elif "kannada" in query:
#         return "Kannada"
#     elif "malayalam" in query:
#         return "Malayalam"
#     elif "french" in query:
#         return "French"
#     elif "german" in query:
#         return "German"
#     elif "spanish" in query:
#         return "Spanish"
#     else:
#         return "English"

# # Wrap the question generation tool with dynamic language injection
# async def generate_questions_with_language(arguments: dict) -> str:
#     content = arguments.get("content", "")
#     language = detect_language_from_query(content)
#     return await generate_questions_by_difficulty(content, language)

# root_agent = Agent(
#     name="sahayak_worksheet_agent",
#     model="gemini-2.0-flash",
#     description="Worksheet agent that creates structured assignments using only the uploaded file or image.",
#     instruction=(
#         "You are a worksheet assignment agent.\n"
#         "ONLY use the uploaded content (file or image) to extract concepts.\n\n"
#         "Generate a structured assignment containing:\n"
#         "- 3 sections: Easy, Medium, and Hard\n"
#         "- Each section must have:\n"
#         "  - 5 multiple choice questions (clearly marked A/B/C/D options)\n"
#         "  - 3 descriptive questions (open-ended)\n\n"
#         "Format strictly like an assignment paper with proper section titles.\n"
#         "Label all parts clearly.\n"
#         "DO NOT hallucinate or use any external knowledge.\n"
#         "If content is insufficient, state clearly which part could not be generated."
#     ),
#     tools=[
#         extract_content_from_upload,
#         generate_questions_with_language,  # this replaces the direct call to generate_questions_by_difficulty
#     ]
# )

# if __name__ == "__main__":
#     print(f"📝 Agent {root_agent.name} initialized successfully.")
#     for i, tool in enumerate(getattr(root_agent, 'tools', []), 1):
#         print(f"  {i}. {tool.__name__}")

#########################################################################



import logging
from google.adk.agents import Agent
from .mcp_servers.worksheet_server import extract_content_from_upload, generate_questions_by_difficulty

logging.basicConfig(level=logging.INFO)

def detect_language_from_query(query: str) -> str:
    query = query.lower()
    if "telugu" in query: return "Telugu"
    if "hindi" in query: return "Hindi"
    if "tamil" in query: return "Tamil"
    if "kannada" in query: return "Kannada"
    if "french" in query: return "French"
    if "german" in query: return "German"
    if "spanish" in query: return "Spanish"
    return "English"

async def generate_questions_with_language(arguments: dict) -> str:
    content = arguments.get("content", "")
    language = detect_language_from_query(content)
    return await generate_questions_by_difficulty(content, language)

root_agent = Agent(
    name="sahayak_worksheet_agent",
    model="gemini-2.0-flash",
    description="Worksheet agent that creates structured assignments from uploaded content only.",
    instruction=(
        "Use only the uploaded file or image to extract topics.\n"
        "Generate an assignment with Easy, Medium, and Hard sections (unless specified otherwise).\n"
        "Default: 5 MCQs + 3 Descriptive in each section.\n"
        "Respect any user-specified language or pattern.\n"
        "Never hallucinate."
    ),
    tools=[
        extract_content_from_upload,
        generate_questions_with_language,
    ]
)

if __name__ == "__main__":
    print(f"📝 Agent {root_agent.name} initialized successfully.")
    for i, tool in enumerate(getattr(root_agent, 'tools', []), 1):
        print(f"  {i}. {tool.__name__}")
