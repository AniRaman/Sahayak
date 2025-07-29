# 🎓 Sahayak - Educational AI Assistant

Sahayak is an intelligent educational content orchestration platform that leverages specialized AI agents to create, retrieve, and manage educational content. Built for educators, students, and educational institutions, it provides a comprehensive suite of AI-powered tools for modern learning environments.

## 🌟 Features

### 🤖 Multi-Agent Architecture
- **Content Agent**: Creates lesson plans, explanations, and educational content
- **Worksheet Agent**: Generates comprehensive worksheets and presentations from uploaded files
- **QA Agent**: Extracts and creates questions from documents and images
- **Image Agent**: Generates educational diagrams, illustrations, and visual content
- **Retrieval Agent**: Searches and retrieves existing educational resources

### 📝 Content Generation
- Automated lesson plan creation
- Curriculum-aligned content generation
- Multi-language support
- Difficulty level customization
- Assessment question generation

### 📋 Document Processing
- PDF and image content extraction
- Worksheet generation with multiple difficulty levels
- Educational slide creation
- Question-answer pair generation

### 🎨 Visual Content
- Educational diagram generation
- Custom illustration creation
- Interactive visual materials

## 🏗️ Architecture

Sahayak follows a coordinator-based architecture where a central coordinator intelligently routes requests to specialized sub-agents based on input analysis:

```
User Request → Coordinator → Agent Selection → Specialized Processing → Response
```

### Agent Routing Logic
- **File uploads** → QA Agent or Worksheet Agent (based on request type)
- **Text with retrieval keywords** → Retrieval Agent
- **Text with content generation keywords** → Content Agent
- **Image generation requests** → Image Agent
- **Default educational queries** → Content Agent

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Google Cloud Platform account
- Gemini API access
- Firebase/Firestore setup

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sahayak.git
   cd sahayak
   ```

2. **Install dependencies**
   ```bash
   cd sahayak-coordinator
   pip install -r requirements.txt
   ```

3. **Environment Setup**
   Create a `.env` file with your API keys:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
   ```

4. **Run the application**
   ```bash
   python -m uvicorn main:app --host 0.0.0.0 --port 8080
   ```

### Web Interface
Access the interactive web interface at `http://localhost:8080` to interact with Sahayak through a user-friendly chat interface.

## 📁 Project Structure

```
sahayak/
├── coordinator_simple.py              # Simplified coordinator for Cloud Run
├── sahayak-coordinator/              # Main coordinator module
│   ├── agent.py                      # Main coordinator agent
│   ├── requirements.txt              # Python dependencies
│   └── sub_agents/                   # Specialized agent modules
│       ├── sahayak_content_agent/    # Educational content generation
│       ├── sahayak_image_agent/      # Image and diagram generation
│       ├── sahayak_qa_agent/         # Question-answer generation
│       ├── sahayak_retrieval_agent/  # Resource search and retrieval
│       └── sahayak_worksheet_agent/  # Worksheet and material creation
├── templates/                        # Web interface templates
│   ├── index.html                    # Main chat interface
│   └── chat.html                     # Chat page
├── static/                          # Static files and generated content
└── AgenticAIHackathon/              # Documentation and project materials
```

## 🛠️ Core Components

### Coordinator Agent
The central hub that:
- Analyzes user input and file uploads
- Routes requests to appropriate specialized agents
- Coordinates multi-agent workflows
- Handles response aggregation

### Content Agent
Specializes in:
- Educational content generation using Gemini AI
- Lesson plan creation
- Curriculum alignment
- Multi-language content support

### Worksheet Agent
Handles:
- PDF content extraction and analysis
- Comprehensive worksheet generation
- Educational slide creation
- Multi-format output (slides + worksheets)

### QA Agent
Focuses on:
- Question extraction from documents
- Answer generation
- Simple Q&A pair creation
- Content-based assessment tools

### Image Agent
Provides:
- Educational diagram generation
- Custom illustration creation
- Visual content for learning materials

### Retrieval Agent
Manages:
- Educational resource search
- Database queries
- Content discovery
- Resource recommendation


## 📊 Dependencies

### Core Libraries
- **FastAPI**: Web framework for API endpoints
- **Google Generative AI**: Gemini AI integration
- **Google Cloud Firestore**: Database operations
- **Pytesseract**: OCR capabilities
- **Pillow**: Image processing
- **PDF2Image**: PDF processing

### Development Tools
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Code linting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Usage Examples

### Creating Educational Content
```python
from sahayak_coordinator.agent import coordinate_request

response = coordinate_request(
    user_input="Create a lesson plan about photosynthesis for grade 8",
    subject="Biology",
    grade_level="8",
    curriculum="NGSS"
)
```

### Processing Educational Documents
```python
response = coordinate_request(
    user_input="Create worksheets from this PDF",
    file_content="base64_encoded_pdf",
    file_type="application/pdf",
    subject="Mathematics"
)
```

### Generating Educational Images
```python
response = coordinate_request(
    user_input="Generate an image of the solar system with labels",
    force_agent="image"
)
```

## 🔒 Security & Privacy

- All file processing is handled securely with temporary storage
- API keys are managed through environment variables
- User data is processed in compliance with educational privacy standards

## 🙏 Acknowledgments

- Built using Google's Generative AI and Cloud Platform
- Leverages the Google Agent Development Kit (ADK)
- Inspired by the need for intelligent educational content creation

---

*Sahayak - Empowering Education through Intelligent AI Assistance* 🚀📚
