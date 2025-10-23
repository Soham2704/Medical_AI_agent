🧑‍⚕️ Post-Discharge Medical AI Assistant
Overview

This project is a Proof of Concept (POC) multi-agent AI system for post-discharge patient care.
It uses LangChain, Streamlit, and ChromaDB to simulate how AI can assist patients after hospital discharge.
The system manages 25+ dummy patient reports, performs Retrieval-Augmented Generation (RAG) using nephrology reference materials, and provides a web-based interface for interaction.

✳️ Features

25+ dummy post-discharge patient reports (JSON)

RAG implementation using nephrology reference PDF

Two AI agents:

Receptionist Agent: Fetches patient report and routes queries

Clinical Agent: Handles medical questions using RAG + web search

Web search fallback for external medical information

Comprehensive logging of all interactions

Simple and interactive Streamlit web UI

📁 Folder Structure
app.py                # Main Streamlit interface
agent_tool.py         # Tools for data retrieval, RAG, and web search
ingest.py             # Builds vector embeddings and ChromaDB
logger.py             # Handles system logging
data/                 # Dummy patient discharge reports
reference_docs/       # Nephrology reference materials
db_chroma/            # Vector database (auto-generated)
requirements.txt      # Project dependencies

⚙️ Installation & Setup
1. Clone the repository
git clone https://github.com/Soham2704/Medical_AI_agent.git
cd Medical_AI_agent

2. Create and activate a virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Add API keys

Create a .env file in the root folder and add:

GEMINI_API_KEY=your_api_key
TAVILY_API_KEY=your_api_key

5. Prepare the reference data

Place the nephrology reference PDF inside reference_docs/.

6. Build the vector database
python ingest.py

7. Run the application
streamlit run app.py

💡 Workflow

Receptionist Agent asks for patient name → retrieves discharge summary.

Clinical Agent answers medical questions using:

RAG (from reference material)

Web search fallback (for latest research)

Citations and disclaimers included in every response

All activities are logged in chat_log.log for transparency.

⚠️ Disclaimer

This AI system is for educational purposes only.
Always consult qualified healthcare professionals for medical advice
