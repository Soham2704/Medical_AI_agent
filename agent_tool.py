import os
import json
from crewai.tools import tool # <-- THE CORRECT IMPORT
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define the directory where patient files are stored
PATIENT_DATA_DIR = "data/"

@tool("Patient Data Retrieval Tool")
def get_patient_report(patient_name: str) -> tuple[str, list | None]:
    """
    Retrieves a patient's discharge report by their full name.
    Handles 'patient not found'. If multiple patients are found,
    it returns a clarification message and the list of matching report data.
    If one is found, returns the report (as JSON string) and None.
    If none found, returns an error message and None.
    """
    print(f"---[Tool Called: get_patient_report with patient_name='{patient_name}']---")

    found_reports_data = [] # Store the actual data dictionaries

    try:
        # Loop through all files in the patient data directory
        for filename in os.listdir(PATIENT_DATA_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(PATIENT_DATA_DIR, filename)

                # Open and read the JSON file
                with open(filepath, 'r') as f:
                    report_data = json.load(f)

                    # Case-insensitive check for the patient name
                    if report_data.get("patient_name", "").lower() == patient_name.lower():
                        found_reports_data.append(report_data) # Store the dictionary

        # --- Handle the results ---

        if len(found_reports_data) == 0:
            # Case 1: No patient found
            print("---[Tool Result: Patient not found]---")
            # Return error message and None for the data list
            return f"Error: No patient report found for the name {patient_name}.", None

        elif len(found_reports_data) > 1:
            # Case 2: Multiple patients found - provide details and the data list
            print(f"---[Tool Result: Found {len(found_reports_data)} patients with same name]---")

            # Build a helpful message listing the options
            clarification_message = f"Multiple patients found with the name {patient_name}. Please specify which report you need by providing the discharge date or diagnosis.\n"
            clarification_message += "Here are the reports I found:\n"
            for i, report in enumerate(found_reports_data):
                name = report.get("patient_name", "N/A")
                diagnosis = report.get("primary_diagnosis", "N/A")
                discharge_date = report.get("discharge_date", "N/A")
                clarification_message += f"  {i+1}. Name: {name}, Diagnosis: {diagnosis}, Discharge Date: {discharge_date}\n"

            # Return the message AND the list of potential reports
            return clarification_message, found_reports_data

        else:
            # Case 3: Exactly one patient found
            print("---[Tool Result: Patient found successfully]---")
            # Convert the single report back to a JSON string
            report_string = json.dumps(found_reports_data[0], indent=2)
            # Return the report string and None for the data list
            return report_string, None

    except Exception as e:
        print(f"---[Tool Error: {e}]---")
        # Return error message and None for the data list
        return f"Error: An unexpected error occurred while searching for the patient: {str(e)}", None

# --- Define constants for the RAG tool ---
PERSIST_DIRECTORY = "db_chroma"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

# --- Initialize the components for the RAG tool ---
# We initialize these components *outside* the tool function
# so they are loaded only once when the app starts, not every time the tool is called.

# 1. Initialize the free, local embedding model
model_kwargs = {'device': 'cpu'}
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs=model_kwargs
)

# 2. Load the persistent vector database
print("---[Loading Vector Database from disk...]---")
vectordb = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)
print("---[Vector Database loaded successfully.]---")


@tool("Nephrology Reference RAG Tool")
def get_rag_context(medical_query: str) -> str:
    """
    Searches the nephrology reference book for context relevant to a medical query.
    Returns the most relevant text chunks from the book.
    """
    print(f"---[Tool Called: get_rag_context with query='{medical_query}']---")

    try:
        # Perform a similarity search in the vector database
        # k=4 means it will return the top 4 most relevant chunks
        relevant_docs = vectordb.similarity_search(medical_query, k=4)

        if not relevant_docs:
            print("---[Tool Result: No relevant context found.]---")
            return "No relevant information found in the reference materials."

        # Format the results into a single string
        context = ""
        for i, doc in enumerate(relevant_docs):
            context += f"--- Relevant Context Chunk {i+1} (Source: {doc.metadata.get('source', 'Unknown')}) ---\n"
            context += doc.page_content
            context += "\n---------------------------------------------------\n"

        print(f"---[Tool Result: Found {len(relevant_docs)} context chunks.]---")
        return context

    except Exception as e:
        print(f"---[Tool Error: {e}]---")
        return f"Error: An unexpected error occurred while searching the reference materials: {str(e)}"