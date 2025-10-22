from dotenv import load_dotenv
load_dotenv() # <-- Load .env file FIRST

import os
import streamlit as st
import json # Need this to parse the report string later
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import TavilySearchResults

# Import our custom tools
from agent_tool import get_patient_report, get_rag_context

# --- IMPORT THE LOGGER ---
from logger import app_logger

# --- INITIALIZE "BRAIN" (LLM) AND TOOLS ---
try:
    llm = ChatGoogleGenerativeAI(model="gemini-pro-latest", google_api_key=os.getenv("GEMINI_API_KEY"))
    web_search_tool = TavilySearchResults(k=3, tavily_api_key=os.getenv("TAVILY_API_KEY"))
    app_logger.info("AI components initialized successfully.")
except Exception as e:
    st.error(f"Error initializing AI components: {e}")
    app_logger.error(f"Failed to initialize AI components: {e}")
    st.stop()
# -------------------------------------------

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Post-Discharge AI Assistant", page_icon="🧑‍⚕️")
st.title("🧑‍⚕️ Post-Discharge Medical AI Assistant")

# Add the required medical disclaimers at the top
st.warning(
    "**Disclaimer:** This is an AI assistant for educational purposes only. "
    "Always consult healthcare professionals for medical advice."
)

# --- Use Streamlit's Session State to store conversation data ---
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""
if "patient_report" not in st.session_state:
    st.session_state.patient_report = None # This will store the FINAL confirmed report string
if "pending_reports" not in st.session_state:
    st.session_state.pending_reports = None # This will store the LIST of dicts if multiple found
if "clarification_needed" not in st.session_state:
    st.session_state.clarification_needed = False # Flag to show the clarification UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =================================================================
# --- "RECEPTIONIST AGENT" LOGIC ---
# =================================================================
st.header("Step 1: Find Your Report")

# --- Initial Name Search ---
# Only show this if clarification is NOT needed
if not st.session_state.clarification_needed:
    name_input = st.text_input(
        "Hello! To get started, please enter your full name:",
        key="name_input_key" # Added key for potential reset later
    )

    if st.button("Get My Discharge Report"):
        if name_input:
            st.session_state.patient_name = name_input # Store the name being searched
            app_logger.info(f"New session started. Searching for patient: '{name_input}'")

            st.info(f"Looking for report for: {name_input}...")
            with st.spinner("Receptionist is finding your file..."):
                # Call the tool - it now returns TWO values
                message, report_data_list = get_patient_report.func(name_input)

            # --- Process the tool's response ---
            if report_data_list is None and "Error:" not in message:
                # Case 1: Exactly ONE report found
                st.success("Report found!")
                st.session_state.patient_report = message # Store the report JSON string
                st.session_state.clarification_needed = False
                st.session_state.pending_reports = None
                app_logger.info(f"Successfully found report for patient: '{name_input}'")

                # Generate initial greeting
                summary_prompt = f"""
                You are a friendly receptionist. A patient's report was just found.
                Patient Report: {message}

                Greet the patient by name and briefly summarize their primary diagnosis and follow-up date.
                Then, ask them a friendly open-ended question like 'How are you feeling today?' or 'Do you have any questions about your discharge instructions?'
                """
                with st.spinner("Generating summary..."):
                    initial_greeting = llm.invoke(summary_prompt).content
                st.session_state.chat_history.append({"role": "assistant", "content": initial_greeting})
                st.rerun() # Rerun to show Step 2

            elif report_data_list is not None:
                # Case 2: MULTIPLE reports found
                st.info(message) # Display the clarification message from the tool
                st.session_state.pending_reports = report_data_list # Store the potential reports
                st.session_state.clarification_needed = True # Set flag to show clarification UI
                st.session_state.patient_report = None
                app_logger.warning(f"Multiple reports found for '{name_input}'. Awaiting clarification.")
                st.rerun() # Rerun to show the clarification input

            else:
                # Case 3: Error (e.g., patient not found)
                st.error(message)
                st.session_state.patient_report = None
                st.session_state.clarification_needed = False
                st.session_state.pending_reports = None
                app_logger.warning(f"Patient search failed for '{name_input}'. Error: {message}")

        else:
            st.error("Please enter your name.")

# --- Clarification Input (if multiple reports were found) ---
if st.session_state.clarification_needed:
    st.info(f"Multiple reports found for {st.session_state.patient_name}. Please help identify the correct one.")

    clarification_detail = st.text_input(
        "Enter the Discharge Date (YYYY-MM-DD) or Primary Diagnosis mentioned in the list above:"
    )

    if st.button("Confirm Report"):
        if clarification_detail and st.session_state.pending_reports:
            found_match = None
            clarification_detail_lower = clarification_detail.lower()

            # Search through the pending reports
            for report in st.session_state.pending_reports:
                diagnosis = report.get("primary_diagnosis", "").lower()
                discharge_date = report.get("discharge_date", "").lower()

                # Check if the input matches either field (case-insensitive)
                if clarification_detail_lower in diagnosis or clarification_detail_lower == discharge_date:
                    if found_match is not None:
                        # Ambiguous clarification if it matches more than one
                        st.error("Your input matches multiple reports. Please be more specific (e.g., provide the full diagnosis or exact date).")
                        app_logger.warning(f"Ambiguous clarification provided: '{clarification_detail}'")
                        found_match = None # Reset match
                        break
                    found_match = report

            if found_match:
                # SUCCESS: We found the specific report
                st.success("Correct report identified!")
                final_report_string = json.dumps(found_match, indent=2)
                st.session_state.patient_report = final_report_string
                st.session_state.clarification_needed = False
                st.session_state.pending_reports = None
                app_logger.info(f"Clarification successful for '{st.session_state.patient_name}' using detail: '{clarification_detail}'")

                # Generate initial greeting
                summary_prompt = f"""
                You are a friendly receptionist. A patient's report was just found after clarification.
                Patient Report: {final_report_string}

                Greet the patient by name and briefly summarize their primary diagnosis and follow-up date.
                Then, ask them a friendly open-ended question like 'How are you feeling today?' or 'Do you have any questions about your discharge instructions?'
                """
                with st.spinner("Generating summary..."):
                    initial_greeting = llm.invoke(summary_prompt).content
                st.session_state.chat_history.append({"role": "assistant", "content": initial_greeting})
                st.rerun() # Rerun to show Step 2

            elif found_match is None and not st.session_state.get("_error_displayed", False):
                 # No match found, and we haven't already shown an error for this attempt
                st.error("Could not find a matching report based on the details provided. Please check the information and try again.")
                app_logger.warning(f"Clarification failed for '{st.session_state.patient_name}' using detail: '{clarification_detail}'")
                st.session_state._error_displayed = True # Flag to prevent repeated error on rerun

        else:
            st.error("Please enter the discharge date or diagnosis.")
            st.session_state._error_displayed = False # Reset error flag if input is empty


# =================================================================
# --- "CLINICAL AI AGENT" LOGIC ---
# =================================================================

# This whole section only appears AFTER the report is successfully confirmed
if st.session_state.patient_report:
    st.header("Step 2: Ask Medical Questions")

    # Display the chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input for medical questions
    if prompt := st.chat_input("Ask a question about your report or condition..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        app_logger.info(f"Patient '{st.session_state.patient_name}' asked new clinical question: '{prompt}'")

        # --- Run the "Clinical Agent" logic ---
        with st.spinner("Clinical Agent is thinking..."):
            # 1. Get context from our RAG tool
            rag_context = get_rag_context.func(prompt) # We call the .func()
            app_logger.info("Retrieved context from RAG tool.")

            # 2. Get context from the web search tool
            web_context = web_search_tool.invoke(prompt)
            app_logger.info("Retrieved context from Web Search tool.")

            # 3. Build a comprehensive prompt for the LLM
            final_prompt = f"""
            You are an expert AI assistant specializing in nephrology.
            Your primary duty is to answer a patient's question.

            Here is the patient's information:
            ---
            PATIENT DISCHARGE REPORT:
            {st.session_state.patient_report}
            ---
            PATIENT'S QUESTION:
            "{prompt}"
            ---

            Here is the context you have retrieved to answer the question:
            ---
            CONTEXT FROM NEPHROLOGY REFERENCE BOOK (RAG):
            {rag_context}
            ---
            CONTEXT FROM WEB SEARCH:
            {web_context}
            ---

            INSTRUCTIONS:
            1.  Answer the patient's question based *first* on the REFERENCE BOOK context and their DISCHARGE REPORT.
            2.  If the question is about new research or information not in the book, use the WEB SEARCH context.
            3.  You *must* cite your sources. Use "(Source: Reference Book)" or "(Source: Web Search)".
            4.  Be helpful, accurate, and safe.
            5.  You *must* end your entire response with the following medical disclaimer:
                "Disclaimer: I am an AI assistant for educational purposes only. Always consult healthcare professionals for medical advice."
            """

            # 4. Call the LLM directly
            response = llm.invoke(final_prompt).content
            app_logger.info("Clinical agent generated final response.")

            # 5. Display the response
            with st.chat_message("assistant"):
                st.markdown(response)
                # Add AI response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})