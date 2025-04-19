import streamlit as st
from graph import build_translation_graph, build_supernode_graph
import uuid
import asyncio

# Streamlit app configuration
st.set_page_config(page_title="Hire a Pro Linguist", layout="wide")

# Sidebar for API configuration
st.sidebar.title("API Configuration")
model = st.sidebar.selectbox("Select Model for Translation", ["Gpt-4o-mini", "Phi-3-small-8k-instruct"], index=0, help="Choose the model for the translation task")
endpoint_url = st.sidebar.text_input("Endpoint URL (optional)", placeholder="Leave blank to use secrets.toml", help="Enter your Azure OpenAI endpoint URL or leave blank")
api_key = st.sidebar.text_input("API Key (optional)", type="password", placeholder="Leave blank to use secrets.toml", help="Enter your Azure OpenAI API key or leave blank")

# Main panel
st.title("Translation Agent System")
st.write("Provide translation details and optional context files.")

# Input fields for translation
with st.form(key="translation_form"):
    col1, col2 = st.columns(2)
    with col1:
        source_language = st.text_input("Source Language", value="English", help="e.g., English")
    with col2:
        target_language = st.text_input("Target Language", value="Farsi", help="e.g., Spanish")
    text = st.text_area("Text to Translate", height=150, help="Enter the text you want to translate")

    # Context options
    st.subheader("Context Options")

    # File uploaders
    grammar_file = st.file_uploader("Upload Grammar Rules (text file)", type=["txt"], key="grammar")
    dictionary_file = st.file_uploader("Upload Dictionary (text file)", type=["txt"], key="dictionary")
    examples_file = st.file_uploader("Upload Examples (text file)", type=["txt"], key="examples")

    # Option to process as single sentence or corpus
    processing_mode = st.radio("Processing Mode", ["Single Sentence", "Large Corpus"], index=0)
    use_llm_judge = st.checkbox("use llm as a judge", value=True)

    submit_button = st.form_submit_button(label="Generate Translation")

# Process the form submission
if submit_button:
    if not source_language or not target_language or not text:
        st.error("Please fill in Source Language, Target Language, and Text to Translate.")
    else:
        try:
            # Get credentials for the selected translation model
            if model == "Gpt-4o-mini":
                translation_endpoint = st.secrets["Gpt-4o-mini"]["ENDPOINT_URL"]
                translation_api_key = st.secrets["Gpt-4o-mini"]["API_KEY"]
                translation_api_version = "2025-01-01-preview"
            else:  # Phi-3-small-8k-instruct
                translation_endpoint = st.secrets["Phi-3-small-8k-instruct"]["ENDPOINT_URL"]
                translation_api_key = st.secrets["Phi-3-small-8k-instruct"]["API_KEY"]
                translation_api_version = "2024-05-01-preview"

            # Override with user-provided credentials if available
            translation_endpoint = endpoint_url if endpoint_url else translation_endpoint
            translation_api_key = api_key if api_key else translation_api_key

            # Get credentials for the fixed model (Gpt-4o-mini)
            fixed_endpoint = st.secrets["Gpt-4o-mini"]["ENDPOINT_URL"]
            fixed_api_key = st.secrets["Gpt-4o-mini"]["API_KEY"]
            fixed_api_version = "2025-01-01-preview"

            # Read content from uploaded files
            grammar_content = grammar_file.read().decode("utf-8") if grammar_file else None
            dictionary_content = dictionary_file.read().decode("utf-8") if dictionary_file else None
            examples_content = examples_file.read().decode("utf-8") if examples_file else None

            # Prepare the initial state for LangGraph
            initial_state = {
                "sentence": text,
                "source_language": source_language,
                "target_language": target_language,
                "dictionary_content_raw": dictionary_content,
                "grammar_content_raw": grammar_content,
                "examples_content_raw": examples_content,
                "translation_model": f"azure/{model}",
                "translation_api_base": translation_endpoint,
                "translation_api_key": translation_api_key,
                "translation_api_version": translation_api_version,
                "fixed_model": "azure/Gpt-4o-mini",
                "fixed_api_base": fixed_endpoint,
                "fixed_api_key": fixed_api_key,
                "fixed_api_version": fixed_api_version,
                "use_judge": use_llm_judge
            }

            # Choose processing mode
            if processing_mode == "Single Sentence":
                graph = build_translation_graph()
                with st.spinner("Generating translation..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(graph.ainvoke(initial_state, config={"configurable": {"thread_id": str(uuid.uuid4())}}))
                    loop.close()
                
                # Display results
                st.subheader("Generated Translation Prompt")
                st.write(result.get("translation_prompt", "No prompt generated."))
                
                st.subheader("Translation Result")
                final_translation = {
                    "translation_model": result.get("translation_model", "No translation model."),
                    "fixed_model": result.get("fixed_model", "No fixed model."),
                    "initial_translation": result.get("initial_translation", "No initial translation."),
                    "judge_feedback": result.get("judge_feedback", "No judge feedback."),
                    "final_translation": result.get("final_translation", "No final translation.")
                }
                st.write(final_translation)
            
            else:  # Large Corpus
                graph = build_supernode_graph()
                with st.spinner("Processing large corpus..."):
                    # Run async graph invocation
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(graph.ainvoke(initial_state, config={"configurable": {"thread_id": str(uuid.uuid4())}}))
                    loop.close()
                
                # Display results
                st.subheader("Chunked Translations")
                st.write({"chunks": result.get("chunks", []), "translations": result.get("chunk_translations", [])})
                
                st.subheader("Final Translation")
                final_translation = {
                    "consistency_status": result.get("consistency_status", "No status."),
                    "final_translation": result.get("final_translation", "No final translation."),
                    "consistency_feedback": result.get("consistency_feedback", "No feedback.")
                }
                st.write(final_translation)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
