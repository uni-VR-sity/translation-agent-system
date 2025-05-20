import streamlit as st
from graph import build_translation_graph, build_supernode_graph
import uuid
import asyncio

# Streamlit app configuration
st.set_page_config(page_title="Hire a Pro Linguist",
                   layout="wide",
                   initial_sidebar_state="collapsed"
                   )

# Load model lists
try:
    with open('data/ollama_models.txt', 'r') as f:
        ollama_models = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    ollama_models = []

try:
    with open('data/openrouter_models.txt', 'r') as f:
        openrouter_models = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    openrouter_models = []

# Sidebar for API configuration
st.sidebar.title("API Configuration")

# Use local Ollama checkbox
use_local_ollama = st.sidebar.checkbox("Use Local Ollama", 
                                     value=st.session_state.get('prev_use_local_ollama', False), 
                                     help="Check to use local Ollama instance instead of remote API")

# Update session state for model provider
if 'prev_use_local_ollama' not in st.session_state:
    st.session_state.prev_use_local_ollama = use_local_ollama

if use_local_ollama != st.session_state.prev_use_local_ollama:
    # Reset models when checkbox changes
    st.session_state.prev_use_local_ollama = use_local_ollama
    # Clear current model selections to force reset
    if 'model' in st.session_state:
        del st.session_state.model
    if 'fixed_model' in st.session_state:
        del st.session_state.fixed_model

# Determine current model options
if use_local_ollama:
    model_options = ollama_models
else:
    model_options = openrouter_models

# Initialize or validate model selections
if not model_options:
    st.error("No models available. Please check the model files.")
    st.stop()

if 'model' not in st.session_state:
    st.session_state.model = model_options[0]
else:
    if st.session_state.model not in model_options:
        st.session_state.model = model_options[0]

if 'fixed_model' not in st.session_state:
    st.session_state.fixed_model = model_options[0]
else:
    if st.session_state.fixed_model not in model_options:
        st.session_state.fixed_model = model_options[0]

# Translation model selection panel
st.sidebar.subheader("Translation Model")
model = st.sidebar.selectbox("Select Model", 
                            options=model_options, 
                            index=model_options.index(st.session_state.model),
                            key='model')

# Fixed model selection panel
st.sidebar.subheader("Fixed Model (for processing/judging)")
fixed_model = st.sidebar.selectbox("Select Fixed Model", 
                                 options=model_options, 
                                 index=model_options.index(st.session_state.fixed_model),
                                 key='fixed_model')


ui_endpoint_url = st.sidebar.text_input("Endpoint URL (optional)", 
                                    placeholder="Leave blank to use secrets.toml", 
                                    help="Enter your Azure OpenAI endpoint URL or leave blank")
ui_api_key = st.sidebar.text_input("API Key (optional)", 
                               type="password", 
                               placeholder="Leave blank to use secrets.toml", 
                               help="Enter your Azure OpenAI API key or leave blank")

# Main panel
st.title("Translation Agent System")
st.write("Provide translation details and optional context files.")

# Input fields for translation
with st.form(key="translation_form"):
    col1, col2 = st.columns(2)
    with col1:
        source_language = st.text_input("Source Language", value="English", help="e.g., English")
    with col2:
        target_language = st.text_input("Target Language", value="Persian", help="e.g., Spanish")
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
            # Determine endpoints based on whether local Ollama is selected
            if use_local_ollama:
                # Using local Ollama for all models
                endpoint = st.secrets["ollama"]["ENDPOINT_URL"]
                api_key = ""
                
            else:
                # Use OpenRouter for non-local models
                endpoint = st.secrets["openrouter"]["ENDPOINT_URL"]
                api_key = st.secrets["openrouter"]["API_KEY"]
                
                # Override with user-provided credentials if available
                if ui_endpoint_url and ui_api_key:
                    endpoint =  ui_endpoint_url
                    api_key = ui_api_key

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
                "translation_model": f'ollama/{model}' if use_local_ollama else f'openrouter/{model}',
                "fixed_model": f'ollama/{fixed_model}' if use_local_ollama else f'openrouter/{fixed_model}',
                "api_base": endpoint,
                "api_key": api_key,
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
