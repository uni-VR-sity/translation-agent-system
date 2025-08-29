import streamlit as st
from graph import build_translation_graph, build_supernode_graph, get_memory_insights
import uuid
import asyncio
from prompts import initialize_dspy_model
from memory_analytics import MemoryAnalytics, MemoryMaintenance
from memory_system import MemoryManager
from memory_config import get_memory_config, apply_preset, get_preset_names
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

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
if 'prev_use_local_ollama' not in st.session_state:
    st.session_state.prev_use_local_ollama = True

use_local_ollama = st.sidebar.checkbox("Use Local Ollama", 
                                     value=st.session_state.prev_use_local_ollama, 
                                     help="Check to use local Ollama instance instead of remote API")

# Check if the checkbox state has changed and force a rerun if it has
if use_local_ollama != st.session_state.prev_use_local_ollama:
    st.session_state.prev_use_local_ollama = use_local_ollama
    # Reset model selections when provider changes
    if 'model' in st.session_state:
        del st.session_state.model
    if 'fixed_model' in st.session_state:
        del st.session_state.fixed_model
    # Force page rerun to update UI
    st.rerun()

# Determine current model options
if use_local_ollama:
    model_options = ollama_models
else:
    model_options = openrouter_models

# Initialize or validate model selections
if not model_options:
    st.error("No models available. Please check the model files.")
    st.stop()

# Set default model if not already set or if current selection is invalid
if 'model' not in st.session_state or st.session_state.model not in model_options:
    st.session_state.model = model_options[0]

if 'fixed_model' not in st.session_state or st.session_state.fixed_model not in model_options:
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

# Memory System Configuration
st.sidebar.subheader("Memory System")
memory_enabled = st.sidebar.checkbox("Enable Memory Learning", value=True,
                                    help="Enable long-term memory and learning features")

if memory_enabled:
    config_preset = st.sidebar.selectbox("Memory Preset",
                                       options=["default"] + get_preset_names(),
                                       help="Choose a memory configuration preset")
    
    if config_preset != "default":
        apply_preset(config_preset)
        st.sidebar.success(f"Applied {config_preset} preset")

# Main panel with tabs
st.title("Translation Agent System with Long-Term Memory")

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["Translation", "Memory Analytics", "Learning Insights", "System Maintenance"])

with tab1:
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
                    api_key = st.secrets["ollama"]["API_KEY"]
                    
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
                    "translation_model": f'openai/{model}' if use_local_ollama else f'openrouter/{model}',
                    "fixed_model": f'openai/{fixed_model}' if use_local_ollama else f'openrouter/{fixed_model}',
                    "api_base": endpoint,
                    "api_key": api_key,
                    "use_judge": use_llm_judge
                }

                initialize_dspy_model(model=initial_state["fixed_model"], api_base= endpoint, api_key=api_key)

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

with tab2:
    st.header("Memory Analytics Dashboard")
    
    if memory_enabled:
        try:
            memory_manager = MemoryManager()
            analytics = MemoryAnalytics(memory_manager)
            
            # Language pair selection for analytics
            col1, col2 = st.columns(2)
            with col1:
                analytics_language_pair = st.selectbox(
                    "Select Language Pair for Analytics",
                    options=["All", "English-Persian", "English-Spanish", "English-French"],
                    help="Choose a specific language pair or view all data"
                )
            
            with col2:
                analytics_days = st.slider("Days to Analyze", min_value=7, max_value=90, value=30)
            
            # Generate report
            if st.button("Generate Analytics Report"):
                with st.spinner("Generating analytics report..."):
                    language_pair = None if analytics_language_pair == "All" else analytics_language_pair
                    report = analytics.generate_performance_report(language_pair, analytics_days)
                    
                    if report:
                        # Summary metrics
                        st.subheader("Summary Statistics")
                        summary = report.get('summary', {})
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Sessions", summary.get('total_sessions', 0))
                        with col2:
                            st.metric("Approval Rate", f"{summary.get('approval_rate', 0):.1%}")
                        with col3:
                            st.metric("Retry Rate", f"{summary.get('retry_rate', 0):.1%}")
                        with col4:
                            st.metric("Avg Quality", f"{summary.get('average_quality_score', 0):.2f}")
                        
                        # Quality trends
                        st.subheader("Quality Trends")
                        quality_trends = report.get('quality_trends', {})
                        daily_averages = quality_trends.get('daily_averages', {})
                        
                        if daily_averages:
                            df = pd.DataFrame(list(daily_averages.items()), columns=['Date', 'Quality'])
                            df['Date'] = pd.to_datetime(df['Date'])
                            
                            fig = px.line(df, x='Date', y='Quality', title='Quality Score Trends Over Time')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Model performance
                        st.subheader("Model Performance")
                        model_perf = report.get('model_performance', {})
                        model_stats = model_perf.get('model_statistics', {})
                        
                        if model_stats:
                            model_df = pd.DataFrame.from_dict(model_stats, orient='index')
                            model_df = model_df.reset_index().rename(columns={'index': 'Model'})
                            
                            fig = px.scatter(model_df, x='avg_processing_time', y='avg_quality',
                                           size='total_uses', hover_name='Model',
                                           title='Model Performance: Quality vs Speed')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Recommendations
                        st.subheader("Recommendations")
                        recommendations = report.get('recommendations', [])
                        for i, rec in enumerate(recommendations, 1):
                            st.write(f"{i}. {rec}")
                    
                    else:
                        st.warning("No data available for the selected criteria.")
            
            # Database statistics
            st.subheader("Database Statistics")
            db_stats = analytics.get_database_stats()
            
            if db_stats:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Translation Sessions", db_stats.get('translation_sessions_count', 0))
                with col2:
                    st.metric("Feedback Patterns", db_stats.get('feedback_patterns_count', 0))
                with col3:
                    st.metric("Successful Patterns", db_stats.get('successful_patterns_count', 0))
                
                st.metric("Database Size", f"{db_stats.get('database_size_mb', 0):.2f} MB")
        
        except Exception as e:
            st.error(f"Error loading analytics: {e}")
    else:
        st.info("Enable memory learning in the sidebar to view analytics.")

with tab3:
    st.header("Learning Insights")
    
    if memory_enabled:
        try:
            # Language pair insights
            col1, col2 = st.columns(2)
            with col1:
                insights_language_pair = st.selectbox(
                    "Language Pair",
                    options=["English-Persian", "English-Spanish", "English-French"],
                    key="insights_lang_pair"
                )
            
            if st.button("Get Learning Insights"):
                with st.spinner("Analyzing learning patterns..."):
                    insights = get_memory_insights(insights_language_pair)
                    
                    if insights:
                        # Total sessions
                        st.metric("Total Learning Sessions", insights.get('total_sessions', 0))
                        
                        # Feedback patterns
                        st.subheader("Common Feedback Patterns")
                        feedback_patterns = insights.get('feedback_patterns', [])
                        
                        if feedback_patterns:
                            for pattern in feedback_patterns:
                                with st.expander(f"{pattern['type'].title()} Issues (Frequency: {pattern['frequency']})"):
                                    st.write("Common Issues:")
                                    for issue in pattern['issues']:
                                        st.write(f"• {issue}")
                        else:
                            st.info("No feedback patterns found yet. Continue using the system to build learning data.")
                        
                        # Model performance insights
                        st.subheader("Model Performance Insights")
                        model_perf = insights.get('model_performance', {})
                        
                        if model_perf.get('best_translation_model'):
                            best_model = model_perf['best_translation_model']
                            st.success(f"Best Translation Model: {best_model['name']} "
                                     f"(Quality: {best_model['quality_score']:.2f}, "
                                     f"Success Rate: {best_model['success_rate']:.1%})")
                        
                        # Recommendations
                        recommendations = insights.get('recommendations', [])
                        if recommendations:
                            st.subheader("Learning-Based Recommendations")
                            for rec in recommendations:
                                st.info(rec)
                    else:
                        st.warning("No learning insights available yet. Use the translation system to generate learning data.")
        
        except Exception as e:
            st.error(f"Error loading learning insights: {e}")
    else:
        st.info("Enable memory learning in the sidebar to view learning insights.")

with tab4:
    st.header("System Maintenance")
    
    if memory_enabled:
        try:
            memory_manager = MemoryManager()
            maintenance = MemoryMaintenance(memory_manager)
            config = get_memory_config()
            
            # Configuration display
            st.subheader("Current Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Learning Settings:**")
                st.write(f"Learning Rate: {config.get('learning_rate')}")
                st.write(f"Quality Threshold: {config.get('high_quality_threshold')}")
                st.write(f"Min Pattern Frequency: {config.get('min_pattern_frequency')}")
            
            with col2:
                st.write("**Memory Management:**")
                st.write(f"History Retention: {config.get('max_session_history_days')} days")
                st.write(f"Cleanup Interval: {config.get('memory_cleanup_interval_days')} days")
                st.write(f"Max Patterns: {config.get('max_patterns_per_language_pair')}")
            
            # Maintenance actions
            st.subheader("Maintenance Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Run Full Maintenance"):
                    with st.spinner("Running maintenance..."):
                        maintenance.run_maintenance()
                        st.success("Maintenance completed successfully!")
            
            with col2:
                cleanup_days = st.number_input("Days to Keep", min_value=7, max_value=365, value=90)
                if st.button("Cleanup Old Data"):
                    with st.spinner("Cleaning up old data..."):
                        analytics = MemoryAnalytics(memory_manager)
                        analytics.cleanup_old_data(cleanup_days)
                        st.success(f"Cleaned up data older than {cleanup_days} days!")
            
            with col3:
                if st.button("Export Analytics Report"):
                    with st.spinner("Generating report..."):
                        analytics = MemoryAnalytics(memory_manager)
                        report = analytics.generate_performance_report(days=30)
                        
                        if report:
                            # Convert to JSON string for download
                            import json
                            report_json = json.dumps(report, indent=2, default=str)
                            
                            st.download_button(
                                label="Download Report",
                                data=report_json,
                                file_name=f"memory_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        else:
                            st.warning("No data available for report generation.")
            
            # System status
            st.subheader("System Status")
            analytics = MemoryAnalytics(memory_manager)
            db_stats = analytics.get_database_stats()
            
            if db_stats:
                status_col1, status_col2 = st.columns(2)
                
                with status_col1:
                    st.write("**Database Health:**")
                    st.write(f"Size: {db_stats.get('database_size_mb', 0):.2f} MB")
                    st.write(f"Sessions: {db_stats.get('translation_sessions_count', 0)}")
                    st.write(f"Patterns: {db_stats.get('feedback_patterns_count', 0)}")
                
                with status_col2:
                    st.write("**Data Range:**")
                    if db_stats.get('oldest_session'):
                        st.write(f"Oldest: {db_stats['oldest_session'][:10]}")
                        st.write(f"Newest: {db_stats['newest_session'][:10]}")
                    else:
                        st.write("No session data available")
        
        except Exception as e:
            st.error(f"Error in maintenance section: {e}")
    else:
        st.info("Enable memory learning in the sidebar to access maintenance features.")