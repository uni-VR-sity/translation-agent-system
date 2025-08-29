import uuid
import asyncio
from typing import List, Optional, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import logging
import instructor
from litellm import completion
import litellm
from pydantic import BaseModel
from datetime import datetime
from prompts import (get_system_prompt, get_translation_prompt,
                      get_dictionary_prompt, get_grammar_prompt,
                      get_examples_prompt, get_retry_prompt, get_judge_prompt,
                      get_chunking_prompt, get_consistency_prompt)

# Import memory system components
from memory_system import MemoryManager, TranslationSession, calculate_text_characteristics
from learning_engine import LearningEngine
from adaptive_components import AdaptivePromptEnhancer, SmartContextSelector, ModelOptimizer


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# litellm._turn_on_debug()
client = instructor.from_litellm(completion)

# Initialize memory system components (global instances)
memory_manager = MemoryManager()
learning_engine = LearningEngine(memory_manager)
prompt_enhancer = AdaptivePromptEnhancer(memory_manager, learning_engine)
context_selector = SmartContextSelector(memory_manager, learning_engine)
model_optimizer = ModelOptimizer(memory_manager, learning_engine)

# Define the state schema
class TranslationState(TypedDict):
    sentence: str
    source_language: str
    target_language: str
    dictionary_content_raw: Optional[str]
    grammar_content_raw: Optional[str]
    examples_content_raw: Optional[str]
    dictionary_content: Optional[List[List[str]]]
    grammar_content: Optional[List[str]]
    examples_content: Optional[List[str]]
    translation_model: str
    fixed_model: str
    api_base: str
    api_key: str
    use_judge: bool
    translation_prompt: Optional[str]
    initial_translation: Optional[str]
    judge_feedback: Optional[str]
    final_translation: Optional[str]
    chunks: Optional[List[str]]
    consistency_feedback: Optional[str]
    consistency_status: Optional[str]
    chunk_translations: Optional[List[str]]
    # Memory-related fields
    session_id: Optional[str]
    processing_start_time: Optional[datetime]
    memory_recommendations: Optional[Dict[str, Any]]

# Pydantic model for dictionary output
class DictionaryOutput(BaseModel):
    word_pairs: List[List[str]]

# Pydantic model for grammar output
class GrammarOutput(BaseModel):
    statements: List[str]

# Pydantic model for examples output
class ExamplesOutput(BaseModel):
    samples: List[str]

# Pydantic model for chunking output
class ChunkOutput(BaseModel):
    chunks: List[str]

# Pydantic model for consistency output
class ConsistencyOutput(BaseModel):
    status: str
    translation: str
    feedback: Optional[str]

# Node to process dictionary
def process_dictionary(state: TranslationState) -> TranslationState:
    if not state.get("dictionary_content_raw"):
        return {"dictionary_content": None}
    
    dictionary_text = state["dictionary_content_raw"]
    prompt = get_dictionary_prompt(source_language=state["source_language"],
                                   target_language=state["target_language"],
                                   text=state["sentence"],
                                   dictionary=dictionary_text)
    
    
    response = client.chat.completions.create(
        model=state["fixed_model"],
        api_base=state["api_base"],
        api_key=state["api_key"],
        messages=[
            {"role": "system", "content": "You are a language processing assistant."},
            {"role": "user", "content": prompt}
        ],
        response_model=DictionaryOutput,
    )
    
    return {"dictionary_content": response.word_pairs}

# Node to process grammar
def process_grammar(state: TranslationState) -> TranslationState:
    if not state.get("grammar_content_raw"):
        return {"grammar_content": None}
    
    grammar_text = state["grammar_content_raw"]
    prompt = get_grammar_prompt(source_language=state["source_language"],
                                target_language=state["target_language"],
                                text=state["sentence"],
                                grammar=grammar_text)
    
    
    response = client.chat.completions.create(
        model=state["fixed_model"],
        api_base=state["api_base"],
        api_key=state["api_key"],
        messages=[
            {"role": "system", "content": "You are a language processing assistant."},
            {"role": "user", "content": prompt}
        ],
        response_model=GrammarOutput,
        max_tokens=200,
        temperature=0.5
    )
    
    return {"grammar_content": response.statements}

# Node to process examples
def process_examples(state: TranslationState) -> TranslationState:
    if not state.get("examples_content_raw"):
        return {"examples_content": None}
    
    examples_text = state["examples_content_raw"]
    prompt = get_examples_prompt(source_language=state["source_language"],
                                 target_language=state["target_language"],
                                 text=state["sentence"],
                                 examples=examples_text)
    
    
    response = client.chat.completions.create(
        model=state["fixed_model"],
        api_base=state["api_base"],
        api_key=state["api_key"],
        messages=[
            {"role": "system", "content": "You are a language processing assistant."},
            {"role": "user", "content": prompt}
        ],
        response_model=ExamplesOutput,
        max_tokens=200,
        temperature=0.5
    )
    
    return {"examples_content": response.samples}

# Node for main translation with memory enhancement
def perform_translation(state: TranslationState) -> TranslationState:
    
    system_prompt = get_system_prompt()
    
    # Get base translation prompt
    base_translation_prompt = get_translation_prompt(
        source_language=state["source_language"],
        target_language=state["target_language"],
        text=state["sentence"],
        dictionary=state.get("dictionary_content"),
        grammar=state.get("grammar_content"),
        examples=state.get("examples_content")
    )
    
    # Enhance prompt with learned patterns
    enhanced_translation_prompt = prompt_enhancer.enhance_translation_prompt(
        base_translation_prompt,
        state["source_language"],
        state["target_language"],
        state["sentence"],
        state.get("dictionary_content"),
        state.get("grammar_content"),
        state.get("examples_content")
    )
    
    chat_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": enhanced_translation_prompt}
    ]
    
    # Use memory-optimized model if available
    recommended_models = model_optimizer.recommend_optimal_models(
        f"{state['source_language']}-{state['target_language']}",
        calculate_text_characteristics(state['sentence']),
        {'translation_model': [state['translation_model']]}
    )
    
    selected_model = recommended_models.get('translation_model', state['translation_model'])
    
    response = client.chat.completions.create(
        model=selected_model,
        api_base=state["api_base"],
        api_key=state["api_key"],
        messages=chat_prompt,
        response_model=None,
    )
    
    translation = response.choices[0].message.content
    return {
        "translation_prompt": enhanced_translation_prompt,
        "initial_translation": translation,
        "final_translation": translation
    }

# Node for judge evaluation with memory learning
def judge_translation(state: TranslationState) -> TranslationState:
    if not state.get("use_judge"):
        # Still learn from non-judged sessions
        if state.get('session_id'):
            _learn_from_session(state, state.get('final_translation'))
        return state
    
    judge_prompt = get_judge_prompt(
        source_language=state["source_language"],
        target_language=state["target_language"],
        sentence=state["sentence"],
        translation=state["initial_translation"],
        translation_prompt=state["translation_prompt"]
    )
    
    response = client.chat.completions.create(
        model=state["fixed_model"],
        api_base=state["api_base"],
        api_key=state["api_key"],
        messages=[
            {"role": "system", "content": "You are a translation quality evaluator."},
            {"role": "user", "content": judge_prompt}
        ],
        response_model=None,
        max_tokens=300,
        temperature=0.5
    )
    
    feedback = response.choices[0].message.content
    
    # Learn from judged sessions immediately if approved
    if feedback == "APPROVED" and state.get('session_id'):
        _learn_from_session(state, state.get('final_translation'), feedback)
    
    return {"judge_feedback": feedback}

# Conditional edge to decide retry or end
def decide_retry(state: TranslationState) -> str:
    if not state.get("use_judge") or state.get("judge_feedback") == "APPROVED":
        return END
    return "retry_translation"

# Node for retry translation with judge feedback and memory learning
def retry_translation(state: TranslationState) -> TranslationState:
    
    system_prompt = get_system_prompt()
    retry_prompt = get_retry_prompt(state['translation_prompt'], state['initial_translation'], state['judge_feedback'])
    
    chat_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": retry_prompt}
    ]
        
    response = client.chat.completions.create(
        model=state["translation_model"],
        api_base=state["api_base"],
        api_key=state["api_key"],
        messages=chat_prompt,
        response_model=None,
        max_tokens=800,
        temperature=0.7,
        top_p=0.95
    )
    
    translation = response.choices[0].message.content
    
    # Learn from the retry session
    if state.get('session_id'):
        _learn_from_session(state, translation)
    
    return {"final_translation": translation}

async def process_resources_parallel(state: TranslationState) -> TranslationState:
    logger.debug("process_resources_parallel: Starting parallel processing")
    
    # Initialize session tracking
    session_id = str(uuid.uuid4())
    processing_start_time = datetime.now()
    
    # Get memory recommendations for context optimization
    language_pair = f"{state['source_language']}-{state['target_language']}"
    text_characteristics = calculate_text_characteristics(state['sentence'])
    recommendations = learning_engine.get_recommendations(language_pair, text_characteristics)
    
    # Optimize context usage based on learned patterns
    available_contexts = {
        'dictionary': state.get('dictionary_content_raw'),
        'grammar': state.get('grammar_content_raw'),
        'examples': state.get('examples_content_raw')
    }
    
    optimized_contexts = context_selector.optimize_context_usage(
        language_pair, state['sentence'], available_contexts
    )
    
    # Update state with optimized contexts
    optimized_state = state.copy()
    optimized_state.update(optimized_contexts)
    
    # Define coroutines for each processing task
    async def run_dictionary():
        return process_dictionary(optimized_state)
    
    async def run_grammar():
        return process_grammar(optimized_state)
    
    async def run_examples():
        return process_examples(optimized_state)
    
    # Run all tasks in parallel
    results = await asyncio.gather(
        run_dictionary(),
        run_grammar(),
        run_examples(),
        return_exceptions=True
    )
    
    # Initialize state updates
    state_updates = {
        "session_id": session_id,
        "processing_start_time": processing_start_time,
        "memory_recommendations": recommendations
    }
    
    # Process results
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"process_resources_parallel: Error in task {i}: {str(result)}")
            continue
        if i == 0:
            state_updates["dictionary_content"] = result.get("dictionary_content")
        elif i == 1:
            state_updates["grammar_content"] = result.get("grammar_content")
        elif i == 2:
            state_updates["examples_content"] = result.get("examples_content")
    
    logger.debug(f"process_resources_parallel: State updates: {state_updates}")
    return state_updates

# Modified build_translation_graph function
def build_translation_graph() -> StateGraph:
    graph = StateGraph(TranslationState)
    
    # Add nodes
    graph.add_node("process_resources_parallel", process_resources_parallel)
    graph.add_node("perform_translation", perform_translation)
    graph.add_node("judge_translation", judge_translation)
    graph.add_node("retry_translation", retry_translation)
    
    # Define edges
    graph.set_entry_point("process_resources_parallel")
    graph.add_edge("process_resources_parallel", "perform_translation")
    graph.add_edge("perform_translation", "judge_translation")
    
    # Conditional edge after judge
    graph.add_conditional_edges(
        "judge_translation",
        decide_retry,
        {
            END: END,
            "retry_translation": "retry_translation"
        }
    )
    
    # Edge from retry to end
    graph.add_edge("retry_translation", END)
    
    # Compile the graph with memory
    return graph.compile(checkpointer=MemorySaver())

def _learn_from_session(state: TranslationState, final_translation: Optional[str] = None, judge_feedback: Optional[str] = None) -> None:
    """Helper function to learn from a translation session"""
    try:
        # Calculate processing time
        processing_time = 0
        if state.get('processing_start_time'):
            processing_time = (datetime.now() - state['processing_start_time']).total_seconds()
        
        # Create translation session record
        session = TranslationSession(
            session_id=state.get('session_id', str(uuid.uuid4())),
            timestamp=datetime.now(),
            source_language=state['source_language'],
            target_language=state['target_language'],
            original_text=state['sentence'],
            initial_translation=state.get('initial_translation', ''),
            final_translation=final_translation or state.get('final_translation', ''),
            judge_feedback=judge_feedback or state.get('judge_feedback'),
            processing_mode='single',  # or 'corpus' for supernode
            models_used={
                'translation_model': state['translation_model'],
                'fixed_model': state['fixed_model']
            },
            context_files_used={
                'dictionary': state.get('dictionary_content_raw') is not None,
                'grammar': state.get('grammar_content_raw') is not None,
                'examples': state.get('examples_content_raw') is not None
            },
            performance_metrics={
                'processing_time': processing_time,
                'retry_needed': state.get('initial_translation') != final_translation,
                'context_count': sum([
                    state.get('dictionary_content_raw') is not None,
                    state.get('grammar_content_raw') is not None,
                    state.get('examples_content_raw') is not None
                ])
            }
        )
        
        # Learn from the session
        learning_results = learning_engine.learn_from_session(session)
        logger.info(f"Learning results: {learning_results}")
        
    except Exception as e:
        logger.error(f"Error in learning from session: {e}")

def get_memory_insights(language_pair: str = None) -> Dict[str, Any]:
    """Get memory system insights for analytics"""
    try:
        insights = {
            'total_sessions': 0,
            'feedback_patterns': [],
            'successful_patterns': [],
            'model_performance': {},
            'recommendations': []
        }
        
        # Get translation history
        history = memory_manager.get_translation_history(language_pair=language_pair, days=30)
        insights['total_sessions'] = len(history)
        
        # Get feedback patterns
        if language_pair:
            feedback_patterns = memory_manager.get_feedback_patterns(language_pair, limit=5)
            insights['feedback_patterns'] = [
                {
                    'type': p.feedback_type,
                    'frequency': p.frequency_count,
                    'issues': p.common_issues[:3]
                } for p in feedback_patterns
            ]
            
            # Get model performance insights
            insights['model_performance'] = model_optimizer.get_model_performance_insights(language_pair)
        
        return insights
        
    except Exception as e:
        logger.error(f"Error getting memory insights: {e}")
        return {}

# Node to split corpus into chunks
async def split_corpus(state: TranslationState) -> TranslationState:
    logger.debug(f"split_corpus: Input state: {state}")
    try:
        prompt = get_chunking_prompt(state["sentence"])
        
        response = client.chat.completions.create(
            model=state["fixed_model"],
            api_base=state["api_base"],
            api_key=state["api_key"],
            messages=[
                {"role": "system", "content": "You are a text processing assistant."},
                {"role": "user", "content": prompt}
            ],
            response_model=ChunkOutput,
            max_tokens=5000,
            temperature=0.5
        )
        chunks = response.chunks
        if not chunks:
            logger.warning("No chunks generated; using original sentence as single chunk")
            chunks = [state["sentence"]]
        logger.debug(f"split_corpus: Generated chunks: {chunks}")
        return {"chunks": chunks}
    except Exception as e:
        logger.error(f"split_corpus: Error during chunking: {str(e)}")
        # Fallback: Use the original sentence as a single chunk
        return {"chunks": [state["sentence"]]}

# Helper to process a single chunk
async def process_chunk(chunk: str, state: TranslationState, graph):
    logger.debug(f"process_chunk: Processing chunk: {chunk}")
    chunk_state = state.copy()
    chunk_state["sentence"] = chunk
    try:
        result = await graph.ainvoke(chunk_state, config={"configurable": {"thread_id": str(uuid.uuid4())}})
        return result["final_translation"]
    except Exception as e:
        logger.error(f"process_chunk: Error processing chunk '{chunk}': {str(e)}")
        return f"Error: {str(e)}"

# Node to process chunks in parallel
async def process_chunks(state: TranslationState) -> TranslationState:
    logger.debug(f"process_chunks: Input state: {state}")
    if "chunks" not in state or not state["chunks"]:
        logger.error("process_chunks: No chunks found in state")
        raise ValueError("No chunks available to process")
    
    graph = build_translation_graph()
    tasks = [process_chunk(chunk, state, graph) for chunk in state["chunks"]]
    translations = await asyncio.gather(*tasks, return_exceptions=True)
    logger.debug(f"process_chunks: Translations: {translations}")
    return {"chunk_translations": translations}

# Node to check consistency
async def check_consistency(state: TranslationState) -> TranslationState:
    logger.debug(f"check_consistency: Input state: {state}")
    try:
        prompt = get_consistency_prompt(
            source_language=state["source_language"],
            target_language=state["target_language"],
            chunks=state["chunks"],
            translations=state["chunk_translations"]
        )
        
        response = client.chat.completions.create(
            model=state["fixed_model"],
            api_base=state["api_base"],
            api_key=state["api_key"],
            messages=[
                {"role": "system", "content": "You are a translation quality evaluator."},
                {"role": "user", "content": prompt}
            ],
            response_model=ConsistencyOutput,
            max_tokens=1000,
            temperature=0.5
        )
        logger.debug(f"check_consistency: Consistency response: {response}")
        return {
            "final_translation": response.translation,
            "consistency_feedback": response.feedback,
            "consistency_status": response.status
        }
    except Exception as e:
        logger.error(f"check_consistency: Error during consistency check: {str(e)}")
        # Fallback: Concatenate translations as-is
        return {
            "final_translation": " ".join(str(t) for t in state["chunk_translations"]),
            "consistency_feedback": f"Consistency check failed: {str(e)}",
            "consistency_status": "ERROR"
        }

# Build the supernode graph
def build_supernode_graph() -> StateGraph:
    graph = StateGraph(TranslationState)
    
    # Add nodes
    graph.add_node("split_corpus", split_corpus)
    graph.add_node("process_chunks", process_chunks)
    graph.add_node("check_consistency", check_consistency)
    
    # Define edges
    graph.set_entry_point("split_corpus")
    graph.add_edge("split_corpus", "process_chunks")
    graph.add_edge("process_chunks", "check_consistency")
    graph.add_edge("check_consistency", END)
    
    # Compile the graph with memory
    return graph.compile(checkpointer=MemorySaver())
