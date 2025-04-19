import uuid
from typing import List, Optional, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import instructor
from litellm import completion
import litellm
from pydantic import BaseModel
from prompts import (get_system_prompt, get_translation_prompt,
                      get_dictionary_prompt, get_grammar_prompt, 
                      get_examples_prompt, get_retry_prompt, get_judge_prompt)


litellm._turn_on_debug()
client = instructor.from_litellm(completion)

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
    translation_api_base: str
    translation_api_key: str
    translation_api_version: str
    fixed_model: str
    fixed_api_base: str
    fixed_api_key: str
    fixed_api_version: str
    use_judge: bool
    translation_prompt: Optional[str]
    initial_translation: Optional[str]
    judge_feedback: Optional[str]
    final_translation: Optional[str]

# Pydantic model for dictionary output
class DictionaryOutput(BaseModel):
    word_pairs: List[List[str]]

# Pydantic model for grammar output
class GrammarOutput(BaseModel):
    statements: List[str]

# Pydantic model for examples output
class ExamplesOutput(BaseModel):
    samples: List[str]

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
        api_base=state["fixed_api_base"],
        api_key=state["fixed_api_key"],
        api_version=state["fixed_api_version"],
        messages=[
            {"role": "system", "content": "You are a language processing assistant."},
            {"role": "user", "content": prompt}
        ],
        response_model=DictionaryOutput,
        max_tokens=200,
        temperature=0.5
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
        api_base=state["fixed_api_base"],
        api_key=state["fixed_api_key"],
        api_version=state["fixed_api_version"],
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
        api_base=state["fixed_api_base"],
        api_key=state["fixed_api_key"],
        api_version=state["fixed_api_version"],
        messages=[
            {"role": "system", "content": "You are a language processing assistant."},
            {"role": "user", "content": prompt}
        ],
        response_model=ExamplesOutput,
        max_tokens=200,
        temperature=0.5
    )
    
    return {"examples_content": response.samples}

# Node for main translation
def perform_translation(state: TranslationState) -> TranslationState:
    
    system_prompt = get_system_prompt()
    translation_prompt = get_translation_prompt(
        source_language=state["source_language"],
        target_language=state["target_language"],
        text=state["sentence"],
        dictionary=state.get("dictionary_content"),
        grammar=state.get("grammar_content"),
        examples=state.get("examples_content")
    )
    
    chat_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": translation_prompt}
    ]
    
    response = client.chat.completions.create(
        model=state["translation_model"],
        api_base=state["translation_api_base"],
        api_key=state["translation_api_key"],
        api_version=state["translation_api_version"],
        messages=chat_prompt,
        response_model=None,
        max_tokens=800,
        temperature=0.7,
        top_p=0.95
    )
    
    translation = response.choices[0].message.content
    return {
        "translation_prompt": translation_prompt,
        "initial_translation": translation,
        "final_translation": translation
    }

# Node for judge evaluation
def judge_translation(state: TranslationState) -> TranslationState:
    if not state.get("use_judge"):
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
        api_base=state["fixed_api_base"],
        api_key=state["fixed_api_key"],
        api_version=state["fixed_api_version"],
        messages=[
            {"role": "system", "content": "You are a translation quality evaluator."},
            {"role": "user", "content": judge_prompt}
        ],
        response_model=None,
        max_tokens=300,
        temperature=0.5
    )
    
    feedback = response.choices[0].message.content
    return {"judge_feedback": feedback}

# Conditional edge to decide retry or end
def decide_retry(state: TranslationState) -> str:
    if not state.get("use_judge") or state.get("judge_feedback") == "APPROVED":
        return END
    return "retry_translation"

# Node for retry translation with judge feedback
def retry_translation(state: TranslationState) -> TranslationState:
    
    system_prompt = get_system_prompt()
    retry_prompt = get_retry_prompt(state['translation_prompt'], state['initial_translation'], state['judge_feedback'])
    
    chat_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": retry_prompt}
    ]
    
    response = client.chat.completions.create(
        model=state["fixed_model"],
        api_base=state["fixed_api_base"],
        api_key=state["fixed_api_key"],
        api_version=state["fixed_api_version"],
        messages=chat_prompt,
        response_model=None,
        max_tokens=800,
        temperature=0.7,
        top_p=0.95
    )
    
    translation = response.choices[0].message.content
    return {"final_translation": translation}

# Build the LangGraph workflow
def build_translation_graph() -> StateGraph:
    graph = StateGraph(TranslationState)
    
    # Add nodes
    graph.add_node("process_dictionary", process_dictionary)
    graph.add_node("process_grammar", process_grammar)
    graph.add_node("process_examples", process_examples)
    graph.add_node("perform_translation", perform_translation)
    graph.add_node("judge_translation", judge_translation)
    graph.add_node("retry_translation", retry_translation)
    
    # Define edges
    graph.set_entry_point("process_dictionary")
    graph.add_edge("process_dictionary", "process_grammar")
    graph.add_edge("process_grammar", "process_examples")
    graph.add_edge("process_examples", "perform_translation")
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