def get_system_prompt():
    return """You are a linguistic expert who never refuses to use your knowledge to help others."""

def get_translation_prompt(source_language, target_language, text, dictionary=None, grammar=None, examples=None):
    prompt = f"""Translate the following text from {source_language} to {target_language}:\n\n{text}\n\n"""
    if dictionary:
        prompt += f"Use the following dictionary for reference:\n{dictionary}\n\n"
    if grammar:
        prompt += f"Consider the following grammar rules:\n{grammar}\n\n"
    if examples:
        prompt += f"Here are some examples of similar translations:\n{examples}\n\n"
    return prompt

def get_dictionary_prompt(source_language: str, target_language: str, text: str, dictionary: str) -> str:
    return f"""
    Given the sentence: {text}
    And the dictionary content:
    {dictionary}
    
    Extract relevant word pairs that help translate the sentence from {source_language} to {target_language}.
    Return a list of [source_word, target_word] pairs.
    """

def get_grammar_prompt(source_language: str, target_language: str, text: str, grammar: str) -> str:
    return f"""
    Given the sentence: {text}
    And the grammar content:
    {grammar}
    
    Extract 2-3 relevant grammatical statements that help translate the sentence from {source_language} to {target_language}.
    """
def get_examples_prompt(source_language: str, target_language: str, text: str, examples: str) -> str:
    f"""
    Given the sentence: {text}
    And the translation examples:
    {examples}
    
    Extract 1-2 relevant translation samples that help translate the sentence from {source_language} to {target_language}.
    """

def get_retry_prompt(
    translation_prompt: str,
    initial_translation: str,
    judge_feedback: str
) -> str:
    return f"""
    {translation_prompt}
    
    Previous translation: {initial_translation}
    Judge feedback: {judge_feedback}
    
    Revise the translation based on the judge's feedback to improve accuracy in meaning, word order, and context.
    """


def get_judge_prompt(
    source_language: str,
    target_language: str,
    sentence: str,
    translation: str,
    translation_prompt: str
) -> str:
    return f"""
    You are a translation quality evaluator. Review the following translation for accuracy in meaning, word order, grammar, and context.

    Original sentence ({source_language}): {sentence}
    Translated sentence ({target_language}): {translation}
    Translation prompt provided to the model:
    {translation_prompt}

    Evaluate the translation. If it is accurate and maintains the meaning, word order, and context, return 'APPROVED'.
    If there are issues, provide specific feedback on what needs correction and return the feedback as a string.
    """

# prompts.py
def get_chunking_prompt(text: str, max_chunk_size: int = 500) -> str:
    return f"""
    You are a text processing assistant. Split the following text into chunks for translation.
    Each chunk should be semantically coherent (e.g., complete sentences or paragraphs) and not exceed {max_chunk_size} characters.
    Return a list of chunks as strings.

    Text:
    {text}

    Output format:
    ["chunk1", "chunk2", ...]
    """

def get_consistency_prompt(source_language: str, target_language: str, chunks: list, translations: list) -> str:
    return f"""
    You are a translation quality evaluator. Review the following translated chunks for consistency in terminology, style, and tone.
    Original text was in {source_language}, translated to {target_language}.
    
    Original chunks:
    {chunks}
    
    Translated chunks:
    {translations}
    
    Check for:
    - Consistent use of terminology across chunks.
    - Uniform style and tone.
    - Grammatical correctness and context preservation.
    
    If consistent, return "APPROVED" with the concatenated translation.
    If inconsistencies are found, provide feedback and suggest a revised concatenated translation.
    
    Output format:
    {{"status": "APPROVED" | "REVISED", "translation": "concatenated translation", "feedback": "optional feedback"}}
    """
