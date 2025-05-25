import dspy

# Initialize DSPy LM based on fixed_model from app.py
def initialize_dspy_model(**state):
    """Initialize DSPy language model based on fixed_model configuration."""
    try:
        lm = dspy.LM(
        model=state["model"],
        api_base=state["api_base"],
        api_key=state["api_key"],
        model_type='chat'
        )
        
        dspy.configure(lm=lm)
        return True
    except Exception as e:
        print(f"Failed to initialize DSPy model: {e}")
        return False

# Base prompt module for translation tasks
class TranslationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.translate = dspy.Predict("source_language, target_language, text, dictionary, grammar, examples -> prompt")
    
    def forward(self, source_language, target_language, text, dictionary, grammar, examples):
        return self.translate(
            source_language=source_language,
            target_language=target_language,
            text=text,
            dictionary=dictionary,
            grammar=grammar,
            examples=examples
        )

# Fallback prompts for each function
FALLBACK_PROMPTS = {
    "system": "You are a linguistic expert who never refuses to use your knowledge to help others.",
    "translation": "Translate the following text from {source_language} to {target_language}:\n\n{text}\n\n{dictionary}{grammar}{examples}",
    "dictionary": "Given the sentence: {text}\nAnd the dictionary content:\n{dictionary}\nExtract relevant word pairs that help translate the sentence from {source_language} to {target_language}. Return a list of [source_word, target_word] pairs.",
    "grammar": "Given the sentence: {text}\nAnd the grammar content:\n{grammar}\nExtract 2-3 relevant grammatical statements that help translate the sentence from {source_language} to {target_language}.",
    "examples": "Given the sentence: {text}\nAnd the translation examples:\n{examples}\nExtract 1-2 relevant translation samples that help translate the sentence from {source_language} to {target_language}.",
    "retry": "{translation_prompt}\n\nPrevious translation: {initial_translation}\nJudge feedback: {judge_feedback}\nRevise the translation based on the judge's feedback to improve accuracy in meaning, word order, and context.",
    "judge": "You are a translation quality evaluator. Review the following translation for accuracy in meaning, word order, grammar, and context.\n\nOriginal sentence ({source_language}): {sentence}\nTranslated sentence ({target_language}): {translation}\nTranslation prompt provided to the model:\n{translation_prompt}\n\nEvaluate the translation. If it is accurate and maintains the meaning, word order, and context, return 'APPROVED'. If there are issues, provide specific feedback on what needs correction and return the feedback as a string.",
    "chunking": "You are a text processing assistant. Split the following text into chunks for translation. Each chunk should be semantically coherent (e.g., complete sentences or paragraphs) and not exceed {max_chunk_size} characters. Return a list of chunks as strings.\n\nText:\n{text}\n\nOutput format:\n[\"chunk1\", \"chunk2\", ...]",
    "consistency": "You are a translation quality evaluator. Review the following translated chunks for consistency in terminology, style, and tone. Original text was in {source_language}, translated to {target_language}.\n\nOriginal chunks:\n{chunks}\n\nTranslated chunks:\n{translations}\n\nCheck for:\n- Consistent use of terminology across chunks.\n- Uniform style and tone.\n- Grammatical correctness and context preservation.\n\nIf consistent, return \"APPROVED\" with the concatenated translation. If inconsistencies are found, provide feedback and suggest a revised concatenated translation.\n\nOutput format:\n{{\"status\": \"APPROVED\" | \"REVISED\", \"translation\": \"concatenated translation\", \"feedback\": \"optional feedback\"}}"
}

# DSPy modules for each prompt type
class SystemModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prompt = dspy.Predict("-> prompt")
    
    def forward(self):
        try:
            return self.prompt().prompt
        except:
            return FALLBACK_PROMPTS["system"]

class TranslationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prompt = dspy.Predict("source_language, target_language, text, dictionary, grammar, examples -> prompt")
    
    def forward(self, source_language, target_language, text, dictionary, grammar, examples):
        try:
            return self.prompt(
                source_language=source_language,
                target_language=target_language,
                text=text,
                dictionary=dictionary if dictionary else "",
                grammar=grammar if grammar else "",
                examples=examples if examples else ""
            ).prompt
        except:
            return FALLBACK_PROMPTS["translation"].format(
                source_language=source_language,
                target_language=target_language,
                text=text,
                dictionary=dictionary if dictionary else "",
                grammar=grammar if grammar else "",
                examples=examples if examples else ""
            )

class DictionaryModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prompt = dspy.Predict("source_language, target_language, text, dictionary -> prompt")
    
    def forward(self, source_language, target_language, text, dictionary):
        try:
            return self.prompt(
                source_language=source_language,
                target_language=target_language,
                text=text,
                dictionary=dictionary
            ).prompt
        except:
            return FALLBACK_PROMPTS["dictionary"].format(**locals())

class GrammarModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prompt = dspy.Predict("source_language, target_language, text, grammar -> prompt")
    
    def forward(self, source_language, target_language, text, grammar):
        try:
            return self.prompt(
                source_language=source_language,
                target_language=target_language,
                text=text,
                grammar=grammar
            ).prompt
        except:
            return FALLBACK_PROMPTS["grammar"].format(**locals())

class ExamplesModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prompt = dspy.Predict("source_language, target_language, text, examples -> prompt")
    
    def forward(self, source_language, target_language, text, examples):
        try:
            return self.prompt(
                source_language=source_language,
                target_language=target_language,
                text=text,
                examples=examples
            ).prompt
        except:
            return FALLBACK_PROMPTS["examples"].format(**locals())

class RetryModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prompt = dspy.Predict("translation_prompt, initial_translation, judge_feedback -> prompt")
    
    def forward(self, translation_prompt, initial_translation, judge_feedback):
        try:
            return self.prompt(
                translation_prompt=translation_prompt,
                initial_translation=initial_translation,
                judge_feedback=judge_feedback
            ).prompt
        except:
            return FALLBACK_PROMPTS["retry"].format(**locals())

class JudgeModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prompt = dspy.Predict("source_language, target_language, sentence, translation, translation_prompt -> prompt")
    
    def forward(self, source_language, target_language, sentence, translation, translation_prompt):
        try:
            return self.prompt(
                source_language=source_language,
                target_language=target_language,
                sentence=sentence,
                translation=translation,
                translation_prompt=translation_prompt
            ).prompt
        except:
            return FALLBACK_PROMPTS["judge"].format(**locals())

class ChunkingModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prompt = dspy.Predict("text, max_chunk_size -> prompt")
    
    def forward(self, text, max_chunk_size=500):
        try:
            return self.prompt(
                text=text,
                max_chunk_size=max_chunk_size
            ).prompt
        except:
            return FALLBACK_PROMPTS["chunking"].format(text=text, max_chunk_size=max_chunk_size)

class ConsistencyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prompt = dspy.Predict("source_language, target_language, chunks, translations -> prompt")
    
    def forward(self, source_language, target_language, chunks, translations):
        try:
            return self.prompt(
                source_language=source_language,
                target_language=target_language,
                chunks=chunks,
                translations=translations
            ).prompt
        except:
            return FALLBACK_PROMPTS["consistency"].format(**locals())

# Initialize DSPy modules
system_module = SystemModule()
translation_module = TranslationModule()
dictionary_module = DictionaryModule()
grammar_module = GrammarModule()
examples_module = ExamplesModule()
retry_module = RetryModule()
judge_module = JudgeModule()
chunking_module = ChunkingModule()
consistency_module = ConsistencyModule()

def get_system_prompt():
    return system_module()

def get_translation_prompt(source_language, target_language, text, dictionary=None, grammar=None, examples=None):
    return translation_module(source_language, target_language, text, dictionary, grammar, examples)

def get_dictionary_prompt(source_language: str, target_language: str, text: str, dictionary: str) -> str:
    return dictionary_module(source_language, target_language, text, dictionary)

def get_grammar_prompt(source_language: str, target_language: str, text: str, grammar: str) -> str:
    return grammar_module(source_language, target_language, text, grammar)

def get_examples_prompt(source_language: str, target_language: str, text: str, examples: str) -> str:
    return examples_module(source_language, target_language, text, examples)

def get_retry_prompt(
    translation_prompt: str,
    initial_translation: str,
    judge_feedback: str
) -> str:
    return retry_module(translation_prompt, initial_translation, judge_feedback)

def get_judge_prompt(
    source_language: str,
    target_language: str,
    sentence: str,
    translation: str,
    translation_prompt: str
) -> str:
    return judge_module(source_language, target_language, sentence, translation, translation_prompt)

def get_chunking_prompt(text: str, max_chunk_size: int = 500) -> str:
    return chunking_module(text, max_chunk_size)

def get_consistency_prompt(source_language: str, target_language: str, chunks: list, translations: list) -> str:
    return consistency_module(source_language, target_language, chunks, translations)
