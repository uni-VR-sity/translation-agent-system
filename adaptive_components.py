import json
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from memory_system import MemoryManager, calculate_text_characteristics
from learning_engine import LearningEngine

logger = logging.getLogger(__name__)

class AdaptivePromptEnhancer:
    """Enhances translation prompts based on learned patterns and feedback"""
    
    def __init__(self, memory_manager: MemoryManager, learning_engine: LearningEngine):
        self.memory_manager = memory_manager
        self.learning_engine = learning_engine
        
        # Template improvements based on common feedback patterns
        self.improvement_templates = {
            'grammar': {
                'prefix': "Pay special attention to grammatical accuracy, including verb tenses, sentence structure, and word order. ",
                'suffix': "\n\nEnsure the translation follows proper grammatical rules for {target_language}."
            },
            'meaning': {
                'prefix': "Focus on preserving the exact meaning and context of the original text. ",
                'suffix': "\n\nMaintain the original intent and semantic nuance in the translation."
            },
            'style': {
                'prefix': "Maintain appropriate style and register for the target language. ",
                'suffix': "\n\nEnsure the translation sounds natural and fluent in {target_language}."
            },
            'terminology': {
                'prefix': "Use precise and appropriate terminology for the subject matter. ",
                'suffix': "\n\nSelect the most accurate technical or domain-specific terms."
            },
            'cultural': {
                'prefix': "Consider cultural context and localization requirements. ",
                'suffix': "\n\nAdapt cultural references and expressions appropriately for the target culture."
            }
        }
        
        # Context enhancement patterns
        self.context_enhancements = {
            'high_complexity': "This text has high linguistic complexity. Take extra care with nuanced expressions and complex sentence structures.",
            'technical_content': "This appears to be technical content. Prioritize accuracy of specialized terminology.",
            'formal_register': "Maintain formal register and professional tone throughout the translation.",
            'multi_sentence': "This is a multi-sentence text. Ensure consistency and coherence across all sentences."
        }
    
    def enhance_translation_prompt(self, base_prompt: str, source_language: str, target_language: str, 
                                 text: str, dictionary: Optional[List] = None, 
                                 grammar: Optional[List] = None, examples: Optional[List] = None) -> str:
        """Enhance the base translation prompt with learned patterns"""
        
        language_pair = f"{source_language}-{target_language}"
        text_characteristics = calculate_text_characteristics(text)
        
        # Get recommendations from learning engine
        recommendations = self.learning_engine.get_recommendations(language_pair, text_characteristics)
        
        # Start with base prompt
        enhanced_prompt = base_prompt
        
        # Add learned improvements
        enhanced_prompt = self._add_feedback_based_improvements(enhanced_prompt, recommendations, target_language)
        
        # Add context-specific enhancements
        enhanced_prompt = self._add_context_enhancements(enhanced_prompt, text_characteristics, text)
        
        # Add successful pattern guidance
        enhanced_prompt = self._add_pattern_guidance(enhanced_prompt, recommendations)
        
        # Add issue avoidance guidance
        enhanced_prompt = self._add_avoidance_guidance(enhanced_prompt, recommendations)
        
        return enhanced_prompt
    
    def _add_feedback_based_improvements(self, prompt: str, recommendations: Dict[str, Any], target_language: str) -> str:
        """Add improvements based on historical feedback patterns"""
        
        # Get feedback patterns to determine what to emphasize
        feedback_patterns = self.memory_manager.get_feedback_patterns(
            recommendations.get('language_pair', ''), limit=5
        )
        
        prefixes = []
        suffixes = []
        
        for pattern in feedback_patterns:
            if pattern.frequency_count >= 2:  # Only use patterns seen multiple times
                feedback_type = pattern.feedback_type
                if feedback_type in self.improvement_templates:
                    template = self.improvement_templates[feedback_type]
                    prefixes.append(template['prefix'])
                    suffixes.append(template['suffix'].format(target_language=target_language))
        
        # Add prefixes to the beginning
        if prefixes:
            prompt = ''.join(prefixes) + prompt
        
        # Add suffixes to the end
        if suffixes:
            prompt = prompt + ''.join(suffixes)
        
        return prompt
    
    def _add_context_enhancements(self, prompt: str, text_characteristics: Dict[str, Any], text: str) -> str:
        """Add context-specific enhancements based on text characteristics"""
        
        enhancements = []
        
        # High complexity text
        if text_characteristics['complexity_score'] > 0.8:
            enhancements.append(self.context_enhancements['high_complexity'])
        
        # Technical content detection
        if self._is_technical_content(text):
            enhancements.append(self.context_enhancements['technical_content'])
        
        # Formal register detection
        if self._is_formal_register(text):
            enhancements.append(self.context_enhancements['formal_register'])
        
        # Multi-sentence text
        if text_characteristics['sentence_count'] > 1:
            enhancements.append(self.context_enhancements['multi_sentence'])
        
        if enhancements:
            prompt = prompt + "\n\nAdditional guidance:\n" + "\n".join(f"- {enhancement}" for enhancement in enhancements)
        
        return prompt
    
    def _add_pattern_guidance(self, prompt: str, recommendations: Dict[str, Any]) -> str:
        """Add guidance based on successful patterns"""
        
        suggested_approaches = recommendations.get('suggested_approaches', [])
        if not suggested_approaches:
            return prompt
        
        # Extract common successful strategies
        strategies = []
        for approach in suggested_approaches[:2]:  # Top 2 approaches
            if approach.get('judge_approved', False):
                strategies.append("This approach has been previously approved by quality review.")
            
            context_usage = approach.get('context_usage', {})
            used_contexts = [k for k, v in context_usage.items() if v]
            if used_contexts:
                strategies.append(f"Consider leveraging {', '.join(used_contexts)} for better accuracy.")
        
        if strategies:
            prompt = prompt + "\n\nBased on successful patterns:\n" + "\n".join(f"- {strategy}" for strategy in strategies)
        
        return prompt
    
    def _add_avoidance_guidance(self, prompt: str, recommendations: Dict[str, Any]) -> str:
        """Add guidance to avoid common issues"""
        
        avoid_issues = recommendations.get('avoid_issues', [])
        if not avoid_issues:
            return prompt
        
        # Filter and format issues
        formatted_issues = []
        for issue in avoid_issues[:3]:  # Top 3 issues to avoid
            if len(issue.strip()) > 3:  # Filter out very short/meaningless issues
                formatted_issues.append(f"Avoid {issue.strip()}")
        
        if formatted_issues:
            prompt = prompt + "\n\nCommon issues to avoid:\n" + "\n".join(f"- {issue}" for issue in formatted_issues)
        
        return prompt
    
    def _is_technical_content(self, text: str) -> bool:
        """Detect if text contains technical content"""
        technical_indicators = [
            r'\b\w+\s*\([^)]*\)',  # Function calls or technical notation
            r'\b[A-Z]{2,}\b',      # Acronyms
            r'\b\d+\.\d+\b',       # Version numbers
            r'\b\w*[Tt]ech\w*\b',  # Tech-related words
            r'\b\w*[Ss]ystem\w*\b', # System-related words
            r'\b\w*[Pp]rocess\w*\b' # Process-related words
        ]
        
        technical_count = sum(1 for pattern in technical_indicators if re.search(pattern, text))
        return technical_count >= 2
    
    def _is_formal_register(self, text: str) -> bool:
        """Detect if text uses formal register"""
        formal_indicators = [
            r'\b(therefore|furthermore|moreover|consequently|nevertheless)\b',
            r'\b(shall|ought|must)\b',
            r'\b(pursuant|regarding|concerning)\b',
            r'\b(hereby|wherein|whereas)\b'
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in formal_indicators)

class SmartContextSelector:
    """Intelligently selects and prioritizes context based on learned patterns"""
    
    def __init__(self, memory_manager: MemoryManager, learning_engine: LearningEngine):
        self.memory_manager = memory_manager
        self.learning_engine = learning_engine
    
    def optimize_context_usage(self, language_pair: str, text: str, 
                             available_contexts: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize context usage based on learned patterns"""
        
        text_characteristics = calculate_text_characteristics(text)
        recommendations = self.learning_engine.get_recommendations(language_pair, text_characteristics)
        
        optimized_contexts = {}
        context_recommendations = recommendations.get('context_recommendations', {})
        
        # Prioritize contexts based on historical success
        for context_type, content in available_contexts.items():
            if content is not None:
                # Get recommendation score for this context type
                recommendation_score = context_recommendations.get(context_type, 0.5)
                
                # Apply content-specific optimization
                if context_type == 'dictionary':
                    optimized_contexts[context_type] = self._optimize_dictionary_content(
                        content, text, recommendation_score
                    )
                elif context_type == 'grammar':
                    optimized_contexts[context_type] = self._optimize_grammar_content(
                        content, text, recommendation_score
                    )
                elif context_type == 'examples':
                    optimized_contexts[context_type] = self._optimize_examples_content(
                        content, text, recommendation_score
                    )
                else:
                    optimized_contexts[context_type] = content
        
        return optimized_contexts
    
    def _optimize_dictionary_content(self, dictionary_content: List[List[str]], 
                                   text: str, recommendation_score: float) -> List[List[str]]:
        """Optimize dictionary content based on text relevance and learned patterns"""
        
        if not dictionary_content or recommendation_score < 0.3:
            return dictionary_content
        
        # Score each dictionary entry based on relevance to the text
        scored_entries = []
        text_words = set(text.lower().split())
        
        for entry in dictionary_content:
            if len(entry) >= 2:
                source_word = entry[0].lower()
                relevance_score = 0
                
                # Direct word match
                if source_word in text_words:
                    relevance_score += 1.0
                
                # Partial word match
                elif any(source_word in word or word in source_word for word in text_words):
                    relevance_score += 0.5
                
                # Apply recommendation boost
                relevance_score *= (1 + recommendation_score)
                
                scored_entries.append((entry, relevance_score))
        
        # Sort by relevance and return top entries
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        max_entries = min(len(scored_entries), int(20 * (1 + recommendation_score)))
        
        return [entry for entry, score in scored_entries[:max_entries]]
    
    def _optimize_grammar_content(self, grammar_content: List[str], 
                                text: str, recommendation_score: float) -> List[str]:
        """Optimize grammar content based on text characteristics and learned patterns"""
        
        if not grammar_content or recommendation_score < 0.3:
            return grammar_content
        
        text_characteristics = calculate_text_characteristics(text)
        
        # Score grammar rules based on relevance
        scored_rules = []
        
        for rule in grammar_content:
            relevance_score = 0.5  # Base score
            
            # Boost score for complex texts
            if text_characteristics['complexity_score'] > 0.7:
                relevance_score += 0.3
            
            # Boost score for multi-sentence texts
            if text_characteristics['sentence_count'] > 1:
                relevance_score += 0.2
            
            # Apply recommendation boost
            relevance_score *= (1 + recommendation_score)
            
            scored_rules.append((rule, relevance_score))
        
        # Sort by relevance and return top rules
        scored_rules.sort(key=lambda x: x[1], reverse=True)
        max_rules = min(len(scored_rules), int(10 * (1 + recommendation_score)))
        
        return [rule for rule, score in scored_rules[:max_rules]]
    
    def _optimize_examples_content(self, examples_content: List[str], 
                                 text: str, recommendation_score: float) -> List[str]:
        """Optimize examples content based on similarity and learned patterns"""
        
        if not examples_content or recommendation_score < 0.3:
            return examples_content
        
        text_characteristics = calculate_text_characteristics(text)
        
        # Score examples based on similarity to current text
        scored_examples = []
        
        for example in examples_content:
            example_chars = calculate_text_characteristics(example)
            similarity_score = self._calculate_text_similarity(text_characteristics, example_chars)
            
            # Apply recommendation boost
            final_score = similarity_score * (1 + recommendation_score)
            
            scored_examples.append((example, final_score))
        
        # Sort by similarity and return top examples
        scored_examples.sort(key=lambda x: x[1], reverse=True)
        max_examples = min(len(scored_examples), int(5 * (1 + recommendation_score)))
        
        return [example for example, score in scored_examples[:max_examples]]
    
    def _calculate_text_similarity(self, chars1: Dict[str, Any], chars2: Dict[str, Any]) -> float:
        """Calculate similarity between two text characteristic profiles"""
        
        # Length similarity
        length_diff = abs(chars1['length'] - chars2['length'])
        length_similarity = max(0, 1 - length_diff / max(chars1['length'], chars2['length'], 1))
        
        # Complexity similarity
        complexity_diff = abs(chars1['complexity_score'] - chars2['complexity_score'])
        complexity_similarity = max(0, 1 - complexity_diff)
        
        # Sentence count similarity
        sentence_diff = abs(chars1['sentence_count'] - chars2['sentence_count'])
        sentence_similarity = max(0, 1 - sentence_diff / max(chars1['sentence_count'], chars2['sentence_count'], 1))
        
        # Feature similarity (numbers, punctuation)
        feature_similarity = 0
        if chars1['has_numbers'] == chars2['has_numbers']:
            feature_similarity += 0.5
        if chars1['has_punctuation'] == chars2['has_punctuation']:
            feature_similarity += 0.5
        
        # Weighted average
        total_similarity = (
            length_similarity * 0.3 +
            complexity_similarity * 0.4 +
            sentence_similarity * 0.2 +
            feature_similarity * 0.1
        )
        
        return total_similarity

class ModelOptimizer:
    """Optimizes model selection based on historical performance data"""
    
    def __init__(self, memory_manager: MemoryManager, learning_engine: LearningEngine):
        self.memory_manager = memory_manager
        self.learning_engine = learning_engine
    
    def recommend_optimal_models(self, language_pair: str, text_characteristics: Dict[str, Any], 
                                available_models: Dict[str, List[str]]) -> Dict[str, str]:
        """Recommend optimal models for translation and processing tasks"""
        
        recommendations = {}
        
        # Get performance-based recommendations
        performance_recommendations = self.learning_engine.get_recommendations(language_pair, text_characteristics)
        best_models = performance_recommendations.get('best_models', {})
        
        # For each task type, recommend the best model
        for task_type, model_list in available_models.items():
            if task_type in best_models and best_models[task_type] in model_list:
                # Use historically best model if available
                recommendations[task_type] = best_models[task_type]
            else:
                # Fall back to heuristic-based selection
                recommendations[task_type] = self._select_model_by_heuristics(
                    task_type, model_list, text_characteristics
                )
        
        return recommendations
    
    def _select_model_by_heuristics(self, task_type: str, available_models: List[str], 
                                  text_characteristics: Dict[str, Any]) -> str:
        """Select model based on heuristics when no performance data is available"""
        
        if not available_models:
            return ""
        
        # Default to first available model
        selected_model = available_models[0]
        
        # Simple heuristics based on task type and text characteristics
        if task_type == 'translation_model':
            # For translation, prefer larger models for complex text
            if text_characteristics['complexity_score'] > 0.8:
                # Look for models that might be larger (crude heuristic)
                larger_models = [m for m in available_models if any(size in m.lower() for size in ['large', 'xl', '70b', '405b'])]
                if larger_models:
                    selected_model = larger_models[0]
        
        elif task_type == 'fixed_model':
            # For processing tasks, prefer faster models
            faster_models = [m for m in available_models if any(size in m.lower() for size in ['small', 'mini', '7b', '8b'])]
            if faster_models:
                selected_model = faster_models[0]
        
        return selected_model
    
    def get_model_performance_insights(self, language_pair: str) -> Dict[str, Any]:
        """Get insights about model performance for a language pair"""
        
        performances = self.memory_manager.get_model_performance(language_pair=language_pair)
        
        insights = {
            'total_models_tested': len(set(p.model_name for p in performances)),
            'best_translation_model': None,
            'best_processing_model': None,
            'performance_trends': {},
            'recommendations': []
        }
        
        if not performances:
            return insights
        
        # Find best models by task type
        translation_models = [p for p in performances if p.task_type == 'translation_model']
        processing_models = [p for p in performances if p.task_type == 'fixed_model']
        
        if translation_models:
            best_translation = max(translation_models, key=lambda p: p.avg_quality_score * p.success_rate)
            insights['best_translation_model'] = {
                'name': best_translation.model_name,
                'quality_score': best_translation.avg_quality_score,
                'success_rate': best_translation.success_rate,
                'total_uses': best_translation.total_uses
            }
        
        if processing_models:
            best_processing = max(processing_models, key=lambda p: p.avg_quality_score * p.success_rate)
            insights['best_processing_model'] = {
                'name': best_processing.model_name,
                'quality_score': best_processing.avg_quality_score,
                'success_rate': best_processing.success_rate,
                'avg_processing_time': best_processing.avg_processing_time
            }
        
        # Generate recommendations
        if insights['best_translation_model']:
            insights['recommendations'].append(
                f"Use {insights['best_translation_model']['name']} for translation tasks "
                f"(Quality: {insights['best_translation_model']['quality_score']:.2f})"
            )
        
        if insights['best_processing_model']:
            insights['recommendations'].append(
                f"Use {insights['best_processing_model']['name']} for processing tasks "
                f"(Speed: {insights['best_processing_model']['avg_processing_time']:.2f}s avg)"
            )
        
        return insights