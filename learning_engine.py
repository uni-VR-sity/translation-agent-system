import re
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from memory_system import (
    MemoryManager, FeedbackPattern, SuccessfulPattern, ModelPerformance,
    TranslationSession, generate_pattern_id, calculate_text_characteristics
)

logger = logging.getLogger(__name__)

class FeedbackAnalyzer:
    """Analyzes judge feedback to extract learning patterns"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        
        # Feedback classification patterns
        self.feedback_patterns = {
            'grammar': [
                r'grammar', r'grammatical', r'syntax', r'tense', r'verb', r'noun',
                r'adjective', r'adverb', r'sentence structure', r'word order'
            ],
            'meaning': [
                r'meaning', r'semantic', r'context', r'interpretation', r'sense',
                r'understanding', r'convey', r'express', r'intent'
            ],
            'style': [
                r'style', r'tone', r'formal', r'informal', r'register', r'voice',
                r'flow', r'natural', r'fluent', r'awkward'
            ],
            'terminology': [
                r'term', r'terminology', r'word choice', r'vocabulary', r'lexical',
                r'specific', r'technical', r'jargon'
            ],
            'cultural': [
                r'cultural', r'culture', r'idiom', r'expression', r'localization',
                r'native', r'cultural context'
            ]
        }
        
        # Common improvement suggestions
        self.improvement_patterns = {
            'use_formal_register': [
                r'more formal', r'formal register', r'formal language', r'formal tone'
            ],
            'improve_word_order': [
                r'word order', r'rearrange', r'sentence structure', r'syntax'
            ],
            'use_better_terminology': [
                r'better word', r'more appropriate', r'specific term', r'technical term'
            ],
            'maintain_consistency': [
                r'consistent', r'consistency', r'same term', r'uniform'
            ],
            'preserve_meaning': [
                r'preserve meaning', r'maintain sense', r'keep intent', r'original meaning'
            ]
        }
    
    def analyze_feedback(self, feedback: str, language_pair: str, session: TranslationSession) -> List[FeedbackPattern]:
        """Analyze judge feedback and extract patterns"""
        if not feedback or feedback == "APPROVED":
            return []
        
        patterns = []
        feedback_lower = feedback.lower()
        
        # Classify feedback type
        feedback_types = self._classify_feedback(feedback_lower)
        
        # Extract common issues
        common_issues = self._extract_issues(feedback_lower)
        
        # Extract improvement suggestions
        improvements = self._extract_improvements(feedback_lower)
        
        for feedback_type in feedback_types:
            pattern_content = f"{language_pair}_{feedback_type}_{hash(feedback)}"
            pattern_id = generate_pattern_id(pattern_content)
            
            pattern = FeedbackPattern(
                pattern_id=pattern_id,
                language_pair=language_pair,
                feedback_type=feedback_type,
                common_issues=common_issues,
                improvement_suggestions=improvements,
                frequency_count=1,
                last_seen=datetime.now(),
                confidence_score=self._calculate_confidence(feedback, feedback_type)
            )
            patterns.append(pattern)
        
        return patterns
    
    def _classify_feedback(self, feedback: str) -> List[str]:
        """Classify feedback into categories"""
        categories = []
        
        for category, patterns in self.feedback_patterns.items():
            for pattern in patterns:
                if re.search(pattern, feedback, re.IGNORECASE):
                    categories.append(category)
                    break
        
        return categories if categories else ['general']
    
    def _extract_issues(self, feedback: str) -> List[str]:
        """Extract specific issues mentioned in feedback"""
        issues = []
        
        # Look for specific issue patterns
        issue_patterns = [
            r'incorrect (.+?)(?:\.|,|$)',
            r'wrong (.+?)(?:\.|,|$)',
            r'should be (.+?)(?:\.|,|$)',
            r'better to (.+?)(?:\.|,|$)',
            r'missing (.+?)(?:\.|,|$)',
            r'awkward (.+?)(?:\.|,|$)'
        ]
        
        for pattern in issue_patterns:
            matches = re.findall(pattern, feedback, re.IGNORECASE)
            issues.extend([match.strip() for match in matches])
        
        return issues[:5]  # Limit to top 5 issues
    
    def _extract_improvements(self, feedback: str) -> List[str]:
        """Extract improvement suggestions from feedback"""
        improvements = []
        
        for improvement_type, patterns in self.improvement_patterns.items():
            for pattern in patterns:
                if re.search(pattern, feedback, re.IGNORECASE):
                    improvements.append(improvement_type)
                    break
        
        # Extract specific suggestions
        suggestion_patterns = [
            r'suggest (.+?)(?:\.|,|$)',
            r'recommend (.+?)(?:\.|,|$)',
            r'try (.+?)(?:\.|,|$)',
            r'consider (.+?)(?:\.|,|$)'
        ]
        
        for pattern in suggestion_patterns:
            matches = re.findall(pattern, feedback, re.IGNORECASE)
            improvements.extend([match.strip() for match in matches])
        
        return improvements[:5]  # Limit to top 5 suggestions
    
    def _calculate_confidence(self, feedback: str, feedback_type: str) -> float:
        """Calculate confidence score for the feedback pattern"""
        base_score = 0.5
        
        # Increase confidence based on specificity
        if len(feedback.split()) > 10:
            base_score += 0.2
        
        # Increase confidence if feedback type patterns are strongly present
        type_patterns = self.feedback_patterns.get(feedback_type, [])
        matches = sum(1 for pattern in type_patterns if re.search(pattern, feedback, re.IGNORECASE))
        base_score += min(matches * 0.1, 0.3)
        
        return min(base_score, 1.0)

class PatternLearner:
    """Learns successful translation patterns from high-quality translations"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
    
    def learn_from_session(self, session: TranslationSession) -> Optional[SuccessfulPattern]:
        """Learn patterns from a successful translation session"""
        # Only learn from high-quality translations
        if not self._is_high_quality(session):
            return None
        
        text_chars = calculate_text_characteristics(session.original_text)
        
        # Create successful approach record
        successful_approach = {
            'prompt_structure': self._extract_prompt_structure(session),
            'context_usage': session.context_files_used,
            'models_used': session.models_used,
            'processing_mode': session.processing_mode,
            'judge_approved': session.judge_feedback == "APPROVED" if session.judge_feedback else False
        }
        
        # Generate pattern ID based on text characteristics and language pair
        pattern_content = f"{session.source_language}-{session.target_language}_{text_chars['complexity_score']:.2f}_{text_chars['length']}"
        pattern_id = generate_pattern_id(pattern_content)
        
        pattern = SuccessfulPattern(
            pattern_id=pattern_id,
            language_pair=f"{session.source_language}-{session.target_language}",
            text_pattern=self._generate_text_pattern(session.original_text, text_chars),
            successful_approach=successful_approach,
            quality_score=session.quality_score or self._estimate_quality_score(session),
            usage_count=1,
            last_used=datetime.now(),
            text_characteristics=text_chars
        )
        
        return pattern
    
    def _is_high_quality(self, session: TranslationSession) -> bool:
        """Determine if a translation session represents high quality"""
        # Judge approved
        if session.judge_feedback == "APPROVED":
            return True
        
        # High quality score
        if session.quality_score and session.quality_score >= 0.8:
            return True
        
        # No judge feedback but no retry needed
        if not session.judge_feedback and session.initial_translation == session.final_translation:
            return True
        
        return False
    
    def _extract_prompt_structure(self, session: TranslationSession) -> Dict[str, Any]:
        """Extract the structure of successful prompts"""
        return {
            'used_dictionary': session.context_files_used.get('dictionary', False),
            'used_grammar': session.context_files_used.get('grammar', False),
            'used_examples': session.context_files_used.get('examples', False),
            'processing_time': session.performance_metrics.get('processing_time', 0),
            'retry_count': 1 if session.initial_translation != session.final_translation else 0
        }
    
    def _generate_text_pattern(self, text: str, characteristics: Dict[str, Any]) -> str:
        """Generate a pattern description for the text"""
        patterns = []
        
        if characteristics['length'] < 50:
            patterns.append('short_text')
        elif characteristics['length'] < 200:
            patterns.append('medium_text')
        else:
            patterns.append('long_text')
        
        if characteristics['has_numbers']:
            patterns.append('contains_numbers')
        
        if characteristics['complexity_score'] > 0.8:
            patterns.append('high_complexity')
        elif characteristics['complexity_score'] > 0.5:
            patterns.append('medium_complexity')
        else:
            patterns.append('low_complexity')
        
        if characteristics['sentence_count'] > 3:
            patterns.append('multi_sentence')
        
        return '_'.join(patterns)
    
    def _estimate_quality_score(self, session: TranslationSession) -> float:
        """Estimate quality score based on available metrics"""
        score = 0.7  # Base score
        
        # Judge approved
        if session.judge_feedback == "APPROVED":
            score = 0.95
        elif session.judge_feedback and "good" in session.judge_feedback.lower():
            score = 0.85
        elif session.judge_feedback:
            score = 0.6  # Had issues but was corrected
        
        # No retry needed
        if session.initial_translation == session.final_translation:
            score += 0.1
        
        # Used context effectively
        context_count = sum(session.context_files_used.values())
        if context_count > 0:
            score += 0.05 * context_count
        
        return min(score, 1.0)

class PerformanceTracker:
    """Tracks and analyzes model performance metrics"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
    
    def track_session_performance(self, session: TranslationSession) -> List[ModelPerformance]:
        """Track performance metrics for models used in a session"""
        performances = []
        
        for model_type, model_name in session.models_used.items():
            # Calculate success rate (1.0 if approved, 0.5 if corrected, 0.0 if failed)
            success_rate = self._calculate_success_rate(session)
            
            # Get processing time
            processing_time = session.performance_metrics.get('processing_time', 0)
            
            # Get quality score
            quality_score = session.quality_score or self._estimate_quality_score(session)
            
            performance = ModelPerformance(
                performance_id=str(uuid.uuid4()),
                model_name=model_name,
                language_pair=f"{session.source_language}-{session.target_language}",
                task_type=model_type,
                success_rate=success_rate,
                avg_quality_score=quality_score,
                avg_processing_time=processing_time,
                total_uses=1,
                last_updated=datetime.now()
            )
            
            performances.append(performance)
        
        return performances
    
    def _calculate_success_rate(self, session: TranslationSession) -> float:
        """Calculate success rate for a session"""
        if session.judge_feedback == "APPROVED":
            return 1.0
        elif session.judge_feedback and session.initial_translation != session.final_translation:
            return 0.7  # Corrected after feedback
        elif not session.judge_feedback:
            return 0.8  # No issues detected
        else:
            return 0.3  # Had issues
    
    def _estimate_quality_score(self, session: TranslationSession) -> float:
        """Estimate quality score (same as PatternLearner)"""
        score = 0.7
        
        if session.judge_feedback == "APPROVED":
            score = 0.95
        elif session.judge_feedback and "good" in session.judge_feedback.lower():
            score = 0.85
        elif session.judge_feedback:
            score = 0.6
        
        if session.initial_translation == session.final_translation:
            score += 0.1
        
        context_count = sum(session.context_files_used.values())
        if context_count > 0:
            score += 0.05 * context_count
        
        return min(score, 1.0)
    
    def get_best_model_for_task(self, language_pair: str, task_type: str) -> Optional[str]:
        """Get the best performing model for a specific task and language pair"""
        performances = self.memory_manager.get_model_performance(
            language_pair=language_pair, 
            task_type=task_type
        )
        
        if not performances:
            return None
        
        # Sort by weighted score (quality * success_rate * usage_factor)
        def score_model(perf):
            usage_factor = min(perf.total_uses / 10, 1.0)  # Cap at 10 uses for full confidence
            return perf.avg_quality_score * perf.success_rate * usage_factor
        
        best_performance = max(performances, key=score_model)
        return best_performance.model_name

class LearningEngine:
    """Main learning engine that coordinates all learning components"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.feedback_analyzer = FeedbackAnalyzer(memory_manager)
        self.pattern_learner = PatternLearner(memory_manager)
        self.performance_tracker = PerformanceTracker(memory_manager)
    
    def learn_from_session(self, session: TranslationSession) -> Dict[str, Any]:
        """Learn from a complete translation session"""
        learning_results = {
            'feedback_patterns': [],
            'successful_patterns': [],
            'performance_updates': [],
            'errors': []
        }
        
        try:
            # Store the session
            if self.memory_manager.store_translation_session(session):
                logger.info(f"Stored translation session: {session.session_id}")
            
            # Analyze feedback patterns
            if session.judge_feedback and session.judge_feedback != "APPROVED":
                feedback_patterns = self.feedback_analyzer.analyze_feedback(
                    session.judge_feedback,
                    f"{session.source_language}-{session.target_language}",
                    session
                )
                
                for pattern in feedback_patterns:
                    if self.memory_manager.store_feedback_pattern(pattern):
                        learning_results['feedback_patterns'].append(pattern.pattern_id)
                    else:
                        learning_results['errors'].append(f"Failed to store feedback pattern: {pattern.pattern_id}")
            
            # Learn successful patterns
            successful_pattern = self.pattern_learner.learn_from_session(session)
            if successful_pattern:
                if self.memory_manager.store_successful_pattern(successful_pattern):
                    learning_results['successful_patterns'].append(successful_pattern.pattern_id)
                else:
                    learning_results['errors'].append(f"Failed to store successful pattern: {successful_pattern.pattern_id}")
            
            # Track performance
            performances = self.performance_tracker.track_session_performance(session)
            for performance in performances:
                if self.memory_manager.update_model_performance(performance):
                    learning_results['performance_updates'].append(f"{performance.model_name}_{performance.task_type}")
                else:
                    learning_results['errors'].append(f"Failed to update performance: {performance.model_name}")
            
        except Exception as e:
            logger.error(f"Error in learning engine: {e}")
            learning_results['errors'].append(str(e))
        
        return learning_results
    
    def get_recommendations(self, language_pair: str, text_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendations based on learned patterns"""
        recommendations = {
            'avoid_issues': [],
            'suggested_approaches': [],
            'best_models': {},
            'context_recommendations': {}
        }
        
        try:
            # Get feedback patterns to avoid
            feedback_patterns = self.memory_manager.get_feedback_patterns(language_pair)
            for pattern in feedback_patterns[:3]:  # Top 3 most common issues
                recommendations['avoid_issues'].extend(pattern.common_issues)
            
            # Get successful patterns for similar text
            successful_patterns = self.memory_manager.get_successful_patterns(language_pair, text_characteristics)
            for pattern in successful_patterns[:2]:  # Top 2 most successful approaches
                recommendations['suggested_approaches'].append(pattern.successful_approach)
            
            # Get best models for different tasks
            for task_type in ['translation_model', 'fixed_model']:
                best_model = self.performance_tracker.get_best_model_for_task(language_pair, task_type)
                if best_model:
                    recommendations['best_models'][task_type] = best_model
            
            # Context recommendations based on successful patterns
            if successful_patterns:
                context_usage = {}
                for pattern in successful_patterns:
                    approach = pattern.successful_approach
                    for context_type in ['dictionary', 'grammar', 'examples']:
                        if approach.get('context_usage', {}).get(context_type, False):
                            context_usage[context_type] = context_usage.get(context_type, 0) + pattern.usage_count
                
                # Recommend contexts used in successful patterns
                total_usage = sum(context_usage.values())
                if total_usage > 0:
                    for context_type, usage in context_usage.items():
                        recommendations['context_recommendations'][context_type] = usage / total_usage
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
        
        return recommendations