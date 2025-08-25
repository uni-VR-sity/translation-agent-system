import unittest
import tempfile
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import json

# Import the memory system components
from memory_system import (
    MemoryManager, TranslationSession, FeedbackPattern, SuccessfulPattern, 
    ModelPerformance, generate_pattern_id, calculate_text_characteristics
)
from learning_engine import LearningEngine, FeedbackAnalyzer, PatternLearner, PerformanceTracker
from adaptive_components import AdaptivePromptEnhancer, SmartContextSelector, ModelOptimizer
from memory_config import MemoryConfig, apply_preset
from memory_analytics import MemoryAnalytics, MemoryMaintenance

class TestMemorySystem(unittest.TestCase):
    """Test suite for the memory system components"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary database for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_memory.db")
        
        # Initialize memory manager with test database
        self.memory_manager = MemoryManager(self.test_db_path)
        self.learning_engine = LearningEngine(self.memory_manager)
        
        # Create test data
        self.test_session = TranslationSession(
            session_id="test-session-1",
            timestamp=datetime.now(),
            source_language="English",
            target_language="Spanish",
            original_text="Hello world",
            initial_translation="Hola mundo",
            final_translation="Hola mundo",
            judge_feedback="APPROVED",
            processing_mode="single",
            models_used={"translation_model": "gpt-4", "fixed_model": "gpt-3.5"},
            context_files_used={"dictionary": True, "grammar": False, "examples": True},
            performance_metrics={"processing_time": 2.5, "retry_needed": False},
            quality_score=0.95
        )
    
    def tearDown(self):
        """Clean up test environment"""
        # Remove temporary files
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        os.rmdir(self.temp_dir)

class TestMemoryManager(TestMemorySystem):
    """Test MemoryManager functionality"""
    
    def test_database_initialization(self):
        """Test database initialization"""
        # Check if database file exists
        self.assertTrue(os.path.exists(self.test_db_path))
        
        # Check if tables are created
        with sqlite3.connect(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = [
                'translation_sessions', 'feedback_patterns', 
                'successful_patterns', 'model_performance'
            ]
            
            for table in expected_tables:
                self.assertIn(table, tables)
    
    def test_store_translation_session(self):
        """Test storing translation sessions"""
        # Store test session
        result = self.memory_manager.store_translation_session(self.test_session)
        self.assertTrue(result)
        
        # Retrieve and verify
        sessions = self.memory_manager.get_translation_history(days=1)
        self.assertEqual(len(sessions), 1)
        
        retrieved_session = sessions[0]
        self.assertEqual(retrieved_session.session_id, self.test_session.session_id)
        self.assertEqual(retrieved_session.source_language, self.test_session.source_language)
        self.assertEqual(retrieved_session.quality_score, self.test_session.quality_score)
    
    def test_store_feedback_pattern(self):
        """Test storing feedback patterns"""
        pattern = FeedbackPattern(
            pattern_id="test-pattern-1",
            language_pair="English-Spanish",
            feedback_type="grammar",
            common_issues=["verb tense", "word order"],
            improvement_suggestions=["use_formal_register", "improve_word_order"],
            frequency_count=1,
            last_seen=datetime.now(),
            confidence_score=0.8
        )
        
        # Store pattern
        result = self.memory_manager.store_feedback_pattern(pattern)
        self.assertTrue(result)
        
        # Retrieve and verify
        patterns = self.memory_manager.get_feedback_patterns("English-Spanish")
        self.assertEqual(len(patterns), 1)
        
        retrieved_pattern = patterns[0]
        self.assertEqual(retrieved_pattern.pattern_id, pattern.pattern_id)
        self.assertEqual(retrieved_pattern.feedback_type, pattern.feedback_type)
        self.assertEqual(retrieved_pattern.common_issues, pattern.common_issues)
    
    def test_store_successful_pattern(self):
        """Test storing successful patterns"""
        pattern = SuccessfulPattern(
            pattern_id="success-pattern-1",
            language_pair="English-Spanish",
            text_pattern="short_text_low_complexity",
            successful_approach={
                "used_dictionary": True,
                "used_grammar": False,
                "used_examples": True
            },
            quality_score=0.9,
            usage_count=1,
            last_used=datetime.now(),
            text_characteristics={"length": 11, "complexity_score": 0.5}
        )
        
        # Store pattern
        result = self.memory_manager.store_successful_pattern(pattern)
        self.assertTrue(result)
        
        # Retrieve and verify
        patterns = self.memory_manager.get_successful_patterns("English-Spanish", {"length": 10})
        self.assertEqual(len(patterns), 1)
        
        retrieved_pattern = patterns[0]
        self.assertEqual(retrieved_pattern.pattern_id, pattern.pattern_id)
        self.assertEqual(retrieved_pattern.quality_score, pattern.quality_score)
    
    def test_update_model_performance(self):
        """Test updating model performance"""
        performance = ModelPerformance(
            performance_id="perf-1",
            model_name="gpt-4",
            language_pair="English-Spanish",
            task_type="translation_model",
            success_rate=0.95,
            avg_quality_score=0.9,
            avg_processing_time=2.5,
            total_uses=1,
            last_updated=datetime.now()
        )
        
        # Store performance
        result = self.memory_manager.update_model_performance(performance)
        self.assertTrue(result)
        
        # Retrieve and verify
        performances = self.memory_manager.get_model_performance(model_name="gpt-4")
        self.assertEqual(len(performances), 1)
        
        retrieved_perf = performances[0]
        self.assertEqual(retrieved_perf.model_name, performance.model_name)
        self.assertEqual(retrieved_perf.success_rate, performance.success_rate)

class TestLearningEngine(TestMemorySystem):
    """Test LearningEngine functionality"""
    
    def test_feedback_analyzer(self):
        """Test feedback analysis"""
        analyzer = FeedbackAnalyzer(self.memory_manager)
        
        feedback = "The translation has incorrect grammar and wrong word order. Should use more formal register."
        patterns = analyzer.analyze_feedback(feedback, "English-Spanish", self.test_session)
        
        self.assertGreater(len(patterns), 0)
        
        # Check if grammar issues are detected
        grammar_patterns = [p for p in patterns if p.feedback_type == "grammar"]
        self.assertGreater(len(grammar_patterns), 0)
    
    def test_pattern_learner(self):
        """Test pattern learning from successful sessions"""
        learner = PatternLearner(self.memory_manager)
        
        # Test with high-quality session
        pattern = learner.learn_from_session(self.test_session)
        self.assertIsNotNone(pattern)
        self.assertGreater(pattern.quality_score, 0.8)
        
        # Test with low-quality session
        low_quality_session = self.test_session
        low_quality_session.quality_score = 0.3
        low_quality_session.judge_feedback = "Poor translation quality"
        
        pattern = learner.learn_from_session(low_quality_session)
        self.assertIsNone(pattern)  # Should not learn from low-quality sessions
    
    def test_performance_tracker(self):
        """Test performance tracking"""
        tracker = PerformanceTracker(self.memory_manager)
        
        performances = tracker.track_session_performance(self.test_session)
        self.assertEqual(len(performances), 2)  # One for each model used
        
        # Check translation model performance
        translation_perf = [p for p in performances if p.task_type == "translation_model"][0]
        self.assertEqual(translation_perf.model_name, "gpt-4")
        self.assertGreater(translation_perf.success_rate, 0.8)
    
    def test_learning_from_session(self):
        """Test complete learning process"""
        # Create session with feedback
        session_with_feedback = self.test_session
        session_with_feedback.judge_feedback = "Good translation but could improve word choice"
        
        # Learn from session
        results = self.learning_engine.learn_from_session(session_with_feedback)
        
        # Check results
        self.assertIn('feedback_patterns', results)
        self.assertIn('successful_patterns', results)
        self.assertIn('performance_updates', results)
        
        # Verify data was stored
        feedback_patterns = self.memory_manager.get_feedback_patterns("English-Spanish")
        self.assertGreater(len(feedback_patterns), 0)

class TestAdaptiveComponents(TestMemorySystem):
    """Test adaptive components"""
    
    def setUp(self):
        super().setUp()
        self.prompt_enhancer = AdaptivePromptEnhancer(self.memory_manager, self.learning_engine)
        self.context_selector = SmartContextSelector(self.memory_manager, self.learning_engine)
        self.model_optimizer = ModelOptimizer(self.memory_manager, self.learning_engine)
    
    def test_prompt_enhancement(self):
        """Test prompt enhancement"""
        base_prompt = "Translate the following text from English to Spanish: Hello world"
        
        enhanced_prompt = self.prompt_enhancer.enhance_translation_prompt(
            base_prompt, "English", "Spanish", "Hello world"
        )
        
        # Enhanced prompt should be longer than base prompt
        self.assertGreaterEqual(len(enhanced_prompt), len(base_prompt))
        self.assertIn("Hello world", enhanced_prompt)
    
    def test_context_optimization(self):
        """Test context optimization"""
        available_contexts = {
            'dictionary': [["hello", "hola"], ["world", "mundo"]],
            'grammar': ["Use formal register", "Maintain word order"],
            'examples': ["Hello -> Hola", "World -> Mundo"]
        }
        
        optimized = self.context_selector.optimize_context_usage(
            "English-Spanish", "Hello world", available_contexts
        )
        
        # Should return optimized contexts
        self.assertIn('dictionary', optimized)
        self.assertIn('grammar', optimized)
        self.assertIn('examples', optimized)
    
    def test_model_optimization(self):
        """Test model optimization"""
        available_models = {
            'translation_model': ['gpt-4', 'gpt-3.5', 'claude-3'],
            'fixed_model': ['gpt-3.5', 'gpt-4']
        }
        
        recommendations = self.model_optimizer.recommend_optimal_models(
            "English-Spanish", 
            calculate_text_characteristics("Hello world"),
            available_models
        )
        
        # Should return recommendations for both model types
        self.assertIn('translation_model', recommendations)
        self.assertIn('fixed_model', recommendations)
        
        # Recommendations should be from available models
        self.assertIn(recommendations['translation_model'], available_models['translation_model'])
        self.assertIn(recommendations['fixed_model'], available_models['fixed_model'])

class TestMemoryConfig(unittest.TestCase):
    """Test memory configuration"""
    
    def test_config_initialization(self):
        """Test configuration initialization"""
        config = MemoryConfig()
        
        # Check default values
        self.assertEqual(config.get('learning_rate'), 0.1)
        self.assertEqual(config.get('high_quality_threshold'), 0.8)
        self.assertTrue(config.get('enable_adaptive_prompts'))
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid learning rate
        with self.assertRaises(AssertionError):
            MemoryConfig({'learning_rate': 1.5})
        
        # Test invalid quality threshold
        with self.assertRaises(AssertionError):
            MemoryConfig({'high_quality_threshold': 1.5})
    
    def test_preset_application(self):
        """Test preset application"""
        # Apply development preset
        apply_preset('development')
        config = MemoryConfig()
        
        # Should have development-specific settings
        self.assertTrue(config.get('debug_mode'))
        self.assertEqual(config.get('log_level'), 'DEBUG')

class TestMemoryAnalytics(TestMemorySystem):
    """Test memory analytics"""
    
    def setUp(self):
        super().setUp()
        self.analytics = MemoryAnalytics(self.memory_manager)
        
        # Add some test data
        self.memory_manager.store_translation_session(self.test_session)
    
    def test_performance_report_generation(self):
        """Test performance report generation"""
        report = self.analytics.generate_performance_report("English-Spanish", days=30)
        
        # Check report structure
        self.assertIn('summary', report)
        self.assertIn('quality_trends', report)
        self.assertIn('model_performance', report)
        self.assertIn('recommendations', report)
        
        # Check summary statistics
        summary = report['summary']
        self.assertEqual(summary['total_sessions'], 1)
        self.assertGreater(summary['approval_rate'], 0)
    
    def test_database_stats(self):
        """Test database statistics"""
        stats = self.analytics.get_database_stats()
        
        # Check stats structure
        self.assertIn('translation_sessions_count', stats)
        self.assertIn('database_size_mb', stats)
        
        # Should have at least one session
        self.assertGreaterEqual(stats['translation_sessions_count'], 1)

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_pattern_id_generation(self):
        """Test pattern ID generation"""
        content1 = "test content"
        content2 = "test content"
        content3 = "different content"
        
        id1 = generate_pattern_id(content1)
        id2 = generate_pattern_id(content2)
        id3 = generate_pattern_id(content3)
        
        # Same content should generate same ID
        self.assertEqual(id1, id2)
        
        # Different content should generate different ID
        self.assertNotEqual(id1, id3)
        
        # ID should be 16 characters
        self.assertEqual(len(id1), 16)
    
    def test_text_characteristics_calculation(self):
        """Test text characteristics calculation"""
        text = "Hello world! This is a test sentence with numbers 123."
        chars = calculate_text_characteristics(text)
        
        # Check required characteristics
        self.assertIn('length', chars)
        self.assertIn('word_count', chars)
        self.assertIn('sentence_count', chars)
        self.assertIn('has_numbers', chars)
        self.assertIn('has_punctuation', chars)
        self.assertIn('complexity_score', chars)
        
        # Verify values
        self.assertEqual(chars['length'], len(text))
        self.assertTrue(chars['has_numbers'])
        self.assertTrue(chars['has_punctuation'])
        self.assertGreater(chars['word_count'], 0)
        self.assertGreater(chars['sentence_count'], 0)

class TestIntegration(TestMemorySystem):
    """Integration tests for the complete memory system"""
    
    def test_complete_learning_workflow(self):
        """Test complete learning workflow"""
        # 1. Store initial session
        result = self.memory_manager.store_translation_session(self.test_session)
        self.assertTrue(result)
        
        # 2. Learn from session
        learning_results = self.learning_engine.learn_from_session(self.test_session)
        self.assertIn('successful_patterns', learning_results)
        self.assertIn('performance_updates', learning_results)
        
        # 3. Get recommendations
        text_chars = calculate_text_characteristics(self.test_session.original_text)
        recommendations = self.learning_engine.get_recommendations(
            "English-Spanish", text_chars
        )
        
        self.assertIn('suggested_approaches', recommendations)
        self.assertIn('best_models', recommendations)
        
        # 4. Generate analytics report
        analytics = MemoryAnalytics(self.memory_manager)
        report = analytics.generate_performance_report("English-Spanish", days=1)
        
        self.assertIn('summary', report)
        self.assertGreater(report['summary']['total_sessions'], 0)
    
    def test_memory_persistence(self):
        """Test memory persistence across sessions"""
        # Store data
        self.memory_manager.store_translation_session(self.test_session)
        
        # Create new memory manager instance (simulating restart)
        new_memory_manager = MemoryManager(self.test_db_path)
        
        # Verify data persists
        sessions = new_memory_manager.get_translation_history(days=1)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].session_id, self.test_session.session_id)

def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMemoryManager,
        TestLearningEngine,
        TestAdaptiveComponents,
        TestMemoryConfig,
        TestMemoryAnalytics,
        TestUtilityFunctions,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    # Run tests when script is executed directly
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)