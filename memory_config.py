import os
from typing import Dict, Any
from pathlib import Path

class MemoryConfig:
    """Configuration settings for the translation memory system"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        # Default configuration
        self.config = {
            # Database settings
            "database_path": "./data/translation_memory.db",
            "backup_database": True,
            "backup_interval_hours": 24,
            
            # Learning settings
            "max_patterns_per_language_pair": 1000,
            "pattern_similarity_threshold": 0.8,
            "min_pattern_frequency": 3,
            "learning_rate": 0.1,
            "confidence_threshold": 0.6,
            
            # Memory management
            "memory_cleanup_interval_days": 7,
            "max_session_history_days": 90,
            "max_feedback_patterns": 500,
            "max_successful_patterns": 300,
            
            # Performance optimization
            "enable_adaptive_prompts": True,
            "enable_smart_context_selection": True,
            "enable_model_optimization": True,
            "context_optimization_threshold": 0.3,
            
            # Analytics and reporting
            "enable_analytics": True,
            "analytics_retention_days": 30,
            "performance_tracking": True,
            
            # Quality thresholds
            "high_quality_threshold": 0.8,
            "low_quality_threshold": 0.4,
            "judge_approval_weight": 1.0,
            "retry_penalty": 0.2,
            
            # Context selection weights
            "dictionary_weight": 0.3,
            "grammar_weight": 0.4,
            "examples_weight": 0.3,
            
            # Model selection criteria
            "quality_weight": 0.6,
            "speed_weight": 0.2,
            "reliability_weight": 0.2,
            
            # Feedback analysis
            "feedback_analysis_frequency": "daily",
            "min_feedback_length": 10,
            "max_feedback_patterns_per_type": 50,
            
            # Pattern matching
            "text_similarity_threshold": 0.7,
            "language_pair_specific_learning": True,
            "cross_language_learning": False,
            
            # System limits
            "max_concurrent_learning_tasks": 5,
            "max_memory_usage_mb": 500,
            "cache_size_limit": 1000,
            
            # Logging and debugging
            "log_level": "INFO",
            "log_memory_operations": False,
            "log_learning_results": True,
            "debug_mode": False
        }
        
        # Override with provided config
        if config_dict:
            self.config.update(config_dict)
        
        # Load from environment variables if available
        self._load_from_env()
        
        # Validate configuration
        self._validate_config()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        env_mappings = {
            "MEMORY_DB_PATH": "database_path",
            "MEMORY_CLEANUP_DAYS": "memory_cleanup_interval_days",
            "MEMORY_HISTORY_DAYS": "max_session_history_days",
            "MEMORY_LEARNING_RATE": "learning_rate",
            "MEMORY_QUALITY_THRESHOLD": "high_quality_threshold",
            "MEMORY_DEBUG": "debug_mode",
            "MEMORY_LOG_LEVEL": "log_level"
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert to appropriate type
                if config_key in ["memory_cleanup_interval_days", "max_session_history_days"]:
                    self.config[config_key] = int(value)
                elif config_key in ["learning_rate", "high_quality_threshold"]:
                    self.config[config_key] = float(value)
                elif config_key == "debug_mode":
                    self.config[config_key] = value.lower() in ["true", "1", "yes"]
                else:
                    self.config[config_key] = value
    
    def _validate_config(self):
        """Validate configuration values"""
        # Ensure database directory exists
        db_path = Path(self.config["database_path"])
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate numeric ranges
        assert 0 < self.config["learning_rate"] <= 1, "Learning rate must be between 0 and 1"
        assert 0 < self.config["high_quality_threshold"] <= 1, "Quality threshold must be between 0 and 1"
        assert self.config["min_pattern_frequency"] >= 1, "Minimum pattern frequency must be at least 1"
        assert self.config["max_session_history_days"] > 0, "History retention must be positive"
        
        # Validate weights sum to 1
        context_weights = [
            self.config["dictionary_weight"],
            self.config["grammar_weight"],
            self.config["examples_weight"]
        ]
        assert abs(sum(context_weights) - 1.0) < 0.01, "Context weights must sum to 1.0"
        
        model_weights = [
            self.config["quality_weight"],
            self.config["speed_weight"],
            self.config["reliability_weight"]
        ]
        assert abs(sum(model_weights) - 1.0) < 0.01, "Model selection weights must sum to 1.0"
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value
        self._validate_config()
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values"""
        self.config.update(updates)
        self._validate_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return self.config.copy()
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load configuration from JSON file"""
        import json
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                file_config = json.load(f)
                self.config.update(file_config)
                self._validate_config()

# Global configuration instance
memory_config = MemoryConfig()

def get_memory_config() -> MemoryConfig:
    """Get the global memory configuration instance"""
    return memory_config

def update_memory_config(updates: Dict[str, Any]):
    """Update the global memory configuration"""
    memory_config.update(updates)

# Configuration presets for different use cases
PRESETS = {
    "development": {
        "debug_mode": True,
        "log_memory_operations": True,
        "memory_cleanup_interval_days": 1,
        "max_session_history_days": 7,
        "min_pattern_frequency": 1,
        "log_level": "DEBUG"
    },
    
    "production": {
        "debug_mode": False,
        "log_memory_operations": False,
        "memory_cleanup_interval_days": 7,
        "max_session_history_days": 90,
        "min_pattern_frequency": 3,
        "log_level": "INFO",
        "backup_database": True
    },
    
    "high_performance": {
        "enable_adaptive_prompts": True,
        "enable_smart_context_selection": True,
        "enable_model_optimization": True,
        "max_concurrent_learning_tasks": 10,
        "cache_size_limit": 2000,
        "learning_rate": 0.15
    },
    
    "conservative": {
        "learning_rate": 0.05,
        "min_pattern_frequency": 5,
        "confidence_threshold": 0.8,
        "high_quality_threshold": 0.9,
        "context_optimization_threshold": 0.5
    },
    
    "aggressive_learning": {
        "learning_rate": 0.2,
        "min_pattern_frequency": 2,
        "confidence_threshold": 0.4,
        "high_quality_threshold": 0.7,
        "context_optimization_threshold": 0.2,
        "cross_language_learning": True
    }
}

def apply_preset(preset_name: str):
    """Apply a configuration preset"""
    if preset_name in PRESETS:
        memory_config.update(PRESETS[preset_name])
    else:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")

def get_preset_names() -> list:
    """Get list of available preset names"""
    return list(PRESETS.keys())