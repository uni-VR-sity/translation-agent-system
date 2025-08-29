import sqlite3
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class TranslationSession:
    """Data class for translation session records"""
    session_id: str
    timestamp: datetime
    source_language: str
    target_language: str
    original_text: str
    initial_translation: str
    final_translation: str
    judge_feedback: Optional[str]
    processing_mode: str  # 'single' or 'corpus'
    models_used: Dict[str, str]
    context_files_used: Dict[str, bool]
    performance_metrics: Dict[str, Any]
    quality_score: Optional[float] = None

@dataclass
class FeedbackPattern:
    """Data class for judge feedback patterns"""
    pattern_id: str
    language_pair: str
    feedback_type: str
    common_issues: List[str]
    improvement_suggestions: List[str]
    frequency_count: int
    last_seen: datetime
    confidence_score: float = 0.0

@dataclass
class SuccessfulPattern:
    """Data class for successful translation patterns"""
    pattern_id: str
    language_pair: str
    text_pattern: str
    successful_approach: Dict[str, Any]
    quality_score: float
    usage_count: int
    last_used: datetime
    text_characteristics: Dict[str, Any]

@dataclass
class ModelPerformance:
    """Data class for model performance metrics"""
    performance_id: str
    model_name: str
    language_pair: str
    task_type: str
    success_rate: float
    avg_quality_score: float
    avg_processing_time: float
    total_uses: int
    last_updated: datetime

class MemoryManager:
    """Central manager for translation memory system"""
    
    def __init__(self, db_path: str = "./data/translation_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Translation sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS translation_sessions (
                    session_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    source_language TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    original_text TEXT NOT NULL,
                    initial_translation TEXT NOT NULL,
                    final_translation TEXT NOT NULL,
                    judge_feedback TEXT,
                    processing_mode TEXT NOT NULL,
                    models_used TEXT NOT NULL,  -- JSON
                    context_files_used TEXT NOT NULL,  -- JSON
                    performance_metrics TEXT NOT NULL,  -- JSON
                    quality_score REAL
                )
            """)
            
            # Feedback patterns table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    language_pair TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    common_issues TEXT NOT NULL,  -- JSON array
                    improvement_suggestions TEXT NOT NULL,  -- JSON array
                    frequency_count INTEGER NOT NULL DEFAULT 1,
                    last_seen TEXT NOT NULL,
                    confidence_score REAL DEFAULT 0.0
                )
            """)
            
            # Successful patterns table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS successful_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    language_pair TEXT NOT NULL,
                    text_pattern TEXT NOT NULL,
                    successful_approach TEXT NOT NULL,  -- JSON
                    quality_score REAL NOT NULL,
                    usage_count INTEGER NOT NULL DEFAULT 1,
                    last_used TEXT NOT NULL,
                    text_characteristics TEXT NOT NULL  -- JSON
                )
            """)
            
            # Model performance table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    performance_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    language_pair TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    success_rate REAL NOT NULL,
                    avg_quality_score REAL NOT NULL,
                    avg_processing_time REAL NOT NULL,
                    total_uses INTEGER NOT NULL DEFAULT 1,
                    last_updated TEXT NOT NULL
                )
            """)
            
            # Create indexes for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_language_pair ON translation_sessions(source_language, target_language)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON translation_sessions(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_language_pair ON feedback_patterns(language_pair)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_patterns_language_pair ON successful_patterns(language_pair)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_model ON model_performance(model_name, language_pair)")
            
            conn.commit()
            
    def store_translation_session(self, session: TranslationSession) -> bool:
        """Store a translation session in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO translation_sessions 
                    (session_id, timestamp, source_language, target_language, original_text,
                     initial_translation, final_translation, judge_feedback, processing_mode,
                     models_used, context_files_used, performance_metrics, quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.timestamp.isoformat(),
                    session.source_language,
                    session.target_language,
                    session.original_text,
                    session.initial_translation,
                    session.final_translation,
                    session.judge_feedback,
                    session.processing_mode,
                    json.dumps(session.models_used),
                    json.dumps(session.context_files_used),
                    json.dumps(session.performance_metrics),
                    session.quality_score
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error storing translation session: {e}")
            return False
    
    def store_feedback_pattern(self, pattern: FeedbackPattern) -> bool:
        """Store or update a feedback pattern"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if pattern exists
                existing = conn.execute(
                    "SELECT frequency_count FROM feedback_patterns WHERE pattern_id = ?",
                    (pattern.pattern_id,)
                ).fetchone()
                
                if existing:
                    # Update existing pattern
                    conn.execute("""
                        UPDATE feedback_patterns 
                        SET frequency_count = frequency_count + 1,
                            last_seen = ?,
                            confidence_score = ?
                        WHERE pattern_id = ?
                    """, (
                        pattern.last_seen.isoformat(),
                        pattern.confidence_score,
                        pattern.pattern_id
                    ))
                else:
                    # Insert new pattern
                    conn.execute("""
                        INSERT INTO feedback_patterns 
                        (pattern_id, language_pair, feedback_type, common_issues,
                         improvement_suggestions, frequency_count, last_seen, confidence_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pattern.pattern_id,
                        pattern.language_pair,
                        pattern.feedback_type,
                        json.dumps(pattern.common_issues),
                        json.dumps(pattern.improvement_suggestions),
                        pattern.frequency_count,
                        pattern.last_seen.isoformat(),
                        pattern.confidence_score
                    ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error storing feedback pattern: {e}")
            return False
    
    def store_successful_pattern(self, pattern: SuccessfulPattern) -> bool:
        """Store or update a successful translation pattern"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if pattern exists
                existing = conn.execute(
                    "SELECT usage_count FROM successful_patterns WHERE pattern_id = ?",
                    (pattern.pattern_id,)
                ).fetchone()
                
                if existing:
                    # Update existing pattern
                    conn.execute("""
                        UPDATE successful_patterns 
                        SET usage_count = usage_count + 1,
                            last_used = ?,
                            quality_score = (quality_score + ?) / 2
                        WHERE pattern_id = ?
                    """, (
                        pattern.last_used.isoformat(),
                        pattern.quality_score,
                        pattern.pattern_id
                    ))
                else:
                    # Insert new pattern
                    conn.execute("""
                        INSERT INTO successful_patterns 
                        (pattern_id, language_pair, text_pattern, successful_approach,
                         quality_score, usage_count, last_used, text_characteristics)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pattern.pattern_id,
                        pattern.language_pair,
                        pattern.text_pattern,
                        json.dumps(pattern.successful_approach),
                        pattern.quality_score,
                        pattern.usage_count,
                        pattern.last_used.isoformat(),
                        json.dumps(pattern.text_characteristics)
                    ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error storing successful pattern: {e}")
            return False
    
    def update_model_performance(self, performance: ModelPerformance) -> bool:
        """Update model performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if performance record exists
                existing = conn.execute("""
                    SELECT total_uses, avg_quality_score, avg_processing_time, success_rate 
                    FROM model_performance 
                    WHERE model_name = ? AND language_pair = ? AND task_type = ?
                """, (performance.model_name, performance.language_pair, performance.task_type)).fetchone()
                
                if existing:
                    # Calculate weighted averages
                    old_uses, old_quality, old_time, old_success = existing
                    total_uses = old_uses + 1
                    new_quality = (old_quality * old_uses + performance.avg_quality_score) / total_uses
                    new_time = (old_time * old_uses + performance.avg_processing_time) / total_uses
                    new_success = (old_success * old_uses + performance.success_rate) / total_uses
                    
                    conn.execute("""
                        UPDATE model_performance 
                        SET success_rate = ?, avg_quality_score = ?, avg_processing_time = ?,
                            total_uses = ?, last_updated = ?
                        WHERE model_name = ? AND language_pair = ? AND task_type = ?
                    """, (
                        new_success, new_quality, new_time, total_uses,
                        performance.last_updated.isoformat(),
                        performance.model_name, performance.language_pair, performance.task_type
                    ))
                else:
                    # Insert new performance record
                    conn.execute("""
                        INSERT INTO model_performance 
                        (performance_id, model_name, language_pair, task_type, success_rate,
                         avg_quality_score, avg_processing_time, total_uses, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        performance.performance_id,
                        performance.model_name,
                        performance.language_pair,
                        performance.task_type,
                        performance.success_rate,
                        performance.avg_quality_score,
                        performance.avg_processing_time,
                        performance.total_uses,
                        performance.last_updated.isoformat()
                    ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
            return False
    
    def get_feedback_patterns(self, language_pair: str, limit: int = 10) -> List[FeedbackPattern]:
        """Retrieve feedback patterns for a language pair"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute("""
                    SELECT * FROM feedback_patterns 
                    WHERE language_pair = ? 
                    ORDER BY frequency_count DESC, confidence_score DESC 
                    LIMIT ?
                """, (language_pair, limit)).fetchall()
                
                patterns = []
                for row in rows:
                    pattern = FeedbackPattern(
                        pattern_id=row[0],
                        language_pair=row[1],
                        feedback_type=row[2],
                        common_issues=json.loads(row[3]),
                        improvement_suggestions=json.loads(row[4]),
                        frequency_count=row[5],
                        last_seen=datetime.fromisoformat(row[6]),
                        confidence_score=row[7]
                    )
                    patterns.append(pattern)
                return patterns
        except Exception as e:
            logger.error(f"Error retrieving feedback patterns: {e}")
            return []
    
    def get_successful_patterns(self, language_pair: str, text_characteristics: Dict[str, Any], limit: int = 5) -> List[SuccessfulPattern]:
        """Retrieve successful patterns for similar text characteristics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute("""
                    SELECT * FROM successful_patterns 
                    WHERE language_pair = ? 
                    ORDER BY quality_score DESC, usage_count DESC 
                    LIMIT ?
                """, (language_pair, limit)).fetchall()
                
                patterns = []
                for row in rows:
                    pattern = SuccessfulPattern(
                        pattern_id=row[0],
                        language_pair=row[1],
                        text_pattern=row[2],
                        successful_approach=json.loads(row[3]),
                        quality_score=row[4],
                        usage_count=row[5],
                        last_used=datetime.fromisoformat(row[6]),
                        text_characteristics=json.loads(row[7])
                    )
                    patterns.append(pattern)
                return patterns
        except Exception as e:
            logger.error(f"Error retrieving successful patterns: {e}")
            return []
    
    def get_model_performance(self, model_name: str = None, language_pair: str = None, task_type: str = None) -> List[ModelPerformance]:
        """Retrieve model performance data with optional filters"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM model_performance WHERE 1=1"
                params = []
                
                if model_name:
                    query += " AND model_name = ?"
                    params.append(model_name)
                if language_pair:
                    query += " AND language_pair = ?"
                    params.append(language_pair)
                if task_type:
                    query += " AND task_type = ?"
                    params.append(task_type)
                
                query += " ORDER BY avg_quality_score DESC, success_rate DESC"
                
                rows = conn.execute(query, params).fetchall()
                
                performances = []
                for row in rows:
                    performance = ModelPerformance(
                        performance_id=row[0],
                        model_name=row[1],
                        language_pair=row[2],
                        task_type=row[3],
                        success_rate=row[4],
                        avg_quality_score=row[5],
                        avg_processing_time=row[6],
                        total_uses=row[7],
                        last_updated=datetime.fromisoformat(row[8])
                    )
                    performances.append(performance)
                return performances
        except Exception as e:
            logger.error(f"Error retrieving model performance: {e}")
            return []
    
    def get_translation_history(self, language_pair: str = None, days: int = 30) -> List[TranslationSession]:
        """Retrieve translation history with optional filters"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_date = datetime.now() - timedelta(days=days)
                
                if language_pair:
                    rows = conn.execute("""
                        SELECT * FROM translation_sessions 
                        WHERE source_language || '-' || target_language = ? 
                        AND timestamp > ?
                        ORDER BY timestamp DESC
                    """, (language_pair, cutoff_date.isoformat())).fetchall()
                else:
                    rows = conn.execute("""
                        SELECT * FROM translation_sessions 
                        WHERE timestamp > ?
                        ORDER BY timestamp DESC
                    """, (cutoff_date.isoformat(),)).fetchall()
                
                sessions = []
                for row in rows:
                    session = TranslationSession(
                        session_id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        source_language=row[2],
                        target_language=row[3],
                        original_text=row[4],
                        initial_translation=row[5],
                        final_translation=row[6],
                        judge_feedback=row[7],
                        processing_mode=row[8],
                        models_used=json.loads(row[9]),
                        context_files_used=json.loads(row[10]),
                        performance_metrics=json.loads(row[11]),
                        quality_score=row[12]
                    )
                    sessions.append(session)
                return sessions
        except Exception as e:
            logger.error(f"Error retrieving translation history: {e}")
            return []
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to maintain database size"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                
                # Clean old sessions
                conn.execute("""
                    DELETE FROM translation_sessions 
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                
                # Clean old feedback patterns with low frequency
                conn.execute("""
                    DELETE FROM feedback_patterns 
                    WHERE last_seen < ? AND frequency_count < 3
                """, (cutoff_date.isoformat(),))
                
                # Clean old successful patterns with low usage
                conn.execute("""
                    DELETE FROM successful_patterns 
                    WHERE last_used < ? AND usage_count < 2
                """, (cutoff_date.isoformat(),))
                
                conn.commit()
                logger.info(f"Cleaned up data older than {days_to_keep} days")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def generate_pattern_id(content: str) -> str:
    """Generate a unique pattern ID based on content hash"""
    return hashlib.md5(content.encode()).hexdigest()[:16]

def calculate_text_characteristics(text: str) -> Dict[str, Any]:
    """Calculate characteristics of input text for pattern matching"""
    return {
        "length": len(text),
        "word_count": len(text.split()),
        "sentence_count": text.count('.') + text.count('!') + text.count('?'),
        "has_numbers": any(char.isdigit() for char in text),
        "has_punctuation": any(char in ".,!?;:" for char in text),
        "avg_word_length": sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0,
        "complexity_score": len(set(text.lower().split())) / len(text.split()) if text.split() else 0
    }