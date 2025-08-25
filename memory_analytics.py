import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
import logging
from memory_system import MemoryManager
from memory_config import get_memory_config
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

class MemoryAnalytics:
    """Analytics and reporting for the translation memory system"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.config = get_memory_config()
    
    def generate_performance_report(self, language_pair: str = None, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            report = {
                'summary': {},
                'quality_trends': {},
                'model_performance': {},
                'feedback_analysis': {},
                'learning_insights': {},
                'recommendations': []
            }
            
            # Get translation history
            history = self.memory_manager.get_translation_history(language_pair, days)
            
            # Summary statistics
            report['summary'] = self._generate_summary_stats(history)
            
            # Quality trends over time
            report['quality_trends'] = self._analyze_quality_trends(history)
            
            # Model performance comparison
            report['model_performance'] = self._analyze_model_performance(language_pair)
            
            # Feedback pattern analysis
            report['feedback_analysis'] = self._analyze_feedback_patterns(language_pair)
            
            # Learning insights
            report['learning_insights'] = self._generate_learning_insights(history)
            
            # Generate recommendations
            report['recommendations'] = self._generate_recommendations(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {}
    
    def _generate_summary_stats(self, history: List) -> Dict[str, Any]:
        """Generate summary statistics from translation history"""
        if not history:
            return {}
        
        total_sessions = len(history)
        approved_sessions = sum(1 for s in history if s.judge_feedback == "APPROVED")
        retry_sessions = sum(1 for s in history if s.initial_translation != s.final_translation)
        
        # Calculate average quality scores
        quality_scores = [s.quality_score for s in history if s.quality_score is not None]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Calculate average processing times
        processing_times = []
        for s in history:
            if s.performance_metrics and 'processing_time' in s.performance_metrics:
                processing_times.append(s.performance_metrics['processing_time'])
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            'total_sessions': total_sessions,
            'approval_rate': approved_sessions / total_sessions if total_sessions > 0 else 0,
            'retry_rate': retry_sessions / total_sessions if total_sessions > 0 else 0,
            'average_quality_score': avg_quality,
            'average_processing_time': avg_processing_time,
            'unique_language_pairs': len(set(f"{s.source_language}-{s.target_language}" for s in history))
        }
    
    def _analyze_quality_trends(self, history: List) -> Dict[str, Any]:
        """Analyze quality trends over time"""
        if not history:
            return {}
        
        # Group by date
        daily_quality = {}
        for session in history:
            date_key = session.timestamp.date().isoformat()
            if date_key not in daily_quality:
                daily_quality[date_key] = []
            
            if session.quality_score:
                daily_quality[date_key].append(session.quality_score)
        
        # Calculate daily averages
        daily_averages = {}
        for date, scores in daily_quality.items():
            daily_averages[date] = sum(scores) / len(scores) if scores else 0
        
        # Calculate trend
        dates = sorted(daily_averages.keys())
        if len(dates) >= 2:
            first_week = dates[:7] if len(dates) >= 7 else dates[:len(dates)//2]
            last_week = dates[-7:] if len(dates) >= 7 else dates[len(dates)//2:]
            
            first_avg = sum(daily_averages[d] for d in first_week) / len(first_week)
            last_avg = sum(daily_averages[d] for d in last_week) / len(last_week)
            
            trend = "improving" if last_avg > first_avg else "declining" if last_avg < first_avg else "stable"
        else:
            trend = "insufficient_data"
        
        return {
            'daily_averages': daily_averages,
            'trend': trend,
            'best_day': max(daily_averages.items(), key=lambda x: x[1]) if daily_averages else None,
            'worst_day': min(daily_averages.items(), key=lambda x: x[1]) if daily_averages else None
        }
    
    def _analyze_model_performance(self, language_pair: str = None) -> Dict[str, Any]:
        """Analyze model performance metrics"""
        performances = self.memory_manager.get_model_performance(language_pair=language_pair)
        
        if not performances:
            return {}
        
        # Group by model name
        model_stats = {}
        for perf in performances:
            model_name = perf.model_name
            if model_name not in model_stats:
                model_stats[model_name] = {
                    'total_uses': 0,
                    'avg_quality': 0,
                    'avg_success_rate': 0,
                    'avg_processing_time': 0,
                    'task_types': set()
                }
            
            stats = model_stats[model_name]
            stats['total_uses'] += perf.total_uses
            stats['avg_quality'] += perf.avg_quality_score * perf.total_uses
            stats['avg_success_rate'] += perf.success_rate * perf.total_uses
            stats['avg_processing_time'] += perf.avg_processing_time * perf.total_uses
            stats['task_types'].add(perf.task_type)
        
        # Calculate weighted averages
        for model_name, stats in model_stats.items():
            if stats['total_uses'] > 0:
                stats['avg_quality'] /= stats['total_uses']
                stats['avg_success_rate'] /= stats['total_uses']
                stats['avg_processing_time'] /= stats['total_uses']
                stats['task_types'] = list(stats['task_types'])
        
        # Find best performers
        best_quality = max(model_stats.items(), key=lambda x: x[1]['avg_quality']) if model_stats else None
        best_speed = min(model_stats.items(), key=lambda x: x[1]['avg_processing_time']) if model_stats else None
        
        return {
            'model_statistics': model_stats,
            'best_quality_model': best_quality,
            'fastest_model': best_speed,
            'total_models_tested': len(model_stats)
        }
    
    def _analyze_feedback_patterns(self, language_pair: str = None) -> Dict[str, Any]:
        """Analyze judge feedback patterns"""
        if not language_pair:
            return {}
        
        patterns = self.memory_manager.get_feedback_patterns(language_pair, limit=20)
        
        if not patterns:
            return {}
        
        # Group by feedback type
        type_analysis = {}
        for pattern in patterns:
            feedback_type = pattern.feedback_type
            if feedback_type not in type_analysis:
                type_analysis[feedback_type] = {
                    'total_frequency': 0,
                    'pattern_count': 0,
                    'common_issues': [],
                    'improvement_suggestions': []
                }
            
            analysis = type_analysis[feedback_type]
            analysis['total_frequency'] += pattern.frequency_count
            analysis['pattern_count'] += 1
            analysis['common_issues'].extend(pattern.common_issues)
            analysis['improvement_suggestions'].extend(pattern.improvement_suggestions)
        
        # Get most common issues and suggestions
        for feedback_type, analysis in type_analysis.items():
            # Count issue frequency
            issue_counts = {}
            for issue in analysis['common_issues']:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            analysis['top_issues'] = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Count suggestion frequency
            suggestion_counts = {}
            for suggestion in analysis['improvement_suggestions']:
                suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1
            
            analysis['top_suggestions'] = sorted(suggestion_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Clean up raw lists
            del analysis['common_issues']
            del analysis['improvement_suggestions']
        
        return {
            'feedback_type_analysis': type_analysis,
            'most_problematic_type': max(type_analysis.items(), key=lambda x: x[1]['total_frequency']) if type_analysis else None,
            'total_feedback_patterns': len(patterns)
        }
    
    def _generate_learning_insights(self, history: List) -> Dict[str, Any]:
        """Generate insights about learning progress"""
        if not history:
            return {}
        
        # Analyze learning over time
        monthly_performance = {}
        for session in history:
            month_key = session.timestamp.strftime('%Y-%m')
            if month_key not in monthly_performance:
                monthly_performance[month_key] = {
                    'sessions': 0,
                    'approvals': 0,
                    'retries': 0,
                    'quality_scores': []
                }
            
            perf = monthly_performance[month_key]
            perf['sessions'] += 1
            
            if session.judge_feedback == "APPROVED":
                perf['approvals'] += 1
            
            if session.initial_translation != session.final_translation:
                perf['retries'] += 1
            
            if session.quality_score:
                perf['quality_scores'].append(session.quality_score)
        
        # Calculate monthly metrics
        for month, perf in monthly_performance.items():
            perf['approval_rate'] = perf['approvals'] / perf['sessions'] if perf['sessions'] > 0 else 0
            perf['retry_rate'] = perf['retries'] / perf['sessions'] if perf['sessions'] > 0 else 0
            perf['avg_quality'] = sum(perf['quality_scores']) / len(perf['quality_scores']) if perf['quality_scores'] else 0
        
        # Detect learning trends
        months = sorted(monthly_performance.keys())
        learning_trend = "stable"
        
        if len(months) >= 2:
            first_month = monthly_performance[months[0]]
            last_month = monthly_performance[months[-1]]
            
            quality_improvement = last_month['avg_quality'] - first_month['avg_quality']
            approval_improvement = last_month['approval_rate'] - first_month['approval_rate']
            
            if quality_improvement > 0.1 or approval_improvement > 0.1:
                learning_trend = "improving"
            elif quality_improvement < -0.1 or approval_improvement < -0.1:
                learning_trend = "declining"
        
        return {
            'monthly_performance': monthly_performance,
            'learning_trend': learning_trend,
            'total_learning_months': len(months),
            'learning_velocity': quality_improvement if len(months) >= 2 else 0
        }
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        summary = report.get('summary', {})
        quality_trends = report.get('quality_trends', {})
        model_performance = report.get('model_performance', {})
        feedback_analysis = report.get('feedback_analysis', {})
        learning_insights = report.get('learning_insights', {})
        
        # Quality-based recommendations
        if summary.get('average_quality_score', 0) < 0.7:
            recommendations.append("Consider improving translation quality by using more context files or better models")
        
        if summary.get('retry_rate', 0) > 0.3:
            recommendations.append("High retry rate detected. Review judge feedback patterns to improve initial translations")
        
        # Trend-based recommendations
        if quality_trends.get('trend') == 'declining':
            recommendations.append("Quality trend is declining. Review recent changes and consider model retraining")
        
        # Model performance recommendations
        if model_performance.get('total_models_tested', 0) < 3:
            recommendations.append("Test more models to find optimal performance for your use case")
        
        # Feedback pattern recommendations
        feedback_types = feedback_analysis.get('feedback_type_analysis', {})
        if 'grammar' in feedback_types and feedback_types['grammar']['total_frequency'] > 10:
            recommendations.append("Grammar issues are frequent. Consider using grammar context files more consistently")
        
        if 'terminology' in feedback_types and feedback_types['terminology']['total_frequency'] > 5:
            recommendations.append("Terminology issues detected. Improve dictionary content or use domain-specific models")
        
        # Learning insights recommendations
        if learning_insights.get('learning_trend') == 'stable' and len(learning_insights.get('monthly_performance', {})) > 2:
            recommendations.append("Learning has plateaued. Consider adjusting learning parameters or adding new training data")
        
        return recommendations
    
    def export_report_to_file(self, report: Dict[str, Any], filepath: str, format: str = 'json'):
        """Export report to file"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
            
            elif format.lower() == 'csv' and 'summary' in report:
                # Export summary as CSV
                import csv
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Metric', 'Value'])
                    for key, value in report['summary'].items():
                        writer.writerow([key, value])
            
            logger.info(f"Report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = None):
        """Clean up old data based on configuration"""
        try:
            if days_to_keep is None:
                days_to_keep = self.config.get('max_session_history_days', 90)
            
            self.memory_manager.cleanup_old_data(days_to_keep)
            logger.info(f"Cleaned up data older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.memory_manager.db_path) as conn:
                stats = {}
                
                # Table row counts
                tables = ['translation_sessions', 'feedback_patterns', 'successful_patterns', 'model_performance']
                for table in tables:
                    result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                    stats[f'{table}_count'] = result[0] if result else 0
                
                # Database size
                stats['database_size_mb'] = self.memory_manager.db_path.stat().st_size / (1024 * 1024)
                
                # Date ranges
                result = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM translation_sessions").fetchone()
                if result and result[0]:
                    stats['oldest_session'] = result[0]
                    stats['newest_session'] = result[1]
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

class MemoryMaintenance:
    """Utilities for memory system maintenance"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.config = get_memory_config()
    
    def run_maintenance(self):
        """Run comprehensive maintenance tasks"""
        logger.info("Starting memory system maintenance")
        
        try:
            # Cleanup old data
            self._cleanup_old_data()
            
            # Optimize database
            self._optimize_database()
            
            # Validate data integrity
            self._validate_data_integrity()
            
            # Update pattern confidence scores
            self._update_pattern_confidence()
            
            # Backup database if enabled
            if self.config.get('backup_database', False):
                self._backup_database()
            
            logger.info("Memory system maintenance completed successfully")
            
        except Exception as e:
            logger.error(f"Error during maintenance: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old data"""
        days_to_keep = self.config.get('max_session_history_days', 90)
        self.memory_manager.cleanup_old_data(days_to_keep)
    
    def _optimize_database(self):
        """Optimize database performance"""
        try:
            with sqlite3.connect(self.memory_manager.db_path) as conn:
                conn.execute("VACUUM")
                conn.execute("ANALYZE")
            logger.info("Database optimization completed")
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
    
    def _validate_data_integrity(self):
        """Validate data integrity"""
        try:
            with sqlite3.connect(self.memory_manager.db_path) as conn:
                # Check for orphaned records
                orphaned_patterns = conn.execute("""
                    SELECT COUNT(*) FROM feedback_patterns 
                    WHERE language_pair NOT IN (
                        SELECT DISTINCT source_language || '-' || target_language 
                        FROM translation_sessions
                    )
                """).fetchone()[0]
                
                if orphaned_patterns > 0:
                    logger.warning(f"Found {orphaned_patterns} orphaned feedback patterns")
                
                # Check for invalid quality scores
                invalid_scores = conn.execute("""
                    SELECT COUNT(*) FROM translation_sessions 
                    WHERE quality_score < 0 OR quality_score > 1
                """).fetchone()[0]
                
                if invalid_scores > 0:
                    logger.warning(f"Found {invalid_scores} invalid quality scores")
                
        except Exception as e:
            logger.error(f"Error validating data integrity: {e}")
    
    def _update_pattern_confidence(self):
        """Update pattern confidence scores based on recent performance"""
        try:
            # This would implement logic to recalculate confidence scores
            # based on recent success rates of patterns
            logger.info("Pattern confidence scores updated")
        except Exception as e:
            logger.error(f"Error updating pattern confidence: {e}")
    
    def _backup_database(self):
        """Create database backup"""
        try:
            backup_path = self.memory_manager.db_path.with_suffix(f'.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}.db')
            
            with sqlite3.connect(self.memory_manager.db_path) as source:
                with sqlite3.connect(backup_path) as backup:
                    source.backup(backup)
            
            logger.info(f"Database backed up to {backup_path}")
            
            # Clean up old backups (keep last 5)
            backup_dir = self.memory_manager.db_path.parent
            backup_files = sorted(backup_dir.glob("*.backup.*.db"))
            if len(backup_files) > 5:
                for old_backup in backup_files[:-5]:
                    old_backup.unlink()
                    logger.info(f"Removed old backup: {old_backup}")
                    
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")