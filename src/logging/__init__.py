"""Enhanced FL-specific logging for ProtoGalaxy.

Provides structured logging with round tracking, metrics logging,
and separate log streams for different components.
"""

import logging
import os
import json
import csv
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import threading


class LogLevel(Enum):
    """Log levels for FL logging"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class FLLogRecord:
    """Structured log record for FL events"""
    timestamp: str
    round_num: int
    component: str  # 'client', 'galaxy', 'global', 'defense'
    event_type: str  # 'training', 'aggregation', 'detection', etc.
    message: str
    metrics: Optional[Dict[str, Any]] = None
    client_id: Optional[str] = None
    galaxy_id: Optional[str] = None
    level: str = "INFO"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())


class FLLogger:
    """Enhanced logger for Federated Learning.
    
    Provides:
    - Structured JSON logging
    - Round-aware log messages
    - Separate log files per component
    - Metrics CSV logging
    - Real-time log streaming
    """
    
    def __init__(
        self,
        name: str,
        log_dir: str = "outputs/logs",
        level: LogLevel = LogLevel.INFO,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: bool = True
    ):
        """Initialize FL logger.
        
        Args:
            name: Logger name (component name)
            log_dir: Directory for log files
            level: Minimum log level
            enable_console: Enable console output
            enable_file: Enable file logging
            enable_json: Enable JSON structured logs
        """
        self.name = name
        self.log_dir = log_dir
        self.level = level
        self.current_round = 0
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Create base logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level.value)
        self.logger.handlers = []  # Clear existing handlers
        
        # Formatter
        self.formatter = logging.Formatter(
            '%(asctime)s | %(name)s | R%(round)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add handlers
        if enable_console:
            self._add_console_handler()
        if enable_file:
            self._add_file_handler()
        if enable_json:
            self._add_json_handler()
        
        # Metrics tracking
        self._metrics_buffer: List[Dict] = []
        self._lock = threading.Lock()
    
    def _add_console_handler(self) -> None:
        """Add console log handler"""
        handler = logging.StreamHandler()
        handler.setLevel(self.level.value)
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)
    
    def _add_file_handler(self) -> None:
        """Add file log handler"""
        log_file = os.path.join(self.log_dir, f"{self.name}.log")
        handler = logging.FileHandler(log_file, mode='a')
        handler.setLevel(self.level.value)
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)
    
    def _add_json_handler(self) -> None:
        """Add JSON file handler"""
        json_file = os.path.join(self.log_dir, f"{self.name}.jsonl")
        self._json_file = open(json_file, 'a')
    
    def set_round(self, round_num: int) -> None:
        """Set current round number for logging.
        
        Args:
            round_num: Current FL round number
        """
        self.current_round = round_num
    
    def _log(
        self,
        level: LogLevel,
        message: str,
        event_type: str = "general",
        metrics: Optional[Dict] = None,
        client_id: Optional[str] = None,
        galaxy_id: Optional[str] = None
    ) -> None:
        """Internal logging method.
        
        Args:
            level: Log level
            message: Log message
            event_type: Type of event
            metrics: Optional metrics dict
            client_id: Optional client ID
            galaxy_id: Optional galaxy ID
        """
        # Standard logging with round info
        extra = {'round': self.current_round}
        self.logger.log(level.value, message, extra=extra)
        
        # JSON structured log
        if hasattr(self, '_json_file') and self._json_file:
            record = FLLogRecord(
                timestamp=datetime.now().isoformat(),
                round_num=self.current_round,
                component=self.name,
                event_type=event_type,
                message=message,
                metrics=metrics,
                client_id=client_id,
                galaxy_id=galaxy_id,
                level=level.name
            )
            try:
                self._json_file.write(record.to_json() + '\n')
                self._json_file.flush()
            except:
                pass
    
    def debug(
        self,
        message: str,
        event_type: str = "debug",
        **kwargs
    ) -> None:
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, event_type, **kwargs)
    
    def info(
        self,
        message: str,
        event_type: str = "info",
        **kwargs
    ) -> None:
        """Log info message"""
        self._log(LogLevel.INFO, message, event_type, **kwargs)
    
    def warning(
        self,
        message: str,
        event_type: str = "warning",
        **kwargs
    ) -> None:
        """Log warning message"""
        self._log(LogLevel.WARNING, message, event_type, **kwargs)
    
    def error(
        self,
        message: str,
        event_type: str = "error",
        **kwargs
    ) -> None:
        """Log error message"""
        self._log(LogLevel.ERROR, message, event_type, **kwargs)
    
    def critical(
        self,
        message: str,
        event_type: str = "critical",
        **kwargs
    ) -> None:
        """Log critical message"""
        self._log(LogLevel.CRITICAL, message, event_type, **kwargs)
    
    # FL-specific logging methods
    
    def log_round_start(
        self,
        round_num: int,
        num_clients: int,
        config: Optional[Dict] = None
    ) -> None:
        """Log start of FL round.
        
        Args:
            round_num: Round number
            num_clients: Number of participating clients
            config: Round configuration
        """
        self.set_round(round_num)
        self.info(
            f"Round {round_num} started with {num_clients} clients",
            event_type="round_start",
            metrics={'num_clients': num_clients, 'config': config}
        )
    
    def log_round_end(
        self,
        round_num: int,
        duration: float,
        metrics: Optional[Dict] = None
    ) -> None:
        """Log end of FL round.
        
        Args:
            round_num: Round number
            duration: Round duration in seconds
            metrics: Round metrics (accuracy, loss, etc.)
        """
        all_metrics = {'duration_sec': duration}
        if metrics:
            all_metrics.update(metrics)
        
        self.info(
            f"Round {round_num} completed in {duration:.2f}s",
            event_type="round_end",
            metrics=all_metrics
        )
    
    def log_training(
        self,
        client_id: str,
        loss: float,
        accuracy: Optional[float] = None,
        epochs: int = 1
    ) -> None:
        """Log client training.
        
        Args:
            client_id: Client identifier
            loss: Training loss
            accuracy: Optional accuracy
            epochs: Number of local epochs
        """
        metrics = {'loss': loss, 'epochs': epochs}
        if accuracy is not None:
            metrics['accuracy'] = accuracy
        
        self.debug(
            f"Client {client_id} training: loss={loss:.4f}",
            event_type="training",
            metrics=metrics,
            client_id=client_id
        )
    
    def log_aggregation(
        self,
        num_gradients: int,
        method: str = "fedavg",
        galaxy_id: Optional[str] = None
    ) -> None:
        """Log gradient aggregation.
        
        Args:
            num_gradients: Number of gradients aggregated
            method: Aggregation method used
            galaxy_id: Galaxy ID (if galaxy-level)
        """
        component = f"galaxy_{galaxy_id}" if galaxy_id else "global"
        self.info(
            f"Aggregated {num_gradients} gradients using {method}",
            event_type="aggregation",
            metrics={'num_gradients': num_gradients, 'method': method},
            galaxy_id=galaxy_id
        )
    
    def log_detection(
        self,
        detected_clients: List[str],
        detection_type: str,
        layer: str = "unknown"
    ) -> None:
        """Log Byzantine detection.
        
        Args:
            detected_clients: List of detected client IDs
            detection_type: Type of detection (statistical, krum, etc.)
            layer: Defense layer that detected
        """
        num_detected = len(detected_clients)
        if num_detected > 0:
            self.warning(
                f"Detected {num_detected} suspicious clients via {detection_type}",
                event_type="detection",
                metrics={
                    'num_detected': num_detected,
                    'clients': detected_clients,
                    'detection_type': detection_type,
                    'layer': layer
                }
            )
    
    def log_reputation_update(
        self,
        client_id: str,
        old_score: float,
        new_score: float,
        reason: str = ""
    ) -> None:
        """Log reputation score update.
        
        Args:
            client_id: Client identifier
            old_score: Previous reputation score
            new_score: Updated reputation score
            reason: Reason for update
        """
        self.debug(
            f"Reputation update for {client_id}: {old_score:.3f} -> {new_score:.3f}",
            event_type="reputation",
            metrics={'old_score': old_score, 'new_score': new_score, 'reason': reason},
            client_id=client_id
        )
    
    def log_quarantine(
        self,
        client_id: str,
        reason: str
    ) -> None:
        """Log client quarantine.
        
        Args:
            client_id: Quarantined client ID
            reason: Reason for quarantine
        """
        self.warning(
            f"Client {client_id} quarantined: {reason}",
            event_type="quarantine",
            client_id=client_id
        )
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        prefix: str = ""
    ) -> None:
        """Log metrics and buffer for CSV export.
        
        Args:
            metrics: Metrics dictionary
            prefix: Optional prefix for metric names
        """
        # Add round and timestamp
        record = {
            'round': self.current_round,
            'timestamp': datetime.now().isoformat()
        }
        
        for key, value in metrics.items():
            metric_name = f"{prefix}_{key}" if prefix else key
            record[metric_name] = value
        
        with self._lock:
            self._metrics_buffer.append(record)
        
        # Log summary
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.info(f"Metrics: {metrics_str}", event_type="metrics", metrics=metrics)
    
    def export_metrics_csv(self, filepath: Optional[str] = None) -> str:
        """Export buffered metrics to CSV.
        
        Args:
            filepath: Output file path (auto-generated if None)
            
        Returns:
            Path to created CSV file
        """
        if filepath is None:
            filepath = os.path.join(self.log_dir, f"{self.name}_metrics.csv")
        
        with self._lock:
            if not self._metrics_buffer:
                return filepath
            
            # Get all unique keys
            all_keys = set()
            for record in self._metrics_buffer:
                all_keys.update(record.keys())
            
            fieldnames = sorted(all_keys)
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._metrics_buffer)
        
        return filepath
    
    def close(self) -> None:
        """Close log handlers and export metrics"""
        try:
            self.export_metrics_csv()
        except:
            pass
        
        if hasattr(self, '_json_file') and self._json_file:
            self._json_file.close()
        
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class FLLoggerFactory:
    """Factory for creating FL loggers with consistent configuration"""
    
    _loggers: Dict[str, FLLogger] = {}
    _log_dir: str = "outputs/logs"
    _default_level: LogLevel = LogLevel.INFO
    
    @classmethod
    def configure(
        cls,
        log_dir: str = "outputs/logs",
        default_level: LogLevel = LogLevel.INFO
    ) -> None:
        """Configure factory settings.
        
        Args:
            log_dir: Base log directory
            default_level: Default log level
        """
        cls._log_dir = log_dir
        cls._default_level = default_level
    
    @classmethod
    def get_logger(
        cls,
        name: str,
        level: Optional[LogLevel] = None
    ) -> FLLogger:
        """Get or create a logger.
        
        Args:
            name: Logger name
            level: Log level (uses default if None)
            
        Returns:
            FLLogger instance
        """
        if name not in cls._loggers:
            cls._loggers[name] = FLLogger(
                name=name,
                log_dir=cls._log_dir,
                level=level or cls._default_level
            )
        return cls._loggers[name]
    
    @classmethod
    def get_client_logger(cls, client_id: int) -> FLLogger:
        """Get logger for a client"""
        return cls.get_logger(f"client_{client_id:03d}")
    
    @classmethod
    def get_galaxy_logger(cls, galaxy_id: int) -> FLLogger:
        """Get logger for a galaxy"""
        return cls.get_logger(f"galaxy_{galaxy_id}")
    
    @classmethod
    def get_global_logger(cls) -> FLLogger:
        """Get global aggregator logger"""
        return cls.get_logger("global")
    
    @classmethod
    def get_defense_logger(cls) -> FLLogger:
        """Get defense layer logger"""
        return cls.get_logger("defense")
    
    @classmethod
    def close_all(cls) -> None:
        """Close all loggers"""
        for logger in cls._loggers.values():
            logger.close()
        cls._loggers.clear()
