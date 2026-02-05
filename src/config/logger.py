"""Logging setup for ProtoGalaxy system"""

import logging
import os
from datetime import datetime


class LoggerSetup:
    """Sets up and manages logging for the system"""
    
    def __init__(self, log_dir: str = "outputs/logs"):
        """Initialize logger setup"""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.loggers = {}
    
    def get_logger(self, name: str, level: int = logging.INFO) -> logging.Logger:
        """Get or create logger with name"""
        if name in self.loggers:
            return self.loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # File handler
        log_file = os.path.join(self.log_dir, f"{name}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        self.loggers[name] = logger
        return logger
    
    def get_client_logger(self, client_id: int) -> logging.Logger:
        """Get logger for a specific client"""
        return self.get_logger(f"client_{client_id:02d}")
    
    def get_galaxy_logger(self, galaxy_id: int) -> logging.Logger:
        """Get logger for a specific galaxy"""
        return self.get_logger(f"galaxy_{galaxy_id}")
    
    def get_global_logger(self) -> logging.Logger:
        """Get logger for global aggregation"""
        return self.get_logger("global")
    
    def get_all_loggers(self) -> dict:
        """Get all registered loggers"""
        return self.loggers.copy()
