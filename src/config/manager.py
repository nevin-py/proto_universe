"""Configuration management for ProtoGalaxy system"""

import yaml
import json
from typing import Any, Dict


class ConfigManager:
    """Manages system configuration from files and environment"""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration manager"""
        self.config = {}
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        return self.config
    
    def load_json_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        return self.config
    
    def save_config(self, config_path: str):
        """Save current configuration to YAML file"""
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dots)"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    def set(self, key: str, value: Any):
        """Set configuration value by key (supports nested keys with dots)"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration"""
        return self.config.copy()
    
    def merge_config(self, new_config: Dict[str, Any]):
        """Merge new configuration with existing"""
        self._deep_merge(self.config, new_config)
    
    def _deep_merge(self, target: dict, source: dict):
        """Deep merge source dict into target dict"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
