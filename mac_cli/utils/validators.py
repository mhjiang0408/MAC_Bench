"""
Validation utilities for MAC_Bench CLI

Provides functions to validate configurations, parameters,
and environment settings.
"""

import os
import sys
import yaml
import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Union


def validate_environment() -> Dict[str, Any]:
    """
    Validate the current environment for MAC_Bench CLI
    
    Returns:
        Dictionary containing validation results
    """
    issues = []
    valid = True
    
    # Check Python version
    min_python = (3, 8)
    current_python = sys.version_info[:2]
    if current_python < min_python:
        issues.append(f"Python {min_python[0]}.{min_python[1]}+ required, found {current_python[0]}.{current_python[1]}")
        valid = False
    
    # Check required packages
    required_packages = [
        'numpy', 'pandas', 'torch', 'transformers', 
        'openai', 'yaml', 'tqdm', 'click', 'colorama'
    ]
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            issues.append(f"Required package '{package}' not found")
            valid = False
    
    # Check project structure
    project_root = Path(__file__).parent.parent.parent
    required_dirs = ['Config', 'Dataset', 'experiment', 'utils']
    
    for dir_name in required_dirs:
        if not (project_root / dir_name).exists():
            issues.append(f"Required directory '{dir_name}' not found")
            valid = False
    
    return {
        'valid': valid,
        'issues': issues,
        'python_version': f"{current_python[0]}.{current_python[1]}",
        'project_root': str(project_root)
    }


def validate_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate configuration file format and required fields
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing validation results
    """
    config_path = Path(config_path)
    issues = []
    valid = True
    
    # Check file exists
    if not config_path.exists():
        return {
            'valid': False, 
            'issues': [f"Configuration file not found: {config_path}"],
            'config': None
        }
    
    # Load and parse config
    try:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            return {
                'valid': False,
                'issues': [f"Unsupported config format: {config_path.suffix}"],
                'config': None
            }
    except Exception as e:
        return {
            'valid': False,
            'issues': [f"Failed to parse config file: {str(e)}"],
            'config': None
        }
    
    # Validate required sections
    required_sections = ['models', 'data']
    for section in required_sections:
        if section not in config:
            issues.append(f"Missing required section: '{section}'")
            valid = False
    
    # Validate models section
    if 'models' in config:
        if not isinstance(config['models'], list) or len(config['models']) == 0:
            issues.append("'models' section must be a non-empty list")
            valid = False
        else:
            for i, model in enumerate(config['models']):
                if not isinstance(model, dict):
                    issues.append(f"Model {i} must be a dictionary")
                    valid = False
                    continue
                
                required_model_fields = ['name', 'api_base', 'api_key', 'prompt_template']
                for field in required_model_fields:
                    if field not in model:
                        issues.append(f"Model {i} missing required field: '{field}'")
                        valid = False
    
    # Validate data section  
    if 'data' in config:
        required_data_fields = ['data_path', 'output_folder', 'num_options', 'type']
        for field in required_data_fields:
            if field not in config['data']:
                issues.append(f"Data section missing required field: '{field}'")
                valid = False
    
    return {
        'valid': valid,
        'issues': issues,
        'config': config if valid else None
    }


def validate_model_config(model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate individual model configuration
    
    Args:
        model: Model configuration dictionary
        
    Returns:
        Dictionary containing validation results
    """
    issues = []
    valid = True
    
    # Check required fields
    required_fields = ['name', 'api_base', 'api_key', 'prompt_template']
    for field in required_fields:
        if field not in model:
            issues.append(f"Missing required field: '{field}'")
            valid = False
    
    # Validate prompt template path
    if 'prompt_template' in model:
        template_path = Path(model['prompt_template'])
        if not template_path.exists():
            issues.append(f"Prompt template not found: {template_path}")
            valid = False
    
    # Validate numeric fields
    numeric_fields = ['num_workers']
    for field in numeric_fields:
        if field in model:
            if not isinstance(model[field], int) or model[field] <= 0:
                issues.append(f"Field '{field}' must be a positive integer")
                valid = False
    
    return {
        'valid': valid,
        'issues': issues
    }


def validate_data_path(data_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate data file exists and has required format
    
    Args:
        data_path: Path to data file
        
    Returns:
        Dictionary containing validation results
    """
    data_path = Path(data_path)
    issues = []
    valid = True
    
    # Check file exists
    if not data_path.exists():
        return {
            'valid': False,
            'issues': [f"Data file not found: {data_path}"],
            'info': None
        }
    
    # Check file format
    if data_path.suffix.lower() != '.csv':
        issues.append(f"Data file must be CSV format, found: {data_path.suffix}")
        valid = False
        return {'valid': valid, 'issues': issues, 'info': None}
    
    # Try to load and validate CSV structure
    try:
        import pandas as pd
        df = pd.read_csv(data_path, nrows=1)  # Only read first row for validation
        
        required_columns = ['journal', 'id', 'question', 'cover_image', 'answer']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
            valid = False
        
        info = {
            'columns': list(df.columns),
            'file_size': data_path.stat().st_size,
            'format': 'CSV'
        }
        
    except Exception as e:
        issues.append(f"Failed to read CSV file: {str(e)}")
        valid = False
        info = None
    
    return {
        'valid': valid,
        'issues': issues,
        'info': info
    }