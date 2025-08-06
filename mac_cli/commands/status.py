"""
Status command implementation for MAC_Bench CLI

Provides system status, environment checks, and resource monitoring.
"""

import click
import sys
import os
import shutil
import subprocess
from pathlib import Path
import psutil
import time
from typing import Dict, Any, Optional

from mac_cli.core.logger import get_logger
from mac_cli.utils.validators import validate_environment


@click.command()
@click.option('--detailed', '-d', is_flag=True, help='Show detailed system information')
@click.option('--check-apis', is_flag=True, help='Test API connectivity (requires config)')
@click.option('--config', type=click.Path(exists=True, path_type=Path), 
              help='Config file for API testing')
@click.option('--json', 'output_json', is_flag=True, help='Output in JSON format')
def status_command(detailed: bool, check_apis: bool, config: Optional[Path], output_json: bool):
    """
    Check MAC_Bench system status and environment
    
    Provides comprehensive system diagnostics including:
    - Python environment and package versions
    - System resources and availability  
    - Project structure validation
    - GPU availability and CUDA status
    - API connectivity testing (with --check-apis)
    
    \b
    Examples:
      mac status                           # Basic status check
      mac status --detailed                # Detailed system info
      mac status --check-apis --config config.yaml  # Include API tests
      mac status --json                    # JSON output format
    """
    
    logger = get_logger()
    
    logger.info("🔍 Checking MAC_Bench system status...")
    
    # Collect status information
    status_info = {
        'environment': _check_environment(),
        'system': _check_system_resources(),
        'project': _check_project_structure(),
        'gpu': _check_gpu_status() if detailed else None,
        'apis': _check_api_connectivity(config) if check_apis and config else None
    }
    
    # Output results
    if output_json:
        import json
        click.echo(json.dumps(status_info, indent=2, default=str))
    else:
        _display_status_report(status_info, detailed, logger)


def _check_environment() -> Dict[str, Any]:
    """Check Python environment and dependencies"""
    
    env_status = validate_environment()
    
    # Get package versions
    packages = {}
    try:
        import numpy
        packages['numpy'] = numpy.__version__
    except ImportError:
        packages['numpy'] = 'Not installed'
    
    try:
        import pandas
        packages['pandas'] = pandas.__version__
    except ImportError:
        packages['pandas'] = 'Not installed'
    
    try:
        import torch
        packages['torch'] = torch.__version__
        packages['torch_cuda'] = torch.cuda.is_available()
    except ImportError:
        packages['torch'] = 'Not installed'
        packages['torch_cuda'] = False
    
    try:
        import transformers
        packages['transformers'] = transformers.__version__
    except ImportError:
        packages['transformers'] = 'Not installed'
    
    try:
        import openai
        packages['openai'] = openai.__version__
    except ImportError:
        packages['openai'] = 'Not installed'
    
    return {
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'platform': sys.platform,
        'executable': sys.executable,
        'packages': packages,
        'validation': env_status
    }


def _check_system_resources() -> Dict[str, Any]:
    """Check system resources and availability"""
    
    # Memory information
    memory = psutil.virtual_memory()
    
    # Disk space
    disk = psutil.disk_usage('/')
    
    # CPU information
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    return {
        'memory': {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_percent': memory.percent
        },
        'disk': {
            'total_gb': round(disk.total / (1024**3), 2),
            'free_gb': round(disk.free / (1024**3), 2),
            'used_percent': round((disk.used / disk.total) * 100, 2)
        },
        'cpu': {
            'cores': cpu_count,
            'usage_percent': cpu_percent
        }
    }


def _check_project_structure() -> Dict[str, Any]:
    """Check project directory structure and key files"""
    
    project_root = Path(__file__).parent.parent.parent
    
    # Required directories and files
    required_structure = {
        'directories': ['Config', 'Dataset', 'experiment', 'utils', 'mac_cli'],
        'files': ['environment.yml', 'README.md'],
        'config_templates': [
            'Config/prompt_template/4_choice_template.json',
            'Config/understanding_config.yaml'
        ]
    }
    
    status = {
        'project_root': str(project_root),
        'structure_valid': True,
        'missing_items': []
    }
    
    # Check directories
    for directory in required_structure['directories']:
        dir_path = project_root / directory
        if not dir_path.exists():
            status['missing_items'].append(f"Directory: {directory}")
            status['structure_valid'] = False
    
    # Check files
    for file_name in required_structure['files']:
        file_path = project_root / file_name
        if not file_path.exists():
            status['missing_items'].append(f"File: {file_name}")
            status['structure_valid'] = False
    
    # Check config templates
    for template in required_structure['config_templates']:
        template_path = project_root / template
        if not template_path.exists():
            status['missing_items'].append(f"Template: {template}")
    
    return status


def _check_gpu_status() -> Dict[str, Any]:
    """Check GPU availability and CUDA status"""
    
    gpu_info = {
        'cuda_available': False,
        'gpu_count': 0,
        'gpu_devices': []
    }
    
    try:
        import torch
        gpu_info['cuda_available'] = torch.cuda.is_available()
        gpu_info['gpu_count'] = torch.cuda.device_count()
        
        for i in range(gpu_info['gpu_count']):
            device_props = torch.cuda.get_device_properties(i)
            gpu_info['gpu_devices'].append({
                'id': i,
                'name': device_props.name,
                'memory_gb': round(device_props.total_memory / (1024**3), 2),
                'compute_capability': f"{device_props.major}.{device_props.minor}"
            })
            
        # CUDA version
        if gpu_info['cuda_available']:
            gpu_info['cuda_version'] = torch.version.cuda
            
    except ImportError:
        pass
    
    # Try nvidia-smi for additional info
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used',
                                '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            nvidia_info = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) == 3:
                    nvidia_info.append({
                        'name': parts[0],
                        'memory_total_mb': int(parts[1]),
                        'memory_used_mb': int(parts[2])
                    })
            gpu_info['nvidia_smi'] = nvidia_info
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return gpu_info


def _check_api_connectivity(config_path: Path) -> Dict[str, Any]:
    """Test API connectivity for configured models"""
    
    api_status = {
        'tested_count': 0,
        'successful_count': 0,
        'results': []
    }
    
    try:
        from mac_cli.utils.validators import validate_config_file
        from utils.test_api import test_model_api
        
        # Load configuration
        config_result = validate_config_file(config_path)
        if not config_result['valid']:
            api_status['error'] = 'Invalid configuration file'
            return api_status
        
        config = config_result['config']
        models = config.get('models', [])
        
        for model in models:
            api_status['tested_count'] += 1
            model_name = model.get('name', 'Unknown')
            
            try:
                # Test API connection
                test_model_api(
                    model_name,
                    model.get('api_base', ''),
                    model.get('api_key', '')
                )
                
                api_status['successful_count'] += 1
                api_status['results'].append({
                    'model': model_name,
                    'status': 'success',
                    'api_base': model.get('api_base', ''),
                    'error': None
                })
                
            except Exception as e:
                api_status['results'].append({
                    'model': model_name,
                    'status': 'failed',
                    'api_base': model.get('api_base', ''),
                    'error': str(e)
                })
    
    except Exception as e:
        api_status['error'] = f'Failed to test APIs: {str(e)}'
    
    return api_status


def _display_status_report(status_info: Dict[str, Any], detailed: bool, logger):
    """Display formatted status report"""
    
    # Environment status
    env = status_info['environment']
    logger.info(f"🐍 Python Environment:")
    logger.info(f"   Version: {env['python_version']}")
    logger.info(f"   Platform: {env['platform']}")
    
    if env['validation']['valid']:
        logger.info("   ✅ Environment: Ready")
    else:
        logger.warning("   ⚠️  Environment: Issues detected")
        for issue in env['validation']['issues']:
            logger.warning(f"      - {issue}")
    
    # Key packages
    packages = env['packages']
    logger.info(f"   📦 Key packages:")
    for pkg in ['torch', 'transformers', 'openai', 'pandas']:
        version = packages.get(pkg, 'Not found')
        status_icon = "✅" if version != 'Not found' and version != 'Not installed' else "❌"
        logger.info(f"      {status_icon} {pkg}: {version}")
    
    # System resources
    system = status_info['system']
    logger.info(f"\n💻 System Resources:")
    logger.info(f"   Memory: {system['memory']['available_gb']:.1f}GB available "
                f"({system['memory']['used_percent']:.1f}% used)")
    logger.info(f"   Disk: {system['disk']['free_gb']:.1f}GB free "
                f"({system['disk']['used_percent']:.1f}% used)")
    logger.info(f"   CPU: {system['cpu']['cores']} cores "
                f"({system['cpu']['usage_percent']:.1f}% usage)")
    
    # Project structure
    project = status_info['project']
    if project['structure_valid']:
        logger.info(f"\n📁 Project Structure: ✅ Valid")
    else:
        logger.warning(f"\n📁 Project Structure: ⚠️  Issues found")
        for missing in project['missing_items']:
            logger.warning(f"   - Missing: {missing}")
    
    # GPU status (if detailed)
    if detailed and status_info['gpu']:
        gpu = status_info['gpu']
        logger.info(f"\n🔥 GPU Status:")
        if gpu['cuda_available']:
            logger.info(f"   ✅ CUDA available (v{gpu.get('cuda_version', 'unknown')})")
            logger.info(f"   🔢 GPU count: {gpu['gpu_count']}")
            
            for device in gpu['gpu_devices']:
                logger.info(f"   GPU {device['id']}: {device['name']} "
                           f"({device['memory_gb']:.1f}GB)")
        else:
            logger.info("   ❌ CUDA not available")
    
    # API connectivity (if tested)
    if status_info['apis']:
        apis = status_info['apis']
        logger.info(f"\n🔌 API Connectivity:")
        
        if 'error' in apis:
            logger.error(f"   ❌ {apis['error']}")
        else:
            success_rate = (apis['successful_count'] / apis['tested_count'] * 100) if apis['tested_count'] > 0 else 0
            logger.info(f"   📊 Success rate: {apis['successful_count']}/{apis['tested_count']} ({success_rate:.1f}%)")
            
            for result in apis['results']:
                status_icon = "✅" if result['status'] == 'success' else "❌"
                logger.info(f"   {status_icon} {result['model']}: {result['status']}")
                if result['error'] and detailed:
                    logger.info(f"      Error: {result['error']}")
    
    # Overall assessment
    logger.info(f"\n🎯 Overall Status:")
    
    issues = []
    if not env['validation']['valid']:
        issues.append("Environment issues")
    if not project['structure_valid']:
        issues.append("Project structure issues")
    if status_info['apis'] and 'error' not in status_info['apis'] and status_info['apis']['successful_count'] == 0:
        issues.append("API connectivity issues")
    
    if not issues:
        logger.info("   ✅ All systems ready for experiments")
    else:
        logger.warning(f"   ⚠️  Issues detected: {', '.join(issues)}")
        logger.info("   Run with --detailed for more information")