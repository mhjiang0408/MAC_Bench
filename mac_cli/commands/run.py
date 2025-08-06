"""
Run command implementation for MAC_Bench CLI

Handles experiment execution with various configuration options
and model settings.
"""

import click
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add project root to path  
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mac_cli.core.logger import get_logger
from mac_cli.utils.validators import validate_config_file, validate_data_path, validate_model_config
from utils.config_loader import ConfigLoader
from experiment.understanding.single_choice import MultiChoiceEvaluation


@click.command()
@click.option('--config', '-c', 
              type=click.Path(exists=True, path_type=Path),
              help='Configuration file path (YAML or JSON)')
@click.option('--models', '-m', 
              multiple=True,
              help='Specific model names to run (can specify multiple)')
@click.option('--data', '-d',
              type=click.Path(exists=True, path_type=Path), 
              help='Override data path from config')
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              help='Override output directory from config')
@click.option('--resume/--no-resume', 
              default=None,
              help='Resume from previous run (overrides config)')
@click.option('--workers', '-w',
              type=int,
              help='Number of parallel workers (overrides config)')
@click.option('--scaling-factor', '-s',
              type=float,
              help='Data scaling factor (0.0-1.0)')
@click.option('--dry-run',
              is_flag=True,
              help='Show what would be run without executing')
@click.option('--verbose', '-v',
              is_flag=True,
              help='Enable verbose output')
def run_command(config: Optional[Path], 
                models: List[str],
                data: Optional[Path],
                output: Optional[Path],
                resume: Optional[bool],
                workers: Optional[int],
                scaling_factor: Optional[float],
                dry_run: bool,
                verbose: bool):
    """
    Run MAC_Bench experiments
    
    Execute MLLM evaluation experiments
    configurations and models.
    
    \b
    Examples:
      mac run --config configs/understanding.yaml
      mac run --config configs/test.yaml --models gpt-4o --dry-run
      mac run -c config.yaml -m qwen-vl-max -m step-1v-8k --workers 4
      mac run --config config.yaml --resume --verbose
    
    \b
    The configuration file should specify:
      - models: List of model configurations
      - data: Dataset configuration  
      - experiment parameters
      
    Use --dry-run to preview what would be executed.
    """
    
    logger = get_logger()
    
    if verbose:
        logger.setLevel('DEBUG')
    
    logger.info("🚀 Starting MAC_Bench experiment run")
    
    # Validate configuration file
    if not config:
        config = _find_default_config()
        if not config:
            logger.error("No configuration file specified and no default found")
            logger.info("Use --config to specify configuration file")
            raise click.Abort()
    
    logger.info(f"📄 Loading configuration: {config}")
    config_result = validate_config_file(config)
    
    if not config_result['valid']:
        logger.error("Configuration validation failed:")
        for issue in config_result['issues']:
            logger.error(f"  ❌ {issue}")
        raise click.Abort()
    
    config_data = config_result['config']
    logger.info("✅ Configuration valid")
    
    # Override config with command line options
    config_data = _apply_overrides(config_data, {
        'data_path': data,
        'output_folder': output,
        'resume': resume,
        'workers': workers,
        'scaling_factor': scaling_factor,
        'models_filter': models
    })
    
    # Filter models if specified
    if models:
        filtered_models = []
        for model_config in config_data['models']:
            if model_config['name'] in models:
                filtered_models.append(model_config)
        
        if not filtered_models:
            logger.error(f"No matching models found for: {', '.join(models)}")
            available = [m['name'] for m in config_data['models']]
            logger.info(f"Available models: {', '.join(available)}")
            raise click.Abort()
        
        config_data['models'] = filtered_models
        logger.info(f"🎯 Running {len(filtered_models)} selected models")
    else:
        logger.info(f"🎯 Running {len(config_data['models'])} models from config")
    
    # Validate data path
    data_path = Path(config_data['data']['data_path'])
    if not data_path.is_absolute():
        data_path = project_root / data_path
        
    data_result = validate_data_path(data_path)
    if not data_result['valid']:
        logger.error("Data validation failed:")
        for issue in data_result['issues']:
            logger.error(f"  ❌ {issue}")
        raise click.Abort()
    
    logger.info(f"📊 Data file: {data_path}")
    if data_result['info']:
        logger.info(f"   Columns: {len(data_result['info']['columns'])}")
        logger.info(f"   Size: {data_result['info']['file_size']} bytes")
    
    # Show experiment plan
    logger.info("\n📋 Experiment Plan:")
    logger.info(f"   Config: {config}")
    logger.info(f"   Data: {data_path}")
    logger.info(f"   Output: {config_data['data']['output_folder']}")
    logger.info(f"   Models: {len(config_data['models'])}")
    logger.info(f"   Type: {config_data['data']['type']}")
    logger.info(f"   Options: {config_data['data']['num_options']}")
    
    if scaling_factor:
        logger.info(f"   Scaling: {scaling_factor}")
    
    # Show model details
    logger.info("\n🤖 Models to run:")
    for i, model in enumerate(config_data['models'], 1):
        workers_str = f" ({model.get('num_workers', 1)} workers)" if 'num_workers' in model else ""
        resume_str = " [RESUME]" if model.get('resume', False) else ""
        logger.info(f"   {i}. {model['name']}{workers_str}{resume_str}")
    
    if dry_run:
        logger.info("\n🔍 Dry run mode - no experiments will be executed")
        logger.info("Remove --dry-run to actually run the experiments")
        return
    
    # Confirm execution  
    if not click.confirm("\nProceed with experiment execution?"):
        logger.info("Experiment cancelled by user")
        return
    
    # Execute experiments
    try:
        _execute_experiments(config_data, verbose)
        logger.info("🎉 All experiments completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Experiment execution failed: {str(e)}")
        if verbose:
            import traceback
            logger.debug(traceback.format_exc())
        raise click.Abort()


def _find_default_config() -> Optional[Path]:
    """Find default configuration file in common locations"""
    default_paths = [
        Path('config.yaml'),
        Path('config.yml'), 
        Path('Config/understanding_config.yaml'),
        Path('configs/default.yaml')
    ]
    
    for path in default_paths:
        if path.exists():
            return path
    return None


def _apply_overrides(config_data: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply command line overrides to configuration"""
    
    # Apply data overrides
    if overrides['data_path']:
        config_data['data']['data_path'] = str(overrides['data_path'])
        
    if overrides['output_folder']:
        config_data['data']['output_folder'] = str(overrides['output_folder'])
        
    if overrides['scaling_factor']:
        config_data['data']['scaling_factor'] = overrides['scaling_factor']
    
    # Apply model overrides
    if overrides['resume'] is not None:
        for model in config_data['models']:
            model['resume'] = overrides['resume']
            
    if overrides['workers']:
        for model in config_data['models']:
            model['num_workers'] = overrides['workers']
    
    return config_data


def _execute_experiments(config_data: Dict[str, Any], verbose: bool = False):
    """Execute experiments based on configuration"""
    
    logger = get_logger()
    
    # Prepare dataset
    logger.info("🔄 Preparing dataset...")
    data = MultiChoiceEvaluation.prepare_dataset(
        data_path=config_data['data']['data_path'],
        scaling_factor=config_data['data'].get('scaling_factor', 1.0),
        seed=config_data['data'].get('random_seed', 42)
    )
    logger.info(f"   Dataset size: {len(data)} samples")
    
    # Execute each model
    successful_runs = 0
    failed_runs = 0
    
    for i, model_config in enumerate(config_data['models'], 1):
        model_name = model_config['name']
        logger.info(f"\n🤖 Running model {i}/{len(config_data['models'])}: {model_name}")
        
        try:
            # Validate model configuration  
            model_result = validate_model_config(model_config)
            if not model_result['valid']:
                logger.error(f"Model configuration invalid:")
                for issue in model_result['issues']:
                    logger.error(f"  ❌ {issue}")
                failed_runs += 1
                continue
            
            # Load prompt template
            loader = ConfigLoader()
            prompt_template = loader.load_config(model_config['prompt_template'])
            
            # Initialize experiment
            experiment = MultiChoiceEvaluation(
                model=model_config['name'],
                api_base=model_config['api_base'], 
                api_key=model_config['api_key'],
                prompt_template=prompt_template,
                num_options=config_data['data']['num_options'],
                type=config_data['data']['type']
            )
            
            # Determine output path
            if model_config.get('resume') and 'resume_path' in model_config:
                output_path = model_config['resume_path']
                resume = True
            else:
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(
                    config_data['data']['output_folder'],
                    f"{model_name.replace('/','_')}_{config_data['data']['num_options']}options_{timestamp}"
                )
                resume = False
            
            # Get worker count
            num_workers = model_config.get('num_workers', 1)
            
            logger.info(f"   Output: {output_path}")
            logger.info(f"   Workers: {num_workers}")
            logger.info(f"   Resume: {'Yes' if resume else 'No'}")
            
            # Run experiment
            experiment.experiment_with_threads(
                data=data,
                output_path=output_path,
                resume=resume, 
                num_workers=num_workers
            )
            
            successful_runs += 1
            logger.info(f"✅ {model_name} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ {model_name} failed: {str(e)}")
            if verbose:
                import traceback
                logger.debug(traceback.format_exc())
            failed_runs += 1
            continue
    
    # Summary
    logger.info(f"\n📊 Execution Summary:")
    logger.info(f"   ✅ Successful: {successful_runs}")
    logger.info(f"   ❌ Failed: {failed_runs}")
    logger.info(f"   📈 Success Rate: {successful_runs/(successful_runs+failed_runs)*100:.1f}%")
    
    if failed_runs > 0:
        logger.warning(f"{failed_runs} experiments failed - check logs for details")