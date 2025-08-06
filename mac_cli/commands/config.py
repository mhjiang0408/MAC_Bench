"""
Config command implementation for MAC_Bench CLI

Handles configuration file management, validation, and template generation.
"""

import click
import yaml
import json
from pathlib import Path
from typing import Optional, Dict, Any

from mac_cli.core.logger import get_logger
from mac_cli.utils.validators import validate_config_file


@click.group(invoke_without_command=True)
@click.pass_context
def config_command(ctx):
    """
    Configuration management for MAC_Bench
    
    Manage configuration files, validate settings, and generate templates.
    
    \b
    Commands:
      validate    - Validate configuration file
      template    - Generate configuration template
      show        - Display configuration content
      edit        - Open configuration in editor
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@config_command.command()
@click.argument('config_path', type=click.Path(exists=True, path_type=Path))
@click.option('--verbose', '-v', is_flag=True, help='Show detailed validation results')
def validate(config_path: Path, verbose: bool):
    """Validate configuration file format and content"""
    
    logger = get_logger()
    
    logger.info(f"🔍 Validating configuration: {config_path}")
    
    result = validate_config_file(config_path)
    
    if result['valid']:
        logger.info("✅ Configuration is valid")
        
        if verbose and result['config']:
            config = result['config']
            logger.info(f"   📊 Models defined: {len(config.get('models', []))}")
            logger.info(f"   📁 Data path: {config.get('data', {}).get('data_path', 'Not specified')}")
            logger.info(f"   🎯 Task type: {config.get('data', {}).get('type', 'Not specified')}")
            logger.info(f"   🔢 Options: {config.get('data', {}).get('num_options', 'Not specified')}")
    else:
        logger.error("❌ Configuration validation failed:")
        for issue in result['issues']:
            logger.error(f"   • {issue}")
        raise click.Abort()


@config_command.command()
@click.option('--output', '-o', 
              type=click.Path(path_type=Path),
              default='config_template.yaml',
              help='Output file for template')
@click.option('--type', 'config_type',
              type=click.Choice(['basic', 'full', 'example']),
              default='basic',
              help='Template type to generate')
def template(output: Path, config_type: str):
    """Generate configuration file template"""
    
    logger = get_logger()
    
    if config_type == 'basic':
        template_config = _generate_basic_template()
    elif config_type == 'full':
        template_config = _generate_full_template()  
    else:  # example
        template_config = _generate_example_template()
    
    # Write template to file
    with open(output, 'w', encoding='utf-8') as f:
        yaml.dump(template_config, f, default_flow_style=False, indent=2)
    
    logger.info(f"📝 Generated {config_type} template: {output}")
    logger.info("   Edit the template with your specific settings before use")


@config_command.command()
@click.argument('config_path', type=click.Path(exists=True, path_type=Path))
@click.option('--format', 'output_format',
              type=click.Choice(['yaml', 'json', 'pretty']),
              default='pretty',
              help='Display format')
def show(config_path: Path, output_format: str):
    """Display configuration file content"""
    
    logger = get_logger()
    
    try:
        # Load config
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            logger.error("Unsupported config format")
            raise click.Abort()
        
        # Display based on format
        if output_format == 'json':
            click.echo(json.dumps(config, indent=2))
        elif output_format == 'yaml':
            click.echo(yaml.dump(config, default_flow_style=False, indent=2))
        else:  # pretty
            _display_pretty_config(config)
            
    except Exception as e:
        logger.error(f"Failed to read configuration: {str(e)}")
        raise click.Abort()


def _generate_basic_template() -> Dict[str, Any]:
    """Generate basic configuration template"""
    
    return {
        'models': [
            {
                'name': 'your-model-name',
                'api_base': 'https://api.example.com/v1',
                'api_key': 'your-api-key-here',
                'prompt_template': 'Config/prompt_template/4_choice_template.json',
                'resume': False,
                'num_workers': 1
            }
        ],
        'data': {
            'data_path': './MAC_Bench/image2text_info.csv',
            'output_folder': './experiment/results/',
            'scaling_factor': 1.0,
            'num_options': 4,
            'type': 'image2text',
            'random_seed': 42
        }
    }


def _generate_full_template() -> Dict[str, Any]:
    """Generate full configuration template with all options"""
    
    return {
        'models': [
            {
                'name': 'gpt-4o',
                'api_base': 'https://api.openai.com/v1',
                'api_key': 'sk-your-openai-key',
                'prompt_template': 'Config/prompt_template/4_choice_template.json',
                'resume': False,
                'resume_path': None,
                'num_workers': 4
            },
            {
                'name': 'qwen-vl-max',
                'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': 'sk-your-qwen-key',
                'prompt_template': 'Config/prompt_template/4_choice_template.json',
                'resume': False,
                'resume_path': None,
                'num_workers': 2
            }
        ],
        'data': {
            'data_path': './MAC_Bench/image2text_info.csv',
            'output_folder': './experiment/results/understanding/',
            'scaling_factor': 1.0,
            'num_options': 4,
            'type': 'image2text',
            'random_seed': 42
        }
    }


def _generate_example_template() -> Dict[str, Any]:
    """Generate example configuration with sample values"""
    
    return {
        'models': [
            {
                'name': 'step-1o-turbo-vision',
                'api_base': 'https://api.stepfun.com/v1',
                'api_key': 'your-stepfun-api-key',
                'prompt_template': 'Config/prompt_template/4_choice_template.json',
                'resume': False,
                'resume_path': None,
                'num_workers': 5
            }
        ],
        'data': {
            'data_path': './MAC_Bench/image2text_info.csv',
            'output_folder': './experiment/results/understanding/',
            'scaling_factor': 0.1,  # Use 10% of data for quick testing
            'num_options': 4,
            'type': 'image2text',
            'random_seed': 42
        }
    }


def _display_pretty_config(config: Dict[str, Any]):
    """Display configuration in a pretty format"""
    
    click.echo("📋 Configuration Summary:")
    click.echo("=" * 50)
    
    # Models section
    if 'models' in config:
        click.echo(f"\n🤖 Models ({len(config['models'])} configured):")
        for i, model in enumerate(config['models'], 1):
            click.echo(f"   {i}. {model.get('name', 'Unnamed')}")
            click.echo(f"      API: {model.get('api_base', 'Not specified')}")
            click.echo(f"      Workers: {model.get('num_workers', 1)}")
            click.echo(f"      Resume: {'Yes' if model.get('resume', False) else 'No'}")
    
    # Data section
    if 'data' in config:
        data = config['data']
        click.echo(f"\n📊 Data Configuration:")
        click.echo(f"   Path: {data.get('data_path', 'Not specified')}")
        click.echo(f"   Output: {data.get('output_folder', 'Not specified')}")
        click.echo(f"   Type: {data.get('type', 'Not specified')}")
        click.echo(f"   Options: {data.get('num_options', 'Not specified')}")
        click.echo(f"   Scaling: {data.get('scaling_factor', 1.0)}")
        click.echo(f"   Seed: {data.get('random_seed', 42)}")
    
    click.echo("\n" + "=" * 50)