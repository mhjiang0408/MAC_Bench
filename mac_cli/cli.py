#!/usr/bin/env python3
"""
MAC_Bench CLI - Main Entry Point

Command-line interface for MAC_Bench .
"""

import click
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mac_cli import __version__, __description__
from mac_cli.core.logger import setup_logger
from mac_cli.utils.validators import validate_environment


class MacCLI:
    """Main CLI context class"""
    
    def __init__(self):
        self.logger = setup_logger()
        self.config = {}
        self.verbose = False


# Create a context object to pass between commands
pass_cli = click.make_pass_decorator(MacCLI, ensure=True)


@click.group(invoke_without_command=True)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--version', is_flag=True, help='Show version information')
@click.pass_context
def cli(ctx, verbose, version):
    """
    MAC_Bench CLI 
    
    A comprehensive evaluation framework
    on scientific journal cover understanding tasks.
    
    \b
    Common workflows:
      mac run --config config.yaml              # Run experiment
      mac analyze results/                       # Analyze results  
      mac config validate                        # Validate configuration
      mac status                                 # Check environment
    
    \b  
    For detailed help on any command:
      mac COMMAND --help
    """
    
    if version:
        click.echo(f"MAC_Bench CLI v{__version__}")
        click.echo(__description__)
        ctx.exit()
    
    # Initialize CLI context
    cli_ctx = ctx.ensure_object(MacCLI)
    cli_ctx.verbose = verbose
    
    # If no command provided, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


# Import and register commands
from mac_cli.commands.run import run_command
from mac_cli.commands.analyze import analyze_command
from mac_cli.commands.config import config_command
from mac_cli.commands.status import status_command

cli.add_command(run_command, name='run')
cli.add_command(analyze_command, name='analyze') 
cli.add_command(config_command, name='config')
cli.add_command(status_command, name='status')


@cli.command()
@pass_cli
def info(cli_ctx):
    """Show system and project information"""
    click.echo(f"🚀 MAC_Bench CLI v{__version__}")
    click.echo(f"📁 Project Root: {project_root}")
    click.echo(f"🐍 Python: {sys.version.split()[0]}")
    click.echo(f"💻 Platform: {sys.platform}")
    
    # Check environment
    env_status = validate_environment()
    if env_status['valid']:
        click.echo("✅ Environment: Ready")
    else:
        click.echo("❌ Environment: Issues found")
        for issue in env_status['issues']:
            click.echo(f"   - {issue}")


if __name__ == '__main__':
    cli()