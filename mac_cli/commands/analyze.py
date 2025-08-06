"""
Analyze command implementation for MAC_Bench CLI

Handles experiment result analysis, statistics computation,
and report generation.
"""

import click
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from mac_cli.core.logger import get_logger


@click.command()
@click.argument('results_path', 
                type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              default='analysis_report',
              help='Output directory for analysis results')
@click.option('--format', 'output_format',
              type=click.Choice(['json', 'csv', 'html', 'all']),
              default='all',
              help='Output format for results')
@click.option('--compare',
              multiple=True,
              type=click.Path(exists=True, path_type=Path),
              help='Additional result files for comparison')
@click.option('--metric',
              type=click.Choice(['accuracy', 'tokens', 'response_quality', 'all']),
              default='all',
              help='Specific metrics to analyze')
@click.option('--group-by',
              type=click.Choice(['model', 'journal', 'option_count', 'none']),
              default='model', 
              help='Group results by specified dimension')
@click.option('--plot/--no-plot',
              default=True,
              help='Generate visualization plots')
@click.option('--detailed',
              is_flag=True,
              help='Include detailed per-sample analysis')
@click.option('--verbose', '-v',
              is_flag=True,
              help='Enable verbose output')
def analyze_command(results_path: Path,
                   output: Path,
                   output_format: str,
                   compare: List[Path],
                   metric: str,
                   group_by: str,
                   plot: bool,
                   detailed: bool,
                   verbose: bool):
    """
    Analyze MAC_Bench experiment results
    
    Generate comprehensive analysis reports including accuracy metrics,
    token usage statistics, error analysis, and comparative studies.
    
    \b
    Examples:
      mac analyze results/experiment_001/results.csv
      mac analyze results/ --output reports/ --format html
      mac analyze exp1.csv --compare exp2.csv exp3.csv --plot
      mac analyze results.csv --group-by journal --detailed
      mac analyze results/ --metric accuracy --no-plot
    
    \b
    RESULTS_PATH can be:
      - Single CSV file with experiment results
      - Directory containing multiple result files  
      - Directory with subdirectories of results
      
    \b
    Analysis includes:
      - Overall accuracy and performance metrics
      - Per-model comparison and ranking
      - Token usage and efficiency analysis  
      - Error pattern identification
      - Statistical significance testing
      - Visualizations and charts
    """
    
    logger = get_logger()
    
    if verbose:
        logger.setLevel('DEBUG')
    
    logger.info("📊 Starting MAC_Bench results analysis")
    
    # Load and validate results
    logger.info(f"📁 Loading results from: {results_path}")
    
    try:
        results_data = _load_results(results_path, logger)
        if not results_data:
            logger.error("No valid results found")
            raise click.Abort()
        
        logger.info(f"✅ Loaded {len(results_data)} result files")
        
    except Exception as e:
        logger.error(f"Failed to load results: {str(e)}")
        raise click.Abort()
    
    # Load comparison data if specified
    comparison_data = []
    if compare:
        logger.info(f"📁 Loading comparison data from {len(compare)} sources")
        for comp_path in compare:
            try:
                comp_data = _load_results(comp_path, logger)
                if comp_data:
                    comparison_data.extend(comp_data)
                    logger.info(f"   ✅ Loaded: {comp_path}")
            except Exception as e:
                logger.warning(f"   ⚠️  Failed to load: {comp_path} - {str(e)}")
    
    # Create output directory
    output.mkdir(parents=True, exist_ok=True)
    logger.info(f"📂 Output directory: {output}")
    
    # Perform analysis
    logger.info("🔍 Performing analysis...")
    
    analyzer = ResultAnalyzer(results_data + comparison_data)
    analysis_results = analyzer.analyze(
        metric_focus=metric,
        group_by=group_by,
        include_detailed=detailed,
        logger=logger
    )
    
    # Generate reports
    logger.info("📝 Generating reports...")
    
    report_generator = ReportGenerator(analysis_results, output)
    
    if output_format in ['json', 'all']:
        json_file = report_generator.generate_json_report()
        logger.info(f"   📄 JSON report: {json_file}")
    
    if output_format in ['csv', 'all']:
        csv_files = report_generator.generate_csv_reports()
        for csv_file in csv_files:
            logger.info(f"   📊 CSV report: {csv_file}")
    
    if output_format in ['html', 'all']:
        html_file = report_generator.generate_html_report()
        logger.info(f"   🌐 HTML report: {html_file}")
    
    # Generate plots
    if plot:
        logger.info("📈 Generating visualizations...")
        plot_files = report_generator.generate_plots()
        for plot_file in plot_files:
            logger.info(f"   📊 Plot: {plot_file}")
    
    # Summary statistics
    logger.info("\n📈 Analysis Summary:")
    summary = analysis_results['summary']
    logger.info(f"   📊 Total samples analyzed: {summary['total_samples']:,}")
    logger.info(f"   🤖 Models evaluated: {summary['num_models']}")
    
    if 'accuracy' in summary:
        logger.info(f"   🎯 Overall accuracy: {summary['accuracy']['mean']:.3f} ± {summary['accuracy']['std']:.3f}")
        logger.info(f"   📊 Best model: {summary['best_model']['name']} ({summary['best_model']['accuracy']:.3f})")
    
    if 'tokens' in summary:
        logger.info(f"   💰 Avg tokens per sample: {summary['tokens']['mean']:.0f}")
        logger.info(f"   💸 Most efficient: {summary['most_efficient']['name']} ({summary['most_efficient']['tokens_per_sample']:.0f}/sample)")
    
    logger.info(f"\n🎉 Analysis complete! Results saved to: {output}")


def _load_results(path: Path, logger) -> List[Dict[str, Any]]:
    """Load experiment results from file or directory"""
    
    results = []
    
    if path.is_file():
        # Single file
        if path.suffix == '.csv':
            try:
                df = pd.read_csv(path)
                results.append({
                    'file_path': path,
                    'data': df,
                    'metadata': _extract_metadata_from_path(path)
                })
            except Exception as e:
                logger.warning(f"Failed to load {path}: {str(e)}")
        
    elif path.is_dir():
        # Directory - find all CSV files
        csv_files = list(path.glob('**/*.csv'))
        logger.info(f"   Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Validate required columns
                required_cols = ['journal', 'question_id', 'ground_truth', 'answer', 'judge']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    logger.warning(f"Skipping {csv_file}: missing columns {missing_cols}")
                    continue
                
                results.append({
                    'file_path': csv_file,
                    'data': df, 
                    'metadata': _extract_metadata_from_path(csv_file)
                })
                
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {str(e)}")
    
    return results


def _extract_metadata_from_path(path: Path) -> Dict[str, Any]:
    """Extract metadata from file path and name"""
    
    metadata = {
        'file_name': path.name,
        'file_size': path.stat().st_size,
        'modification_time': path.stat().st_mtime
    }
    
    # Try to extract model name and parameters from filename
    name_parts = path.stem.split('_')
    
    if len(name_parts) >= 2:
        metadata['model'] = name_parts[0]
        
        # Look for option count
        for part in name_parts:
            if 'options' in part:
                try:
                    metadata['num_options'] = int(part.replace('options', ''))
                except:
                    pass
        
        # Look for timestamp
        for part in name_parts:
            if len(part) == 15 and part.isdigit():  # YYYYMMDD_HHMMSS format
                metadata['timestamp'] = part
    
    return metadata


class ResultAnalyzer:
    """Core analysis functionality for experiment results"""
    
    def __init__(self, results_data: List[Dict[str, Any]]):
        self.results_data = results_data
        self.combined_df = self._combine_results()
    
    def _combine_results(self) -> pd.DataFrame:
        """Combine all result dataframes with metadata"""
        
        combined_data = []
        
        for result in self.results_data:
            df = result['data'].copy()
            
            # Add metadata columns
            for key, value in result['metadata'].items():
                df[f'meta_{key}'] = value
            
            combined_data.append(df)
        
        if not combined_data:
            return pd.DataFrame()
        
        return pd.concat(combined_data, ignore_index=True)
    
    def analyze(self, metric_focus: str = 'all', 
                group_by: str = 'model',
                include_detailed: bool = False,
                logger = None) -> Dict[str, Any]:
        """Perform comprehensive analysis"""
        
        if self.combined_df.empty:
            return {'error': 'No data to analyze'}
        
        analysis = {
            'summary': self._compute_summary_stats(),
            'by_model': self._analyze_by_model(),
            'accuracy_metrics': self._compute_accuracy_metrics(),
            'token_analysis': self._analyze_token_usage(),
            'error_analysis': self._analyze_errors()
        }
        
        # Group-specific analysis
        if group_by != 'none':
            analysis[f'by_{group_by}'] = self._analyze_by_group(group_by)
        
        # Detailed analysis
        if include_detailed:
            analysis['detailed'] = self._detailed_analysis()
        
        # Statistical tests
        if len(self.results_data) > 1:
            analysis['statistical_tests'] = self._statistical_tests()
        
        return analysis
    
    def _compute_summary_stats(self) -> Dict[str, Any]:
        """Compute overall summary statistics"""
        
        df = self.combined_df
        
        summary = {
            'total_samples': len(df),
            'num_models': df['meta_model'].nunique() if 'meta_model' in df.columns else 1,
            'num_files': len(self.results_data)
        }
        
        # Accuracy statistics
        if 'judge' in df.columns:
            accuracy_stats = {
                'mean': df['judge'].mean(),
                'std': df['judge'].std(),
                'min': df['judge'].min(),
                'max': df['judge'].max(),
                'median': df['judge'].median()
            }
            summary['accuracy'] = accuracy_stats
            
            # Best performing model
            if 'meta_model' in df.columns:
                model_acc = df.groupby('meta_model')['judge'].mean()
                best_model = model_acc.idxmax()
                summary['best_model'] = {
                    'name': best_model,
                    'accuracy': model_acc[best_model]
                }
        
        # Token statistics
        if 'total_tokens' in df.columns:
            token_stats = {
                'mean': df['total_tokens'].mean(),
                'std': df['total_tokens'].std(),
                'total': df['total_tokens'].sum(),
                'median': df['total_tokens'].median()
            }
            summary['tokens'] = token_stats
            
            # Most efficient model
            if 'meta_model' in df.columns:
                model_tokens = df.groupby('meta_model')['total_tokens'].mean()
                most_efficient = model_tokens.idxmin()
                summary['most_efficient'] = {
                    'name': most_efficient,
                    'tokens_per_sample': model_tokens[most_efficient]
                }
        
        return summary
    
    def _analyze_by_model(self) -> Dict[str, Any]:
        """Analyze results grouped by model"""
        
        df = self.combined_df
        
        if 'meta_model' not in df.columns:
            return {}
        
        model_analysis = {}
        
        for model in df['meta_model'].unique():
            model_df = df[df['meta_model'] == model]
            
            analysis = {
                'sample_count': len(model_df),
                'accuracy': model_df['judge'].mean() if 'judge' in df.columns else None,
                'accuracy_std': model_df['judge'].std() if 'judge' in df.columns else None
            }
            
            if 'total_tokens' in df.columns:
                analysis['avg_tokens'] = model_df['total_tokens'].mean()
                analysis['total_tokens'] = model_df['total_tokens'].sum()
            
            model_analysis[model] = analysis
        
        return model_analysis
    
    def _compute_accuracy_metrics(self) -> Dict[str, Any]:
        """Compute detailed accuracy metrics"""
        
        df = self.combined_df
        
        if 'judge' not in df.columns:
            return {}
        
        metrics = {}
        
        # Overall metrics
        metrics['overall'] = {
            'accuracy': df['judge'].mean(),
            'correct_count': df['judge'].sum(),
            'total_count': len(df),
            'confidence_interval': self._compute_confidence_interval(df['judge'])
        }
        
        # By journal if available
        if 'journal' in df.columns:
            journal_acc = df.groupby('journal')['judge'].agg(['mean', 'count', 'std'])
            metrics['by_journal'] = journal_acc.to_dict('index')
        
        return metrics
    
    def _analyze_token_usage(self) -> Dict[str, Any]:
        """Analyze token usage patterns"""
        
        df = self.combined_df
        
        if 'total_tokens' not in df.columns:
            return {}
        
        analysis = {
            'statistics': {
                'mean': df['total_tokens'].mean(),
                'median': df['total_tokens'].median(),
                'std': df['total_tokens'].std(),
                'total': df['total_tokens'].sum(),
                'min': df['total_tokens'].min(),
                'max': df['total_tokens'].max()
            },
            'distribution': {
                'q25': df['total_tokens'].quantile(0.25),
                'q75': df['total_tokens'].quantile(0.75),
                'iqr': df['total_tokens'].quantile(0.75) - df['total_tokens'].quantile(0.25)
            }
        }
        
        # Efficiency metrics (tokens per accuracy)
        if 'judge' in df.columns:
            correct_df = df[df['judge'] == 1]
            if not correct_df.empty:
                analysis['efficiency'] = {
                    'tokens_per_correct_answer': correct_df['total_tokens'].mean(),
                    'cost_efficiency_ratio': df['judge'].mean() / (df['total_tokens'].mean() / 1000)
                }
        
        return analysis
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze error patterns and failure modes"""
        
        df = self.combined_df
        
        if 'judge' not in df.columns:
            return {}
        
        error_df = df[df['judge'] == 0]  # Incorrect answers
        
        analysis = {
            'error_rate': len(error_df) / len(df),
            'error_count': len(error_df),
            'total_count': len(df)
        }
        
        # Error distribution by answer choice
        if 'answer' in df.columns and 'ground_truth' in df.columns:
            error_patterns = error_df.groupby(['ground_truth', 'answer']).size()
            analysis['error_patterns'] = error_patterns.to_dict()
        
        # Errors by journal
        if 'journal' in df.columns:
            error_by_journal = error_df['journal'].value_counts()
            analysis['errors_by_journal'] = error_by_journal.to_dict()
        
        return analysis
    
    def _analyze_by_group(self, group_by: str) -> Dict[str, Any]:
        """Analyze results grouped by specified dimension"""
        
        df = self.combined_df
        group_col = f'meta_{group_by}' if group_by in ['model'] else group_by
        
        if group_col not in df.columns:
            return {}
        
        grouped = df.groupby(group_col)
        analysis = {}
        
        for group, group_df in grouped:
            group_analysis = {
                'sample_count': len(group_df),
                'accuracy': group_df['judge'].mean() if 'judge' in df.columns else None
            }
            
            if 'total_tokens' in df.columns:
                group_analysis['avg_tokens'] = group_df['total_tokens'].mean()
            
            analysis[str(group)] = group_analysis
        
        return analysis
    
    def _detailed_analysis(self) -> Dict[str, Any]:
        """Perform detailed per-sample analysis"""
        
        df = self.combined_df
        
        # Find challenging samples (high error rate across models)
        challenging = {}
        
        if 'question_id' in df.columns and 'judge' in df.columns:
            question_performance = df.groupby('question_id')['judge'].agg(['mean', 'count'])
            # Questions with low accuracy and multiple attempts
            difficult_questions = question_performance[
                (question_performance['mean'] < 0.5) & 
                (question_performance['count'] > 1)
            ]
            challenging['difficult_questions'] = difficult_questions.head(10).to_dict('index')
        
        return challenging
    
    def _statistical_tests(self) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        
        # Placeholder for statistical tests
        # Would implement t-tests, ANOVA, etc.
        return {'note': 'Statistical tests not implemented in this version'}
    
    def _compute_confidence_interval(self, data, confidence=0.95):
        """Compute confidence interval for proportion"""
        n = len(data)
        p = data.mean()
        
        import math
        z = 1.96  # 95% confidence
        margin = z * math.sqrt(p * (1 - p) / n)
        
        return {
            'lower': max(0, p - margin),
            'upper': min(1, p + margin),
            'margin': margin
        }


class ReportGenerator:
    """Generate various report formats from analysis results"""
    
    def __init__(self, analysis_results: Dict[str, Any], output_dir: Path):
        self.results = analysis_results
        self.output_dir = output_dir
    
    def generate_json_report(self) -> Path:
        """Generate JSON report"""
        
        report_path = self.output_dir / 'analysis_report.json'
        
        # Convert numpy types to native Python types for JSON serialization
        json_results = self._convert_for_json(self.results)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        return report_path
    
    def generate_csv_reports(self) -> List[Path]:
        """Generate CSV reports"""
        
        csv_files = []
        
        # Summary report
        if 'by_model' in self.results:
            summary_path = self.output_dir / 'model_summary.csv'
            
            summary_data = []
            for model, stats in self.results['by_model'].items():
                row = {'model': model}
                row.update(stats)
                summary_data.append(row)
            
            df = pd.DataFrame(summary_data)
            df.to_csv(summary_path, index=False)
            csv_files.append(summary_path)
        
        return csv_files
    
    def generate_html_report(self) -> Path:
        """Generate HTML report"""
        
        report_path = self.output_dir / 'analysis_report.html'
        
        html_content = self._generate_html_content()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
    
    def generate_plots(self) -> List[Path]:
        """Generate visualization plots"""
        
        plot_files = []
        
        try:
            # Accuracy comparison plot
            if 'by_model' in self.results:
                plot_path = self._plot_model_comparison()
                if plot_path:
                    plot_files.append(plot_path)
            
            # Token usage plot
            if 'token_analysis' in self.results:
                plot_path = self._plot_token_analysis()
                if plot_path:
                    plot_files.append(plot_path)
        
        except Exception as e:
            # Plotting is optional, don't fail the entire analysis
            pass
        
        return plot_files
    
    def _convert_for_json(self, obj):
        """Convert numpy types to JSON-serializable types"""
        
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _generate_html_content(self) -> str:
        """Generate HTML report content"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MAC_Bench Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 MAC_Bench Analysis Report</h1>
                <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Summary Statistics</h2>
                {self._format_summary_html()}
            </div>
            
            <div class="section">
                <h2>🤖 Model Performance</h2>
                {self._format_model_table_html()}
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _format_summary_html(self) -> str:
        """Format summary statistics as HTML"""
        
        summary = self.results.get('summary', {})
        
        metrics_html = []
        
        if 'total_samples' in summary:
            metrics_html.append(f'<div class="metric"><strong>Total Samples:</strong> {summary["total_samples"]:,}</div>')
        
        if 'accuracy' in summary:
            acc = summary['accuracy']
            metrics_html.append(f'<div class="metric"><strong>Accuracy:</strong> {acc["mean"]:.3f} ± {acc["std"]:.3f}</div>')
        
        if 'tokens' in summary:
            tokens = summary['tokens']
            metrics_html.append(f'<div class="metric"><strong>Avg Tokens:</strong> {tokens["mean"]:.0f}</div>')
        
        return '\n'.join(metrics_html)
    
    def _format_model_table_html(self) -> str:
        """Format model performance table as HTML"""
        
        if 'by_model' not in self.results:
            return '<p>No model-specific data available.</p>'
        
        table_html = ['<table>', '<tr><th>Model</th><th>Samples</th><th>Accuracy</th><th>Avg Tokens</th></tr>']
        
        for model, stats in self.results['by_model'].items():
            accuracy = f"{stats.get('accuracy', 0):.3f}" if stats.get('accuracy') is not None else 'N/A'
            tokens = f"{stats.get('avg_tokens', 0):.0f}" if stats.get('avg_tokens') is not None else 'N/A'
            
            table_html.append(
                f'<tr><td>{model}</td><td>{stats.get("sample_count", 0)}</td><td>{accuracy}</td><td>{tokens}</td></tr>'
            )
        
        table_html.append('</table>')
        
        return '\n'.join(table_html)
    
    def _plot_model_comparison(self) -> Optional[Path]:
        """Create model comparison plot"""
        
        if 'by_model' not in self.results:
            return None
        
        try:
            models = []
            accuracies = []
            
            for model, stats in self.results['by_model'].items():
                if stats.get('accuracy') is not None:
                    models.append(model)
                    accuracies.append(stats['accuracy'])
            
            if not models:
                return None
            
            plt.figure(figsize=(10, 6))
            plt.bar(models, accuracies)
            plt.title('Model Performance Comparison')
            plt.xlabel('Model')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plot_path = self.output_dir / 'model_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
        
        except Exception:
            return None
    
    def _plot_token_analysis(self) -> Optional[Path]:
        """Create token usage analysis plot"""
        
        # Placeholder for token analysis plot
        return None