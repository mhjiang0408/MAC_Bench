"""
Analyze command implementation for MAC_Bench CLI

Handles experiment result analysis, calculating accuracy, ECE, NLL, and RMS metrics.
Outputs results in CSV format only.
"""

import click
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import json
import os

from mac_cli.core.logger import get_logger
from utils.parse_jsonString import parse_probabilities


def calculate_rms_calibration_error(predictions, ground_truths, min_samples_per_bin=30):
    """
    Calculate RMS calibration error using adaptive binning
    
    Args:
        predictions: List of prediction probabilities, each element is a dict {option: probability}
        ground_truths: List of true labels
        min_samples_per_bin: Minimum samples per bin for adaptive binning
        
    Returns:
        float: RMS calibration error
    """
    if not predictions or len(predictions) != len(ground_truths):
        return None
    
    # Extract maximum probability and correctness for each prediction
    confidences = []
    accuracies = []
    
    for pred, gt in zip(predictions, ground_truths):
        # Normalize prediction probabilities
        if not sum(pred.values()) == 0:
            total = sum(pred.values())
            pred = {k: v/total for k, v in pred.items()}
        
        if gt in pred:
            # Get highest probability and its corresponding option
            max_prob_option = max(pred.items(), key=lambda x: x[1])
            confidence = max_prob_option[1]
            is_correct = float(max_prob_option[0] == gt)
            
            confidences.append(confidence)
            accuracies.append(is_correct)
    
    if not confidences:
        return None
    
    # Convert to numpy arrays
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    
    # Sort by confidence
    sort_indices = np.argsort(confidences)
    confidences = confidences[sort_indices]
    accuracies = accuracies[sort_indices]
    
    # Adaptive binning
    n_samples = len(confidences)
    n_bins = max(n_samples // min_samples_per_bin, 1)
    
    squared_errors = []
    current_pos = 0
    
    while current_pos < n_samples:
        end_pos = min(current_pos + min_samples_per_bin, n_samples)
        
        # Calculate average confidence and accuracy for current bin
        bin_confidences = confidences[current_pos:end_pos]
        bin_accuracies = accuracies[current_pos:end_pos]
        
        avg_confidence = np.mean(bin_confidences)
        avg_accuracy = np.mean(bin_accuracies)
        
        # Calculate squared calibration error for this bin
        bin_error = (avg_confidence - avg_accuracy) ** 2
        bin_weight = (end_pos - current_pos) / n_samples
        squared_errors.append(bin_error * bin_weight)
        
        current_pos = end_pos
    
    # Calculate weighted RMS error
    if squared_errors:
        rms = np.sqrt(np.sum(squared_errors))
        return rms
    else:
        return None


def calculate_ece(predictions, ground_truths, n_bins=15):
    """
    Calculate Expected Calibration Error (ECE)
    
    Args:
        predictions: List of prediction probabilities, each element is a dict {option: probability}
        ground_truths: List of true labels
        n_bins: Number of bins, default 15
        
    Returns:
        float: ECE value
    """
    if not predictions or len(predictions) != len(ground_truths):
        return None
    
    # Extract maximum probability and correctness for each prediction
    confidences = []
    accuracies = []
    
    for pred, gt in zip(predictions, ground_truths):
        # Normalize first
        if not sum(pred.values()) == 0:
            for key, value in pred.items():
                pred[key] = value / sum(pred.values())
        
        if not pred:
            continue
            
        # Get highest probability and its corresponding option
        max_prob_option = max(pred.items(), key=lambda x: x[1])
        confidence = max_prob_option[1]
        is_correct = float(max_prob_option[0] == gt)
        
        confidences.append(confidence)
        accuracies.append(is_correct)
    
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    
    # Bin probability values
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def calculate_nll(predictions, ground_truths):
    """
    Calculate Negative Log-Likelihood
    
    Args:
        predictions: List of prediction probabilities, each element is a dict {option: probability}
        ground_truths: List of true labels
        
    Returns:
        float: Average NLL value
    """
    if not predictions or len(predictions) != len(ground_truths):
        return None
    
    nlls = []
    eps = 1e-15  # Prevent log(0)
    
    for pred, gt in zip(predictions, ground_truths):
        # Normalize prediction probabilities
        if not sum(pred.values()) == 0:
            total = sum(pred.values())
            pred = {k: v/total for k, v in pred.items()}
        
        # Get prediction probability for true label and ensure valid range
        prob = pred.get(gt, eps)
        prob = max(min(prob, 1-eps), eps)  # Clip to [eps, 1-eps]
        
        # Calculate single sample NLL
        nll = -np.log(prob)
        nlls.append(nll)
    
    if nlls:
        return np.mean(nlls)
    else:
        return None


@click.command()
@click.argument('results_path', 
                type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              default='analysis_report',
              help='Output directory for analysis results')
@click.option('--format', 'output_format',
              type=click.Choice(['csv']),
              default='csv',
              help='Output format for results (CSV only)')
@click.option('--compare',
              multiple=True,
              type=click.Path(exists=True, path_type=Path),
              help='Additional result files for comparison')
@click.option('--metric',
              type=click.Choice(['all']),
              default='all',
              help='Analyze accuracy, ECE, NLL, and RMS metrics')
@click.option('--group-by',
              type=click.Choice(['model', 'journal', 'option_count', 'none']),
              default='model', 
              help='Group results by specified dimension')
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
                   detailed: bool,
                   verbose: bool):
    """
    Analyze MAC_Bench experiment results
    
    Generate CSV analysis reports with accuracy, ECE, NLL, and RMS calibration error metrics.
    
    \\b
    Examples:
      mac analyze results/experiment_001/results.csv
      mac analyze results/ --output reports/
      mac analyze exp1.csv --compare exp2.csv exp3.csv
      mac analyze results.csv --group-by journal --detailed
    
    \\b
    RESULTS_PATH can be:
      - Single CSV file with experiment results
      - Directory containing multiple result files  
      - Directory with subdirectories of results
      
    \\b
    Analysis includes:
      - Overall accuracy metrics
      - Expected Calibration Error (ECE)
      - Negative Log-Likelihood (NLL)
      - RMS Calibration Error
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
    
    all_results = results_data + comparison_data
    analysis_results = []
    
    for result in all_results:
        logger.info(f"   Analyzing: {result['file_path'].name}")
        
        # Calculate metrics
        total_samples, accuracy, ece, nll, rms = _calculate_metrics(result['data'])
        
        # Get experiment name from file path
        experiment_name = result['metadata'].get('model', result['file_path'].stem)
        
        analysis_results.append({
            'experiment': experiment_name,
            'path': str(result['file_path']),
            'total_samples': total_samples,
            'accuracy': accuracy,
            'ece': ece,
            'nll': nll,
            'rms': rms
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(analysis_results)
    
    # Generate CSV report
    logger.info("📝 Generating CSV report...")
    csv_output_path = output / 'analysis_summary.csv'
    results_df.to_csv(csv_output_path, index=False)
    logger.info(f"   📊 CSV report: {csv_output_path}")
    
    # Print summary
    logger.info("\\n📈 Analysis Summary:")
    for _, row in results_df.iterrows():
        logger.info(f"   🔬 {row['experiment']}: ACC={row['accuracy']:.4f}, ECE={row['ece']:.4f}, NLL={row['nll']:.4f}, RMS={row['rms']:.4f}")
    
    logger.info(f"\\n🎉 Analysis complete! Results saved to: {csv_output_path}")


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
                required_cols = ['journal', 'question_id', 'ground_truth', 'response']
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
    
    # Extract model name from parent directory name
    # Example: experiment/results/understanding/gemini-2.5-pro_4options_20250806_203634/results.csv
    # Extract: gemini-2.5-pro
    parent_dir = path.parent.name
    
    if '_' in parent_dir:
        # Split by underscore and take the first part as model name
        model_name = parent_dir.split('_')[0]
        metadata['model'] = model_name
        
        # Look for option count in directory name
        dir_parts = parent_dir.split('_')
        for part in dir_parts:
            if 'options' in part:
                try:
                    metadata['num_options'] = int(part.replace('options', ''))
                except:
                    pass
        
        # Look for timestamp in directory name
        for part in dir_parts:
            if len(part) == 15 and part.isdigit():  # YYYYMMDD_HHMMSS format
                metadata['timestamp'] = part
    else:
        # Fallback: use directory name as model name
        metadata['model'] = parent_dir
    
    return metadata


def _calculate_metrics(df: pd.DataFrame) -> Tuple[int, float, float, float, float]:
    """
    Calculate accuracy, ECE, NLL, and RMS metrics for a dataset
    
    Args:
        df: DataFrame containing the experiment results
    
    Returns:
        tuple: (total_samples, accuracy, ece, nll, rms)
    """
    
    total_samples = len(df)
    correct = 0
    all_predictions = []
    all_ground_truths = []
    
    # Process each row
    for _, row in df.iterrows():
        try:
            # Parse response to get probabilities
            probabilities = parse_probabilities(row['response'])
            if not probabilities:
                continue
                
            # Get ground truth
            ground_truth = row['ground_truth']
            
            # Store predictions and ground truth
            all_predictions.append(probabilities)
            all_ground_truths.append(ground_truth)
            
            # Calculate accuracy (top1)
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            top1 = sorted_probs[0][0]
            
            if top1 == ground_truth:
                correct += 1
                
        except Exception as e:
            # Skip problematic rows
            continue
    
    # Calculate accuracy
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    
    # Calculate calibration metrics
    ece = calculate_ece(all_predictions, all_ground_truths)
    nll = calculate_nll(all_predictions, all_ground_truths)
    rms = calculate_rms_calibration_error(all_predictions, all_ground_truths)
    
    # Handle None values
    ece = ece if ece is not None else 0.0
    nll = nll if nll is not None else 0.0
    rms = rms if rms is not None else 0.0
    
    return total_samples, accuracy, ece, nll, rms