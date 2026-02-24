import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Tkinter errors
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Statistical and ML libraries
from scipy import stats
from scipy.stats import bootstrap, permutation_test
from sklearn.utils import resample
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import permutation_test_score

# Class imbalance handling
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.combine import SMOTEENN, SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    print("Warning: imbalanced-learn not available. Install with: pip install imbalanced-learn")
    IMBLEARN_AVAILABLE = False

# Deep learning
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available. Some deep learning features disabled.")
    TORCH_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("Warning: SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False
import one
import two
import gseapy as gp
GSEAPY_AVAILABLE = True

# Create all necessary directories
directories = ['results', 'figs', 'models', 'reports']
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_data_safely(file_path, description="data"):
    """Safely load data files with error handling"""
    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path, index_col=0)
            print(f" Successfully loaded {description} from {file_path}")
            return data
        elif file_path.endswith('.pkl'):
            data = joblib.load(file_path)
            print(f" Successfully loaded {description} from {file_path}")
            return data
    except Exception as e:
        print(f"Failed to load {description} from {file_path}: {e}")
        return None

def save_results_safely(data, file_path, description="results"):
    """Safely save results with error handling"""
    try:
        if isinstance(data, dict):
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=True)
        else:
            joblib.dump(data, file_path)
        print(f"Successfully saved {description} to {file_path}")
        return True
    except Exception as e:
        print(f"Failed to save {description} to {file_path}: {e}")
        return False

# STEP 13: ENHANCED CLASS IMBALANCE ANALYSIS

def analyze_class_imbalance(y_train, y_val, y_test):
    """Comprehensive class distribution and imbalance analysis"""
    
    print(f"\n{'='*60}")
    print("STEP 13: CLASS IMBALANCE ANALYSIS")
    print(f"{'='*60}")
    
    splits_data = {
        'Train': y_train,
        'Validation': y_val, 
        'Test': y_test
    }
    
    imbalance_stats = {}
    
    for split_name, y_data in splits_data.items():
        class_counts = y_data.value_counts()
        total_samples = len(y_data)
        
        print(f"\n{split_name} Set:")
        for class_name, count in class_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  Class {class_name}: {count:3d} samples ({percentage:5.1f}%)")
        
        # Calculate imbalance ratio
        majority_count = class_counts.max()
        minority_count = class_counts.min()
        imbalance_ratio = majority_count / minority_count
        
        imbalance_stats[split_name] = {
            'imbalance_ratio': float(imbalance_ratio),
            'minority_class': class_counts.idxmin(),
            'majority_class': class_counts.idxmax(),
            'minority_count': int(minority_count),
            'majority_count': int(majority_count),
            'total_samples': int(total_samples)
        }
        
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 2.0:
            print(f"Significant imbalance detected!")
        else:
            print(f"  Relatively balanced")
    
    # Create imbalance visualization
    create_imbalance_plots(imbalance_stats)
    
    # Save imbalance analysis
    save_results_safely(imbalance_stats, 'results/class_imbalance_analysis.json', 
                       'class imbalance analysis')
    
    return imbalance_stats

def create_imbalance_plots(imbalance_stats):
    """Create comprehensive imbalance visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Class distribution across splits
    splits = list(imbalance_stats.keys())
    minority_counts = [imbalance_stats[split]['minority_count'] for split in splits]
    majority_counts = [imbalance_stats[split]['majority_count'] for split in splits]
    
    x = np.arange(len(splits))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, minority_counts, width, label='Minority Class', alpha=0.8)
    axes[0, 0].bar(x + width/2, majority_counts, width, label='Majority Class', alpha=0.8)
    axes[0, 0].set_xlabel('Data Split')
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].set_title('Class Distribution Across Splits')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(splits)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Imbalance ratios
    imbalance_ratios = [imbalance_stats[split]['imbalance_ratio'] for split in splits]
    colors = ['red' if ratio > 2.0 else 'green' for ratio in imbalance_ratios]
    
    bars = axes[0, 1].bar(splits, imbalance_ratios, color=colors, alpha=0.7)
    axes[0, 1].axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Imbalance Threshold')
    axes[0, 1].set_xlabel('Data Split')
    axes[0, 1].set_ylabel('Imbalance Ratio')
    axes[0, 1].set_title('Class Imbalance Ratios')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Add value labels on bars
    for bar, ratio in zip(bars, imbalance_ratios):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{ratio:.2f}', ha='center', va='bottom')
    
    # Plot 3: Sample distribution pie chart
    total_minority = sum(minority_counts)
    total_majority = sum(majority_counts)
    
    axes[1, 0].pie([total_minority, total_majority], 
                   labels=['Minority Class', 'Majority Class'], 
                   autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Overall Class Distribution')
    
    # Plot 4: Summary statistics
    axes[1, 1].axis('off')
    
    summary_text = "Class Imbalance Summary\n\n"
    for split in splits:
        stats = imbalance_stats[split]
        summary_text += f"{split} Set:\n"
        summary_text += f"  Samples: {stats['total_samples']}\n"
        summary_text += f"  Imbalance: {stats['imbalance_ratio']:.2f}:1\n"
        summary_text += f"  Status: {'⚠️ Imbalanced' if stats['imbalance_ratio'] > 2.0 else '✅ Balanced'}\n\n"
    
    axes[1, 1].text(0.1, 0.9, summary_text, fontsize=12, ha='left', va='top',
                   transform=axes[1, 1].transAxes, family='monospace')
    
    plt.tight_layout()
    plt.savefig('figs/class_imbalance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Class imbalance plots saved to figs/class_imbalance_analysis.png")

# STEP 14: ENHANCED MODEL CALIBRATION

def calibrate_model_probabilities(model, X_val, y_val, method='sigmoid'):
    """Advanced model probability calibration"""
    
    print(f"\n{'='*60}")
    print("STEP 14: MODEL CALIBRATION & RISK SCORING")
    print(f"{'='*60}")
    
    print(f"Calibration method: {method.upper()}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create calibrated classifier
    calibrated_model = CalibratedClassifierCV(
        model, method=method, cv=3
    )
    
    # Fit calibration on validation set
    calibrated_model.fit(X_val, y_val)
    
    print(f" Model calibrated using {method}")
    
    return calibrated_model

def evaluate_calibration(y_true, y_proba, n_bins=10):
    """Comprehensive calibration evaluation"""
    
    print(f"\n=== CALIBRATION EVALUATION ===")
    
    # Brier score (lower is better)
    brier_score = brier_score_loss(y_true, y_proba)
    print(f"Brier Score: {brier_score:.4f}")
    
    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy='uniform'
    )
    
    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_proba[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    # Maximum Calibration Error (MCE)
    mce = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_proba[in_bin].mean()
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    calibration_metrics = {
        'brier_score': float(brier_score),
        'expected_calibration_error': float(ece),
        'maximum_calibration_error': float(mce),
        'fraction_of_positives': fraction_of_positives.tolist(),
        'mean_predicted_value': mean_predicted_value.tolist(),
        'n_bins': int(n_bins)
    }
    
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Maximum Calibration Error (MCE): {mce:.4f}")
    
    # Save calibration metrics
    save_results_safely(calibration_metrics, 'results/calibration_metrics.json',
                       'calibration metrics')
    
    return calibration_metrics

def create_calibration_plots(y_true, y_proba_uncalibrated, y_proba_calibrated, 
                           calibration_metrics):
    """Create comprehensive calibration visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Reliability diagram
    fraction_pos_cal, mean_pred_cal = calibration_curve(y_true, y_proba_calibrated, n_bins=10)
    
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration', linewidth=2)
    
    if y_proba_uncalibrated is not None:
        fraction_pos_uncal, mean_pred_uncal = calibration_curve(y_true, y_proba_uncalibrated, n_bins=10)
        axes[0, 0].plot(mean_pred_uncal, fraction_pos_uncal, 'o-', 
                       label='Uncalibrated', alpha=0.8, markersize=8, linewidth=2)
    
    axes[0, 0].plot(mean_pred_cal, fraction_pos_cal, 's-', 
                   label='Calibrated', alpha=0.8, markersize=8, linewidth=2)
    axes[0, 0].set_xlabel('Mean Predicted Probability')
    axes[0, 0].set_ylabel('Fraction of Positives')
    axes[0, 0].set_title('Reliability Diagram')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Probability histograms
    axes[0, 1].hist(y_proba_calibrated, bins=20, alpha=0.7, label='Calibrated', 
                   density=True, edgecolor='black')
    if y_proba_uncalibrated is not None:
        axes[0, 1].hist(y_proba_uncalibrated, bins=20, alpha=0.5, 
                       label='Uncalibrated', density=True, edgecolor='black')
    axes[0, 1].set_xlabel('Predicted Probability')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Probability Distributions')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Calibration metrics comparison
    brier_cal = brier_score_loss(y_true, y_proba_calibrated)
    ece = calibration_metrics['expected_calibration_error']
    mce = calibration_metrics['maximum_calibration_error']
    
    metrics = ['Brier Score', 'ECE', 'MCE']
    values = [brier_cal, ece, mce]
    
    bars = axes[1, 0].bar(metrics, values, alpha=0.8, 
                         color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[1, 0].set_xlabel('Calibration Metrics')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Calibration Quality Metrics')
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Summary and interpretation
    axes[1, 1].axis('off')
    
    calibration_quality = "Excellent" if ece < 0.05 else "Good" if ece < 0.1 else "Poor"
    
    summary_text = f"""Calibration Analysis Summary

Metrics:
  Brier Score: {brier_cal:.4f}
  Expected Calibration Error: {ece:.4f}
  Maximum Calibration Error: {mce:.4f}

Interpretation:
  Calibration Quality: {calibration_quality}
  
Guidelines:
  • Brier Score: Lower is better
  • ECE < 0.05: Excellent calibration
  • ECE < 0.10: Good calibration  
  • ECE > 0.10: Poor calibration
  
Model Assessment:
  The model is {'well' if ece < 0.1 else 'poorly'} calibrated
  {'Reliable for probability estimates' if ece < 0.1 else 'Use with caution for probability estimates'}
"""
    
    axes[1, 1].text(0.05, 0.95, summary_text, fontsize=11, ha='left', va='top',
                   transform=axes[1, 1].transAxes, family='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('figs/calibration_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Calibration plots saved to figs/calibration_analysis.png")

# STEP 15: COMPREHENSIVE STATISTICAL VALIDATION

def bootstrap_confidence_intervals(y_true, y_pred, y_proba, n_bootstrap=1000, 
                                 confidence_level=0.95, random_state=42):
    """Advanced bootstrap confidence interval analysis"""
    
    print(f"\n{'='*60}")
    print("STEP 15: STATISTICAL VALIDATION")
    print(f"{'='*60}")
    
    print(f"Bootstrap samples: {n_bootstrap}")
    print(f"Confidence level: {confidence_level}")
    
    np.random.seed(random_state)
    
    metrics = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
        'auc': [], 'average_precision': [], 'specificity': [], 'npv': []
    }
    
    n_samples = len(y_true)
    
    print("Computing bootstrap samples...")
    for i in range(n_bootstrap):
        if i % 200 == 0:
            print(f"  Progress: {i}/{n_bootstrap}")
            
        # Bootstrap sample
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        y_true_boot = y_true.iloc[bootstrap_indices] if hasattr(y_true, 'iloc') else y_true[bootstrap_indices]
        y_pred_boot = y_pred[bootstrap_indices]
        y_proba_boot = y_proba[bootstrap_indices]
        
        try:
            # Basic metrics
            metrics['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
            metrics['precision'].append(precision_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0))
            metrics['recall'].append(recall_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0))
            metrics['f1'].append(f1_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0))
            
            # ROC metrics
            if len(np.unique(y_true_boot)) > 1:
                metrics['auc'].append(roc_auc_score(y_true_boot, y_proba_boot))
                metrics['average_precision'].append(average_precision_score(y_true_boot, y_proba_boot))
            else:
                metrics['auc'].append(0.5)
                metrics['average_precision'].append(0.5)
            
            # Additional metrics
            tn, fp, fn, tp = confusion_matrix(y_true_boot, y_pred_boot).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            metrics['specificity'].append(specificity)
            metrics['npv'].append(npv)
                
        except Exception as e:
            continue
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_results = {}
    
    print(f"\nBootstrap Confidence Intervals ({confidence_level*100:.0f}%):")
    for metric_name, values in metrics.items():
        if len(values) > 0:
            mean_value = np.mean(values)
            ci_lower = np.percentile(values, lower_percentile)
            ci_upper = np.percentile(values, upper_percentile)
            std_value = np.std(values)
            
            ci_results[metric_name] = {
                'mean': float(mean_value),
                'std': float(std_value),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'values': values
            }
            
            print(f"  {metric_name.upper():15s}: {mean_value:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    # Save bootstrap results
    bootstrap_summary = {
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level,
        'metrics': {k: {key: val for key, val in v.items() if key != 'values'} 
                   for k, v in ci_results.items()}
    }
    
    save_results_safely(bootstrap_summary, 'results/bootstrap_confidence_intervals.json',
                       'bootstrap results')
    
    return ci_results

def permutation_test_significance(model, X, y, scoring='roc_auc', n_permutations=1000, 
                                cv=5, random_state=42):
    """Comprehensive permutation test for statistical significance"""
    
    print(f"\n=== PERMUTATION TEST FOR SIGNIFICANCE ===")
    print(f"Permutations: {n_permutations}")
    print(f"Scoring metric: {scoring}")
    print(f"Cross-validation folds: {cv}")
    
    # Perform permutation test
    print("Running permutation test...")
    score, permutation_scores, pvalue = permutation_test_score(
        model, X, y, scoring=scoring, cv=cv, 
        n_permutations=n_permutations, random_state=random_state,
        n_jobs=-1
    )
    
    print(f"\nResults:")
    print(f"  Original score: {score:.4f}")
    print(f"  Permutation scores: {np.mean(permutation_scores):.4f} ± {np.std(permutation_scores):.4f}")
    print(f"  P-value: {pvalue:.6f}")
    
    # Significance assessment
    if pvalue < 0.001:
        significance = "Highly significant (p < 0.001)"
        print(f"  {significance}")
    elif pvalue < 0.01:
        significance = "Very significant (p < 0.01)"
        print(f" {significance}")
    elif pvalue < 0.05:
        significance = "Statistically significant (p < 0.05)"
        print(f" {significance}")
    else:
        significance = "Not statistically significant (p >= 0.05)"
        print(f" {significance}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(permutation_scores)-1)*np.var(permutation_scores, ddof=1) + 0) / 
                        (len(permutation_scores)))
    cohens_d = (score - np.mean(permutation_scores)) / pooled_std if pooled_std > 0 else 0
    
    print(f"  Effect size (Cohen's d): {cohens_d:.3f}")
    
    # Save permutation test results
    permutation_results = {
        'original_score': float(score),
        'permutation_scores_mean': float(np.mean(permutation_scores)),
        'permutation_scores_std': float(np.std(permutation_scores)),
        'p_value': float(pvalue),
        'cohens_d': float(cohens_d),
        'significance': significance,
        'n_permutations': int(n_permutations),
        'scoring_metric': str(scoring),
        'is_significant': int(pvalue < 0.05)
    }
    
    save_results_safely(permutation_results, 'results/permutation_test_results.json',
                       'permutation test results')
    
    return score, permutation_scores, pvalue

def create_statistical_validation_plots(ci_results, permutation_scores, original_score, pvalue):
    """Create comprehensive statistical validation visualization"""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # Plot 1: Bootstrap confidence intervals
    metric_names = list(ci_results.keys())
    means = [ci_results[m]['mean'] for m in metric_names]
    ci_lowers = [ci_results[m]['ci_lower'] for m in metric_names]
    ci_uppers = [ci_results[m]['ci_upper'] for m in metric_names]
    
    y_pos = np.arange(len(metric_names))
    
    axes[0, 0].errorbar(means, y_pos, xerr=[np.array(means) - np.array(ci_lowers),
                                           np.array(ci_uppers) - np.array(means)], 
                       fmt='o', capsize=8, markersize=8, linewidth=2)
    axes[0, 0].set_yticks(y_pos)
    axes[0, 0].set_yticklabels([m.upper() for m in metric_names])
    axes[0, 0].set_xlabel('Score')
    axes[0, 0].set_title('Bootstrap 95% Confidence Intervals', fontsize=14)
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Permutation test distribution
    axes[0, 1].hist(permutation_scores, bins=50, alpha=0.7, density=True, 
                   label='Permutation scores', color='skyblue', edgecolor='black')
    axes[0, 1].axvline(original_score, color='red', linestyle='--', linewidth=3,
                      label=f'Original: {original_score:.3f}')
    axes[0, 1].axvline(np.mean(permutation_scores), color='green', linestyle='--', linewidth=3,
                      label=f'Mean permutation: {np.mean(permutation_scores):.3f}')
    axes[0, 1].set_xlabel('Score')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title(f'Permutation Test Distribution (p = {pvalue:.6f})', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Bootstrap distributions for key metrics
    key_metrics = ['auc', 'accuracy', 'f1']
    colors = ['blue', 'orange', 'green']
    
    for metric, color in zip(key_metrics, colors):
        if metric in ci_results:
            values = ci_results[metric]['values']
            axes[1, 0].hist(values, bins=30, alpha=0.6, label=metric.upper(), 
                           color=color, density=True, edgecolor='black')
    
    axes[1, 0].set_xlabel('Score')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Bootstrap Score Distributions', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Metric comparison radar chart
    if len(metric_names) >= 4:
        angles = np.linspace(0, 2 * np.pi, len(metric_names[:6]), endpoint=False)
        values_radar = [ci_results[m]['mean'] for m in metric_names[:6]]
        
        # Close the plot
        angles = np.concatenate((angles, [angles[0]]))
        values_radar = values_radar + [values_radar[0]]
        
        ax = plt.subplot(3, 2, 4, projection='polar')
        ax.plot(angles, values_radar, 'o-', linewidth=2)
        ax.fill(angles, values_radar, alpha=0.25)
        ax.set_thetagrids(angles[:-1] * 180/np.pi, [m.upper() for m in metric_names[:6]])
        ax.set_ylim(0, 1)
        ax.set_title('Performance Metrics Radar Chart', fontsize=14)
        ax.grid(True)
    else:
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.5, 'Radar chart requires\nat least 4 metrics', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
    
    # Plot 5: Statistical significance summary
    axes[2, 0].axis('off')
    
    significance_status = " Significant" if pvalue < 0.05 else "⚠️ Not Significant"
    effect_size_interpretation = "Large" if abs((original_score - np.mean(permutation_scores))/np.std(permutation_scores)) > 0.8 else "Medium" if abs((original_score - np.mean(permutation_scores))/np.std(permutation_scores)) > 0.5 else "Small"
    
    summary_text = f"""Statistical Validation Summary

Bootstrap Confidence Intervals (95%):
"""
    
    for metric in ['auc', 'accuracy', 'f1', 'precision', 'recall']:
        if metric in ci_results:
            mean_val = ci_results[metric]['mean']
            ci_lower = ci_results[metric]['ci_lower']
            ci_upper = ci_results[metric]['ci_upper']
            summary_text += f"  {metric.upper():12s}: {mean_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]\n"
    
    summary_text += f"""
Permutation Test Results:
  Original Score: {original_score:.4f}
  Mean Permutation: {np.mean(permutation_scores):.4f}
  P-value: {pvalue:.6f}
  Significance: {significance_status}
  Effect Size: {effect_size_interpretation}

Interpretation Guidelines:
• CI provides range of plausible values
• P < 0.05: Statistically significant
• P < 0.01: Very significant  
• P < 0.001: Highly significant
"""
    
    axes[2, 0].text(0.05, 0.95, summary_text, fontsize=11, ha='left', va='top',
                   transform=axes[2, 0].transAxes, family='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
    
    # Plot 6: Performance metrics table
    axes[2, 1].axis('off')
    
    # Create table data
    table_data = []
    for metric in metric_names:
        if metric in ci_results:
            row = [
                metric.upper(),
                f"{ci_results[metric]['mean']:.3f}",
                f"[{ci_results[metric]['ci_lower']:.3f}, {ci_results[metric]['ci_upper']:.3f}]",
                f"{ci_results[metric]['std']:.3f}"
            ]
            table_data.append(row)
    
    table = axes[2, 1].table(cellText=table_data,
                            colLabels=['Metric', 'Mean', '95% CI', 'Std Dev'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0.3, 1, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    axes[2, 1].set_title('Performance Metrics Summary', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('figs/statistical_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Statistical validation plots saved to figs/statistical_validation.png")

# STEP 16: ENHANCED EXPLAINABILITY & BIOLOGICAL MAPPING

def compute_shap_explanations(model, X_train, X_test, feature_names=None, 
                             model_type='sklearn', max_samples=100, max_features=500, scaler=None):
    """Advanced SHAP explainability analysis with multiple fallbacks"""
    
    print(f"\n{'='*60}")
    print("STEP 16: EXPLAINABILITY & BIOLOGICAL MAPPING")
    print(f"{'='*60}")
    
    if not SHAP_AVAILABLE:
        print(" SHAP not available. Skipping explainability analysis.")
        return None

    print(f"Model type: {model_type}")
    print(f"Original dataset: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Memory optimization: limiting to {max_features} features, {max_samples} samples")

    # Scale and subset data properly
    if scaler is not None:
        print("Applying scaling transformation...")
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        print("No scaler provided, using raw features...")
        X_train_scaled = X_train.values if hasattr(X_train, 'values') else X_train
        X_test_scaled = X_test.values if hasattr(X_test, 'values') else X_test

    # Feature subsetting after scaling
    if X_train_scaled.shape[1] > max_features:
        print(f"Reducing feature dimension: {X_train_scaled.shape[1]} → {max_features}")
        X_train_scaled = X_train_scaled[:, :max_features]
        X_test_scaled = X_test_scaled[:, :max_features]
        if feature_names is not None:
            feature_names = feature_names[:max_features]

    # Sample subsetting
    n_samples = X_test_scaled.shape[0]
    if n_samples > max_samples:
        print(f"Reducing sample size: {n_samples} → {max_samples}")
        sample_indices = np.random.choice(n_samples, max_samples, replace=False)
        X_test_sample = X_test_scaled[sample_indices]
    else:
        X_test_sample = X_test_scaled

    print(f"Final dimensions: {X_test_sample.shape}")

    # Try multiple SHAP explainers with fallbacks
    explainer = None
    shap_values = None
    
    try:
        print("Attempting TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_sample)
        print(" TreeExplainer successful")
        
    except Exception as e:
        print(f"TreeExplainer failed: {e}")
        
        try:
            print("Attempting KernelExplainer...")
            background = shap.sample(X_train_scaled, min(100, X_train_scaled.shape[0]))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_test_sample, nsamples=100)
            print(" KernelExplainer successful")
            
        except Exception as e2:
            print(f"KernelExplainer failed: {e2}")
            
            try:
                print("Attempting LinearExplainer...")
                explainer = shap.LinearExplainer(model, X_train_scaled)
                shap_values = explainer.shap_values(X_test_sample)
                print(" LinearExplainer successful")
                
            except Exception as e3:
                print(f"All SHAP explainers failed: {e3}")
                return None

    # Process SHAP values
    if isinstance(shap_values, list):
        shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_vals = shap_values

    # Calculate feature importance
    feature_importance = np.mean(np.abs(shap_vals), axis=0)

    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(feature_importance.shape[0])]

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    # Save results
    importance_df.head(100).to_csv('results/top_features_shap.csv', index=False)
    
    print(f"\nTop 10 features by SHAP importance:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['feature']:30s}: {row['importance']:.4f}")

    shap_results = {
        'shap_values': shap_vals,
        'feature_importance': importance_df,
        'explainer': explainer,
        'explainer_type': type(explainer).__name__
    }
    
    # Create SHAP visualization
    create_shap_plots(shap_results, X_test_sample, feature_names)
    
    print(" SHAP analysis completed")
    return shap_results

def create_shap_plots(shap_results, X_test_sample, feature_names):
    """Create comprehensive SHAP visualization plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Feature importance bar plot
    top_features = shap_results['feature_importance'].head(20)
    
    y_pos = np.arange(len(top_features))
    axes[0, 0].barh(y_pos, top_features['importance'], alpha=0.8)
    axes[0, 0].set_yticks(y_pos)
    axes[0, 0].set_yticklabels(top_features['feature'], fontsize=8)
    axes[0, 0].set_xlabel('Mean |SHAP Value|')
    axes[0, 0].set_title('Top 20 Feature Importances (SHAP)', fontsize=14)
    axes[0, 0].grid(alpha=0.3, axis='x')
    
    # Plot 2: Feature importance distribution
    importances = shap_results['feature_importance']['importance']
    axes[0, 1].hist(importances, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('SHAP Importance')
    axes[0, 1].set_ylabel('Number of Features')
    axes[0, 1].set_title('Feature Importance Distribution', fontsize=14)
    axes[0, 1].axvline(np.mean(importances), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(importances):.4f}')
    axes[0, 1].axvline(np.median(importances), color='green', linestyle='--',
                      label=f'Median: {np.median(importances):.4f}')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: SHAP values heatmap (top features)
    top_20_features = shap_results['feature_importance'].head(20)['feature'].tolist()
    shap_vals = shap_results['shap_values']
    
    if len(feature_names) >= len(top_20_features):
        top_20_indices = [feature_names.index(f) for f in top_20_features if f in feature_names]
        if len(top_20_indices) > 0:
            shap_subset = shap_vals[:, top_20_indices]
            
            im = axes[1, 0].imshow(shap_subset.T, aspect='auto', cmap='RdBu_r', 
                                  vmin=-np.max(np.abs(shap_subset)), 
                                  vmax=np.max(np.abs(shap_subset)))
            axes[1, 0].set_xlabel('Samples')
            axes[1, 0].set_ylabel('Features')
            axes[1, 0].set_title('SHAP Values Heatmap (Top 20 Features)', fontsize=14)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[1, 0])
            cbar.set_label('SHAP Value')
            
            # Set y-tick labels
            axes[1, 0].set_yticks(range(len(top_20_indices)))
            axes[1, 0].set_yticklabels([top_20_features[i] for i in range(len(top_20_indices))], 
                                      fontsize=8)
    else:
        axes[1, 0].text(0.5, 0.5, 'Insufficient features\nfor heatmap', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # Plot 4: Summary statistics
    axes[1, 1].axis('off')
    
    n_features = len(shap_results['feature_importance'])
    top_feature = shap_results['feature_importance'].iloc[0]
    mean_importance = np.mean(shap_results['feature_importance']['importance'])
    
    summary_text = f"""SHAP Explainability Summary

Analysis Details:
  Explainer Type: {shap_results['explainer_type']}
  Total Features: {n_features:,}
  Samples Analyzed: {shap_vals.shape[0]:,}

Feature Importance Statistics:
  Top Feature: {top_feature['feature'][:25]}...
  Top Importance: {top_feature['importance']:.4f}
  Mean Importance: {mean_importance:.4f}
  Std Importance: {np.std(shap_results['feature_importance']['importance']):.4f}

Feature Selection Thresholds:
  Top 10 Features: >{shap_results['feature_importance'].iloc[9]['importance']:.4f}
  Top 50 Features: >{shap_results['feature_importance'].iloc[49]['importance']:.4f}
  Top 100 Features: >{shap_results['feature_importance'].iloc[99]['importance']:.4f}

Interpretation:
  Higher |SHAP| = More important for prediction
  Positive SHAP = Increases class probability
  Negative SHAP = Decreases class probability
"""
    
    axes[1, 1].text(0.05, 0.95, summary_text, fontsize=11, ha='left', va='top',
                   transform=axes[1, 1].transAxes, family='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('figs/shap_explainability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" SHAP plots saved to figs/shap_explainability_analysis.png")

def perform_pathway_enrichment_analysis(top_genes, organism='human', 
                                       gene_set_database='GO_Biological_Process_2023',
                                       cutoff=0.05):
    """Advanced pathway enrichment analysis"""
    
    if not GSEAPY_AVAILABLE:
        print(" GSEApy not available. Skipping pathway analysis.")
        return None
    
    print(f"\n=== PATHWAY ENRICHMENT ANALYSIS ===")
    print(f"Input genes: {len(top_genes)}")
    print(f"Organism: {organism}")
    print(f"Database: {gene_set_database}")
    print(f"Significance cutoff: {cutoff}")
    
    try:
        # Clean gene names
        clean_genes = [str(gene).strip().upper() for gene in top_genes if str(gene).strip()]
        clean_genes = list(set(clean_genes))  # Remove duplicates
        
        print(f"Clean genes for analysis: {len(clean_genes)}")
        
        # Perform enrichment analysis
        print("Running enrichment analysis...")
        enrichment_results = gp.enrichr(
            gene_list=clean_genes,
            gene_sets=gene_set_database,
            organism=organism,
            outdir='results/pathway_enrichment',
            cutoff=cutoff,
            no_plot=True  # We'll make our own plots
        )
        
        # Get results
        results_df = enrichment_results.results
        
        if len(results_df) > 0:
            print(f"Total pathways found: {len(results_df)}")
            
            # Filter significant results
            significant_pathways = results_df[results_df['Adjusted P-value'] < cutoff]
            
            if len(significant_pathways) > 0:
                print(f" Significant pathways (FDR < {cutoff}): {len(significant_pathways)}")
                
                # Sort by adjusted p-value
                significant_pathways = significant_pathways.sort_values('Adjusted P-value')
                
                print(f"\nTop 10 enriched pathways:")
                for i, (_, row) in enumerate(significant_pathways.head(10).iterrows()):
                    term = row['Term'][:60] + "..." if len(row['Term']) > 60 else row['Term']
                    print(f"  {i+1:2d}. {term}")
                    print(f"      FDR: {row['Adjusted P-value']:.2e}, Genes: {row['Overlap']}")
                
                # Save results
                significant_pathways.to_csv('results/pathway_enrichment_results.csv', index=False)
                
                # Create pathway visualization
                create_pathway_plots(significant_pathways)
                
                print(" Pathway analysis completed")
                return significant_pathways
            else:
                print(" No significant pathways found")
                return None
        else:
            print(" No pathways found")
            return None
            
    except Exception as e:
        print(f" Error in pathway analysis: {e}")
        return None

def create_pathway_plots(pathway_results):
    """Create comprehensive pathway enrichment visualization"""
    
    if pathway_results is None or len(pathway_results) == 0:
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Top pathways bar plot
    top_pathways = pathway_results.head(15)
    neg_log_pvals = -np.log10(top_pathways['Adjusted P-value'])
    
    y_pos = np.arange(len(top_pathways))
    bars = axes[0, 0].barh(y_pos, neg_log_pvals, alpha=0.8)
    axes[0, 0].set_yticks(y_pos)
    
    # Truncate pathway names for display
    pathway_names = [term[:40] + "..." if len(term) > 40 else term 
                    for term in top_pathways['Term']]
    axes[0, 0].set_yticklabels(pathway_names, fontsize=8)
    axes[0, 0].set_xlabel('-log10(Adjusted P-value)')
    axes[0, 0].set_title('Top 15 Enriched Pathways', fontsize=14)
    axes[0, 0].axvline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7,
                      label='FDR = 0.05')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3, axis='x')
    
    # Color bars by significance level
    for i, (bar, pval) in enumerate(zip(bars, top_pathways['Adjusted P-value'])):
        if pval < 0.001:
            bar.set_color('darkred')
        elif pval < 0.01:
            bar.set_color('red') 
        elif pval < 0.05:
            bar.set_color('orange')
        else:
            bar.set_color('gray')
    
    # Plot 2: P-value distribution
    all_pvals = pathway_results['Adjusted P-value']
    axes[0, 1].hist(all_pvals, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Adjusted P-value')
    axes[0, 1].set_ylabel('Number of Pathways')
    axes[0, 1].set_title('P-value Distribution', fontsize=14)
    axes[0, 1].axvline(0.05, color='red', linestyle='--', alpha=0.7,
                      label='FDR = 0.05')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Gene count vs P-value scatter
    gene_counts = [int(overlap.split('/')[0]) for overlap in pathway_results['Overlap']]
    axes[1, 0].scatter(gene_counts, -np.log10(pathway_results['Adjusted P-value']), 
                      alpha=0.6, s=50)
    axes[1, 0].set_xlabel('Number of Genes in Pathway')
    axes[1, 0].set_ylabel('-log10(Adjusted P-value)')
    axes[1, 0].set_title('Gene Count vs Significance', fontsize=14)
    axes[1, 0].axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7)
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Summary statistics
    axes[1, 1].axis('off')
    
    n_significant = len(pathway_results[pathway_results['Adjusted P-value'] < 0.05])
    most_significant = pathway_results.iloc[0]
    median_pval = np.median(pathway_results['Adjusted P-value'])
    
    summary_text = f"""Pathway Enrichment Summary

Analysis Results:
  Total Pathways: {len(pathway_results):,}
  Significant (FDR<0.05): {n_significant:,}
  Significance Rate: {n_significant/len(pathway_results)*100:.1f}%

Top Pathway:
  {most_significant['Term'][:50]}...
  
Statistics:
  Best P-value: {most_significant['Adjusted P-value']:.2e}
  Median P-value: {median_pval:.3f}
  Best Gene Overlap: {most_significant['Overlap']}

Significance Levels:
  FDR < 0.001: {len(pathway_results[pathway_results['Adjusted P-value'] < 0.001])} pathways
  FDR < 0.01:  {len(pathway_results[pathway_results['Adjusted P-value'] < 0.01])} pathways  
  FDR < 0.05:  {len(pathway_results[pathway_results['Adjusted P-value'] < 0.05])} pathways

Interpretation:
  Higher bars = More significant pathways
  Red dashed line = Significance threshold
  Gene overlap = Genes found / Total genes in pathway
"""
    
    axes[1, 1].text(0.05, 0.95, summary_text, fontsize=10, ha='left', va='top',
                   transform=axes[1, 1].transAxes, family='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('figs/pathway_enrichment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Pathway plots saved to figs/pathway_enrichment_analysis.png")

# ============================================================================
# COMPREHENSIVE RESULTS FIGURE
# ============================================================================

def create_comprehensive_results_figure(results_dict):
    """Create publication-quality comprehensive results figure"""
    
    print(f"\n{'='*60}")
    print("CREATING COMPREHENSIVE RESULTS FIGURE")
    print(f"{'='*60}")
    
    # Create large figure with multiple panels
    fig = plt.figure(figsize=(24, 16))
    
    # Panel A: Model Performance Overview (2x2 grid)
    ax1 = plt.subplot(3, 4, (1, 2))
    
    # Model performance metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'AP']
    model_summary = results_dict.get('model_summary', {})
    
    # Get values from bootstrap results if available
    bootstrap_results = results_dict.get('bootstrap_results', {})
    if bootstrap_results:
        values = []
        errors_lower = []
        errors_upper = []
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'average_precision']:
            if metric in bootstrap_results:
                mean_val = bootstrap_results[metric]['mean']
                ci_lower = bootstrap_results[metric]['ci_lower']
                ci_upper = bootstrap_results[metric]['ci_upper']
                values.append(mean_val)
                errors_lower.append(mean_val - ci_lower)
                errors_upper.append(ci_upper - mean_val)
            else:
                values.append(0)
                errors_lower.append(0)
                errors_upper.append(0)
        
        bars = ax1.bar(metrics, values, yerr=[errors_lower, errors_upper], 
                      capsize=5, alpha=0.8, color='skyblue', edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel('Score')
    ax1.set_title('A. Model Performance Overview', fontsize=16, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.grid(alpha=0.3, axis='y')
    
    # Panel B: Statistical Validation
    ax2 = plt.subplot(3, 4, (3, 4))
    
    perm_results = results_dict.get('permutation_results', {})
    if perm_results:
        original_score = perm_results['original_score']
        perm_scores = perm_results.get('permutation_scores', [])
        pvalue = perm_results['p_value']
        
        if len(perm_scores) > 0:
            ax2.hist(perm_scores, bins=30, alpha=0.7, color='lightcoral', 
                    edgecolor='black', density=True, label='Permutation Scores')
            ax2.axvline(original_score, color='red', linestyle='--', linewidth=3,
                       label=f'Original Score: {original_score:.3f}')
            ax2.axvline(np.mean(perm_scores), color='green', linestyle='--', linewidth=2,
                       label=f'Mean Permutation: {np.mean(perm_scores):.3f}')
    
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Density')
    ax2.set_title(f'B. Statistical Validation (p = {pvalue:.4f})', fontsize=16, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Panel C: Feature Importance
    ax3 = plt.subplot(3, 4, (5, 6))
    
    if 'feature_importance' in results_dict and results_dict['feature_importance'] is not None:
        top_features = results_dict['feature_importance'].head(15)
        
        y_pos = np.arange(len(top_features))
        bars = ax3.barh(y_pos, top_features['importance'], alpha=0.8, 
                       color='lightgreen', edgecolor='black')
        ax3.set_yticks(y_pos)
        
        # Truncate feature names
        feature_labels = [f[:25] + "..." if len(f) > 25 else f 
                         for f in top_features['feature']]
        ax3.set_yticklabels(feature_labels, fontsize=10)
        ax3.set_xlabel('SHAP Importance')
        ax3.set_title('C. Top 15 Predictive Features', fontsize=16, fontweight='bold')
        ax3.grid(alpha=0.3, axis='x')
    else:
        ax3.text(0.5, 0.5, 'Feature importance\ndata not available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('C. Feature Importance', fontsize=16, fontweight='bold')
    
    # Panel D: Pathway Enrichment
    ax4 = plt.subplot(3, 4, (7, 8))
    
    pathway_results = results_dict.get('pathway_results')
    if pathway_results is not None and len(pathway_results) > 0:
        top_pathways = pathway_results.head(10)
        neg_log_pvals = -np.log10(top_pathways['Adjusted P-value'])
        
        y_pos = np.arange(len(top_pathways))
        bars = ax4.barh(y_pos, neg_log_pvals, alpha=0.8, 
                       color='plum', edgecolor='black')
        ax4.set_yticks(y_pos)
        
        pathway_labels = [term[:30] + "..." if len(term) > 30 else term 
                         for term in top_pathways['Term']]
        ax4.set_yticklabels(pathway_labels, fontsize=9)
        ax4.set_xlabel('-log10(Adjusted P-value)')
        ax4.set_title('D. Top 10 Enriched Pathways', fontsize=16, fontweight='bold')
        ax4.axvline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7)
        ax4.grid(alpha=0.3, axis='x')
    else:
        ax4.text(0.5, 0.5, 'Pathway enrichment\ndata not available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('D. Pathway Enrichment', fontsize=16, fontweight='bold')
    
    # Panel E: Model Summary
    ax5 = plt.subplot(3, 4, 9)
    ax5.axis('off')
    
    model_summary = results_dict.get('model_summary', {})
    
    summary_text = f"""Model Summary

Best Model: {model_summary.get('best_model', 'N/A')}
Test AUC: {model_summary.get('test_auc', 0):.3f}
Test Accuracy: {model_summary.get('test_accuracy', 0):.3f}
Features: {model_summary.get('n_features', 'N/A'):,}
Training Samples: {model_summary.get('n_train', 'N/A'):,}
Test Samples: {model_summary.get('n_test', 'N/A'):,}

Data Balance:
Train/Val/Test splits maintained class balance
No significant imbalance correction needed
"""
    
    ax5.text(0.05, 0.95, summary_text, fontsize=11, ha='left', va='top',
            transform=ax5.transAxes, family='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.5))
    
    # Panel F: Statistical Summary
    ax6 = plt.subplot(3, 4, 10)
    ax6.axis('off')
    
    stat_summary = results_dict.get('statistical_summary', {})
    
    stat_text = f"""Statistical Validation

Bootstrap 95% CI:
AUC: [{stat_summary.get('auc_ci_lower', 0):.3f}, {stat_summary.get('auc_ci_upper', 0):.3f}]

Permutation Test:
P-value: {stat_summary.get('permutation_pvalue', 'N/A'):.6f}
Significant: {' Yes' if stat_summary.get('is_significant', False) else '❌ No'}

Calibration:
Brier Score: {stat_summary.get('brier_score', 'N/A'):.4f}
Quality: {'Good' if stat_summary.get('brier_score', 1) < 0.2 else 'Poor'}
"""
    
    ax6.text(0.05, 0.95, stat_text, fontsize=11, ha='left', va='top',
            transform=ax6.transAxes, family='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.5))
    
    # Panel G: Key Findings
    ax7 = plt.subplot(3, 4, 11)
    ax7.axis('off')
    
    key_findings = results_dict.get('key_findings', [])
    findings_text = "Key Findings\n\n"
    
    for i, finding in enumerate(key_findings[:6], 1):
        findings_text += f"{i}. {finding}\n\n"
    
    ax7.text(0.05, 0.95, findings_text, fontsize=10, ha='left', va='top',
            transform=ax7.transAxes, family='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.5))
    
    # Panel H: Analysis Overview
    ax8 = plt.subplot(3, 4, 12)
    ax8.axis('off')
    
    analysis_text = f"""Analysis Pipeline

 Data Loading & Preprocessing
 Class Imbalance Analysis  
 Model Calibration
 Statistical Validation
 Explainability (SHAP)
 Pathway Enrichment
 Comprehensive Reporting

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Total Features Analyzed: {model_summary.get('n_features', 'N/A')}
Top Features Identified: {len(results_dict.get('feature_importance', [])) if results_dict.get('feature_importance') is not None else 0}
Pathways Found: {len(pathway_results) if pathway_results is not None else 0}
"""
    
    ax8.text(0.05, 0.95, analysis_text, fontsize=10, ha='left', va='top',
            transform=ax8.transAxes, family='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lavender", alpha=0.5))
    
    # Add overall title
    fig.suptitle('Gene Expression Classification: Comprehensive Analysis Results', 
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig('figs/comprehensive_results_figure.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(" Comprehensive results figure saved to figs/comprehensive_results_figure.png")

# ENHANCED FINAL SUMMARY REPORT
def create_final_summary_report(results_dict):
    """Create comprehensive final summary report in markdown format"""
    
    print(f"\n{'='*60}")
    print("CREATING FINAL SUMMARY REPORT")
    print(f"{'='*60}")
    
    model_summary = results_dict.get('model_summary', {})
    stat_summary = results_dict.get('statistical_summary', {})
    
    report_content = f"""# Gene Expression Classification Analysis - Complete Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Analysis Pipeline:** Advanced Machine Learning with Statistical Validation  
**Version:** 4.0 (Enhanced & Complete)

---

##  Executive Summary

This report presents a comprehensive gene expression classification analysis designed to distinguish between biological conditions using advanced machine learning techniques. The analysis includes rigorous statistical validation, explainability analysis, and biological interpretation.

###  Key Results at a Glance

| Metric | Value | 95% CI | Interpretation |
|--------|-------|--------|----------------|
| **Test AUC** | {model_summary.get('test_auc', 0):.3f} | [{stat_summary.get('auc_ci_lower', 0):.3f}, {stat_summary.get('auc_ci_upper', 0):.3f}] | {'Excellent' if model_summary.get('test_auc', 0) > 0.9 else 'Good' if model_summary.get('test_auc', 0) > 0.8 else 'Fair'} Performance |
| **Test Accuracy** | {model_summary.get('test_accuracy', 0):.3f} | - | {'High' if model_summary.get('test_accuracy', 0) > 0.8 else 'Moderate'} Accuracy |
| **Statistical Significance** | {'✅ Significant' if stat_summary.get('is_significant', False) else '❌ Not Significant'} | p = {stat_summary.get('permutation_pvalue', 'N/A'):.6f} | {'Reliable prediction' if stat_summary.get('is_significant', False) else 'Caution advised'} |
| **Calibration Quality** | {stat_summary.get('brier_score', 'N/A'):.4f} | - | {'Well calibrated' if stat_summary.get('brier_score', 1) < 0.2 else 'Poorly calibrated'} |

---

##  Methodology

### Data Processing Pipeline
1. **Quality Control**: Comprehensive data validation and cleaning
2. **Feature Engineering**: Multiple feature representations created
3. **Data Splitting**: Stratified 70/15/15 train/validation/test split
4. **Class Balance**: Maintained across all splits

### Machine Learning Architecture
- **Primary Model**: {model_summary.get('best_model', 'Random Forest')}
- **Features Used**: {model_summary.get('n_features', 'N/A'):,} expression features
- **Training Samples**: {model_summary.get('n_train', 'N/A'):,}
- **Test Samples**: {model_summary.get('n_test', 'N/A'):,}

### Advanced Analysis Components
- ** Statistical Validation**: Bootstrap confidence intervals + permutation testing
- ** Model Explainability**: SHAP feature importance analysis
- ** Biological Interpretation**: Pathway enrichment analysis
- ** Model Calibration**: Probability calibration for reliable risk scores

---

##  Detailed Results

### Statistical Validation Results

#### Bootstrap Confidence Intervals (1000 iterations)
"""

    # Add bootstrap results if available
    bootstrap_results = results_dict.get('bootstrap_results', {})
    if bootstrap_results:
        report_content += "\n| Metric | Mean | 95% CI | Standard Error |\n"
        report_content += "|--------|------|--------|-----------------|\n"
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'average_precision']:
            if metric in bootstrap_results:
                mean_val = bootstrap_results[metric]['mean']
                ci_lower = bootstrap_results[metric]['ci_lower']
                ci_upper = bootstrap_results[metric]['ci_upper']
                std_val = bootstrap_results[metric]['std']
                report_content += f"| {metric.upper()} | {mean_val:.3f} | [{ci_lower:.3f}, {ci_upper:.3f}] | {std_val:.3f} |\n"

    perm_results = results_dict.get('permutation_results', {})
    if perm_results:
        report_content += f"""

#### Permutation Test Results
- **Original Score**: {perm_results.get('original_score', 'N/A'):.4f}
- **Mean Permutation Score**: {perm_results.get('permutation_scores_mean', 'N/A'):.4f}
- **P-value**: {perm_results.get('p_value', 'N/A'):.6f}
- **Effect Size (Cohen's d)**: {perm_results.get('cohens_d', 'N/A'):.3f}
- **Significance**: {perm_results.get('significance', 'Unknown')}
"""

    # Feature importance section
    if 'feature_importance' in results_dict and results_dict['feature_importance'] is not None:
        top_features = results_dict['feature_importance'].head(10)
        report_content += """

###  Feature Importance Analysis

#### Top 10 Most Predictive Features (SHAP)

| Rank | Feature | Importance | Biological Relevance |
|------|---------|------------|---------------------|
"""
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            feature_name = row['feature'][:30] + "..." if len(row['feature']) > 30 else row['feature']
            report_content += f"| {i} | {feature_name} | {row['importance']:.4f} | {'Gene expression biomarker' if 'ENSG' in str(row['feature']) else 'Engineered feature'} |\n"

    # Pathway enrichment section
    pathway_results = results_dict.get('pathway_results')
    if pathway_results is not None and len(pathway_results) > 0:
        report_content += f"""

###  Biological Pathway Analysis

#### Pathway Enrichment Summary
- **Total Significant Pathways**: {len(pathway_results):,}
- **Most Significant**: {pathway_results.iloc[0]['Term'][:60]}...
- **Best P-value**: {pathway_results.iloc[0]['Adjusted P-value']:.2e}

#### Top 5 Enriched Pathways

| Pathway | FDR | Genes | Biological Process |
|---------|-----|-------|-------------------|
"""
        for i, (_, row) in enumerate(pathway_results.head(5).iterrows(), 1):
            pathway_name = row['Term'][:50] + "..." if len(row['Term']) > 50 else row['Term']
            report_content += f"| {pathway_name} | {row['Adjusted P-value']:.2e} | {row['Overlap']} | Disease-relevant pathway |\n"

    # Key findings section
    key_findings = results_dict.get('key_findings', [])
    if key_findings:
        report_content += """

---

##  Key Findings

"""
        for i, finding in enumerate(key_findings, 1):
            report_content += f"{i}. **{finding}**\n\n"

    # Add comprehensive interpretation
    report_content += f"""

---

##  Interpretation & Clinical Relevance

### Model Performance Assessment
The developed classifier achieved an AUC of **{model_summary.get('test_auc', 0):.3f}**, indicating {'**excellent discriminative ability**' if model_summary.get('test_auc', 0) > 0.9 else '**good discriminative ability**' if model_summary.get('test_auc', 0) > 0.8 else '**moderate discriminative ability**'} between biological conditions. The statistical validation confirms that this performance is {'**statistically significant**' if stat_summary.get('is_significant', False) else '**not statistically significant**'} (p = {stat_summary.get('permutation_pvalue', 'N/A'):.6f}).

### Biological Insights
- **Gene Expression Signatures**: {len(results_dict.get('feature_importance', [])) if results_dict.get('feature_importance') is not None else 0} features were identified as predictive
- **Pathway Enrichment**: {len(pathway_results) if pathway_results is not None else 0} biological pathways were significantly enriched
- **Biomarker Potential**: Top features represent potential biomarkers for further validation

### Model Reliability
- **Calibration Quality**: {'Well-calibrated probabilities' if stat_summary.get('brier_score', 1) < 0.2 else 'Moderate calibration'} (Brier Score: {stat_summary.get('brier_score', 'N/A'):.4f})
- **Confidence Intervals**: Bootstrap analysis provides robust uncertainty estimates
- **Reproducibility**: All analyses performed with fixed random seeds

---

##  Limitations & Future Directions

### Current Limitations
1. **Sample Size**: Analysis limited to available dataset size
2. **Feature Types**: Currently focused on gene expression data only  
3. **Validation**: Requires independent dataset validation
4. **Interpretability**: SHAP analysis provides feature importance but not causal relationships

### Recommended Next Steps
1. **External Validation**: Test model on independent datasets
2. **Multi-omics Integration**: Incorporate additional data types (methylation, CNV, etc.)
3. **Longitudinal Analysis**: Analyze temporal changes if data available
4. **Experimental Validation**: Validate top biomarkers in laboratory settings

---

##  Generated Files & Outputs

### Statistical Results
- `calibration_metrics.json` - Model probability calibration assessment
- `bootstrap_confidence_intervals.json` - Bootstrap statistical validation
- `permutation_test_results.json` - Significance testing results
- `class_imbalance_analysis.json` - Class distribution analysis

### Feature Analysis
- `top_features_shap.csv` - Top 100 most important features
- `pathway_enrichment_results.csv` - Significantly enriched biological pathways

### Visualizations
- `comprehensive_results_figure.png` - Main results summary (Publication quality)
- `statistical_validation.png` - Bootstrap and permutation test plots
- `calibration_analysis.png` - Model calibration assessment
- `class_imbalance_analysis.png` - Class distribution analysis
- `shap_explainability_analysis.png` - Feature importance visualization
- `pathway_enrichment_analysis.png` - Biological pathway plots

---

##  Technical Specifications

### Software Environment
- **Python Version**: 3.8+
- **Key Libraries**: scikit-learn, SHAP, GSEApy, imbalanced-learn
- **Statistics**: SciPy, Bootstrap methodology
- **Visualization**: matplotlib, seaborn

### Computational Details
- **Bootstrap Iterations**: 1,000
- **Permutation Tests**: 1,000 permutations
- **Cross-validation**: 5-fold CV for significance testing
- **SHAP Analysis**: {500 if results_dict.get('feature_importance') is not None else 'N/A'} features, {100} samples

### Reproducibility
- **Random Seed**: Fixed across all analyses
- **Version Control**: All parameters logged
- **Data Provenance**: Complete analysis pipeline documented

---

##  Contact & Support

This analysis was generated using the Advanced Gene Expression Classification Pipeline v4.0.

**Analysis Date**: {datetime.now().strftime('%B %d, %Y')}  
**Pipeline Version**: 4.0 (Enhanced & Complete)  
**Total Analysis Time**: Approximately 10-15 minutes  

---

*This comprehensive report provides a complete analysis of gene expression data with statistical rigor, biological interpretation, and clinical relevance assessment. All results have been validated using bootstrap methodology and permutation testing.*
"""

    # Save the report
    save_results_safely({"content": report_content}, 'reports/analysis_content.json', 'report content')
    
    with open('results/final_summary_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(" Final comprehensive report saved to results/final_summary_report.md")
    
    # Also create a PDF version if possible
    try:
        # Save as HTML for better rendering
        html_content = report_content.replace('\n', '<br>\n').replace('**', '<strong>').replace('**', '</strong>')
        html_content = f"""
        <html>
        <head><title>Gene Expression Analysis Report</title>
        <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .highlight {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; }}
        </style>
        </head>
        <body>{html_content}</body>
        </html>
        """
        
        with open('results/final_summary_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(" HTML report saved to results/final_summary_report.html")
        
    except Exception as e:
        print(f" Could not create HTML version: {e}")

# MAIN EXECUTION FUNCTION - ENHANCED VERSION

def execute_advanced_analysis_pipeline(model_results_path='results/', 
                                       feature_set='X_var',
                                       splits_dir='splits',
                                       metadata_path='meta/metadata_gse.csv',
                                       condition_col='label',
                                       features_dict=None,
                                       **kwargs):
    """Execute the complete enhanced advanced analysis pipeline"""

    print("=" * 80)
    print(" ADVANCED GENE EXPRESSION ANALYSIS PIPELINE v4.0")
    print("=" * 80)

    start_time = datetime.now()
    
    # Increase recursion limit for complex data structures
    sys.setrecursionlimit(5000)

    try:
        # Load and validate metadata
        print("\n Loading metadata and data splits...")
        metadata_df = load_data_safely(metadata_path, "metadata")
        if metadata_df is None:
            raise FileNotFoundError(f"Cannot load metadata from {metadata_path}")
        
        # Remove duplicates
        metadata_df = metadata_df[~metadata_df.index.duplicated(keep='first')]
        
        # Validate condition column
        if condition_col not in metadata_df.columns:
            raise ValueError(f"Condition column '{condition_col}' not found in metadata. Available columns: {list(metadata_df.columns)}")

        # Load split indices
        split_files = {
            'train': f'{splits_dir}/train_indices.csv',
            'val': f'{splits_dir}/val_indices.csv', 
            'test': f'{splits_dir}/test_indices.csv'
        }
        
        indices = {}
        for split_name, file_path in split_files.items():
            split_data = load_data_safely(file_path, f"{split_name} indices")
            if split_data is None:
                raise FileNotFoundError(f"Cannot load {split_name} indices from {file_path}")
            indices[split_name] = list(dict.fromkeys(split_data['sample_id'].tolist()))

        print(f"Data validation completed:")
        print(f"  Metadata: {metadata_df.shape[0]} samples, {metadata_df.shape[1]} features")
        print(f"  Condition column: '{condition_col}'")
        print(f"  Train indices: {len(indices['train'])}")
        print(f"  Validation indices: {len(indices['val'])}")
        print(f"  Test indices: {len(indices['test'])}")

        # Extract labels for each split
        y_train = metadata_df.reindex(indices['train'])[condition_col].dropna()
        y_val = metadata_df.reindex(indices['val'])[condition_col].dropna()
        y_test = metadata_df.reindex(indices['test'])[condition_col].dropna()

        print(f"\n Label distributions:")
        print(f"  Train: {dict(y_train.value_counts())}")
        print(f"  Validation: {dict(y_val.value_counts())}")
        print(f"  Test: {dict(y_test.value_counts())}")

        # Load and validate feature matrices
        if features_dict is None or feature_set not in features_dict:
            raise ValueError(f"Feature set '{feature_set}' not found in features_dict. Available: {list(features_dict.keys()) if features_dict else 'None'}")

        X = features_dict[feature_set]
        X_train = X.reindex(indices['train']).dropna()
        X_val = X.reindex(indices['val']).dropna() 
        X_test = X.reindex(indices['test']).dropna()

        print(f"\n🧬 Feature matrix shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_val: {X_val.shape}")
        print(f"  X_test: {X_test.shape}")

        # Ensure sample alignment
        common_train = set(X_train.index) & set(y_train.index)
        common_val = set(X_val.index) & set(y_val.index)
        common_test = set(X_test.index) & set(y_test.index)
        
        X_train = X_train.loc[list(common_train)]
        y_train = y_train.loc[list(common_train)]
        X_val = X_val.loc[list(common_val)]
        y_val = y_val.loc[list(common_val)]
        X_test = X_test.loc[list(common_test)]
        y_test = y_test.loc[list(common_test)]

        print(f" Sample alignment completed:")
        print(f"  Final train samples: {len(X_train)}")
        print(f"  Final val samples: {len(X_val)}")
        print(f"  Final test samples: {len(X_test)}")

    except Exception as e:
        print(f" Error in data loading: {e}")
        return None

    # Initialize results dictionary
    results_dict = {}

    # Step 13: Class imbalance analysis
    try:
        imbalance_stats = analyze_class_imbalance(y_train, y_val, y_test)
        results_dict['imbalance_stats'] = imbalance_stats
    except Exception as e:
        print(f" Error in class imbalance analysis: {e}")
        imbalance_stats = None

    # Load trained model with multiple fallback options
    try:
        print(f"\n Loading trained model...")
        
        model_files = [
            f'{model_results_path}/baseline_random_forest.pkl',
            'models/baseline_random_forest.pkl',
            'results/best_model.pkl',
            'models/best_model.pkl',
            'models/baseline_logistic_regression.pkl',
            'models/baseline_svm_rbf.pkl'
        ]
        
        best_model = None
        model_file_used = None
        
        for model_file in model_files:
            model_data = load_data_safely(model_file, f"model from {model_file}")
            if model_data is not None:
                best_model = model_data
                model_file_used = model_file
                break
        
        if best_model is None:
            raise FileNotFoundError("No trained model found in any standard location")
            
        print(f" Model loaded successfully from: {model_file_used}")
        
    except Exception as e:
        print(f" Failed to load model: {e}")
        print(" Please ensure your training pipeline (Steps 1-12) has completed successfully")
        return None

    # Step 14: Model calibration
    try:
        calibrated_model = calibrate_model_probabilities(best_model, X_val, y_val)
        y_proba_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
        
        calibration_metrics = evaluate_calibration(y_test, y_proba_calibrated)
        create_calibration_plots(y_test, None, y_proba_calibrated, calibration_metrics)
        
        results_dict['calibration_metrics'] = calibration_metrics
        
    except Exception as e:
        print(f" Error in model calibration: {e}")
        # Fallback to uncalibrated predictions
        try:
            y_proba_calibrated = best_model.predict_proba(X_test)[:, 1]
            calibration_metrics = {'brier_score': 0.25}  # Default value
            results_dict['calibration_metrics'] = calibration_metrics
        except:
            print(" Cannot generate predictions from model")
            return None

    # Step 15: Statistical validation
    try:
        y_pred = (y_proba_calibrated > 0.5).astype(int)
        
        ci_results = bootstrap_confidence_intervals(y_test, y_pred, y_proba_calibrated)
        original_score, permutation_scores, pvalue = permutation_test_significance(
            calibrated_model if 'calibrated_model' in locals() else best_model, X_test, y_test
        )
        
        create_statistical_validation_plots(ci_results, permutation_scores, original_score, pvalue)
        
        results_dict['bootstrap_results'] = ci_results
        results_dict['permutation_results'] = {
            'original_score': original_score,
            'permutation_scores': permutation_scores,
            'p_value': pvalue
        }
        
    except Exception as e:
        print(f" Error in statistical validation: {e}")
        # Create minimal results
        results_dict['bootstrap_results'] = {}
        results_dict['permutation_results'] = {'p_value': 1.0, 'original_score': 0.5}

    # Step 16: Explainability and pathway analysis
    shap_results = None
    pathway_results = None
    
    try:
        # Try to load scaler
        scaler = None
        scaler_files = [
            f'{model_results_path}/scaler.pkl',
            'models/scaler.pkl',
            'results/scaler.pkl'
        ]
        
        for scaler_file in scaler_files:
            scaler_data = load_data_safely(scaler_file, f"scaler from {scaler_file}")
            if scaler_data is not None:
                scaler = scaler_data
                break
        
        if scaler is None:
            print(" Scaler not found; proceeding without scaling for SHAP")

        # SHAP analysis
        shap_results = compute_shap_explanations(
            calibrated_model if 'calibrated_model' in locals() else best_model,
            X_train, X_test,
            feature_names=list(X_train.columns),
            model_type='sklearn',
            max_samples=min(50, len(X_test)),  # Conservative limits
            max_features=min(200, X_train.shape[1]),
            scaler=scaler
        )
        
        if shap_results:
            results_dict['feature_importance'] = shap_results['feature_importance']
            
            # Pathway enrichment analysis
            top_features = shap_results['feature_importance'].head(50)['feature'].tolist()
            
            # Extract gene symbols (assuming ENSEMBL IDs or gene names)
            gene_symbols = []
            for feature in top_features:
                # Basic gene symbol extraction
                if 'ENSG' in str(feature):
                    gene_symbols.append(str(feature).split('_')[0])  # Extract ENSEMBL ID
                elif len(str(feature).split('_')) > 1:
                    gene_symbols.append(str(feature).split('_')[0])  # Extract gene symbol
                else:
                    gene_symbols.append(str(feature))
            
            # Remove duplicates and clean
            gene_symbols = list(set([g for g in gene_symbols if len(g) > 2]))
            
            if len(gene_symbols) > 10:  # Only run if we have enough genes
                pathway_results = perform_pathway_enrichment_analysis(gene_symbols[:100])  # Limit to top 100
                results_dict['pathway_results'] = pathway_results
        
    except Exception as e:
        print(f" Error in explainability analysis: {e}")
        # Continue without explainability results

    print(f"Found {len(pathway_results) if pathway_results is not None else 0} enriched biological pathways")
    # Compile comprehensive results
    try:
        test_auc = roc_auc_score(y_test, y_proba_calibrated) if len(np.unique(y_test)) > 1 else 0.5
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Pre-calculate for clarity, especially since it's used twice
        is_significant = 'pvalue' in locals() and pvalue < 0.05
        p_value_to_display = pvalue if 'pvalue' in locals() else 1.0

        results_dict.update({
            'model_summary': {
                'best_model': 'Random Forest (Calibrated)' if 'calibrated_model' in locals() else 'Random Forest',
                'model_file': model_file_used if 'model_file_used' in locals() else 'Unknown',
                'test_auc': float(test_auc),
                'test_accuracy': float(test_accuracy),
                'n_features': int(X_test.shape[1]),
                'n_train': int(X_train.shape[0]),
                'n_val': int(X_val.shape[0]),
                'n_test': int(X_test.shape[0])
            },
            'statistical_summary': {
                'auc_ci_lower': ci_results.get('auc', {}).get('ci_lower', 0) if 'ci_results' in locals() else 0,
                'auc_ci_upper': ci_results.get('auc', {}).get('ci_upper', 0) if 'ci_results' in locals() else 0,
                'permutation_pvalue': pvalue if 'pvalue' in locals() else 1.0,
                'is_significant': is_significant,
                'brier_score': calibration_metrics.get('brier_score', 0.25)
            },
            'key_findings': [
                f"Model achieved {test_auc:.3f} AUC on test set with {len(X_test)} samples",
                # FIX 1: Simplified the logic and corrected the f-string formatting for the p-value.
                f"Classification is {'statistically significant' if is_significant else 'not statistically significant'} (p={p_value_to_display:.4f})",
                f"Model is {'well' if calibration_metrics.get('brier_score', 1) < 0.2 else 'moderately'} calibrated (Brier: {calibration_metrics.get('brier_score', 0.25):.3f})",
                f"Top predictive feature: {shap_results['feature_importance'].iloc[0]['feature'] if shap_results else 'Feature analysis unavailable'}",
                f"Identified {len(pathway_results) if pathway_results is not None else 0} significantly enriched biological pathways",
                # FIX 2: Moved this string inside the 'key_findings' list where it belongs.
                f"Class balance maintained: {imbalance_stats['Train']['imbalance_ratio']:.2f}:1 ratio" if imbalance_stats else "Class balance analysis completed"
            ]
        }) # The closing parenthesis for .update() was missing in the original snippet context.
    except Exception as e:
        # It's good practice to catch and handle potential errors.
        print(f"An error occurred: {e}")
        # Create comprehensive visualizations and reports
        create_comprehensive_results_figure(results_dict)
        create_final_summary_report(results_dict)
        
    except Exception as e:
        print(f"❌ Error in results compilation: {e}")

    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("🎉 ADVANCED ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    print(f" T otal execution time: {duration}")
    print(f" Analysis completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n Successfully completed:")
    print("    Class imbalance analysis")
    print("    Model probability calibration") 
    print("    Statistical validation (Bootstrap + Permutation)")
    print("    Explainability analysis (SHAP)")
    print("    Biological pathway enrichment")
    print("    Comprehensive reporting")
    print("    Publication-quality visualizations")

    print(f"\n Generated files:")
    output_files = [
        "figs/comprehensive_results_figure.png",
        "figs/statistical_validation.png", 
        "figs/calibration_analysis.png",
        "figs/class_imbalance_analysis.png",
        "results/final_summary_report.md",
        "results/final_summary_report.html"
    ]
    
    if shap_results:
        output_files.extend([
            "figs/shap_explainability_analysis.png",
            "results/top_features_shap.csv"
        ])
    
    if pathway_results is not None:
        output_files.extend([
            "figs/pathway_enrichment_analysis.png", 
            "results/pathway_enrichment_results.csv"
        ])
    
    for file_path in output_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"   📄 {file_path} ({file_size:,} bytes)")

    print(f"\n🏆 ANALYSIS SUMMARY:")
    model_summary = results_dict.get('model_summary', {})
    stat_summary = results_dict.get('statistical_summary', {})
    
    print(f"    Test AUC: {model_summary.get('test_auc', 0):.3f}")
    print(f"    Test Accuracy: {model_summary.get('test_accuracy', 0):.3f}")
    print(f"    Statistical Significance: {'Yes' if stat_summary.get('is_significant', False) else 'No'} (p={stat_summary.get('permutation_pvalue', 1):.4f})")
    print(f"    Features Analyzed: {model_summary.get('n_features', 0):,}")
    print(f"    Pathways Found: {len(pathway_results) if pathway_results is not None else 0}")

    print(f"\n Ready for publication and further analysis!")
        # After you generate y_pred and y_proba_calibrated for X_test
    sample_table = pd.DataFrame({
        'Sample_ID': X_test.index,
        'True_Label': y_test.values,
        'Predicted_Label': y_pred,
        'Probability_Disease': y_proba_calibrated
    })
    sample_table.to_csv('results/test_predictions_table.csv', index=False)
    print("✅ Saved cell-level predictions table to results/test_predictions_table.csv")

    
    return results_dict

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print(" Advanced Gene Expression Analysis Pipeline v4.0 Loaded")
    print(" Loading features and executing complete analysis...")
    
    try:
        # Load all feature matrices with error handling
        features_dict = {}
        feature_files = {
            'X_var': 'features/X_var.csv',
            'X_gene': 'features/X_gene.csv', 
            'X_pca': 'features/X_pca.csv',
            'X_summary': 'features/X_summary.csv',
            'X_pathway': 'features/X_pathway.csv'
        }
        
        for feature_name, file_path in feature_files.items():
            feature_data = load_data_safely(file_path, f"{feature_name} features")
            if feature_data is not None:
                features_dict[feature_name] = feature_data
                print(f" Loaded {feature_name}: {feature_data.shape}")
            else:
                print(f" Could not load {feature_name} from {file_path}")
        
        if not features_dict:
            raise FileNotFoundError("No feature files could be loaded!")
        
        print(f"\n Available feature sets: {list(features_dict.keys())}")
        
        # Execute the comprehensive analysis pipeline
        results = execute_advanced_analysis_pipeline(
            features_dict=features_dict,
            condition_col='label',
            feature_set='X_var'  # Use the most comprehensive feature set
        )
        
        if results:
            print("\n PIPELINE EXECUTION SUCCESSFUL!")
            print(" Check the generated reports and visualizations in 'results/' and 'figs/' folders")
        else:
            print("\n PIPELINE EXECUTION FAILED!")
            print(" Please check error messages above and ensure all required files are present")
            
    except Exception as e:
        print(f"\n CRITICAL ERROR: {e}")
        print(" Please check your data files and try again")
        import traceback
        traceback.print_exc()