import os
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import seaborn as sns
import joblib
import json
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, classification_report
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Tkinter errors
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(filename='pipeline.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

ROOT_DIR = os.getcwd()  # or specify your root folder
REQUIRED_DIRS = ['data', 'meta', 'interim', 'artifacts', 'results', 'features', 'figs', 'splits', 'models', 'notebooks']

def setup_directory_structure():
    for dir_path in REQUIRED_DIRS:
        os.makedirs(os.path.join(ROOT_DIR, dir_path), exist_ok=True)
    logging.info("Directory structure created successfully")

def calculate_file_hash(filepath):

    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def create_data_provenance(file_paths, sources, download_dates):
    provenance_data = []
    
    for filepath, source, download_date in zip(file_paths, sources, download_dates):
        if os.path.exists(filepath):
            file_hash = calculate_file_hash(filepath)
            file_size = os.path.getsize(filepath)
            
            provenance_data.append({
                'filename': os.path.basename(filepath),
                'filepath': filepath,
                'source': source,
                'download_date': download_date,
                'file_size_bytes': file_size,
                'sha256_hash': file_hash,
                'verification_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
    # Save provenance information
    provenance_df = pd.DataFrame(provenance_data)
    provenance_df.to_csv('meta/data_provenance.csv', index=False)
    
    # Create README
    readme_content = f"""
# Data Provenance Documentation

## Overview
This document tracks all data files used in the gene expression analysis pipeline.

## Files Processed
{len(provenance_data)} files have been processed and verified.

## Verification
All file hashes verified on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Data Sources
- GSE107011: Gene Expression Omnibus
- Zenodo: Public repository datasets

## File Integrity
SHA256 hashes calculated for all files to ensure data integrity.
"""
    
    with open('README_data.md', 'w') as f:
        f.write(readme_content)
    
    return provenance_df

# Example usage
setup_directory_structure()




def inspect_raw_matrix(filepath, sample_name="dataset"):
    
    # Load the data
    if filepath.endswith('.txt') or filepath.endswith('.tsv'):
        df = pd.read_csv(filepath, sep='\t', index_col=0)
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath, index_col=0)
    else:
        raise ValueError("Unsupported file format")
    
    # Basic shape information
    n_genes, n_samples = df.shape
    print(f"{sample_name} Matrix Dimensions:")
    print(f"Genes (G): {n_genes}")
    print(f"Samples (S): {n_samples}")
    print(f"Total data points: {n_genes * n_samples}")
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    print(f"\nMissing values: {missing_values} ({missing_values/(n_genes*n_samples)*100:.2f}%)")
    
    # Check for duplicate gene IDs
    duplicate_genes = df.index.duplicated().sum()
    print(f"Duplicate gene IDs: {duplicate_genes}")
    
    # Data type and basic statistics
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    
    # Sample of first few rows and columns
    print(f"\nFirst 6x6 sample:")
    print(df.iloc[:6, :6])
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    stats = df.describe()
    print(stats.iloc[:, :5])  # Show first 5 samples
    
    # Save artifacts
    shape_info = {
        'dataset': sample_name,
        'n_genes': n_genes,
        'n_samples': n_samples,
        'missing_values': missing_values,
        'duplicate_genes': duplicate_genes,
        'min_expression': df.min().min(),
        'max_expression': df.max().max(),
        'mean_expression': df.mean().mean()
    }
    
    # Save shape information
    with open(f'artifacts/initial_shape_{sample_name}.txt', 'w') as f:
        for key, value in shape_info.items():
            f.write(f"{key}: {value}\n")
    
    # Save head sample table
    df.head(10).to_csv(f'artifacts/head_sample_table_{sample_name}.csv')
    
    return df, shape_info
import pandas as pd

def clean_ensembl_ids(gene_ids):
    """Strip Ensembl version suffix (e.g., ENSG00000162704.5 → ENSG00000162704)."""
    return [str(gid).split('.')[0] for gid in gene_ids]

def annotate_genes_biomart(gene_ids, chunk_size=150):
    import biomart
    server = biomart.BiomartServer("http://ensembl.org/biomart")
    mart = server.datasets['hsapiens_gene_ensembl']
    attributes = [
        'ensembl_gene_id', 'external_gene_name', 'gene_biotype',
        'chromosome_name', 'start_position', 'end_position', 'strand'
    ]
    results = []
    cleaned = clean_ensembl_ids(gene_ids)
    for start in range(0, len(cleaned), chunk_size):
        chunk = cleaned[start:start+chunk_size]
        try:
            response = mart.search({'filters': {'ensembl_gene_id': chunk},
                                   'attributes': attributes})
            for line in response.iter_lines():
                if line:
                    results.append(line.decode('utf-8').split('\t'))
        except Exception as e:
            print(f"Chunk {start//chunk_size + 1} error: {e}")
            continue
    ann_df = pd.DataFrame(results, columns=attributes)
    ann_df = ann_df[ann_df['ensembl_gene_id'] != '']
    ann_df.drop_duplicates(subset='ensembl_gene_id', inplace=True)
    ann_df.to_csv('interim/ensembl_to_symbol.csv', index=False)
    return ann_df

def add_annotations(expr_df, ann_df, keep_biotype='protein_coding'):
    # Ensure index matches
    expr_df = expr_df.copy()
    expr_df.index = clean_ensembl_ids(expr_df.index)
    # Merge annotation
    merged = expr_df.merge(ann_df[['ensembl_gene_id', 'external_gene_name', 'gene_biotype']], 
                           left_index=True, right_on='ensembl_gene_id', how='left')
    # Optionally filter by biotype
    if keep_biotype:
        merged = merged[merged['gene_biotype'] == keep_biotype]
    merged.to_csv('interim/expr_with_symbols.tsv', sep='\t')
    return merged

# Example usage:
# gene_ids = expr_df.index.tolist()
# ann_df = annotate_genes_biomart(gene_ids, chunk_size=150)
# expr_annotated = add_annotations(expr_df, ann_df, keep_biotype='protein_coding')




import numpy as np

def filter_genes(expr_df, threshold=1, min_fraction=0.1, log_data=False):
    
    n_samples = expr_df.shape[1]
    min_count = int(n_samples * min_fraction)
    # If log2(x+1), threshold should be 1 (e.g., log2(2))
    th = np.log2(threshold+1) if log_data else threshold
    # Boolean mask (genes with enough samples above threshold)
    mask = (expr_df > th).sum(axis=1) >= min_count
    filtered = expr_df[mask]
    filtered_ids = filtered.index.tolist()
    # Save artifacts
    pd.DataFrame({'gene_id': filtered_ids}).to_csv('interim/filtered_genes_list.csv', index=False)
    stats = {
        "genes_before": expr_df.shape[0],
        "genes_after": filtered.shape[0],
        "threshold": th,
        "min_fraction": min_fraction
    }
    with open('artifacts/filter_stats.txt', 'w') as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
    return filtered

# Example usage:
# filtered_expr = filter_genes(expr_annotated.set_index('ensembl_gene_id').iloc[:, :-2], threshold=1, min_fraction=0.1, log_data=False)




from rnanorm import TPM, CPM
import matplotlib.pyplot as plt
import seaborn as sns

def normalize_expression_data(expr_df, method='tpm', gtf_path=None, 
                            apply_log_transform=True, apply_zscore=False, 
                            save_plots=True):
    
    print(f"Starting normalization with method: {method}")
    print(f"Input shape: {expr_df.shape}")
    
    # Store original for plotting
    original_df = expr_df.copy()
    
    if method == 'tpm':
        if gtf_path is None:
            raise ValueError("GTF file path required for TPM normalization")
        normalizer = TPM(gtf_path)
        # TPM expects samples in rows, genes in columns
        expr_transposed = expr_df.T
        normalized_transposed = normalizer.fit_transform(expr_transposed)
        normalized_df = normalized_transposed.T
        print("Applied TPM normalization")
        
    elif method == 'cpm':
        normalizer = CPM()
        # CPM expects samples in rows, genes in columns
        expr_transposed = expr_df.T
        normalized_transposed = normalizer.fit_transform(expr_transposed)
        normalized_df = normalized_transposed.T
        print("Applied CPM normalization")
        
    elif method == 'assume_normalized':
        normalized_df = expr_df.copy()
        print("Assuming data is already normalized")
        
    else:
        raise ValueError("Method must be 'tpm', 'cpm', or 'assume_normalized'")
    
    # Apply log transformation
    if apply_log_transform:
        # Check if data appears to be already log-transformed
        max_val = normalized_df.max().max()
        if max_val < 50:  # Likely already log-transformed
            print("Data appears to be log-transformed already (max value < 50)")
            log_transformed_df = normalized_df.copy()
        else:
            log_transformed_df = np.log2(normalized_df + 1)
            print("Applied log2(x+1) transformation")
    else:
        log_transformed_df = normalized_df.copy()
    
    # Apply z-score normalization per gene (across samples)
    if apply_zscore:
        zscore_df = log_transformed_df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        final_df = zscore_df
        print("Applied per-gene z-score normalization")
    else:
        final_df = log_transformed_df
    
    # Save intermediate results
    normalized_df.to_csv('interim/expr_normalized.tsv', sep='\t')
    if apply_log_transform:
        log_transformed_df.to_csv('interim/expr_log2.tsv', sep='\t')
    if apply_zscore:
        final_df.to_csv('interim/expr_final_normalized.tsv', sep='\t')
    
    # Create comprehensive visualization
    if save_plots:
        create_normalization_plots(original_df, normalized_df, log_transformed_df, 
                                 final_df, method)
    
    print(f"Final normalized shape: {final_df.shape}")
    print(f"Final value range: {final_df.min().min():.3f} to {final_df.max().max():.3f}")
    
    return final_df

def create_normalization_plots(original_df, normalized_df, log_df, final_df, method):
    """Create comprehensive normalization visualization plots"""
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Distribution before and after normalization
    plt.subplot(3, 4, 1)
    plt.hist(original_df.values.flatten(), bins=100, alpha=0.7, density=True)
    plt.title('Original Distribution')
    plt.xlabel('Expression Value')
    plt.ylabel('Density')
    plt.yscale('log')
    
    plt.subplot(3, 4, 2)
    plt.hist(normalized_df.values.flatten(), bins=100, alpha=0.7, density=True, color='green')
    plt.title(f'After {method.upper()} Normalization')
    plt.xlabel('Expression Value')
    plt.ylabel('Density')
    plt.yscale('log')
    
    plt.subplot(3, 4, 3)
    plt.hist(log_df.values.flatten(), bins=100, alpha=0.7, density=True, color='orange')
    plt.title('After Log2 Transformation')
    plt.xlabel('Log2 Expression Value')
    plt.ylabel('Density')
    
    plt.subplot(3, 4, 4)
    plt.hist(final_df.values.flatten(), bins=100, alpha=0.7, density=True, color='red')
    plt.title('Final Normalized')
    plt.xlabel('Final Expression Value')
    plt.ylabel('Density')
    
    # 2. Sample-wise distributions
    sample_subset = min(10, original_df.shape[1])
    
    plt.subplot(3, 4, 5)
    for i in range(sample_subset):
        plt.hist(original_df.iloc[:, i], bins=50, alpha=0.3, density=True)
    plt.title('Original: Sample Distributions')
    plt.xlabel('Expression')
    plt.ylabel('Density')
    
    plt.subplot(3, 4, 6)
    for i in range(sample_subset):
        plt.hist(normalized_df.iloc[:, i], bins=50, alpha=0.3, density=True)
    plt.title(f'{method.upper()}: Sample Distributions')
    plt.xlabel('Expression')
    plt.ylabel('Density')
    
    plt.subplot(3, 4, 7)
    for i in range(sample_subset):
        plt.hist(log_df.iloc[:, i], bins=50, alpha=0.3, density=True)
    plt.title('Log2: Sample Distributions')
    plt.xlabel('Expression')
    plt.ylabel('Density')
    
    plt.subplot(3, 4, 8)
    for i in range(sample_subset):
        plt.hist(final_df.iloc[:, i], bins=50, alpha=0.3, density=True)
    plt.title('Final: Sample Distributions')
    plt.xlabel('Expression')
    plt.ylabel('Density')
    
    # 3. Sample correlation heatmaps
    sample_corr_orig = original_df.T.corr()
    sample_corr_final = final_df.T.corr()
    
    plt.subplot(3, 4, 9)
    sns.heatmap(sample_corr_orig.iloc[:20, :20], cmap='viridis', cbar=False)
    plt.title('Original: Sample Correlations')
    
    plt.subplot(3, 4, 10)
    sns.heatmap(sample_corr_final.iloc[:20, :20], cmap='viridis', cbar=False)
    plt.title('Final: Sample Correlations')
    
    # 4. Library size effects
    plt.subplot(3, 4, 11)
    lib_sizes_orig = original_df.sum(axis=0)
    lib_sizes_final = final_df.sum(axis=0)
    plt.scatter(lib_sizes_orig, lib_sizes_final, alpha=0.6)
    plt.xlabel('Original Library Size')
    plt.ylabel('Final Library Size')
    plt.title('Library Size Changes')
    
    # 5. Variance stabilization
    plt.subplot(3, 4, 12)
    gene_means_orig = original_df.mean(axis=1)
    gene_vars_orig = original_df.var(axis=1)
    gene_means_final = final_df.mean(axis=1)
    gene_vars_final = final_df.var(axis=1)
    
    plt.scatter(gene_means_orig, gene_vars_orig, alpha=0.3, s=1, label='Original')
    plt.scatter(gene_means_final, gene_vars_final, alpha=0.3, s=1, label='Final')
    plt.xlabel('Mean Expression')
    plt.ylabel('Variance')
    plt.title('Mean-Variance Relationship')
    plt.legend()
    plt.loglog()
    
    plt.tight_layout()
    plt.savefig('figs/density_before_after.png', dpi=300, bbox_inches='tight')
    plt.close()

# Example usage
# normalized_expr = normalize_expression_data(filtered_expr, method='tpm', 
#                                           gtf_path='path/to/genes.gtf')



def execute_advanced_analysis(trained_models, splits, features_dict, best_model='Random_Forest',
                         handle_imbalance=True, imbalance_method='smote',
                         calibrate_model=True, calibration_method='platt',
                         explain_model=True, model_type_for_shap='tree',
                         n_bootstrap=1000, create_publication_materials=True):
    """
    Execute advanced analysis on trained models with various enhancements.
    
    Parameters:
    -----------
    trained_models : dict
        Dictionary of trained models
    splits : dict
        Dictionary containing train/val/test splits
    features_dict : dict
        Dictionary of feature sets
    best_model : str
        Name of the best model to use for advanced analysis
    handle_imbalance : bool
        Whether to handle class imbalance
    imbalance_method : str
        Method to use for handling imbalance ('smote', 'adasyn', etc.)
    calibrate_model : bool
        Whether to calibrate model probabilities
    calibration_method : str
        Method for calibration ('platt', 'isotonic')
    explain_model : bool
        Whether to generate model explanations
    model_type_for_shap : str
        Model type for SHAP explanations ('tree', 'linear', etc.)
    n_bootstrap : int
        Number of bootstrap iterations for confidence intervals
    create_publication_materials : bool
        Whether to create publication-ready figures
        
    Returns:
    --------
    dict
        Dictionary containing all advanced analysis results
    """
    logging.info("Starting advanced analysis pipeline")
    results = {}
    
    # Extract data from splits
    X_train, y_train = splits.get('train', (None, None))
    X_val, y_val = splits.get('val', (None, None))
    X_test, y_test = splits.get('test', (None, None))
    
    # Check if we have the required data
    if X_train is None or y_train is None or X_test is None or y_test is None:
        logging.error("Missing required train or test data")
        return {"error": "Missing required data splits"}
    
    # Get the best model
    if best_model not in trained_models:
        logging.error(f"Model {best_model} not found in trained models")
        return {"error": f"Model {best_model} not found"}
    
    model = trained_models[best_model]
    logging.info(f"Using {best_model} for advanced analysis")
    
    # Handle class imbalance if requested
    if handle_imbalance:
        logging.info(f"Handling class imbalance with {imbalance_method}")
        if imbalance_method.lower() == 'smote':
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            logging.info(f"Resampled training data from {X_train.shape} to {X_train_resampled.shape}")
            
            # Train a new model on the resampled data
            model_resampled = joblib.load(f"models/{best_model}.joblib")  # Load a fresh copy
            model_resampled.fit(X_train_resampled, y_train_resampled)
            model = model_resampled
            
            # Save the resampled model
            joblib.dump(model, f"models/{best_model}_resampled.joblib")
            results['resampled_model_path'] = f"models/{best_model}_resampled.joblib"
    
    # Calibrate probabilities if requested
    if calibrate_model:
        logging.info(f"Calibrating model with {calibration_method}")
        if calibration_method.lower() == 'platt':
            method = 'sigmoid'
        else:
            method = 'isotonic'
            
        calibrated_model = CalibratedClassifierCV(model, method=method, cv='prefit')
        calibrated_model.fit(X_val if X_val is not None else X_train, 
                            y_val if y_val is not None else y_train)
        
        # Save the calibrated model
        joblib.dump(calibrated_model, f"models/{best_model}_calibrated.joblib")
        model = calibrated_model
        results['calibrated_model_path'] = f"models/{best_model}_calibrated.joblib"
    
    # Calculate performance metrics
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Basic metrics
    results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
    results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
    
    # ROC and PR curves if probabilities are available
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        results['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
    
    # Save results to file
    with open(f"results/advanced_analysis_{best_model}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations if requested
    if create_publication_materials:
        logging.info("Creating publication materials")
        
        # Confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {best_model}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"figs/confusion_matrix_{best_model}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC curve plot
        if 'roc_curve' in results:
            plt.figure(figsize=(10, 8))
            plt.plot(results['roc_curve']['fpr'], results['roc_curve']['tpr'], 
                    label=f'{best_model} (AUC = {results["classification_report"]["macro avg"]["f1-score"]:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='best')
            plt.savefig(f"figs/roc_curve_{best_model}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    logging.info("Advanced analysis completed successfully")
    return results

if __name__ == "__main__":
    setup_directory_structure()
    
    file_paths = [
        'data/GSE107011_Processed_data_TPM.txt',
        'data/GSE122459_tpm.txt',
        'data/GSE122459_ann.txt',
        'data/GSE122459_cnt.txt',
        'data/GSE107011_tpm.txt'
    ]
    sources = ['Gene Expression Omnibus', 'Zenodo']
    dates = ['2024-05-01', '2024-05-02']  # Replace with actual download dates
    
    provenance_df = create_data_provenance(file_paths, sources, dates)
    
    for fp in file_paths:
        df, shape_info = inspect_raw_matrix(fp, sample_name=os.path.splitext(os.path.basename(fp))[0])
        
        # Limit to first 1000 genes only for faster processing
        df = df.iloc[:1000, :]
        
        gene_ids = df.index.tolist()
        ann_df = annotate_genes_biomart(gene_ids)
        expr_annot = add_annotations(df, ann_df)
        
        # Adjust filtering to filtered expression with first 1000 genes
        filtered_expr = filter_genes(expr_annot.set_index('ensembl_gene_id').iloc[:, :-2], threshold=1, min_fraction=0.1, log_data=False)
        
        normalized_expr = normalize_expression_data(filtered_expr,
                                                method='tpm',
                                                save_plots=True)

