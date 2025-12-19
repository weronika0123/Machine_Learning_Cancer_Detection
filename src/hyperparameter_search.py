import subprocess
import json
import random
import csv
from pathlib import Path
from datetime import datetime
import sys

# Search space definition
SEARCH_SPACE = {
    'hidden_layers': [
        [128, 64],
        [256, 128, 64],
        [512, 256],
        [64, 32],
        [256, 128],
        [128, 64, 32],
        [512, 256, 128],
        [256, 64]
    ],
    'learning_rate': [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    'batch_size': [16, 32, 64, 128],
    'activation': ['relu', 'elu', 'selu', 'tanh'],
    'epochs': [50, 75, 100] 
}

FIXED_PARAMS = {
    'data': 'src/data_sources\\liquid_biopsy_data.csv',
    'use_validation': 'separate',
    'model': 'DNN',
    'preprocess': '[]',
    'preprocess_params': '{}',
    'postprocess': False,
    'postprocess_params': '{}',
    'eval': "['accuracy','F1','Precision','Recall','AUC ROC','AUC PR','Confusion Matrix']",
    'xai': False
}

def generate_random_config(experiment_id):
    config = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'hidden_layers': random.choice(SEARCH_SPACE['hidden_layers']),
        'learning_rate': random.choice(SEARCH_SPACE['learning_rate']),
        'dropout_rate': random.choice(SEARCH_SPACE['dropout_rate']),
        'batch_size': random.choice(SEARCH_SPACE['batch_size']),
        'activation': random.choice(SEARCH_SPACE['activation']),
        'epochs': random.choice(SEARCH_SPACE['epochs'])
    }
    return config


def run_experiment(config, experiment_num, total_experiments):
    print(f"\n{'='*80}")
    print(f"EXPERIMENT {experiment_num}/{total_experiments}")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  - Hidden Layers: {config['hidden_layers']}")
    print(f"  - Learning Rate: {config['learning_rate']}")
    print(f"  - Dropout Rate: {config['dropout_rate']}")
    print(f"  - Batch Size: {config['batch_size']}")
    print(f"  - Activation: {config['activation']}")
    print(f"  - Epochs: {config['epochs']}")
    print(f"{'='*80}\n")
    model_params = {
        'hidden_layers': config['hidden_layers'],
        'learning_rate': config['learning_rate'],
        'dropout_rate': config['dropout_rate'],
        'batch_size': config['batch_size'],
        'activation': config['activation'],
        'epochs': config['epochs']
    }

    cmd = [
        sys.executable, 
        'main.py',
        '--data', '../' + FIXED_PARAMS['data'],
        '--use_validation', FIXED_PARAMS['use_validation'],
        '--model', FIXED_PARAMS['model'],
        '--model_params', str(model_params),
        '--preprocess', FIXED_PARAMS['preprocess'],
        '--eval', FIXED_PARAMS['eval']
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path.cwd() / 'src', 
            timeout=1800  
        )
        metrics = parse_metrics_from_output(result.stdout)
        metrics.update({
            'experiment_id': config['experiment_id'],
            'timestamp': config['timestamp'],
            'hidden_layers': str(config['hidden_layers']),
            'learning_rate': config['learning_rate'],
            'dropout_rate': config['dropout_rate'],
            'batch_size': config['batch_size'],
            'activation': config['activation'],
            'epochs': config['epochs'],
            'status': 'success' if result.returncode == 0 else 'failed',
            'error': result.stderr if result.returncode != 0 else ''
        })
        return metrics

    except subprocess.TimeoutExpired:
        print(f"Experiment {experiment_num} timed out!")
        return {
            'experiment_id': config['experiment_id'],
            'timestamp': config['timestamp'],
            'hidden_layers': str(config['hidden_layers']),
            'learning_rate': config['learning_rate'],
            'dropout_rate': config['dropout_rate'],
            'batch_size': config['batch_size'],
            'activation': config['activation'],
            'epochs': config['epochs'],
            'status': 'timeout',
            'error': 'Experiment exceeded 30 minute timeout'
        }
    except Exception as e:
        print(f"Experiment {experiment_num} failed with error: {e}")
        return {
            'experiment_id': config['experiment_id'],
            'timestamp': config['timestamp'],
            'hidden_layers': str(config['hidden_layers']),
            'learning_rate': config['learning_rate'],
            'dropout_rate': config['dropout_rate'],
            'batch_size': config['batch_size'],
            'activation': config['activation'],
            'epochs': config['epochs'],
            'status': 'error',
            'error': str(e)
        }


def parse_metrics_from_output(output):
    try:
        lines = output.strip().split('\n')
        json_started = False
        json_lines = []
        
        for line in lines:
            if line.strip().startswith('{'):
                json_started = True
                json_lines = [line]
            elif json_started:
                json_lines.append(line)
                if line.strip().endswith('}'):
                    break
        
        if json_lines:
            json_str = '\n'.join(json_lines)
            result = json.loads(json_str)
            
            metrics = result.get('metrics', {})
            cm = metrics.get('Confusion matrix', None)
            tn, fp, fn, tp = None, None, None, None
            if cm is not None and len(cm) == 2 and len(cm[0]) == 2:
                tn, fp = cm[0][0], cm[0][1]
                fn, tp = cm[1][0], cm[1][1]
            
            return {
                'accuracy': metrics.get('accuracy', None),
                'f1': metrics.get('f1', None),
                'precision': metrics.get('precision', None),
                'recall': metrics.get('recall', None),
                'auc_roc': metrics.get('AUC ROC', None),
                'auc_pr': metrics.get('AUC PR', None),
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'tp': tp
            }
    except:
        pass
    
    return {
        'accuracy': None,
        'f1': None,
        'precision': None,
        'recall': None,
        'auc_roc': None,
        'auc_pr': None,
        'tn': None,
        'fp': None,
        'fn': None,
        'tp': None
    }


def save_results(results, output_file):
    if not results:
        print("No results to save!")
        return
    fieldnames = [
        'experiment_id', 'timestamp', 'status',
        'hidden_layers', 'learning_rate', 'dropout_rate', 
        'batch_size', 'activation', 'epochs',
        'accuracy', 'f1', 'precision', 'recall', 'auc_roc', 'auc_pr',
        'tn', 'fp', 'fn', 'tp',
        'error'
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {output_file}")


def main():
    n_experiments = 40
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER SEARCH - PHASE 3")
    print(f"{'='*80}")
    print(f"Total experiments: {n_experiments}")
    print(f"Search space size: {len(SEARCH_SPACE['hidden_layers'])} x "
          f"{len(SEARCH_SPACE['learning_rate'])} x "
          f"{len(SEARCH_SPACE['dropout_rate'])} x "
          f"{len(SEARCH_SPACE['batch_size'])} x "
          f"{len(SEARCH_SPACE['activation'])} x "
          f"{len(SEARCH_SPACE['epochs'])} = "
          f"{len(SEARCH_SPACE['hidden_layers']) * len(SEARCH_SPACE['learning_rate']) * len(SEARCH_SPACE['dropout_rate']) * len(SEARCH_SPACE['batch_size']) * len(SEARCH_SPACE['activation']) * len(SEARCH_SPACE['epochs'])} combinations")
    print(f"Estimated time: {n_experiments * 10} - {n_experiments * 15} minutes")
    print(f"{'='*80}\n")

    configs = [generate_random_config(i+1) for i in range(n_experiments)]
    results = []
    start_time = datetime.now()
    
    for i, config in enumerate(configs, 1):
        experiment_start = datetime.now()
        result = run_experiment(config, i, n_experiments)
        results.append(result)
        experiment_end = datetime.now()
        experiment_duration = (experiment_end - experiment_start).total_seconds()
        
        print(f"\nExperiment {i}/{n_experiments} completed in {experiment_duration:.1f}s")
        if result.get('accuracy'):
            print(f"   Metrics: Acc={result['accuracy']:.4f}, F1={result['f1']:.4f}, AUC={result['auc_roc']:.4f}")
        experiments_dir = Path('experiments')
        save_results(results, experiments_dir / 'experiments_log.csv')
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER SEARCH COMPLETED!")
    print(f"{'='*80}")
    print(f"Total time: {total_duration/60:.1f} minutes")
    print(f"Successful experiments: {sum(1 for r in results if r['status'] == 'success')}/{n_experiments}")
    print(f"Failed experiments: {sum(1 for r in results if r['status'] != 'success')}/{n_experiments}")
    
    successful_results = [r for r in results if r.get('auc_pr') is not None]
    if successful_results:
        best = max(successful_results, key=lambda x: x.get('auc_pr', 0))
        print(f"\nBEST EXPERIMENT (selected by AUC PR):")
        print(f"   ID: {best['experiment_id']}")
        print(f"   AUC PR (Primary Metric): {best['auc_pr']:.4f}")
        print(f"   AUC ROC: {best['auc_roc']:.4f}")
        print(f"   Accuracy: {best['accuracy']:.4f}")
        print(f"   F1 Score: {best['f1']:.4f}")
        print(f"   Config: layers={best['hidden_layers']}, lr={best['learning_rate']}, "
              f"dropout={best['dropout_rate']}, batch={best['batch_size']}, act={best['activation']}")
        
        best_config_path = Path('experiments') / 'best_run_info.json'
        with open(best_config_path, 'w', encoding='utf-8') as f:
            json.dump(best, f, indent=2)
        print(f"\nBest configuration saved to: {best_config_path}")
        print("\nNote: Best model selected based on AUC PR (optimal for imbalanced datasets)")
    
    print(f"\n{'='*80}\n")
    print("Next steps:")
    print("1.Run: python src/analyze_experiments.py")
    print("2.View results in experiments/plots/")
    print("3.Use TensorBoard: tensorboard --logdir runs")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
