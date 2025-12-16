"""
Quick test script - runs just 1 experiment to verify everything works
"""
import subprocess
import sys
from pathlib import Path

print("🧪 Testing single experiment...")
print("="*80)

cmd = [
    sys.executable,
    'main.py',
    '--data', '../src/data_sources/liquid_biopsy_data.csv',
    '--use_validation', 'separate',
    '--model', 'DNN',
    '--model_params', "{'hidden_layers': [128, 64], 'activation': 'relu', 'epochs': 5, 'batch_size': 32, 'learning_rate': 0.001, 'dropout_rate': 0.3}",
    '--preprocess', '[]',
    '--eval', "['accuracy','F1','AUC ROC']"
]

try:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path('src'),
        timeout=300  # 5 min timeout
    )
    
    print("STDOUT:")
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)  # Last 2000 chars
    
    print("\n" + "="*80)
    print("STDERR:")
    print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)  # Last 1000 chars
    
    print("\n" + "="*80)
    print(f"Return code: {result.returncode}")
    
    if result.returncode == 0:
        print("✅ SUCCESS! Single experiment completed.")
    else:
        print("❌ FAILED! Check errors above.")
        
except subprocess.TimeoutExpired:
    print("⏱️ Timeout - experiment took too long (>5 min)")
except Exception as e:
    print(f"❌ Error: {e}")

print("="*80)
