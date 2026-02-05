"""Analyze and summarize simulation results"""

import os
import json
import pandas as pd
from src.storage.manager import StorageManager


def main():
    storage = StorageManager()
    
    print("ProtoGalaxy Results Analysis")
    print("=" * 50)
    
    # Load metrics
    metrics = storage.load_metrics('summary')
    if metrics:
        print(f"Final Accuracy: {metrics.get('final_accuracy', 'N/A'):.4f}")
        print(f"Final Loss: {metrics.get('final_loss', 'N/A'):.4f}")
        print(f"Average Accuracy: {metrics.get('avg_accuracy', 'N/A'):.4f}")
        print(f"Total Rounds: {metrics.get('total_rounds', 'N/A')}")
        print(f"Detections: {metrics.get('total_detected', 'N/A')}")
    else:
        print("No metrics found. Run simulation first.")
    
    # List available models
    models = storage.get_available_models()
    print(f"\nAvailable Models: {len(models)}")
    for model in models[:5]:
        print(f"  - {model}")


if __name__ == '__main__':
    main()
