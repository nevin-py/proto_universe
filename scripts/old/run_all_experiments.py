#!/usr/bin/env python3
"""
Comprehensive Experiment Suite - All Byzantine Detection Tests
Live logging with real-time updates to console and file
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.models.mnist import create_mnist_model
from src.crypto.zkp_prover import TrainingProofProver


class LiveLogger:
    """Logger that outputs to both console and file in real-time"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        
        # Create logger
        self.logger = logging.getLogger('ExperimentSuite')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler with immediate flush
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console_format = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
        console.setFormatter(console_format)
        self.logger.addHandler(console)
        
        # File handler with immediate flush
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Force immediate flush
        for handler in self.logger.handlers:
            handler.flush()
    
    def info(self, msg):
        self.logger.info(msg)
        # Force flush after each message for live updates
        for handler in self.logger.handlers:
            handler.flush()
    
    def error(self, msg):
        self.logger.error(msg)
        for handler in self.logger.handlers:
            handler.flush()
    
    def separator(self):
        self.info("=" * 80)
    
    def line(self):
        self.info("-" * 80)


class ExperimentSuite:
    """Complete experiment suite with live logging"""
    
    def __init__(self, output_dir):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"experiments_{self.timestamp}"
        self.results_dir = self.output_dir / "results"
        self.logs_dir = self.output_dir / "logs"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup live logger
        self.log = LiveLogger(self.output_dir / "main.log")
        
        # Results tracking
        self.all_results = {}
        self.total_experiments = 0
        self.passed_experiments = 0
        self.failed_experiments = 0
    
    def run_byzantine_detection_test(self, model_type, input_dim, num_classes=10):
        """Test Byzantine detection for a specific model architecture"""
        
        exp_name = f"byzantine_{model_type}"
        self.total_experiments += 1
        
        self.log.separator()
        self.log.info(f"EXPERIMENT {self.total_experiments}: Byzantine Detection - {model_type.upper()}")
        self.log.separator()
        self.log.info(f"Model: {model_type}")
        self.log.info(f"Input dim: {input_dim}, Classes: {num_classes}")
        self.log.info("")
        
        start_time = time.time()
        
        try:
            # Create honest and malicious models
            self.log.info("Creating models...")
            honest_model = create_mnist_model(model_type, num_classes)
            malicious_model = create_mnist_model(model_type, num_classes)
            
            # Get final layer parameters
            self.log.info("Extracting model parameters...")
            
            if model_type == "linear":
                honest_weights = honest_model.linear.weight.detach()
                honest_bias = honest_model.linear.bias.detach()
                malicious_weights = malicious_model.linear.weight.detach()
                malicious_bias = malicious_model.linear.bias.detach()
            elif model_type == "mlp":
                honest_weights = honest_model.fc3.weight.detach()
                honest_bias = honest_model.fc3.bias.detach()
                malicious_weights = malicious_model.fc3.weight.detach()
                malicious_bias = malicious_model.fc3.bias.detach()
            elif model_type == "cnn":
                honest_weights = honest_model.fc2.weight.detach()
                honest_bias = honest_model.fc2.bias.detach()
                malicious_weights = malicious_model.fc2.weight.detach()
                malicious_bias = malicious_model.fc2.bias.detach()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Poison malicious model weights
            self.log.info("Poisoning malicious model...")
            malicious_weights = malicious_weights + torch.randn_like(malicious_weights) * 0.5
            
            # Test Byzantine detection
            self.log.info("Initializing ZKP prover...")
            prover = TrainingProofProver()
            
            # Compute honest fingerprint with adaptive sample size
            self.log.info("Computing honest model fingerprint...")
            round_number = 1
            r_vec = prover.generate_random_vector(round_number)
            sample_size = min(100, input_dim)
            
            honest_fp, _ = prover.compute_model_fingerprint(
                honest_weights, honest_bias, r_vec, round_number, sample_size=sample_size
            )
            self.log.info(f"  Honest fingerprint: {honest_fp}")
            
            # Compute malicious fingerprint
            self.log.info("Computing malicious model fingerprint...")
            malicious_fp, _ = prover.compute_model_fingerprint(
                malicious_weights, malicious_bias, r_vec, round_number, sample_size=sample_size
            )
            self.log.info(f"  Malicious fingerprint: {malicious_fp}")
            
            # Check if fingerprints differ
            fp_diff = abs(honest_fp - malicious_fp)
            detected = fp_diff > 0
            
            self.log.info("")
            self.log.info("RESULTS:")
            self.log.line()
            self.log.info(f"  Fingerprint difference: {fp_diff}")
            self.log.info(f"  Byzantine detected: {detected}")
            self.log.info(f"  Detection time: {(time.time() - start_time) * 1000:.2f}ms")
            
            # Save results
            result = {
                "experiment": exp_name,
                "model_type": model_type,
                "input_dim": input_dim,
                "num_classes": num_classes,
                "honest_fingerprint": int(honest_fp),
                "malicious_fingerprint": int(malicious_fp),
                "fingerprint_diff": int(fp_diff),
                "byzantine_detected": detected,
                "detection_time_ms": (time.time() - start_time) * 1000,
                "timestamp": self.timestamp,
                "status": "PASS" if detected else "FAIL"
            }
            
            self.all_results[exp_name] = result
            
            # Save individual result
            result_file = self.results_dir / f"{exp_name}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            self.log.info(f"  Result saved: {result_file}")
            
            if detected:
                self.log.info("")
                self.log.info(f"✅ {exp_name}: PASS")
                self.passed_experiments += 1
                return True
            else:
                self.log.error("")
                self.log.error(f"❌ {exp_name}: FAIL - Byzantine not detected!")
                self.failed_experiments += 1
                return False
                
        except Exception as e:
            self.log.error("")
            self.log.error(f"❌ {exp_name}: EXCEPTION - {str(e)}")
            self.log.error(f"   {type(e).__name__}: {str(e)}")
            self.failed_experiments += 1
            
            # Save error result
            result = {
                "experiment": exp_name,
                "status": "ERROR",
                "error": str(e),
                "timestamp": self.timestamp
            }
            self.all_results[exp_name] = result
            
            return False
    
    def run_all_experiments(self):
        """Run complete experiment suite"""
        
        self.log.separator()
        self.log.info("COMPREHENSIVE EXPERIMENT SUITE")
        self.log.separator()
        self.log.info(f"Timestamp: {self.timestamp}")
        self.log.info(f"Output directory: {self.output_dir}")
        self.log.info(f"Results directory: {self.results_dir}")
        self.log.info(f"Logs directory: {self.logs_dir}")
        self.log.info("")
        
        # ====================================================================
        # PART 1: Byzantine Detection - All Architectures
        # ====================================================================
        
        self.log.info("")
        self.log.separator()
        self.log.info("PART 1: BYZANTINE DETECTION VALIDATION")
        self.log.separator()
        self.log.info("")
        
        experiments = [
            ("linear", 784, "MNISTLinearRegression"),
            ("mlp", 64, "SimpleMLP"),
            ("cnn", 128, "MNISTCnn"),
        ]
        
        for model_type, input_dim, model_class in experiments:
            self.log.info(f"Testing: {model_class} (input_dim={input_dim})")
            self.run_byzantine_detection_test(model_type, input_dim)
            self.log.info("")
        
        # ====================================================================
        # FINAL SUMMARY
        # ====================================================================
        
        self.log.info("")
        self.log.separator()
        self.log.info("EXPERIMENT SUITE COMPLETE")
        self.log.separator()
        self.log.info("")
        self.log.info("STATISTICS:")
        self.log.info(f"  Total experiments: {self.total_experiments}")
        self.log.info(f"  Passed: {self.passed_experiments}")
        self.log.info(f"  Failed: {self.failed_experiments}")
        self.log.info(f"  Success rate: {self.passed_experiments * 100 // self.total_experiments}%")
        self.log.info("")
        
        # Save combined results
        combined_file = self.results_dir / "all_results.json"
        with open(combined_file, 'w') as f:
            json.dump({
                "timestamp": self.timestamp,
                "total": self.total_experiments,
                "passed": self.passed_experiments,
                "failed": self.failed_experiments,
                "experiments": self.all_results
            }, f, indent=2)
        
        self.log.info(f"Combined results saved: {combined_file}")
        self.log.info(f"Main log: {self.output_dir / 'main.log'}")
        self.log.info("")
        
        # List all result files
        self.log.info("Individual result files:")
        for result_file in sorted(self.results_dir.glob("*.json")):
            size = result_file.stat().st_size
            self.log.info(f"  - {result_file.name} ({size} bytes)")
        
        self.log.info("")
        self.log.separator()
        
        if self.failed_experiments == 0:
            self.log.info("🎉 ALL EXPERIMENTS PASSED!")
            self.log.separator()
            return 0
        else:
            self.log.error(f"⚠️  {self.failed_experiments} EXPERIMENT(S) FAILED")
            self.log.separator()
            return 1


def main():
    """Main entry point"""
    
    # Output directory
    output_dir = "outputs"
    
    # Create and run experiment suite
    suite = ExperimentSuite(output_dir)
    exit_code = suite.run_all_experiments()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
