#!/usr/bin/env python3
"""
Falcon-3B Model Validation Script

This script downloads, loads, and validates the Falcon-3B base model,
measuring baseline performance metrics including memory usage and inference speed.
"""

import time
import psutil
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple
import json
import os

class Falcon3BValidator:
    def __init__(self, model_name: str = "tiiuae/Falcon3-3B-Base-1.58bit"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metrics = {}
        
    def download_and_load_model(self) -> bool:
        """Download and load the Falcon-3B model and tokenizer."""
        print(f"🔄 Loading Falcon-3B model: {self.model_name}")
        print(f"📍 Target device: {self.device}")
        
        try:
            # Record initial memory
            initial_memory = self._get_memory_usage()
            
            # Load tokenizer
            print("📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimal settings
            print("📥 Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Record post-load memory
            final_memory = self._get_memory_usage()
            
            # Calculate model size
            model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**3)
            
            self.metrics['model_loading'] = {
                'model_name': self.model_name,
                'device': self.device,
                'model_size_gb': round(model_size, 2),
                'memory_before_mb': initial_memory,
                'memory_after_mb': final_memory,
                'memory_increase_mb': final_memory - initial_memory
            }
            
            print(f"✅ Model loaded successfully!")
            print(f"📊 Model size: {model_size:.2f} GB")
            print(f"💾 Memory usage increased by: {final_memory - initial_memory:.1f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def validate_basic_inference(self) -> bool:
        """Test basic inference functionality."""
        if not self.model or not self.tokenizer:
            print("❌ Model not loaded")
            return False
            
        print("\n🧪 Testing basic inference...")
        
        test_prompts = [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly,",
            "Machine learning has revolutionized"
        ]
        
        try:
            for i, prompt in enumerate(test_prompts, 1):
                print(f"\n📝 Test {i}: '{prompt}'")
                
                # Measure inference time
                start_time = time.time()
                
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt")
                # Remove token_type_ids if present (not used by Falcon models)
                if 'token_type_ids' in inputs:
                    del inputs['token_type_ids']
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                inference_time = time.time() - start_time
                
                print(f"🔍 Output: {response}")
                print(f"⏱️  Inference time: {inference_time:.2f}s")
                
                # Store metrics for first test
                if i == 1:
                    self.metrics['basic_inference'] = {
                        'prompt': prompt,
                        'response': response,
                        'inference_time_seconds': round(inference_time, 3),
                        'input_tokens': len(inputs['input_ids'][0]),
                        'output_tokens': len(outputs[0])
                    }
            
            print("\n✅ Basic inference validation passed!")
            return True
            
        except Exception as e:
            print(f"❌ Basic inference failed: {e}")
            return False
    
    def measure_performance_metrics(self) -> Dict:
        """Measure detailed performance metrics."""
        if not self.model or not self.tokenizer:
            print("❌ Model not loaded")
            return {}
            
        print("\n📊 Measuring performance metrics...")
        
        # Test prompt for consistent measurements
        test_prompt = "Artificial intelligence and machine learning are transforming"
        
        try:
            # Warm-up run
            inputs = self.tokenizer(test_prompt, return_tensors="pt")
            # Remove token_type_ids if present (not used by Falcon models)
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                torch.cuda.empty_cache()
            
            # Memory before inference
            memory_before = self._get_memory_usage()
            if self.device == "cuda":
                gpu_memory_before = torch.cuda.memory_allocated() / (1024**2)
            
            # Performance test
            inference_times = []
            num_runs = 5
            
            for run in range(num_runs):
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,  # Deterministic for consistent timing
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            
            # Memory after inference
            memory_after = self._get_memory_usage()
            if self.device == "cuda":
                gpu_memory_after = torch.cuda.memory_allocated() / (1024**2)
            
            # Calculate metrics
            avg_inference_time = sum(inference_times) / len(inference_times)
            tokens_per_second = 100 / avg_inference_time  # 100 new tokens generated
            
            performance_metrics = {
                'inference_times_seconds': [round(t, 3) for t in inference_times],
                'average_inference_time_seconds': round(avg_inference_time, 3),
                'tokens_per_second': round(tokens_per_second, 2),
                'memory_before_inference_mb': memory_before,
                'memory_after_inference_mb': memory_after,
                'memory_delta_mb': memory_after - memory_before
            }
            
            if self.device == "cuda":
                performance_metrics.update({
                    'gpu_memory_before_mb': round(gpu_memory_before, 2),
                    'gpu_memory_after_mb': round(gpu_memory_after, 2),
                    'gpu_memory_delta_mb': round(gpu_memory_after - gpu_memory_before, 2),
                    'gpu_total_memory_mb': round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 2)
                })
            
            self.metrics['performance'] = performance_metrics
            
            print(f"⚡ Average inference time: {avg_inference_time:.3f}s")
            print(f"🔤 Tokens per second: {tokens_per_second:.2f}")
            print(f"💾 Memory usage: {memory_before:.1f} → {memory_after:.1f} MB")
            
            return performance_metrics
            
        except Exception as e:
            print(f"❌ Performance measurement failed: {e}")
            return {}
    
    def test_different_configurations(self) -> bool:
        """Test model with different precision and configuration settings."""
        if not self.tokenizer:
            print("❌ Tokenizer not loaded")
            return False
            
        print("\n🔧 Testing different model configurations...")
        
        configurations = [
            {"name": "bfloat16", "dtype": torch.bfloat16},
            {"name": "float16", "dtype": torch.float16},
        ]
        
        if self.device == "cpu":
            configurations.append({"name": "float32", "dtype": torch.float32})
        
        config_results = []
        test_prompt = "The advancement of technology"
        
        for config in configurations:
            print(f"\n🧪 Testing {config['name']} configuration...")
            
            try:
                # Load model with specific configuration
                start_time = time.time()
                test_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=config['dtype'],
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                if self.device == "cpu":
                    test_model = test_model.to(self.device)
                
                load_time = time.time() - start_time
                
                # Test inference
                inputs = self.tokenizer(test_prompt, return_tensors="pt")
                # Remove token_type_ids if present (not used by Falcon models)
                if 'token_type_ids' in inputs:
                    del inputs['token_type_ids']
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                inference_start = time.time()
                with torch.no_grad():
                    outputs = test_model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                inference_time = time.time() - inference_start
                
                # Calculate model size
                model_size = sum(p.numel() * p.element_size() for p in test_model.parameters()) / (1024**3)
                
                result = {
                    'configuration': config['name'],
                    'load_time_seconds': round(load_time, 2),
                    'inference_time_seconds': round(inference_time, 3),
                    'model_size_gb': round(model_size, 2),
                    'status': 'success'
                }
                
                config_results.append(result)
                print(f"✅ {config['name']}: Load {load_time:.2f}s, Inference {inference_time:.3f}s, Size {model_size:.2f}GB")
                
                # Clean up
                del test_model
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                result = {
                    'configuration': config['name'],
                    'status': 'failed',
                    'error': str(e)
                }
                config_results.append(result)
                print(f"❌ {config['name']}: {e}")
        
        self.metrics['configurations'] = config_results
        return len([r for r in config_results if r['status'] == 'success']) > 0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    
    def save_metrics(self, filename: str = "falcon3b_validation_metrics.json"):
        """Save all metrics to a JSON file."""
        try:
            # Add system information
            self.metrics['system_info'] = {
                'device': self.device,
                'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'total_ram_gb': round(psutil.virtual_memory().total / (1024**3), 2)
            }
            
            if torch.cuda.is_available():
                self.metrics['system_info']['gpu_name'] = torch.cuda.get_device_name(0)
                self.metrics['system_info']['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            
            with open(filename, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            print(f"📁 Metrics saved to {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save metrics: {e}")
            return False
    
    def run_full_validation(self) -> bool:
        """Run complete validation suite."""
        print("🚀 Starting Falcon-3B Model Validation")
        print("=" * 60)
        
        # Step 1: Download and load model
        if not self.download_and_load_model():
            return False
        
        # Step 2: Basic inference validation
        if not self.validate_basic_inference():
            return False
        
        # Step 3: Performance metrics
        self.measure_performance_metrics()
        
        # Step 4: Configuration testing
        self.test_different_configurations()
        
        # Step 5: Save results
        self.save_metrics()
        
        print("\n" + "=" * 60)
        print("🎉 Falcon-3B validation completed successfully!")
        print(f"📊 Results saved to falcon3b_validation_metrics.json")
        
        return True

def main():
    """Main execution function."""
    validator = Falcon3BValidator()
    success = validator.run_full_validation()
    
    if not success:
        print("❌ Validation failed")
        exit(1)
    else:
        print("✅ All validations passed")
        exit(0)

if __name__ == "__main__":
    main()