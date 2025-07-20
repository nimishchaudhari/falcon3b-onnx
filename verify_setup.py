#!/usr/bin/env python3
"""
Falcon3B ONNX Development Environment Verification Script

This script verifies that all required components for the Falcon3B ONNX project
are properly installed and functioning.
"""

import sys
import importlib
from typing import List, Tuple

class EnvironmentVerifier:
    def __init__(self):
        self.results = []
        self.errors = []
    
    def test_component(self, name: str, test_func) -> bool:
        """Test a component and record results."""
        try:
            success, message = test_func()
            self.results.append((name, success, message))
            if success:
                print(f"✅ {name}: {message}")
            else:
                print(f"❌ {name}: {message}")
                self.errors.append(f"{name}: {message}")
            return success
        except Exception as e:
            error_msg = f"Error testing {name}: {e}"
            self.results.append((name, False, error_msg))
            print(f"❌ {name}: {error_msg}")
            self.errors.append(error_msg)
            return False
    
    def test_python_version(self) -> Tuple[bool, str]:
        """Test Python version compatibility."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 9:
            return True, f"Python {version.major}.{version.minor}.{version.micro}"
        else:
            return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.9+)"
    
    def test_pytorch(self) -> Tuple[bool, str]:
        """Test PyTorch installation and CUDA support."""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_available else 0
            
            version_info = f"PyTorch {torch.__version__}"
            if cuda_available:
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                return True, f"{version_info}, CUDA available with {device_count} device(s): {device_name}"
            else:
                return True, f"{version_info}, CUDA not available (CPU only)"
        except ImportError:
            return False, "PyTorch not installed"
    
    def test_onnx(self) -> Tuple[bool, str]:
        """Test ONNX and ONNX Runtime installations."""
        try:
            import onnx
            import onnxruntime as ort
            
            providers = ort.get_available_providers()
            has_gpu = any('CUDA' in provider or 'GPU' in provider for provider in providers)
            
            return True, f"ONNX {onnx.__version__}, ONNX Runtime {ort.__version__} (GPU: {has_gpu})"
        except ImportError as e:
            return False, f"ONNX components not available: {e}"
    
    def test_transformers(self) -> Tuple[bool, str]:
        """Test HuggingFace transformers installation."""
        try:
            import transformers
            return True, f"Transformers {transformers.__version__}"
        except ImportError:
            return False, "HuggingFace transformers not installed"
    
    def test_falcon_model_access(self) -> Tuple[bool, str]:
        """Test ability to load Falcon model components."""
        try:
            from transformers import AutoTokenizer
            
            # Test with a small Falcon model
            model_name = "tiiuae/falcon-rw-1b"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Test tokenization
            test_text = "Hello, world!"
            tokens = tokenizer(test_text, return_tensors="pt")
            
            return True, f"Falcon model access working (tested {model_name})"
        except Exception as e:
            return False, f"Falcon model access failed: {e}"
    
    def test_onebitllms(self) -> Tuple[bool, str]:
        """Test onebitllms installation."""
        try:
            import onebitllms
            from onebitllms import replace_linear_with_bitnet_linear
            return True, f"onebitllms {onebitllms.__version__}"
        except ImportError:
            return False, "onebitllms not installed"
    
    def test_additional_dependencies(self) -> Tuple[bool, str]:
        """Test additional required dependencies."""
        missing = []
        required_modules = [
            'numpy', 'scipy', 'matplotlib', 'tqdm', 
            'einops', 'safetensors', 'accelerate'
        ]
        
        for module in required_modules:
            try:
                importlib.import_module(module)
            except ImportError:
                missing.append(module)
        
        if missing:
            return False, f"Missing modules: {', '.join(missing)}"
        else:
            return True, f"All additional dependencies available"
    
    def test_bitnet_cpp(self) -> Tuple[bool, str]:
        """Test BitNet.cpp availability."""
        import os
        bitnet_path = "/home/nimish/falcon3b-onnx/BitNet"
        
        if os.path.exists(bitnet_path):
            # Check for key files
            required_files = ['setup_env.py', 'run_inference.py']
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(bitnet_path, f))]
            
            if missing_files:
                return False, f"BitNet.cpp incomplete - missing: {', '.join(missing_files)}"
            else:
                return True, "BitNet.cpp repository available"
        else:
            return False, "BitNet.cpp not cloned"
    
    def run_all_tests(self):
        """Run all verification tests."""
        print("🔍 Falcon3B ONNX Environment Verification")
        print("=" * 50)
        
        # Core components
        self.test_component("Python Version", self.test_python_version)
        self.test_component("PyTorch", self.test_pytorch)
        self.test_component("ONNX", self.test_onnx)
        self.test_component("HuggingFace Transformers", self.test_transformers)
        self.test_component("Falcon Model Access", self.test_falcon_model_access)
        self.test_component("Additional Dependencies", self.test_additional_dependencies)
        
        # Quantization tools
        self.test_component("onebitllms", self.test_onebitllms)
        self.test_component("BitNet.cpp", self.test_bitnet_cpp)
        
        # Summary
        print("\n" + "=" * 50)
        total_tests = len(self.results)
        passed_tests = sum(1 for _, success, _ in self.results if success)
        
        print(f"📊 Summary: {passed_tests}/{total_tests} tests passed")
        
        if self.errors:
            print(f"\n❌ Issues found:")
            for error in self.errors:
                print(f"  - {error}")
        else:
            print(f"\n✅ All components verified successfully!")
            print(f"🚀 Environment ready for Falcon3B ONNX development!")
        
        return len(self.errors) == 0

def main():
    verifier = EnvironmentVerifier()
    success = verifier.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()