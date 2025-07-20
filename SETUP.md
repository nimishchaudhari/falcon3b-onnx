# Falcon3B ONNX Development Environment Setup

This guide will help you set up a complete development environment for the Falcon3B ONNX project, including PyTorch, ONNX, HuggingFace transformers, and BitNet quantization tools.

## Prerequisites

- Python 3.9 or later (Python 3.12+ recommended)
- NVIDIA GPU with CUDA support (recommended for optimal performance)
- Git for cloning repositories

## Quick Setup

### 1. Clone the Repository
```bash
git clone https://github.com/nimishchaudhari/falcon3b-onnx.git
cd falcon3b-onnx
```

### 2. Create Virtual Environment
```bash
python -m venv falcon3b_env
source falcon3b_env/bin/activate  # On Windows: falcon3b_env\Scripts\activate
```

### 3. Install Core Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install ONNX dependencies
pip install onnx>=1.14.0 onnxruntime>=1.15.0 onnxruntime-gpu>=1.15.0

# Install HuggingFace and Falcon dependencies
pip install transformers>=4.30.0 tokenizers>=0.13.0 datasets>=2.12.0 accelerate>=0.20.0 einops>=0.6.0 safetensors>=0.3.0

# Install additional dependencies
pip install numpy>=1.24.0 scipy>=1.10.0 matplotlib>=3.7.0 tqdm>=4.65.0

# Install development tools
pip install jupyter>=1.0.0 notebook>=6.5.0 pytest>=7.3.0 black>=23.0.0 flake8>=6.0.0
```

### 4. Install Quantization Tools

#### onebitllms (1-bit LLM training)
```bash
pip install onebitllms
```

#### BitNet.cpp (1-bit LLM inference)
```bash
# Clone BitNet repository
git clone --recursive https://github.com/microsoft/BitNet.git

# Install BitNet Python dependencies
cd BitNet
pip install -r requirements.txt

# Note: After installation, you may need to reinstall PyTorch with CUDA:
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu118
cd ..
```

### 5. Verify Installation
Run the verification script to ensure all components are working:
```bash
python verify_setup.py
```

## Detailed Component Information

### PyTorch with CUDA
- **Version**: 2.7.1+cu118
- **Purpose**: Core deep learning framework with GPU acceleration
- **Verification**: Check `torch.cuda.is_available()` returns `True`

### ONNX Ecosystem
- **ONNX**: 1.18.0 - Open Neural Network Exchange format
- **ONNX Runtime**: 1.22.1 - High-performance inference engine
- **Purpose**: Model conversion and optimized inference

### HuggingFace Transformers
- **Version**: 4.53.2+
- **Purpose**: Access to Falcon models and tokenizers
- **Models**: Supports all Falcon model variants (1B, 3B, 7B, etc.)

### Quantization Tools

#### onebitllms
- **Version**: 0.0.3
- **Purpose**: Training and fine-tuning 1-bit quantized models
- **Requirements**: CUDA-compatible GPU
- **Usage**: Convert linear layers to BitNet for training

#### BitNet.cpp
- **Purpose**: Optimized inference for 1-bit quantized models
- **Features**: 
  - CPU and GPU inference
  - Memory-efficient model serving
  - Cross-platform support

## Usage Examples

### Loading a Falcon Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Falcon model
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Test inference
text = "The future of AI is"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

### Using onebitllms for Quantization
```python
import torch
from transformers import AutoModelForCausalLM
from onebitllms import replace_linear_with_bitnet_linear

# Load pre-quantized model
model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/Falcon-E-1B-Base",
    revision="prequantized",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Convert to BitNet linear layers
model = replace_linear_with_bitnet_linear(model)
```

## Troubleshooting

### CUDA Issues
If CUDA is not detected:
1. Verify NVIDIA drivers: `nvidia-smi`
2. Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
3. Check CUDA compatibility with your GPU

### Memory Issues
For large models:
1. Use `torch_dtype=torch.bfloat16` or `torch.float16`
2. Enable `device_map="auto"` for automatic device placement
3. Consider gradient checkpointing for training

### BitNet.cpp Compilation Issues
If BitNet.cpp fails to build:
1. Install build tools: `sudo apt-get install build-essential cmake`
2. For Clang: Follow instructions at https://apt.llvm.org/llvm.sh
3. Alternative: Use onebitllms for quantization without BitNet.cpp

## Development Workflow

1. **Environment Activation**: Always activate the virtual environment
   ```bash
   source falcon3b_env/bin/activate
   ```

2. **Verification**: Run verification script after any changes
   ```bash
   python verify_setup.py
   ```

3. **Development**: Use Jupyter notebooks for experimentation
   ```bash
   jupyter notebook
   ```

## Performance Optimization

### For Training
- Use mixed precision: `torch.cuda.amp`
- Enable gradient checkpointing
- Use optimized data loaders with `num_workers`

### For Inference
- Use ONNX Runtime for deployment
- Consider TensorRT for NVIDIA GPUs
- Use BitNet.cpp for 1-bit model inference

## Next Steps

1. Explore Falcon model variants in the `notebooks/` directory
2. Run quantization experiments with onebitllms
3. Set up ONNX conversion pipelines
4. Experiment with BitNet.cpp inference optimization

## Support

- Run `python verify_setup.py` to diagnose issues
- Check GitHub issues for common problems
- Refer to component documentation:
  - [PyTorch](https://pytorch.org/docs/)
  - [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
  - [ONNX](https://onnx.ai/onnx/)
  - [BitNet](https://github.com/microsoft/BitNet)
  - [onebitllms](https://github.com/tiiuae/onebitllms)