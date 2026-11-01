Quantization-Aware Weight Streaming
Memory-Constrained LLM Inference Using Quantization and CPU-GPU Weight Streaming
Project Objective

This project investigates whether quantization-aware weight streaming can reduce GPU memory requirements for Large Language Model inference on memory-constrained GPUs.

The system compares an FP16 baseline with low-bit quantized inference using CPU-resident weights and CPU-to-GPU weight streaming.

System Pipeline
LLM
 │
 ├── FP16 Baseline
 │
 └── Quantization
       │
       ├── INT8
       ├── INT4
       ├── INT3
       └── INT2
              │
              ▼
        CPU RAM Storage
              │
              ▼
       CPU → GPU Streaming
              │
              ▼
       GPU Dequantization
              │
              ▼
          GPU Inference
              │
              ▼
        Benchmark Metrics
Hardware
NVIDIA RTX 4050 Laptop GPU
6 GB VRAM
Intel Core i7-13620H
16 GB RAM
Default Model
Qwen/Qwen2.5-0.5B-Instruct

The model is intentionally kept small so that the complete experimental pipeline can be demonstrated on a 6 GB laptop GPU.

Technologies
Python
PyTorch
CUDA
Hugging Face Transformers
Quantization
CPU-GPU Weight Streaming
Streamlit
OpenMPI
mpi4py
Main Metrics

The current implementation measures:

Inference latency
Generated tokens
Tokens per second
Peak GPU memory
Allocated GPU memory
Reserved GPU memory
Process RAM
Quantization reconstruction sensitivity
Review 2 Scope

Approximately 75% of the proposed system is implemented:

Model loading
CUDA execution
FP16 baseline
INT8 quantization
INT4 quantization
CPU-resident quantized weights
CPU-to-GPU weight streaming
GPU dequantization
GPU inference
Benchmarking
Memory measurement
Streamlit visualization
MPI-based experiment orchestration
Future Work

The remaining work includes:

Calibration-based quantization
Perplexity evaluation
Task-level accuracy evaluation
Better layer sensitivity analysis
Optimized asynchronous weight streaming
PCIe bandwidth measurement
Energy per token
Pareto frontier analysis
Multi-GPU distributed inference
More extensive benchmark repetitions
Running the Project
1. Create environment
python -m venv .venv
Windows
.venv\Scripts\activate
WSL
source .venv/bin/activate
2. Install dependencies
pip install -r requirements.txt
3. Verify GPU
python scripts/verify_installation.py
4. Run FP16 baseline
python scripts/run_experiment.py --model qwen --precision FP16
5. Run INT8
python scripts/run_experiment.py --model qwen --precision INT8
6. Run INT4
python scripts/run_experiment.py --model qwen --precision INT4
7. Start dashboard
streamlit run ui/app.py
8. Run MPI experiment
mpirun -np 2 python scripts/run_mpi_experiments.py
Important MPI Note

OpenMPI is used primarily for experiment orchestration in the current laptop configuration.

Multiple MPI ranks should not independently load large GPU models onto the same 6 GB RTX 4050 because this can cause GPU memory contention or out-of-memory errors.

MPI is therefore not claimed as a direct GPU acceleration mechanism in the current single-GPU implementation.

Project Claim

The project demonstrates a practical approach for reducing GPU memory pressure during LLM inference by combining low-bit weight representation with CPU-resident weight storage and CPU-to-GPU weight streaming.