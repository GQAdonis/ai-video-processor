# Video Processing with Video-LLaVA and NV-Embed

A comprehensive video processing system that segments videos, interprets content using Video-LLaVA, and stores semantic embeddings using NV-Embed-v2.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with at least 16GB VRAM (tested on T4)
- CUDA 11.8 compatible system
- Sufficient storage for model caching

### System Dependencies

1. Install NVIDIA Driver and CUDA Toolkit:
```bash
sudo apt update && sudo apt upgrade
sudo apt install nvidia-driver-535
sudo apt install nvidia-cuda-toolkit
```

2. Install FFmpeg with CUDA support:
```bash
# Install build dependencies
sudo apt install build-essential yasm cmake libtool libc6 libc6-dev unzip wget libnuma1 libnuma-dev

# Download and install NASM
wget https://www.nasm.us/pub/nasm/releasebuilds/2.15.05/nasm-2.15.05.tar.gz
tar xvf nasm-2.15.05.tar.gz
cd nasm-2.15.05
./configure
make -j$(nproc)
sudo make install
cd ..

# Install NVENC SDK headers
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
make
sudo make install
cd ..

# Build FFmpeg with CUDA support
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg/
cd ffmpeg
./configure --enable-cuda-nvcc \
    --enable-cuvid \
    --enable-nvenc \
    --enable-nonfree \
    --enable-libnpp \
    --extra-cflags=-I/usr/local/cuda/include \
    --extra-ldflags=-L/usr/local/cuda/lib64
make -j$(nproc)
sudo make install
```

## Environment Setup

1. Create conda environment:
```bash
conda create -n video-processor python=3.11.2 -y
conda activate video-processor
```

2. Install PyTorch ecosystem:
```bash
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. Install ML dependencies:
```bash
conda install -c conda-forge transformers=4.42.4
conda install -c conda-forge flash-attn=2.2.0
conda install -c conda-forge sentence-transformers=2.7.0
```

4. Install additional dependencies:
```bash
conda install -c conda-forge ffmpeg-python python-dotenv openai qdrant-client av tqdm
```

## Configuration

Create a `.env` file in the project root:
```env
DEEPSEEK_API_KEY=your_api_key_here
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
VIDEO_PATH=input.mp4
USER_PROMPT="Describe the main events and actions in this video"
```

## Usage

1. Place your input video in the project directory
2. Run the processor:
```bash
python main.py
```

## Features

- Video segmentation into 5-second chunks
- Frame extraction and processing using PyAV
- Video interpretation using Video-LLaVA
- Semantic embeddings using NV-Embed-v2
- Vector storage using Qdrant
- Context-aware interpretation using DeepSeek

## Output

The system generates:
- Timestamped interpretation files
- Vector embeddings for semantic search
- Processed video segments

## Limitations

- Requires 16GB+ VRAM for optimal performance
- Processing time depends on video length
- Currently optimized for 5-second segments

## License

This project uses components with various licenses:
- NV-Embed-v2: Non-commercial license
- Video-LLaVA: Apache 2.0
- Other components: See respective licenses

## Troubleshooting

If you encounter CUDA errors:
```bash
# Verify CUDA installation
nvidia-smi
# Verify FFmpeg CUDA support
ffmpeg -encoders | grep nvenc
```

For memory issues, adjust chunk_duration and frames_per_chunk in the VideoInterpreter class initialization.

## Verifying CUDA Installation

To verify that CUDA is properly configured for this project to run, execute:

`python verify_cuda.py`

This should return something like this:

```bash
python verify_cuda.py
2025-01-05 17:48:12,986 - INFO - NVIDIA Driver Info:
Sun Jan  5 17:48:12 2025       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       Off | 00000001:00:00.0 Off |                    0 |
| N/A   38C    P0              26W /  70W |    438MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A       853      C   .../miniconda3/envs/comfyui/bin/python      152MiB |
|    0   N/A  N/A   1234589      C   /usr/local/bin/python                       282MiB |
+---------------------------------------------------------------------------------------+

2025-01-05 17:48:13,040 - INFO - CUDA Toolkit Info:
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Aug_15_22:02:13_PDT_2023
Cuda compilation tools, release 12.2, V12.2.140
Build cuda_12.2.r12.2/compiler.33191640_0

2025-01-05 17:48:13,040 - INFO - PyTorch version: 2.2.1
2025-01-05 17:48:13,062 - INFO - CUDA available: True
2025-01-05 17:48:13,063 - INFO - CUDA version: 11.8
2025-01-05 17:48:13,077 - INFO - GPU device: _CudaDeviceProperties(name='Tesla T4', major=7, minor=5, total_memory=14930MB, multi_processor_count=40)
2025-01-05 17:48:13,077 - INFO - Current GPU device: 0
2025-01-05 17:48:13,077 - INFO - GPU count: 1
2025-01-05 17:48:13,077 - INFO - 
Verification Summary:
2025-01-05 17:48:13,077 - INFO - NVIDIA Driver: ✓ PASSED
2025-01-05 17:48:13,077 - INFO - CUDA Toolkit: ✓ PASSED
2025-01-05 17:48:13,077 - INFO - PyTorch CUDA: ✓ PASSED
2025-01-05 17:48:13,077 - INFO - 
CUDA installation verified successfully!
```


