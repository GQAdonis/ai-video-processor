# Create conda environment with Python 3.11.2
conda create -n video-processor python=3.11.2 -y
conda activate video-processor

# Install PyTorch ecosystem with CUDA support
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install additional dependencies
conda install -c conda-forge transformers=4.42.4
conda install -c conda-forge flash-attn=2.2.0
conda install -c conda-forge sentence-transformers=2.7.0
conda install -c conda-forge ffmpeg-python
conda install -c conda-forge python-dotenv
conda install -c conda-forge openai
conda install -c conda-forge qdrant-client
conda install -c conda-forge av
conda install -c conda-forge tqdm
