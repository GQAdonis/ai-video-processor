import os
import subprocess
import torch
import numpy as np
import av
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from transformers import (
    VideoLlavaForConditionalGeneration, 
    VideoLlavaProcessor,
    BitsAndBytesConfig,
    AutoModel,
    AutoTokenizer,
    HfFolder
)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
import logging
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoInterpreter:
    def __init__(
        self, 
        model_cache_dir: str = "model_cache",
        chunk_duration: int = 5,
        frames_per_chunk: int = 8,
        collection_name: str = "video_segments",
        embedding_dimension: int = 1024
    ):
        self.model_cache_dir = Path(model_cache_dir)
        self.chunk_duration = chunk_duration
        self.frames_per_chunk = frames_per_chunk
        self.output_dir = Path("hls_chunks")
        self.chat_history = []
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        
        # Get API keys and tokens from environment
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        if not all([self.deepseek_api_key, self.qdrant_url, self.qdrant_api_key, self.hf_token]):
            raise ValueError("Missing required environment variables")
        
        # Set HuggingFace token
        HfFolder.save_token(self.hf_token)
        
        # Initialize clients and models
        self._initialize_clients()
        self._initialize_models()
        self._initialize_vector_store()

    def _initialize_clients(self):
        """Initialize API clients"""
        try:
            # Initialize DeepSeek client
            self.deepseek_client = OpenAI(
                api_key=self.deepseek_api_key,
                base_url="https://api.deepseek.com"
            )
            
            # Initialize Qdrant client
            self.qdrant = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key
            )
            
        except Exception as error:
            logger.error(f"Error initializing clients: {str(error)}")
            raise

    def _initialize_models(self):
        """Initialize all required models"""
        try:
            logger.info("Initializing models...")
            
            # Initialize Video-LLaVA
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            video_model_path = self.model_cache_dir / "video_llava"
            embedding_model_path = self.model_cache_dir / "nv_embed"
            
            # Download and initialize Video-LLaVA
            if not video_model_path.exists():
                logger.info("Downloading Video-LLaVA model...")
                self.video_model = VideoLlavaForConditionalGeneration.from_pretrained(
                    "LanguageBind/Video-LLaVA-7B-hf",
                    quantization_config=quantization_config,
                    device_map="auto",
                    cache_dir=str(video_model_path),
                    token=self.hf_token
                )
                self.processor = VideoLlavaProcessor.from_pretrained(
                    "LanguageBind/Video-LLaVA-7B-hf",
                    cache_dir=str(video_model_path),
                    token=self.hf_token
                )
                self.video_model.save_pretrained(str(video_model_path))
                self.processor.save_pretrained(str(video_model_path))
            else:
                logger.info("Loading Video-LLaVA from cache...")
                self.video_model = VideoLlavaForConditionalGeneration.from_pretrained(
                    str(video_model_path),
                    quantization_config=quantization_config,
                    device_map="auto"
                )
                self.processor = VideoLlavaProcessor.from_pretrained(str(video_model_path))
            
            # Download and initialize NV-Embed
            if not embedding_model_path.exists():
                logger.info("Downloading NV-Embed model...")
                self.embedding_model = AutoModel.from_pretrained(
                    "nvidia/nv-embed-v2",
                    trust_remote_code=True,
                    device_map="auto",
                    cache_dir=str(embedding_model_path),
                    token=self.hf_token
                )
                self.embedding_tokenizer = AutoTokenizer.from_pretrained(
                    "nvidia/nv-embed-v2",
                    cache_dir=str(embedding_model_path),
                    token=self.hf_token
                )
                self.embedding_model.save_pretrained(str(embedding_model_path))
                self.embedding_tokenizer.save_pretrained(str(embedding_model_path))
            else:
                logger.info("Loading NV-Embed from cache...")
                self.embedding_model = AutoModel.from_pretrained(
                    str(embedding_model_path),
                    trust_remote_code=True,
                    device_map="auto"
                )
                self.embedding_tokenizer = AutoTokenizer.from_pretrained(str(embedding_model_path))
            
            logger.info("Models initialized successfully")
            
        except Exception as error:
            logger.error(f"Error initializing models: {str(error)}")
            raise

    # [Rest of the methods remain unchanged]

def main():
    try:
        # Load environment variables
        load_dotenv()
        
        # Check for required environment variables
        required_vars = [
            "DEEPSEEK_API_KEY",
            "QDRANT_URL",
            "QDRANT_API_KEY",
            "HUGGINGFACE_TOKEN"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        video_path = os.getenv("VIDEO_PATH", "input.mp4")
        user_prompt = os.getenv("USER_PROMPT", "Describe the main events and actions in this video")
        
        interpreter = VideoInterpreter()
        result = interpreter.process_video(video_path, user_prompt)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"video_interpretation_{timestamp}.txt"
        
        with open(output_file, "w") as file:
            file.write(result)
        
        logger.info(f"Results saved to {output_file}")
        print("\nVideo Interpretation:")
        print(result)
        
    except Exception as error:
        logger.error(f"Error in main execution: {str(error)}")
        raise

if __name__ == "__main__":
    main()
