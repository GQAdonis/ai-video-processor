import os
import subprocess
import torch
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_nvidia_driver():
    """Check NVIDIA driver installation"""
    try:
        output = subprocess.check_output(['nvidia-smi']).decode()
        logger.info("NVIDIA Driver Info:\n%s", output)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("NVIDIA driver not found or not properly installed")
        return False

def check_cuda_toolkit():
    """Check CUDA toolkit installation"""
    try:
        output = subprocess.check_output(['nvcc', '--version']).decode()
        logger.info("CUDA Toolkit Info:\n%s", output)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("CUDA toolkit not found or not properly installed")
        return False

def check_pytorch_cuda():
    """Check PyTorch CUDA configuration"""
    logger.info("PyTorch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        logger.info("CUDA version: %s", torch.version.cuda)
        logger.info("GPU device: %s", torch.cuda.get_device_properties(0))
        logger.info("Current GPU device: %s", torch.cuda.current_device())
        logger.info("GPU count: %s", torch.cuda.device_count())
        return True
    return False

def verify_cuda_installation():
    """Run all verification checks"""
    checks = {
        "NVIDIA Driver": check_nvidia_driver(),
        "CUDA Toolkit": check_cuda_toolkit(),
        "PyTorch CUDA": check_pytorch_cuda()
    }
    
    logger.info("\nVerification Summary:")
    all_passed = True
    for check, passed in checks.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info("%s: %s", check, status)
        all_passed = all_passed and passed
    
    return all_passed

if __name__ == "__main__":
    try:
        if verify_cuda_installation():
            logger.info("\nCUDA installation verified successfully!")
        else:
            logger.error("\nCUDA installation verification failed!")
            exit(1)
    except Exception as error:
        logger.error("Error during verification: %s", str(error))
        exit(1)
