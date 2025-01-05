# Architecture Overview

## System Purpose
This video processing system segments and analyzes videos using a combination of state-of-the-art models optimized for commodity hardware (16GB VRAM GPUs). It provides detailed interpretations while maintaining temporal context across video segments.

## Core Components

### Video Processing Pipeline
1. **FFmpeg Segmentation**
   - Splits videos into 5-second segments
   - Maintains video quality through copy mode
   - Enables parallel processing capabilities

2. **PG-Video-LLaVA Integration**
   - Chosen for superior temporal understanding
   - Provides pixel-level grounding capabilities
   - Integrates audio context through transcription
   - 4-bit quantization enables running on 16GB VRAM

3. **NV-Embed-v2 for Embeddings**
   - 1024-dimensional embeddings
   - State-of-the-art performance on MTEB benchmark
   - Optimized for generalist embedding tasks
   - Based on Mistral 7B architecture

## Model Selection Rationale

### Video Understanding
```
┌────────────────────┐
│    Input Video     │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│  FFmpeg Segments   │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│  PG-Video-LLaVA    │◄──── 4-bit Quantization
└─────────┬──────────┘
          ▼
┌────────────────────┐
│    NV-Embed-v2     │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│  Qdrant Storage    │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│  DeepSeek Chat     │
└────────────────────┘
```

### Model Comparison Table
| Feature | PG-Video-LLaVA | DeepSeek-VL2 |
|---------|---------------|---------------|
| Temporal Understanding | Native | Limited |
| Audio Integration | Yes | No |
| VRAM Usage (4-bit) | ~12GB | ~14GB |
| Object Tracking | Yes | No |
| Context Window | 4096 tokens | 2048 tokens |

## Memory Management

### Vector Storage Strategy
- Qdrant chosen for:
  - High-performance similarity search
  - Efficient metadata storage
  - Scalable collection management
  - Support for concurrent operations

### Temporal Context
- Memory bank maintains interpretations
- Sequential processing preserves temporal relationships
- Embeddings enable semantic search across segments

## Optimization Choices

### Quantization Strategy
- 4-bit quantization reduces VRAM usage by 75%
- Minimal accuracy loss (1-2%)
- Enables running on consumer GPUs
- Uses group-wise quantization for better quality

### Processing Efficiency
- 5-second segments balance:
  - Context preservation
  - Memory efficiency
  - Processing speed
  - Temporal coherence

## Future Considerations
- Dynamic segment length based on content
- Parallel processing of segments
- Integration with streaming pipelines
- Enhanced audio-visual synchronization

This architecture provides a balanced approach to video understanding, optimizing for both performance and resource constraints while maintaining high-quality interpretations.


