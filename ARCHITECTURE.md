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

# Expected Performance

Based on web search results, the accuracy of our implementation using PG-Video-LLaVA can be broken down across several metrics:

## Core Performance Metrics
- MSRVD-QA: 64.1% accuracy
- MSRVTT-QA: 51.6% accuracy
- TGIF-QA: 66.8% accuracy
- ActivityNet-QA: 39.9% accuracy[3]

## Quality Dimensions
- Correctness: 2.84 out of 5
- Detail Orientation: 2.97 out of 5
- Contextual Understanding: 3.22 out of 5
- Temporal Understanding: 2.54 out of 5
- Consistency: 3.56 out of 5[4]

## Spatial Grounding Accuracy
- VidSTG benchmark: 35.1% accuracy
- HC-STVG benchmark: 27.3% accuracy[4]

The model's performance is enhanced when using audio integration, but it's important to note that it performs best with:
- Shorter video clips
- Consistent camera views
- Clear audio transcription
- Well-defined temporal segments

The accuracy can degrade with longer videos or complex scenes that require extensive temporal reasoning[3].

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/1678275/d8db9e9a-51b2-4b48-a2d0-01f962900ac3/paste.txt
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/1678275/37bf69a7-970c-48b1-99a4-d4b5dec5780d/paste.txt
[3] https://arxiv.org/html/2311.13435v2
[4] https://openreview.net/pdf/5b8e8e8192fda2be20e6e0a4d04972a60bd7fdbc.pdf
[5] https://openreview.net/pdf/ee697e8c0d4f92a44e85ffbdda620ce31973713d.pdf
[6] https://www.youtube.com/watch?v=ZbOgzFxLPD4
[7] https://github.com/mbzuai-oryx/video-llava
