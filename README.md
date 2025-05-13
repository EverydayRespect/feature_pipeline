# Feature Pipeline

A distributed video feature extraction pipeline that processes videos using state-of-the-art vision-language models and stores the extracted features in a vector database.

## Overview

This pipeline is designed to:
1. Process multiple videos in parallel using GPU workers
2. Extract features using various vision-language models (CLIP, VL3-SigLIP-NaViT)
3. Store the extracted features in a Milvus vector database
4. Handle distributed processing with proper error handling and logging

## System Requirements

- Python 3.x
- CUDA-compatible GPU (for optimal performance)
- MacOS/Linux (MPS/CUDA support)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd feature_pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download models:
```
cd models
python download.py
```

## Configuration

The pipeline is configured using YAML files located in the `config/` directory. The default configuration file is `config/navit-config.yaml`.

Change the following configs when running pipeline:
- video_input.path
- gpus (according to server gpu type and number)

Example configuration:
```yaml
model:
  name: "VL3-SigLIP-NaViT"
  path: "models/VL3-SigLIP-NaViT"
  source: "huggingface"
  features: ["video_embedding"]

video_input:
  path: "./video_samples/"

gpus:
  - cuda:0
  - cuda:1
  - cuda:2

db:
  type: "milvus"
  name: "navit_video_feature.db"
  batch_size: 1000
  collections:
    - name: "video_embedding_collection"
      fields:
        - name: id
          dtype: INT64
          is_primary: true
          auto_id: true
        - name: video_path
          dtype: VARCHAR
          max_length: 512
        - name: frame_id
          dtype: INT16
        - name: row_idx
          dtype: INT16
        - name: col_idx
          dtype: INT16
        - name: embeddings
          dtype: FLOAT_VECTOR
          dim: 1152
```

## Usage

Run the pipeline:
```bash
python main.py --config_path config/navit-config.yaml
```

## Project Structure

```
feature_pipeline/
├── main.py              # Main entry point
├── worker.py            # GPU and DB worker implementations
├── model.py             # Model implementations
├── database.py          # Database interface
├── utils.py             # Utility functions
├── logger.py            # Logging configuration
├── config/              # Configuration files
├── models/              # Model checkpoints
├── video_samples/       # Input videos
├── logs/                # Log files
└── db/                  # Database files
```

## Components

### Models
- **CLIP**: OpenAI's CLIP model for video feature extraction
- **VL3-SigLIP-NaViT**: Vision-language model encoder for VideoLLaMA3

### Workers
- **GPU Worker**: Processes videos and extracts features
- **DB Worker**: Handles database operations

### Database
- Uses Milvus for efficient vector storage and retrieval
- Supports different collection schemas for various feature types

## Logging

The pipeline uses a comprehensive logging system that tracks:
- Video processing progress
- Model loading and inference
- Database operations
- Error handling

Logs are stored in the `logs/` directory with timestamps.

## Error Handling

The pipeline includes robust error handling for:
- Model loading failures
- Video processing errors
- Database connection issues
- Worker thread management
