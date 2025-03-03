# BERT MLU Implementation

---

## Overview

This repository provides an MLU implementation of BERT based on [NVIDIA BERT for PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT).

---

## Support Matrix

### Training Support

| Models | Framework  | Supported MLU | Supported Data Precision | Multi-GPUs |
| ------ | ---------- | ------------- | ------------------------ | ---------- |
| BERT   | PyTorch 1.6 | MLU370-X8     | FP32                     | Yes        |

### Inference Support

| Models | Framework  | Supported MLU | Supported Data Precision |
| ------ | ---------- | ------------- | ------------------------ |
| BERT   | PyTorch 1.6 | MLU370-S4/X4  | FP32                     |

---

## Default Configuration

### Training Parameters

#### Optimizer

The model uses a custom Adam optimizer for BERT with the following parameters:

- **Learning Rate**: 3e-5 (for batch size 4)
- **Epsilon**: 1e-6
- **Weight Decay**: 0.01
- **Epochs**: 2

---

## Environment Requirements

- **Operating System**: Linux (e.g., Ubuntu 16.04, Ubuntu 18.04, CentOS 7.x) with Docker (>= v18.00.0) installed.
- **Hardware**: MLU370-X8 accelerator cards.
- **Software**:
  - Cambricon Driver >= v4.20.6
  - CNToolKit >= 2.8.3
  - CNNL >= 1.10.2
  - CNCL >= 1.1.1
  - CNLight >= 0.12.0
  - CNPyTorch >= 1.3.0
- **Alternative**: If the above requirements are not met, you can register and try the Cambricon Cloud Platform [@TODO].

---

## Quick Start Guide

### Directory Structure

- **run_scripts/**: Contains shell scripts for one-click training and inference.
- **models/**: Includes the original model repository files.
  - `run_squad.py`: Training and inference script. Use `python run_squad.py -h` for more details.
  - `run_squad.sh`: Wrapper script for `run_squad.py`.

### Dataset Preparation

1. Download the [SQuAD v1.1 dataset](https://rajpurkar.github.io/SQuAD-explorer/) and extract it. Alternatively, use the provided script:

   ```bash
   bash models/data/squad/squad_download.sh
   ```

   The extracted dataset should include:

   ```plain-text
   |——dev-v1.1.json
   |——evaluate-v1.1.py
   |——train-v1.1.json
   ```

2. Set the dataset environment variable:

   ```bash
   export SQUAD_DIR=YOUR_DATASET_PATH
   ```

### Docker Setup (Optional)

#### Import Docker Image

```bash
docker load -i xxx.tar.gz
```

#### Start Test Container

Modify `run_docker.sh` to map your host data directory to the container:

```bash
# Update /your/data:/your/data in run_docker.sh
bash run_docker.sh
```

#### Configure Container Environment

1. Set `SQUAD_DIR` in `env.sh` to the dataset path inside the container.
2. Set `BERT_MODEL` in `env.sh` to the pre-trained model path (see "Pre-trained Model Preparation" below).
3. Set `BERT_INFER_MODEL` in `env.sh` to the inference model path.
4. Activate the environment:

   ```bash
   source env.sh
   source /torch/venv3/pytorch/bin/activate
   ```

#### Build Docker Image

```bash
export IMAGE_NAME=demo_bert
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../
```

#### Start Container

```bash
docker run -it --ipc=host -v /data:/data -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name mlu_bert --network=host $IMAGE_NAME
```

### Pre-trained Model Preparation

1. Download the pre-trained [BERT model](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip) and extract it. The contents should include:

   ```plain-text
   .
   ├── bert_config.json
   ├── bert_model.ckpt.data-00000-of-00001
   ├── bert_model.ckpt.index
   ├── bert_model.ckpt.meta
   └── vocab.txt
   ```

2. Convert the TensorFlow checkpoint to PyTorch format:
   - Install dependencies:

     ```bash
     pip install protobuf==3.20.0 tensorflow==2.15.0
     pip install transformers
     ```

   - Run the conversion script:

     ```bash
     python convert_bert_original_tf_checkpoint_to_pytorch.py --tf_checkpoint_path /path/to/bert_model.ckpt --bert_config_file /path/to/bert_config.json --pytorch_dump_path /path/to/pytorch_model.pt
     ```

3. Set environment variables:

   ```bash
   # Before Training
   export BERT_MODEL=/path/to/pytorch_model.pt
   # After Training
   export BERT_INFER_MODEL=/path/to/output/pytorch_model.bin
   ```

### Run Training or Inference

```bash
bash run_scripts/BERT_FP32_2E_4MLUs_Train.sh
```

---

## One-Click Scripts

### Training

| Models | Framework | MLU       | Data Precision | Cards | Description                  | Run                                          |
| ------ | --------- | --------- | -------------- | ----- | ---------------------------- | -------------------------------------------- |
| BERT   | PyTorch   | MLU370-X8 | FP32           | 4     | Fine-tune training using 4 MLUs | `bash run_scripts/BERT_FP32_2E_4MLUs_Train.sh` |

### Inference

Before running inference, set the `BERT_INFER_MODEL` environment variable to the trained model path (e.g., `models/output/pytorch_model.bin`).

> Before Inferencing, you should run `export CNCL_MLU_DIRECT_LEVEL=1` to enable MLU Direct RDMA.

| Models | Framework | MLU          | Data Precision | Description      | Run                            |
| ------ | --------- | ------------ | -------------- | ---------------- | ------------------------------ |
| BERT   | PyTorch   | MLU370-S4/X4 | FP32           | Inference script | `bash run_scripts/BERT_Infer.sh` |

---

## Results

### Training Accuracy (MLU370-X8)

| Models | Data Precision | F1 Score |
| ------ | -------------- | -------- |
| BERT   | FP32           | 88.51    |

---

## Disclaimer

The software, data, or models linked below are provided and maintained by third parties. The appearance of any third-party names, trademarks, logos, products, or services does not constitute an endorsement, guarantee, or recommendation. You acknowledge and agree that the use of any third-party software, data, or models, including any information or personal data you provide (intentionally or unintentionally), is subject to the terms, licenses, privacy policies, or other agreements governing such use. All risks associated with using the linked resources are your sole responsibility.

- **Dataset**: [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)
- **Pre-trained Model**: [BERT](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)
