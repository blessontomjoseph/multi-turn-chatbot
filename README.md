 # Multiturn Chatbot System 

<!-- ![Chatbot Banner](https://example.com/chatbot_banner.png) -->

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Model Implementation](#model-implementation)
- [Evaluation Metrics](#evaluation-metrics)
- [Logging and Visualization](#logging-and-visualization)
- [Parallel Processing](#parallel-processing)
- [Training and Inference](#training-and-inference)
- [System Requirements](#system-requirements)
- [Usage](#usage)
- [Reads](#Reads)

## Introduction

Welcome to our Multiturn Chatbot System! This project implements a chatbot capable of handling multiple turns of conversation, and it can be trained on various large language models (LLMs). The current implementation supports two specific LLMs: GPT Neo 125M by EleutherAI and Open Llama 3B. The chatbot is designed to process and train on three different conversational datasets: Daily Dialog Dataset, Empathetic Dialog Dataset, and Human Chat Dataset.

The chatbot's main features include:
- Multiturn conversation handling
- Training on GPT Neo 125M and Open Llama 3B
- Evaluation using ROUGE-1, ROUGE-2, ROUGE-L, and BLEU scores
- Logging training and evaluation outputs to Weights and Biases
- Checkpointing the trained model to Hugging Face Hub
- Option for parallel processing using Lightning code (GPT Neo implementation)
- Single GPU training for GPT Neo without using Qlora
- Limited context retention during inference for effective responses

## Datasets

The following datasets are used for training the chatbot:

1. [**Daily Dialog Dataset**](https://huggingface.co/datasets/daily_dialog) from HuggingFace
2. [**Empathetic Dialog Dataset**](https://huggingface.co/datasets/empathetic_dialogues) from HuggingFace
3. [**Human Chat Dataset**](https://www.kaggle.com/datasets/projjal1/human-conversation-training-data) from Kaggle

the preprocessed datasets are given in the [processed data](processed_data) directory

## Model Implementation

The chatbot is implemented using Python and leverages the powerful capabilities of GPT Neo 125M and Open Llama 3B. The codebase is structured to accommodate easy integration with other LLMs if desired. The implementation is built on top of Hugging Face's `transformers` library.

## Evaluation Metrics

To evaluate the chatbot's performance, we use the following metrics:

- **ROUGE-1**: Measures unigram overlap between generated responses and reference responses.
- **ROUGE-2**: Measures bigram overlap between generated responses and reference responses.
- **ROUGE-L**: Longest Common Subsequence (LCS) based metric that captures long-range sequence similarity.
- **BLEU**: Bilingual Evaluation Understudy, which measures n-gram precision against reference responses.

## Logging and Visualization

The training and evaluation outputs are logged and visualized using Weights and Biases (W&B). This allows for easy monitoring of the training progress and performance metrics. You can access the dashboards for the chatbot's training and evaluation at the following link: [Dashboard Link](https://wandb.ai/theothertom/lightning_logs?workspace=user-theothertom)

## Parallel Processing

By default, the codebase does not utilize parallel processing. However, if you have access to multiple GPUs, you can leverage parallel processing using the provided [Lightning code](lightning.py). This is particularly useful for large models like LLaMa, which may require 32 GB of VRAM for full model fine-tuning. For single-GPU users with 16 GB VRAM,LLaMa can be trained using QLoRa but the implementaion is not provided here.

## Training and Inference

The training process involves pre-processing the datasets using the tokenizer associated to the LLM and fine-tuning the selected LLM on the multiturn chatbot task. After training, the model is checkpointed into Hugging Face Hub for easy access and sharing. [Trained model checkpoint](https://huggingface.co/theothertom/gpt_neo_extended_retrain/tree/main)

During inference, the chatbot takes a certain amount of conversation history as input to maintain context for a limited number of chat turns. This ensures that the chatbot provides more relevant and coherent responses.

## System Requirements

To run the chatbot system, make sure you have the following:

- Python 3.6 or higher
- GPU with at least 16 GB VRAM (32 GB VRAM recommended for LLaMa training)
- Datasets: Daily Dialog Dataset, Empathetic Dialog Dataset, Human Chat Dataset

## Usage

To use the chatbot system, follow these steps:

1. Clone the repository: `git clone https://github.com/your_username/your_chatbot_repo.git`
2. Install the required dependencies: `pip install -r requirements.txt`
4. dont have to run the preprocessing script as it is already in the repo
5. if loging to wandb and checkpoining to hf is required, provide proper login keys in the [train file](train.py)
6. Fine-tune the selected LLM on the multiturn chatbot task.
7. Evaluate the trained model using the provided evaluation metrics.


## Reads

1. tokenization for chat bot and contextual inference: 
    - https://itnext.io/building-a-multi-turn-chatbot-with-gpt-and-sagemaker-a-step-by-step-guide-7d75f33ccea1
 
2. QLoRa: 
    - https://arxiv.org/abs/2305.14314
    - https://huggingface.co/blog/4bit-transformers-bitsandbytes
