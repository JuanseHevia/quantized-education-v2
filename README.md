# Repository Name

## Introduction

This repository contains the implementation of various models and pipelines for retrieval-augmented generation (RAG) and other machine learning tasks. The project is structured to facilitate easy experimentation and development of different components.

## Main Directories

- **commons/**: Contains common modules used across the project.
  - `model.py`: Defines the machine learning models.
  - `pipeline.py`: Implements the data processing and model pipelines.
  - `retrieval.py`: Handles the retrieval mechanisms for the RAG pipeline.

- **lora/**: Contains scripts and utilities for low-rank adaptation (LoRA).
  - `playground.ipynb`: Jupyter notebook for experimenting with LoRA.
  - `utils.py`: Utility functions for LoRA.

- **notebooks/**: Contains Jupyter notebooks for various experiments and analyses.
  - `03_compare_conversation_models.ipynb`: Notebook for comparing different conversation models.
  - `inspect_architecture.ipynb`: Notebook for inspecting model architectures.
  - `load_model.ipynb`: Notebook for loading and testing models.
  - `rag.ipynb`: Notebook for RAG pipeline experiments.

- **quantization/**: Directory for quantization-related scripts and experiments.

- **rag/**: Directory for RAG-specific modules and data.
  - `data/`: Contains data files and scripts for the RAG pipeline.

- **requirements.txt**: Lists the dependencies required to run the project.

## Data

Please download the reference book we are using to inform our RAG pipeline from the following link:
[Download Reference Book](#)

## Installation

To install the required dependencies, run the following command:

```sh
pip install -r requirements.txt