# MSPatch: A Multi-scale Patch Mixing Framework for Multivariate Time Series Forecasting

MSPatch is a state-of-the-art framework designed for multivariate time series forecasting. By decomposing time series data into multi-scale patches, MSPatch captures both short-term and long-term temporal patterns, providing accurate and robust predictions across various domains such as traffic management, weather forecasting, and more.

## Features

- **Multi-scale Patch Embedding (MSPE) Module**: Decomposes time series data into patches at different scales, capturing both fine-grained and broad temporal features.
- **Patch Linear Attention (PLA) Module**: Efficiently captures dependencies within and across patches using linear attention mechanisms.
- **Hybrid Patch Convolution (HPC) Module**: Fuses information from patches of different lengths, enhancing the model's ability to capture local and global dependencies.

## Installation

To install and run MSPatch, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/zjucaoyz/MSPatch.git
   cd MSPatch
2. Training the Model

    ```bash
   python run.py --model MSPatch --data ETTh1 --pred_len 96

## Results
MSPatch has been extensively tested on multiple benchmark datasets, including ETTh1, Traffic, and Weather. The framework consistently achieves state-of-the-art performance, particularly in long-term forecasting tasks.

## Results
We appreciate the following GitHub repos a lot for their valuable code and efforts.

Time-Series-Library (https://github.com/thuml/Time-Series-Library)

Autoformer (https://github.com/thuml/Autoformer)
