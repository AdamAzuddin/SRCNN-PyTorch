# SRCNN Implementation

This repository contains an implementation of the **Image Super-Resolution Using Deep Convolutional Networks** paper by Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang. The implementation is done using PyTorch and includes a Jupyter notebook demonstrating the training and evaluation of the SRCNN model for image super-resolution.

For more detail about how SRCNN works, you can check out my blog post series on dev.to [here](https://dev.to/adamazuddin/series/27213)

## Contents

- `srcnn_model.py`: Python script defining the architecture of the SRCNN model.
- `train_srcnn.ipynb`: Jupyter notebook containing code for training and evaluating the SRCNN model.
- `requirements.txt`: File listing the required Python packages for running the code.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/AdamAzuddin/SRCNN-PyTorch.git
   cd srcnn-implementation
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Download train.h5 and test.h5 by running the matlab scripts generate_train and generate_test from [here](https://mmlab.ie.cuhk.edu.hk/projects/SRCNN/SRCNN_train.zip) 
3. Open and run `train_srcnn.ipynb` in Jupyter Notebook to train the SRCNN model.
4. After training, use the trained model to perform image super-resolution on new images.

## References

- Original Paper: [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092)
- Source Code for the original paper: [https://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html]
- PyTorch: [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
