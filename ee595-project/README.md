# Multithreaded FNO-backed Inference Application
## Overview
This program implements <a href="https://github.com/gengshuoliu/EE595_MFG">FNO model</a> inference in C++ with multithreading, allowing faster inferences than the vanilla Pytorch inferences.

## Dependencies
This package depends on the C++ distributions of Pytorch. Install the distribution using these command:


        wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
        unzip libtorch-shared-with-deps-latest.zip
Set DCMAKE_PREFIX_PATH = /path/to/libtorch in build.sh

## Model
This application uses the <a href="https://github.com/gengshuoliu/EE595_MFG">FNO model</a> after <a href="https://pytorch.org/tutorials/advanced/cpp_export.html#step-1-converting-your-pytorch-model-to-torch-script">serializing</a> it. 
## Build
Build with the following command


        sh build.sh
## Usage
This program accepts tensors(.pt files) of size (n, 256, 3) as input. Put the input files inside this directory. Then call 

        
        ./build/ee595-project <input_file> <num_thread>
By default, this program runs with num_thread = 1.