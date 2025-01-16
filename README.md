# Visual Evolutionary Surrogate-Assisted Prescription Architecture

The goal of this project is to incorporate image data into the context of ESP.
The architecture being used is a CNN context encoder, whose output is concatenated with the actions vector, passed through a fully connected network to predict outcomes. Then, this encoder is reused to precompute the entire encoded context which is used to evolved simple FCN prescriptors.

## Installation
This was made with python 3.12.8,

To install, use `pip install -r requirements.txt`.

Make sure to set `export PYTHONPATH=$PWD` before running any experiments.

## MNIST
The first use-case is classifying the MNIST dataset into even or odd using synthetic actions and outcomes. The context is the digit, the outcome is randomly assigned as 0 or 1 (incorrect or correct) and the actions are assigned based off the outcome as guessing even or odd (0 or 1). For example, 2 valid (C, A, O) pairs are (7, 1, 1), and (7, 0, 0) because if 7 is guessed to be odd then the outcome must be correct. If 7 is guessed to be even the answer must be wrong.

To run evolution on MNIST use `python mnist/run_evolution.py --config mnist/config.yml`.

