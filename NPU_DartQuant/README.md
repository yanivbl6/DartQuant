# Code for ``DartQuant: Efficient Rotational Distribution Calibration for LLM Quantization''

## 1. Requirements:
- python 3.10, pytorch >= 2.0
- install pytorch with NPU.
- pip install -r requirement.txt

## Guidelines

- The ``fake_quant'' folder contains the code for fusing the calibrated rotation matrix and performing the quantization test. The usage is described in detail in the Readme.md file in the directory.

- The ``calibrater'' folder contains the code for obtaining the calibration set and the calibration rotation matrix. The specific usage is described in the Readme.md in this directory.