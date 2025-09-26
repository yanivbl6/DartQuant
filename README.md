# Code for ``DartQuant: Efficient Rotational Distribution Calibration for LLM Quantization''

## 1. Requirements:
- python 3.10, pytorch >= 2.0
- install pytorch with cuda from https://pytorch.org/get-started/locally/, it is prerequisite for fast-hadamard-transform package.
- pip install -r requirement.txt
  
  install fast-hadamard-transform
  
  ```python
  cd third-part
  git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
  cd fast-hadamard-transform
  pip install .
  ```

  install lm-eval
  
  ```python
  git clone https://github.com/EleutherAI/lm-evaluation-harness.git
  cd lm-evaluation-harness
  pip install -e .
  ```

## Guidelines

- The ``fake_quant'' folder contains the code for fusing the calibrated rotation matrix and performing the quantization test. The usage is described in detail in the Readme.md file in the directory.

- The ``calibrater'' folder contains the code for obtaining the calibration set and the calibration rotation matrix. The specific usage is described in the Readme.md in this directory.
