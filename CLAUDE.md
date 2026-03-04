This project is testing quantizations over LLMs. It was meant to test an experimental (published) method called dart, that learns allegedly better rotations than the quarot method, aimed to reduce outliers in LLMs.

quarot adds r1/r2/r3/r4 rotation operations to attention. 
dart computes r1/r2. It's in 
/workspace/DartQuant/calibrater/r1_base_qr.py
/workspace/DartQuant/calibrater/r2_base_qr.py

which can be both run using:
calibrater/calibrate_act_scales.py

we also do gptq with a specified number of bits. 



We have made several changes for:
1. easier testing (in fake_quant Scripts), with caching and visualization.
2. support of the 1B model (didn't work OOB)
3. bypass huggingface downloads to use local datasets
4. add support for static configuration, which is a big and important change
5. Added the Piecewise Linear activation support based on the halio-sdk repository, and tested it.

The quantization of the hailo-sdk repository is in
`/home/yanivbl/phase2-sdk/model_optimization/model_optimization_production/hailo_model_optimization/flows/optimization_flow.py`


6. Integer GEMM with capped accumulator (`fake_quant/int_acc_gemm.py`): Triton kernel + PyTorch reference that simulates hardware integer GEMM with limited-width accumulator. Enabled via `--int_gemm --acc_bits N --acc_block_k N`. Integrated into inference, GPTQ propagation, and calibration.

for static configuration, we use `calibrater/calibrate_model.sh` before the run. it accepts r1/r2 and also runs gptq, which we cache.
`multi_calibration.py` runs `calibrater/calibrate_model.sh` for 3 different experiments.


when changing enviroment variables/ update the dockerfile to match.

example run:
`sh Script/experiments.sh -m 1b` 
but it's very slow and uses shared GPU, so don't run it yourself

