# Fake Quantization in DartQuant


In this directory, we provide the torch scripts for the experiments in DartQuant. 


## Language Generation and Zero-Shot Evaluations

Currently, we only support **LLaMa** models. You can simply run the `main_for_test.py` to reproduce the results in the paper. The most important arguments are:

- `--model`: the model name (or path to the weights)
- `--bsz`: the batch size for PPL evaluation
- `--fuse_norm`: whether we want to fuse the layer norm
- `--use_r1`: whether we want to use R1 for rotate attention, up-projection and gate-projection inputs.
- `--r1_path`: we are going to use the calibrated R1 path
- `--use_r2`: whether we want to use R2 for rotate out-projection inputs.
- `--r2_path`: we are going to use the calibrated R2 path
- `--use_r3`: whether we want to use R1
- `--use_r4`: whether we want to use R1
- `--ppl_eval`: whether we want to run PPL test
- `--ppl_eval_dataset`: the tasks for PPL
- `--lm_eval`: whether we want to run LM-Eval for Zero-Shot tasks
- `--tasks`: the tasks for LM-Eval
- `--cal_dataset`: the calibration dataset for GPTQ quantization
- `--a_bits`: the number of bits for activation quantization
- `--w_bits`: the number of bits for weight quantization
- `--v_bits`: the number of bits for value quantization
- `--k_bits`: the number of bits for key quantization
- `--w_clip`: Whether we want to clip the weights
- `--a_clip_ratio`: The ratio of clipping for activation
- `--k_clip_ratio`: The ratio of clipping for key
- `--v_clip_ratio`: The ratio of clipping for value
- `--w_asym`: Whether we want to use asymmetric quantization for weights
- `--a_asym`: Whether we want to use asymmetric quantization for activation
- `--v_asym`: Whether we want to use asymmetric quantization for value
- `--k_asym`: Whether we want to use asymmetric quantization for key
- `--a_groupsize`: The group size for activation quantization
- `--w_groupsize`: The group size for weight quantization
- `--v_groupsize`: The group size for value quantization
- `--k_groupsize`: The group size for key quantization
  
For example, to run a model with quantizing all weights and activations, you can run the following command:

```bash
bash Script/dart_gptq_wxaykvz_test.sh <GPU_ID> <MODEL> <W_BITS> <A_BITS> <KV_BITS> <R2_PATH> <R1_PATH>
```
