# Rotational distribution calibration in DartQuant


In this directory, we provide the torch scripts for the experiments in DartQuant. 


## one-shot calibration data collection 

Currently, we only support **LLaMa** models. You can simply run the `get_train_data.py` to collect the calibration data. The most important arguments are:

- `--model`: the model name (or path to the weights)
- `--calib_dataset`: dataset used to obtain the calibration set
- `--r_path`: where to save the r1&2 calibration data
- `--r_list`: which rotation matrix calibration data to obtain
- `--nsamples`: number of samples
- `--seqlen`: sequence length
  
For example, to get the calibration data for Llama 2-7b, you can run the following command:

```bash
python get_train_data.py --model meta-llama/Llama-2-7b-hf --calib_dataset wikitext2 --nsamples 128 --seqlen 2048
```

## iterative rotation calibration


Currently, we only support **LLaMa** models. You can simply run the `r1_base_qr.py` to calibrate R1. The most important arguments are:


- `--model`: the model name (or path to the weights)
- `--ep`: number of epochs for training
- `--bsz`: batch-size for training
- `--lr`: learning rate for training
  
For example, to calibrate R1 for Llama 2-7b, you can run the following command:

```bash
python r1_base_qr.py --model meta-llama/Llama-2-7b-hf --ep 10 --bsz 64 --lr 1.5e-3
```

You can simply run the `r2_base_qr.py` to calibrate R2. The most important arguments are:


- `--model`: the model name (or path to the weights)
- `--ep`: number of epochs for training
- `--bsz`: batch-size for training
- `--lr`: learning rate for training

For example, to calibrate R1 for Llama 2-7b, you can run the following command:

```bash
python r2_base_qr.py --model meta-llama/Llama-2-7b-hf --ep 10 --bsz 64 --lr 1e-3
```