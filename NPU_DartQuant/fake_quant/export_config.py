from collections import defaultdict

class DotDict(defaultdict):
    def __init__(self, *args, **kwargs):
        super().__init__(DotDict, *args, **kwargs)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

llama2_70b_dict_dart = DotDict()

# General settings
llama2_70b_dict_dart.model = "meta-llama/Llama-2-70b-hf"
llama2_70b_dict_dart.seed = 0
llama2_70b_dict_dart.hf_token = None

# Rotation settings
llama2_70b_dict_dart.fuse_norm = True
llama2_70b_dict_dart.smooth = None
llama2_70b_dict_dart.use_r1 = True
llama2_70b_dict_dart.r1_path = "/home/chenrenxing/NPU_DartQuant/data/trained_rotation/wikitext2_128samples/Llama-2-70b-hf/r2/sgd.0.001.0.9.10.64.2/sgd.0.001.0.9.10.64.2.pt"
llama2_70b_dict_dart.use_r2 = 'offline'
llama2_70b_dict_dart.r2_path = "/home/chenrenxing/NPU_DartQuant/data/trained_rotation/wikitext2_128samples/Llama-2-70b-hf/r1/sgd.0.0015.0.9.10.64.0.1.1/sgd.0.0015.0.9.10.64.0.1.1.pt"
llama2_70b_dict_dart.use_r3 = True
llama2_70b_dict_dart.use_r4 = True
llama2_70b_dict_dart.rotate_mode = "hadamard"
llama2_70b_dict_dart.fp32_had = False

# Activation quantization
llama2_70b_dict_dart.a_bits = 4
llama2_70b_dict_dart.a_groupsize = -1
llama2_70b_dict_dart.a_asym = True
llama2_70b_dict_dart.a_clip_ratio = 0.9
llama2_70b_dict_dart.a_residual = False

# Weight quantization
llama2_70b_dict_dart.w_bits = 4
llama2_70b_dict_dart.w_groupsize = 128
llama2_70b_dict_dart.w_static_groups = False
llama2_70b_dict_dart.w_asym = False
llama2_70b_dict_dart.w_rtn = False
llama2_70b_dict_dart.w_clip = True
llama2_70b_dict_dart.nsamples = 128
llama2_70b_dict_dart.cal_dataset = "wikitext2"
llama2_70b_dict_dart.percdamp = 0.01
llama2_70b_dict_dart.act_order = False

# Per-layer overrides
llama2_70b_dict_dart.w_bits_down_proj = None
llama2_70b_dict_dart.a_bits_down_proj = None
llama2_70b_dict_dart.o_per_head = True

# KV-cache quantization
llama2_70b_dict_dart.v_bits = 16
llama2_70b_dict_dart.v_groupsize = 128
llama2_70b_dict_dart.v_asym = True
llama2_70b_dict_dart.v_clip_ratio = 1.0

llama2_70b_dict_dart.k_bits = 16
llama2_70b_dict_dart.k_groupsize = 128
llama2_70b_dict_dart.k_asym = True
llama2_70b_dict_dart.k_pre_rope = False
llama2_70b_dict_dart.k_clip_ratio = 1.0

# Quantized model I/O
llama2_70b_dict_dart.load_qmodel_path = "/home/chenrenxing/NPU_DartQuant/model/Llama-2-7b-hf_w4_r1_r2_r3_r4_gptq_clip_g128"
llama2_70b_dict_dart.save_qmodel_path = None

# WandB
llama2_70b_dict_dart.wandb = False
llama2_70b_dict_dart.wandb_id = None
llama2_70b_dict_dart.wandb_project = None

# Experiment settings
llama2_70b_dict_dart.save_name = None  # can be set later with datetime.now().strftime("%Y%m%d_%H%M%S")
llama2_70b_dict_dart.log_to_console = True

# Layer I/O capture
llama2_70b_dict_dart.capture_layer_io = False
llama2_70b_dict_dart.layer_idx = 10

# PPL eval
llama2_70b_dict_dart.ppl_eval = False
llama2_70b_dict_dart.ppl_eval_dataset = ["wikitext2", "ptb", "c4"]
llama2_70b_dict_dart.ppl_eval_batch_size = 1

# LM eval
llama2_70b_dict_dart.lm_eval = False
llama2_70b_dict_dart.tasks = [
    "piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande",
    "lambada_openai", "social_iqa", "openbookqa", "mmlu"
]
llama2_70b_dict_dart.lm_eval_batch_size = 8
llama2_70b_dict_dart.distribute = True

# Weight fine-tuning
llama2_70b_dict_dart.w_ft = False
llama2_70b_dict_dart.ft_percdamp = 0.0

# Optional custom logic you may want:
# llama2_70b_dict_dart.save_path = os.path.join("experiments", llama2_70b_dict_dart.model, llama2_70b_dict_dart.save_name or datetime.now().strftime("%Y%m%d_%H%M%S"))

