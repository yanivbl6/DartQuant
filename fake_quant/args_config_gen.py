import argparse
import pprint
import os
from datetime import datetime
import logging

from utils import supported_models, supported_datasets, config_logging


def parser_gen():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                        help='Model to load;')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for HuggingFace and PyTorch')
    parser.add_argument('--hf_token', type=str, default=None)

    # Rotation Arguments
    parser.add_argument('--fuse_norm', action=argparse.BooleanOptionalAction, default=True,
                        help='Fuse the normalization layer with the linear layer.')
    parser.add_argument('--smooth', type=str, default=None,
                        help='Smooth the rotation matrix.')
    parser.add_argument('--use_r1', action=argparse.BooleanOptionalAction, default=True,
                        help='''Use R1 for rotate attention, up-projection and gate-projection inputs.''')
    parser.add_argument('--r1_path', type=str, default=None,
                        help='''Path to the R1 rotation matrix. Deafult is None.
                        If not specified, R1 will generated as "rotate_mode".''')
    parser.add_argument('--use_r2', type=str, default='offline',
                        choices=['offline', 'online', 'none'],
                        help='''Use R2 for rotate out-projection inputs.''')
    parser.add_argument('--r2_path', type=str, default=None,
                        help='''Path to the R2 rotation matrix. Deafult is None.
                        If not specified, R2 will generated as "rotate_mode".''')
    parser.add_argument('--use_r3', action=argparse.BooleanOptionalAction, default=True,
                        help='''Use R3 for rotate Q/K Online.''')
    parser.add_argument('--use_r4', action=argparse.BooleanOptionalAction, default=True,
                        help='''Use R4 for rotate down-projection inputs Online.''')
    parser.add_argument('--kv_ex', type=int, default=0,
                        help='When non-zero, disable R3 and quantize K-cache to N bits (no rotation).')
    parser.add_argument('--proj_ex', type=int, default=0,
                        help='When non-zero, disable R4 and quantize down_proj input to N bits. Shorthand for --no_r4 --down_bits X.')
    parser.add_argument('--no_r4', action='store_true',
                        help='Disable R4 rotation on down_proj (without changing bits)')
    parser.add_argument('--late_rot4', action='store_true', default=False,
                        help='Quantize weights first, then apply R4 rotation: Q(W)@H instead of Q(W@H)')
    parser.add_argument('--down_bits', type=int, default=None,
                        help='Override down_proj input activation bits (without disabling R4)')
    parser.add_argument('--oproj_bits', type=int, default=None,
                        help='Override o_proj input activation bits (e.g. 16 for int16 decomposition)')
    parser.add_argument('--eq', action='store_true',
                        help='Enable per-channel equalization on down_proj inputs '
                             '(legacy: online division at down_proj input)')
    parser.add_argument('--ud_eq', action='store_true',
                        help='Branch equalization (up + down): per-channel scale on '
                             'up_proj output, folded into W_up rows and W_down cols. '
                             'Mutually exclusive with --eq and --ugd_eq.')
    parser.add_argument('--ugd_eq', action='store_true',
                        help='Branch equalization (up + gate + down): independent '
                             'per-channel scales on silu(gate) and up_proj output. '
                             'With --pwl_act the gate factor rides PWL per-channel '
                             's_out (HW-accurate); without --pwl_act it is applied '
                             'via an in-fp EqActivation wrapper (fake-quant). '
                             'Mutually exclusive with --eq and --ud_eq.')
    parser.add_argument('--quant_out', type=str, default='none',
                        choices=['none', 'up', 'mlp', 'spec', 'speco', 'all', 'r4', 'res', 'mm', 'ex'],
                        help='Output quantization: none, up, mlp, spec, speco, all, r4, res (residuals), mm (Q in attn), ex (all+res+mm)')
    parser.add_argument('--rotate_mode', type=str, default='hadamard', choices=['hadamard', 'random'])
    # parser.add_argument('--rotation_seed', type=int, default=-1,
    #                     help='Random Seed for generating random matrix!!')
    parser.add_argument('--fp32_had', action=argparse.BooleanOptionalAction, default=False,
                        help='Apply Hadamard rotation in FP32 (default: False)')

    # Activation Quantization Arguments
    parser.add_argument('--a_bits', type=int, default=16,
                        help='''Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)''')
    parser.add_argument('--a_groupsize', type=int, default=-1,
                        help='Groupsize for activation quantization. Note that this should be the same as w_groupsize')
    parser.add_argument('--a_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric Activation quantization (default: False)')
    parser.add_argument('--a_clip_ratio', type=float, default=1.0,
                        help='Clip ratio for activation quantization. new_max = max * clip_ratio')
    parser.add_argument('--a_residual', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether use residual quant for activation quantization (default: False)')
    parser.add_argument('--act_scales_path', type=str, default=None,
                        help='Path to pre-calibrated activation scales (.pt). '
                             'When set, static quantization is used instead of dynamic.')
    parser.add_argument('--selective-dyn', type=str, default=None,
                        help='Comma-separated layer name patterns to force dynamic quantization. '
                             'E.g., "v_proj,o_proj" matches any layer whose name contains those substrings.')

    # PWL Activation Approximation Arguments
    parser.add_argument('--pwl_act', action=argparse.BooleanOptionalAction, default=False,
                        help='Replace non-linear activations (SiLU/GELU) with piecewise-linear '
                             'approximations matching Hailo hardware behavior.')
    parser.add_argument('--pwl_n_segments', type=int, default=9,
                        help='Number of PWL segments (default: 9, matching Hailo SiLU/GELU)')
    parser.add_argument('--pwl_input_bits', type=int, default=16,
                        help='Bit-width for PWL input quantization (default: 16 = no quant)')
    parser.add_argument('--pwl_output_bits', type=int, default=16,
                        help='Bit-width for PWL output quantization (default: 16 = no quant)')
    parser.add_argument('--pwl_mantissa_bits', type=int, default=10,
                        help='Slope mantissa precision bits (default: 10, Hailo HW)')
    parser.add_argument('--pwl_exp_bits', type=int, default=4,
                        help='Slope exponent bits (default: 4, Hailo HW)')
    parser.add_argument('--pwl_offset_bits', type=int, default=13,
                        help='Offset precision bits (default: 13, Hailo HW)')
    parser.add_argument('--pwl_no_hw_sim', action=argparse.BooleanOptionalAction, default=False,
                        help='Disable HW precision simulation for PWL (pure float PWL)')

    # Softmax Output Quantization
    parser.add_argument('--smq', type=int, default=0,
                        help='Softmax output quantization bit-width (0=disabled)')

    # Integer GEMM / Capped Accumulator Arguments
    parser.add_argument('--int_gemm', action=argparse.BooleanOptionalAction, default=False,
                        help='Use integer GEMM with capped accumulator instead of float matmul. '
                             'Requires symmetric activation quantization and a_bits/w_bits <= 8.')
    parser.add_argument('--acc_bits', type=int, default=32,
                        help='Accumulator bit-width for integer GEMM (e.g. 16, 20, 32). '
                             '32 means no capping. (default: 32)')
    parser.add_argument('--acc_block_k', type=int, default=32,
                        help='K-dimension block size for accumulator capping granularity. '
                             'Smaller = more frequent capping = more realistic HW simulation. (default: 32)')
    parser.add_argument('--acc_wrap', action=argparse.BooleanOptionalAction, default=False,
                        help='Use two\'s-complement wrap-around on accumulator overflow instead of '
                             'saturation (clamp). Default: False (saturation).')
    parser.add_argument('--acc_dtype', type=str, default='float',
                        help='Tier-2 accumulator dtype for integer GEMM. '
                             'Accepts int<N> (e.g. int16, int24, int32), float/fp32, half/fp16, bfloat/bf16. '
                             'Non-float32 requires static activation scales or a16. (default: float)')
    parser.add_argument('--lsb_mac_shift', type=int, default=0,
                        help='Right-shift tl.dot by N bits in LSB int16 kernel (default: 0)')
    parser.add_argument('--int_gemm_use_triton', action=argparse.BooleanOptionalAction, default=True,
                        help='Use Triton kernel for integer GEMM (default: True). '
                             'Set --no-int_gemm_use_triton for pure-PyTorch reference.')
    parser.add_argument('--t1_msb_scan', action='store_true',
                        help='Diagnostic: detect tier-1 accumulator overflow per layer (aborts on mismatch)')

    parser.add_argument('--ig_compare', action=argparse.BooleanOptionalAction, default=False,
                        help='Compare int_gemm output vs normal fake-quant GEMM per layer. '
                             'Prints relative error and uses the float path for PPL.')

    # FP32 model precision
    parser.add_argument('--fp32', action=argparse.BooleanOptionalAction, default=False,
                        help='Run model in float32 instead of float16 (isolate precision effects)')

    # Real integer quantization (bypass the 16-bit passthrough)
    parser.add_argument('--realint', action=argparse.BooleanOptionalAction, default=False,
                        help='Force real integer quantize/dequantize even at 16 bits')

    # Static vs Dynamic comparison diagnostic
    parser.add_argument('--sd_check', type=float, default=0,
                        help='Compare static vs dynamic quantization per-layer. '
                             'Warn if relative error exceeds this threshold. 0=off.')
    parser.add_argument('--sd_check_norm', type=str, default='inf',
                        choices=['1', '2', 'inf'],
                        help='Norm for sd_check relative error (default: inf/max)')

    # Stochastic quantization
    parser.add_argument('--stochastic_quant', action='store_true', default=False,
                        help='Use stochastic rounding for all activation quantizers (unbiased)')

    # Semi-int GEMM diagnostic kernel
    parser.add_argument('--semi_int_gemm', type=str, default=None,
                        help='Diagnostic GEMM with toggleable precision stages. '
                             'Bitmask string (e.g. "111111"=int_gemm, "000000"=float) '
                             'or keywords: full, none. '
                             'Bits: act_quant|int16_decomp|dot_round|acc_cap|wzp_round|t2_int')

    # Hardware-aligned activation scales
    parser.add_argument('--hw_align', action='store_true', default=False,
                        help='Convert all static activation scales to per-group '
                             '(aligned with --w_groupsize). Makes fake_quant '
                             'hardware-accurate by ensuring scales factor out of '
                             'dot products.')
    parser.add_argument('--hw_accurate', action='store_true', default=False,
                        help='Hardware-accurate activation scales: collapse all static scales '
                             'to per-tensor. Combine with --eq for per-channel down_proj '
                             '(full hardware match). Mutually exclusive with --hw_align.')

    # R4 diagnostic stats
    parser.add_argument('--r4_stats', type=str, default=None,
                        help='Collect R4 diagnostic stats on down_proj and save JSON to this path')
    parser.add_argument('--r4_stats_batches', type=int, default=0,
                        help='Limit stats collection to first N eval batches (0=all)')

    # Weight Quantization Arguments
    parser.add_argument('--w_bits', type=int, default=16,
                        help='Number of bits for weights of the Linear layers')
    parser.add_argument('--w_groupsize', type=int, default=-1,
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize')
    parser.add_argument('--w_static_groups', action=argparse.BooleanOptionalAction, default=False,
                        help='''Static Grouping for weight quantization.''')
    parser.add_argument('--w_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric weight quantization (default: False)')
    parser.add_argument('--w_rtn', action=argparse.BooleanOptionalAction, default=False,
                        help='Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ')
    parser.add_argument('--w_clip', action=argparse.BooleanOptionalAction, default=False,
                        help='''Clipping the weight quantization!
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization''')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for GPTQ.')
    parser.add_argument('--cal_dataset', type=str, default='wikitext2',
                        help='calibration data samples for GPTQ.', choices=supported_datasets)
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--act_order', action=argparse.BooleanOptionalAction, default=False,
                        help='act-order in GPTQ')
    parser.add_argument('--weights_stats', type=str, default=None,
                        help='Path to output file for weight sparsity stats. '
                             'When set, reports per-layer zero counts and effective sparsity.')
    parser.add_argument('--gptq_strength', type=float, default=1.0,
                        help='GPTQ error-propagation strength (0.0=no compensation, 1.0=full). '
                             'Values < 1.0 weaken GPTQ for bring-your-own-weights flows.')
    parser.add_argument('--gscaler', type=str, default=None,
                        help='Group scale format: M5S3, M6E4b2, M6S4l2, etc. '
                             '(default: None = FP32 scales)')
    parser.add_argument('--hwscale', type=str, default=None,
                        help='Merged (a_gscale * w_gscale) per-group scale snapped '
                             'to M<m>S<s>[b<z>|l<z>] at model-load for inference only. '
                             'Applied only on layers with per-group a+w scales '
                             '(down_proj under hw_accurate; all matching layers under '
                             'hw_align). Trailing bz sets a per-layer FP32 global '
                             'scale = 2^-z applied at T2 (default: None).')
    parser.add_argument('--adaquant', type=str, nargs='?', const='default', default=None,
                        help='Use AdaQuant instead of GPTQ. No value = defaults. '
                             'Inline params string to customise '
                             '(e.g., "lr.0.001_ep.20_optWSX_adam_cos")')

    # General Quantization Arguments
    parser.add_argument('--w_bits_down_proj', type=int, default=None,
                        help='''Use special weight quantization bit width for Down Projection!
                        Default: w_bits.''')
    parser.add_argument('--a_bits_down_proj', type=int, default=None,
                        help='''Use special activation quantization bit width for Down Projection!
                        Default: a_bits.''')
    parser.add_argument('--o_per_head', action=argparse.BooleanOptionalAction, default=False,
                        help='Per-head quantization for out-projection')
    # parser.add_argument('--int8_down_proj', action=argparse.BooleanOptionalAction, default=False,
    #                     help='''Use INT8 for Down Projection! If this set,
    #                     both weights and activations of this layer will be in INT8''')

    # KV-Cache Quantization Arguments
    parser.add_argument('--v_bits', type=int, default=16,
                        help='''Number of bits for V-cache quantization.
                        Note that quantizing the V-cache does not need any other rotation''')
    parser.add_argument('--v_groupsize', type=int, default=-1)
    parser.add_argument('--v_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric V-cache quantization')
    parser.add_argument('--v_clip_ratio', type=float, default=1.0,
                        help='Clip ratio for v-cache quantization. new_max = max * clip_ratio')

    parser.add_argument('--k_bits', type=int, default=16,
                        help='''Number of bits for K-cache quantization.
                        Note that quantizing the K-cache needs another rotation for the keys/queries''')
    parser.add_argument('--k_groupsize', type=int, default=-1)
    parser.add_argument('--k_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric K-cache quantization')
    parser.add_argument('--k_pre_rope', action=argparse.BooleanOptionalAction, default=False,
                        help='Pre-RoPE quantization for K-cache (not Supported yet!)')
    parser.add_argument('--k_clip_ratio', type=float, default=1.0,
                        help='Clip ratio for k-cache quantization. new_max = max * clip_ratio')

    # GGUF imitation (per-layer bit-width matching)
    parser.add_argument('--imitate_gguf', type=str, default=None,
                        help='Match per-layer weight bit-widths from a GGUF file. '
                             'Pass a quant scheme name (e.g. Q4_K_S, Q4_K_M, Q4_K_L) '
                             'to lookup <ModelName>-<scheme>.gguf, or an explicit .gguf path.')

    # GGUF Pre-quantized Weights
    parser.add_argument('--gguf_path', type=str, default=None,
                        help='Path to .gguf file with pre-quantized weights. '
                             'Weights are dequantized and loaded before rotations/GPTQ.')
    parser.add_argument('--quant_warnings', action='store_true', default=False,
                        help='Warn when quantization params (w_bits, w_groupsize, w_sym) '
                             'mismatch GGUF tensor quantization specs.')

    # Save/Load Quantized Model Arguments
    parser.add_argument('--load_qmodel_path', type=str, default=None,
                        help='Load the quantized model from the specified path!')
    parser.add_argument('--save_qmodel_path', type=str, default=None,
                        help='Save the quantized model to the specified path!')
    parser.add_argument('--gptq_checkpoint_path', type=str, default=None,
                        help='Auto-checkpoint after GPTQ: if path exists, load; otherwise run GPTQ and save.')

    # WandB Arguments
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--wandb_id', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default=None)

    # Experiments Arguments
    parser.add_argument('--save_name', type=str, default=None,
                        help='''The path to save experiment data,
                        including quantized models, dumped layer inputs, etc.
                        The data will be saved in experiments/[model]/save_name.
                        Default: [datetime].''')
    parser.add_argument('--log_to_console', action=argparse.BooleanOptionalAction, default=True,
                        help='Log to console')
    # Capture Layer Input/Output Arguments
    parser.add_argument('--capture_layer_io', action=argparse.BooleanOptionalAction, default=False,
                        help='Capture the input and output of the specified decoder layer and dump into a file')
    parser.add_argument('--layer_idx', type=int, default=10, help='Which decoder layer to capture')

    # Result Caching Arguments
    parser.add_argument('--cache_path', type=str, default=None,
                        help='Path to the JSON result cache file (e.g. data/cached_results/quarot_results.pb). '
                             'When set, completed eval results are saved and reused across runs.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Ignore existing cached results and re-run all evaluations.')

    # PPL Eval Arguments
    parser.add_argument("--ppl_eval", action="store_true", help="Evaluate the model PPL")
    parser.add_argument('--ppl_eval_dataset', type=str, nargs='+', default=['wikitext2', 'ptb', 'c4'],
                        help='Dataset for Evaluation (default: wikitext2)', choices=supported_datasets,)
    parser.add_argument('--ppl_eval_batch_size', type=int, default=1,
                        help='Batch-size for PPL evaluation (default:1)')

    # LM Eval Arguments
    parser.add_argument("--lm_eval", action="store_true", help="Evaluate the model on LM Eval tasks.")
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy",
                 "arc_challenge", "winogrande", "lambada_openai",
                 "social_iqa", "openbookqa", "mmlu"],  # boolq
    )
    parser.add_argument('--lm_eval_batch_size', type=str, default='32',
                        help='Batch size for evaluating with lm eval harness.')
    parser.add_argument(
        "--distribute",
        action="store_true",
        help="Distribute the model on multiple GPUs for evaluation.",
    )

    # 权重微调参数:
    parser.add_argument('--w_ft', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to fine-tune weights to adapt to quantized activations(default: False).')
    parser.add_argument('--ft_percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')

    args = parser.parse_args()

    # Mutual exclusion: at most one of {--eq, --ud_eq, --ugd_eq}
    _eq_modes = [name for name in ('eq', 'ud_eq', 'ugd_eq')
                 if getattr(args, name, False)]
    if len(_eq_modes) > 1:
        parser.error(
            f"--eq, --ud_eq, and --ugd_eq are mutually exclusive "
            f"(got: {', '.join('--' + m for m in _eq_modes)})")
    # --ugd_eq supports both PWL and non-PWL: with --pwl_act the gate factor
    # rides PWL per-channel s_out; without, it's applied via an EqActivation
    # wrapper that divides silu(gate) output by g[c] in fp.

    if args.weights_stats:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.weights_stats)), exist_ok=True)
            with open(args.weights_stats, 'a'):
                pass
        except IOError as e:
            parser.error(f'Cannot write to --weights_stats path: {args.weights_stats} ({e})')

    # Parse and validate --gscaler early so we fail fast on bad format
    from quant_utils import parse_gscaler
    args.gscaler_parsed = parse_gscaler(args.gscaler)
    # --hwscale reuses the same grammar
    args.hwscale_parsed = parse_gscaler(args.hwscale)

    if args.lm_eval:
        from lm_eval.tasks import TaskManager   # lm_eval==0.4.3
        task_manager = TaskManager()
        task_names = task_manager.match_tasks(args.tasks)
        for task in [task for task in args.tasks if task not in task_names]:
            raise ValueError(f"Invalid task: {task}")

    if args.save_name is None:
        args.save_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = os.path.basename(args.model.rstrip('/')) if os.path.isabs(args.model) else args.model
    setattr(args, 'save_path',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', model_tag, args.save_name))

    os.makedirs(args.save_path, exist_ok=True)

    config_logging(os.path.join(args.save_path, f'{args.save_name}.txt'),
                   to_console=args.log_to_console)

    # assert args.a_groupsize == args.w_groupsize, 'a_groupsize should be the same as w_groupsize!'
    assert args.k_pre_rope == False, 'Pre-RoPE quantization is not supported yet!'

    if args.int_gemm:
        assert args.a_bits <= 8, 'Integer GEMM requires activation bits <= 8'
        assert args.w_bits <= 8, 'Integer GEMM requires weight bits <= 8'
        if args.w_groupsize > 0:
            assert args.acc_block_k <= args.w_groupsize, (
                f'acc_block_k ({args.acc_block_k}) must be <= w_groupsize ({args.w_groupsize}) '
                'to avoid straddling weight group boundaries in the kernel')
        # Validate --acc_dtype
        from int_acc_gemm import parse_acc_dtype
        acc_kind, acc_type_bits, _acc_frac = parse_acc_dtype(args.acc_dtype)
        if not (acc_kind == 'float' and acc_type_bits == 32):
            # Non-float32 tier-2 requires per-token activation scales (not per-group)
            is_static = getattr(args, 'act_scales_path', None) is not None
            is_a16 = args.a_bits >= 16
            assert is_static or is_a16, (
                f'--acc_dtype {args.acc_dtype} requires static activation scales '
                '(--act_scales_path) or unquantized activations (a_bits >= 16). '
                'Per-group activation scales inside the K-loop are incompatible '
                'with non-float32 tier-2 accumulators.')

    if hasattr(args, 'acc_dtype') and not args.int_gemm:
        from int_acc_gemm import parse_acc_dtype
        acc_kind, acc_type_bits, _acc_frac = parse_acc_dtype(args.acc_dtype)
        if not (acc_kind == 'float' and acc_type_bits == 32):
            assert False, '--acc_dtype requires --int_gemm'

    if args.model == 'facebook/opt-125m' or args.model == 'facebook/opt-1.3b':
        logging.warning('Warning: OPT-125M/1.3B is only for debugging purposes!!')

    if args.wandb:
        assert args.wandb_id is not None and args.wandb_project is not None, 'WandB ID/project is not provided!'

    logging.info('Arguments: ')
    logging.info(pprint.pformat(vars(args)))
    logging.info('--' * 30)
    return args
