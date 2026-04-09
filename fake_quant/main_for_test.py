import utils
import torch
import model_utils
import data_utils
import transformers
import quant_utils
import rotation_utils
import gptq_utils
import eval_utils
import args_config_gen
import hadamard_utils
import logging
import os
from result_cache import ResultCache


def main():
    args = args_config_gen.parser_gen()

    if args.r1_path and '.pt' not in args.r1_path and '.bin' not in args.r1_path:
        args.r1_path += '/' + args.r1_path.split('/')[-1] + '.pt'

    if args.r2_path and '.pt' not in args.r2_path and '.bin' not in args.r2_path:
        args.r2_path += '/' + args.r2_path.split('/')[-1] + '.pt'

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config.update(args)

    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token)
    if getattr(args, 'fp32', False):
        model = model.float()
    model.eval()
    model.model_name = args.model.split('/')[-1]

    # --- Resolve imitate_gguf: build per-layer bit-width map ---
    if getattr(args, 'imitate_gguf', None):
        import gguf_utils
        # imitate_gguf may be a resolved path (from shell script) or shorthand
        gguf_imitate_path = args.imitate_gguf
        if not os.path.isfile(gguf_imitate_path):
            import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from experiment_config import resolve_gguf_path
            gguf_imitate_path = resolve_gguf_path(args.imitate_gguf, args.model)
        args.w_bits_map, _imit_label = gguf_utils.get_gguf_bits_map(gguf_imitate_path)
        logging.info("imitate_gguf: loaded %d layer bit-widths from %s",
                     len(args.w_bits_map), gguf_imitate_path)

    # --- Load GGUF pre-quantized weights (before rotations) ---
    if args.gguf_path:
        import gguf_utils
        gguf_utils.load_gguf_weights(
            model, args.gguf_path,
            quant_warnings=args.quant_warnings,
            w_bits=args.w_bits,
            w_groupsize=args.w_groupsize,
            w_sym=not args.w_asym,
        )

    # --- kv_ex / proj_ex overrides (must be before rotate_model) ---
    if args.kv_ex != 0:
        logging.info("kv_ex=%d: disabling R3, setting k_bits=%d", args.kv_ex, args.kv_ex)
        args.use_r3 = False
        args.k_bits = args.kv_ex

    # Expand --proj_ex into --no_r4 + --down_bits
    if args.proj_ex != 0:
        args.no_r4 = True
        if getattr(args, 'down_bits', None) is None:
            args.down_bits = args.proj_ex
        logging.info("proj_ex=%d: setting no_r4=True, down_bits=%d", args.proj_ex, args.down_bits)

    if getattr(args, 'late_rot4', False):
        logging.info("late_rot4: disabling R4 for rotation phase (will apply post-GPTQ)")
        args.use_r4 = False

    if getattr(args, 'no_r4', False):
        logging.info("no_r4: disabling R4 rotation on down_proj")
        args.use_r4 = False

    if getattr(args, 'down_bits', None) is not None:
        logging.info("down_bits=%d: overriding down_proj input bits", args.down_bits)
        args.a_bits_down_proj = args.down_bits

    # Enable softmax output quantization (replaces SDPA globally)
    if args.smq > 0:
        import smq_utils
        smq_utils.enable_smq(args.smq)
        logging.info("Enabled softmax output quantization: %d bits", args.smq)

    # Rotate the weights
    if args.fuse_norm:
        logging.info("Fuse LayerNorms")
        logging.info("Rotate the model use_r1={}, use_r2={}, use_r4={}, use_r3={}".format(
            args.use_r1, args.use_r2, args.use_r4, args.use_r3))

        rotation_utils.fuse_layer_norms(model)
        if args.use_r1 or args.use_r2 != 'none' or args.use_r4:
            rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if (args.use_r4 or getattr(args, 'late_rot4', False)) and 'down_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
            if args.use_r2 == 'online' and 'o_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = model.config.hidden_size // model.config.num_attention_heads
                qlayers[name].fp32_had = args.fp32_had
    elif args.a_bits < 16:
        logging.info("Add activation quantization: a_bits={}, a_groupsize={}, a_sym={}, a_clip_ratio={}".format(
            args.a_bits, args.a_groupsize, not (args.a_asym), args.a_clip_ratio))
        # Add Activation Wrapper to the model as the rest of the code assumes it is present
        quant_utils.add_actquant(model)

    # --- Equalization: load factors and apply to weights + online scaling ---
    if getattr(args, 'eq', False) and args.act_scales_path:
        import equalization as eq_module
        _eq_data = torch.load(args.act_scales_path, map_location='cpu',
                              weights_only=True)
        if '__eq_factors__' in _eq_data:
            _eq_factors = _eq_data['__eq_factors__']
            logging.info("Applying equalization factors for %d layers",
                         len(_eq_factors))
            eq_module.apply_eq_to_weights(model, _eq_factors)
            eq_module.setup_eq_online(model, _eq_factors)
        else:
            logging.warning("--eq enabled but no eq_factors found in %s",
                            args.act_scales_path)

    # Replace activations with PWL approximation (before GPTQ so Hessians see PWL)
    if args.pwl_act:
        import pwl_utils
        act_name = getattr(model.config, 'hidden_act', 'silu')
        hw_config = None if args.pwl_no_hw_sim else pwl_utils.HWConfig(
            mantissa_bits=args.pwl_mantissa_bits,
            exp_bits=args.pwl_exp_bits,
            offset_bits=args.pwl_offset_bits)
        replaced = pwl_utils.replace_activation_with_pwl(
            model, act_name=act_name,
            n_segments=args.pwl_n_segments,
            hw_config=hw_config,
            input_bits=args.pwl_input_bits,
            output_bits=args.pwl_output_bits)
        logging.info("Replaced %d activations with PWL (%s, %d segments, hw_sim=%s)",
                     len(replaced), act_name, args.pwl_n_segments, hw_config is not None)
        if getattr(args, 'realint', False):
            for pwl_mod in pwl_utils.find_pwl_activations(model).values():
                pwl_mod.input_quantizer.realint = True
                pwl_mod.output_quantizer.realint = True

    _has_w_bits_map = bool(getattr(args, 'w_bits_map', None))
    if args.w_bits < 16 or _has_w_bits_map:
        logging.info("Add weight quantization: w_rtn = {}, w_bits = {}, w_groupsize = {}, w_sym = {}, w_clip = {}, w_bits_map = {}".format(
            args.w_rtn, args.w_bits, args.w_groupsize, not (args.w_asym), args.w_clip, _has_w_bits_map))

        save_dict = {}

        # Resolve GPTQ checkpoint path
        _gptq_ckpt = None
        if args.gptq_checkpoint_path:
            w_suffix = 'w0' if _has_w_bits_map else f'w{args.w_bits}'
            _gptq_ckpt = os.path.join(
                args.gptq_checkpoint_path,
                f'{model.model_name}_{w_suffix}'
            )

        # Snapshot weights for GPTQ strength blending (strength 0→RTN, 1→full GPTQ)
        gptq_strength = getattr(args, 'gptq_strength', 1.0)
        orig_weights = {}
        if 0.0 < gptq_strength < 1.0:
            for pname, param in model.named_parameters():
                if pname.startswith('model.layers.') and 'weight' in pname and param.dim() == 2:
                    orig_weights[pname] = param.data.clone()

        if args.load_qmodel_path:  # Load Quantized Rotated Model
            # assert args.fuse_norm, "Model should be fused to load a quantized model!"
            assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
            logging.info("Load quantized model from: {}.".format(args.load_qmodel_path))
            utils.load_model_in_parts(model, args.load_qmodel_path)
            # save_dict = torch.load(args.load_qmodel_path, map_location='cpu')
            # model.load_state_dict(save_dict["model"])

        elif getattr(args, 'adaquant', None) is not None:  # AdaQuant Weight Quantization
            import adaquant_utils
            _adaquant_ckpt = None
            if args.gptq_checkpoint_path:
                # Use parallel directory for AdaQuant caches
                w_suffix = 'w0' if _has_w_bits_map else f'w{args.w_bits}'
                _adaquant_ckpt = os.path.join(
                    args.gptq_checkpoint_path.replace('gptq_checkpoints', 'adaquant_checkpoints'),
                    f'{model.model_name}_{w_suffix}'
                )

            if _adaquant_ckpt and os.path.isdir(_adaquant_ckpt) and any(
                    f.endswith('.pth') for f in os.listdir(_adaquant_ckpt)):
                logging.info("Loading AdaQuant checkpoint from: {}".format(_adaquant_ckpt))
                utils.load_model_in_parts(model, _adaquant_ckpt)
                if args.w_asym and args.int_gemm:
                    from int_acc_gemm import load_gptq_w_params
                    load_gptq_w_params(model, _adaquant_ckpt)
                # Load optimised activation scales if present
                _xscales_path = os.path.join(_adaquant_ckpt, '_adaquant_x_scales.pt')
                if os.path.isfile(_xscales_path):
                    logging.info("Loading AdaQuant x_scales from: %s", _xscales_path)
                logging.info("AdaQuant checkpoint loaded – skipping quantization.")
            else:
                assert "llama" in args.model, "Only llama is supported for AdaQuant!"
                aq_params = adaquant_utils.AdaQuantParams.from_string(args.adaquant)
                aq_nsamples = aq_params.nsamples if aq_params.nsamples is not None else args.nsamples
                trainloader = data_utils.get_loaders(
                    args.cal_dataset, nsamples=aq_nsamples,
                    seed=args.seed, model=args.model,
                    seqlen=model.seqlen, eval_mode=False
                )
                quantizers, x_scales = adaquant_utils.adaquant_fwrd(
                    model, trainloader, utils.DEV, args, aq_params)
                save_dict["w_quantizers"] = quantizers

                # Auto-save AdaQuant checkpoint
                if _adaquant_ckpt:
                    if os.path.isdir(_adaquant_ckpt) and any(
                            f.endswith('.pth') for f in os.listdir(_adaquant_ckpt)):
                        logging.info("AdaQuant checkpoint already exists (written by another run) – skipping save: %s", _adaquant_ckpt)
                    else:
                        os.makedirs(_adaquant_ckpt, exist_ok=True)
                        logging.info("Saving AdaQuant checkpoint to: {}".format(_adaquant_ckpt))
                        utils.save_model_in_parts(model, _adaquant_ckpt,
                                                  prefix=f'{model.model_name}_part')
                        if x_scales:
                            torch.save(x_scales, os.path.join(_adaquant_ckpt, '_adaquant_x_scales.pt'))
                        if args.w_asym and args.int_gemm:
                            from int_acc_gemm import save_gptq_w_params
                            save_gptq_w_params(model, _adaquant_ckpt)

        elif gptq_strength > 0.0 and _gptq_ckpt and os.path.isdir(_gptq_ckpt) and any(
                f.endswith('.pth') for f in os.listdir(_gptq_ckpt)):
            logging.info("Loading GPTQ checkpoint from: {}".format(_gptq_ckpt))
            utils.load_model_in_parts(model, _gptq_ckpt)
            # Load per-group GPTQ scale/zero for w_asym + int_gemm
            if args.w_asym and args.int_gemm:
                from int_acc_gemm import load_gptq_w_params
                load_gptq_w_params(model, _gptq_ckpt)
            logging.info("GPTQ checkpoint loaded – skipping quantization.")

            # late_rot4: apply R4 rotation to already-quantized down_proj weights
            if getattr(args, 'late_rot4', False):
                logging.info("late_rot4: applying R4 rotation to GPTQ-quantized down_proj weights")
                for _name, _module in model.named_modules():
                    if 'down_proj' in _name and isinstance(_module, torch.nn.Linear):
                        hadamard_utils.apply_exact_had_to_linear(_module, had_dim=-1, output=False)
                logging.info("late_rot4: R4 rotation applied post-quantization")

        elif gptq_strength > 0.0 and not args.w_rtn:  # GPTQ Weight Quantization
            assert "llama" in args.model, "Only llama is supported for GPTQ!"

            trainloader = data_utils.get_loaders(
                args.cal_dataset, nsamples=args.nsamples,
                seed=args.seed, model=args.model,
                seqlen=model.seqlen, eval_mode=False
            )
            # 精度补偿：
            if args.w_ft:
                w_fine_tuning.w_ft(model, trainloader, utils.DEV, args)
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers

            # Auto-save GPTQ checkpoint
            if _gptq_ckpt:
                # Guard against parallel runs with the same tag: if another
                # process already wrote the checkpoint while we were running,
                # skip saving to avoid partial-overwrite collisions.
                if os.path.isdir(_gptq_ckpt) and any(
                        f.endswith('.pth') for f in os.listdir(_gptq_ckpt)):
                    logging.info("GPTQ checkpoint already exists (written by another run) – skipping save: %s", _gptq_ckpt)
                else:
                    os.makedirs(_gptq_ckpt, exist_ok=True)
                    logging.info("Saving GPTQ checkpoint to: {}".format(_gptq_ckpt))
                    utils.save_model_in_parts(model, _gptq_ckpt,
                                              prefix=f'{model.model_name}_part')
                    # Save per-group GPTQ scale/zero for w_asym + int_gemm
                    if args.w_asym and args.int_gemm:
                        from int_acc_gemm import save_gptq_w_params
                        save_gptq_w_params(model, _gptq_ckpt)

        else:  # RTN Weight Quantization (also used when gptq_strength=0.0)

            if args.w_ft:  # 精度补偿：
                trainloader = data_utils.get_loaders(
                    args.cal_dataset, nsamples=args.nsamples,
                    seed=args.seed, model=args.model,
                    seqlen=model.seqlen, eval_mode=False
                )
                w_fine_tuning.w_ft(model, trainloader, utils.DEV, args)
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers

        # Apply GPTQ strength blending: interpolate original ↔ GPTQ, then RTN re-quantize
        if orig_weights:
            logging.info("Applying GPTQ strength %.2f (blending + stochastic re-quantize)", gptq_strength)
            for pname, param in model.named_parameters():
                if pname in orig_weights:
                    w_orig = orig_weights[pname].float()
                    w_gptq = param.data.float()
                    param.data = (w_orig + gptq_strength * (w_gptq - w_orig)).to(param.data.dtype)
            del orig_weights
            # Snap blended weights back to the quantization grid (stochastic to avoid bias)
            gptq_utils.rtn_fwrd(model, utils.DEV, args, stochastic=True)

        if args.save_qmodel_path:
            folder_name = f'{model.model_name}'
            folder_name += f'_w{args.w_bits}'
            folder_name += '_r1' if args.use_r1 else ''
            folder_name += '_r2' if args.use_r2 != 'none' else ''
            folder_name += '' if args.r2_path else '_r'
            folder_name += '_r3' if args.use_r3 else ''
            folder_name += '_r4' if args.use_r4 else ''
            folder_name += '_rtn' if args.w_rtn else '_gptq'
            folder_name += '_clip' if args.w_clip else ''
            folder_name += f'_g{args.w_groupsize}' if args.w_groupsize > 0 else ''
            folder_name += '_asym' if args.w_asym else ''
            folder_name += '_smooth' if args.smooth else ''
            folder_name += '_ft' if args.w_ft else ''
            args.save_qmodel_path = os.path.join(args.save_qmodel_path, folder_name)
            if not os.path.exists(args.save_qmodel_path):
                os.makedirs(args.save_qmodel_path)
            logging.info("Save quantized model to: {}.".format(args.save_qmodel_path))
            utils.save_model_in_parts(model, args.save_qmodel_path, prefix=f'{model.model_name}_part')
            # save_dict["model"] = model.state_dict()
            # torch.save(save_dict, args.save_qmodel_path)

    # --- Weight sparsity stats ---
    if args.weights_stats and args.w_bits < 16:
        import json
        qlayers_ws = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        stats_rows = []
        for name, qlayer in qlayers_ws.items():
            w = qlayer.module.weight.data
            N, K = w.shape
            total = w.numel()
            n_zeros = (w == 0).sum().item()
            pct_zero = n_zeros / total if total > 0 else 0.0
            ops = 2 * N * K
            stats_rows.append({
                'layer': name,
                'shape': [N, K],
                'total': total,
                'zeros': n_zeros,
                'pct_zero': pct_zero,
                'ops': ops,
            })

        total_ops = sum(r['ops'] for r in stats_rows)
        effective_sparsity = (
            sum(r['ops'] * r['pct_zero'] for r in stats_rows) / total_ops
            if total_ops > 0 else 0.0
        )

        with open(args.weights_stats, 'w') as f:
            f.write(f"{'Layer':<60} {'Shape':>14} {'Total':>10} {'Zeros':>10} {'%Zero':>8} {'OPs':>14}\n")
            f.write('-' * 120 + '\n')
            for r in stats_rows:
                shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
                f.write(f"{r['layer']:<60} {shape_str:>14} {r['total']:>10} {r['zeros']:>10} {r['pct_zero']:>8.4f} {r['ops']:>14}\n")
            f.write('-' * 120 + '\n')
            f.write(f"Effective sparsity (ops-weighted): {effective_sparsity:.6f}\n")
            f.write(f"Total OPs: {total_ops}\n")
            f.write('\n--- JSON ---\n')
            json.dump({
                'layers': stats_rows,
                'effective_sparsity': effective_sparsity,
                'total_ops': total_ops,
            }, f, indent=2)
            f.write('\n')

        logging.info("Weight stats written to %s (effective sparsity: %.6f)",
                     args.weights_stats, effective_sparsity)

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16 or getattr(args, 'realint', False):
        logging.info("Add v quantization: v_bits={}, v_groupsize={}, v_sym={}, v_clip_ratio={}".format(
            args.v_bits, args.v_groupsize, not (args.v_asym), args.v_clip_ratio))

        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0 and "llama" in args.model:
            down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio
            residual = args.a_residual

            if 'v_proj' in name and (args.v_bits < 16 or getattr(args, 'realint', False)):  # Set the v_proj precision
                qlayers[name].out_quantizer.configure(bits=args.v_bits,
                                                      groupsize=args.v_groupsize,
                                                      sym=not (args.v_asym),
                                                      clip_ratio=args.v_clip_ratio)

            if 'lm_head' in name:  # Skip lm_head quantization
                layer_input_bits = 16

            if args.o_per_head and 'o_proj' in name:  # Set the o_proj precision
                num_heads = model.config.num_attention_heads
                model_dim = model.config.hidden_size
                layer_groupsize = model_dim // num_heads

            if 'down_proj' in name:  # Set the down_proj precision
                if args.a_bits_down_proj is not None:
                    layer_input_bits = args.a_bits_down_proj
                layer_groupsize = down_proj_groupsize

            qlayers[name].quantizer.configure(bits=layer_input_bits,
                                              groupsize=layer_groupsize,
                                              sym=layer_a_sym,
                                              clip_ratio=layer_a_clip,
                                              residual=residual)

            if getattr(args, 'realint', False):
                qlayers[name].quantizer.realint = True
                # Only set realint on out_quantizer if it was already configured (bits < 16)
                if qlayers[name].out_quantizer.maxq != 0:
                    qlayers[name].out_quantizer.realint = True

            # Configure output quantization (--quant_out)
            quant_out = getattr(args, 'quant_out', 'none')
            if quant_out != 'none' and quant_utils.should_quant_out(name, quant_out):
                # Don't override existing out_quantizer (e.g. v_proj V-cache)
                if qlayers[name].out_quantizer.maxq == 0:
                    qlayers[name].out_quantizer.configure(bits=16, groupsize=-1, sym=True, clip_ratio=1.0)
                    qlayers[name].out_quantizer.realint = True  # Force actual quant at 16 bits
            if quant_out != 'none' and quant_utils.should_quant_pre(name, quant_out):
                qlayers[name].pre_quantizer.configure(bits=16, groupsize=-1, sym=True, clip_ratio=1.0)
                qlayers[name].pre_quantizer.realint = True

    # Link quantizer→quantizer chains (no-op for Llama, future-ready)
    if getattr(args, 'quant_out', 'none') != 'none':
        quant_utils.link_adjacent_quantizers(model)

    # Setup residual quantizers (--quant_out res/ex)
    quant_out = getattr(args, 'quant_out', 'none')
    if quant_utils.needs_residual_quant(quant_out):
        quant_utils.setup_residual_quantizers(model)

    # Setup Q quantizer in attention (--quant_out mm/ex)
    if quant_utils.needs_mm_quant(quant_out):
        layers = model_utils.get_layers(model)
        rope_fn = model_utils.get_rope_function_name(model)
        for layer in layers:
            wrapper_attr = f'{rope_fn}_qk_rotation_wrapper'
            if hasattr(layer.self_attn, wrapper_attr):
                wrapper = getattr(layer.self_attn, wrapper_attr)
                wrapper.q_quantizer.configure(bits=16, groupsize=-1, sym=True, clip_ratio=1.0)
                wrapper.q_quantizer.realint = True

    # --- Stochastic quantization ---
    if getattr(args, 'stochastic_quant', False):
        logging.info("Enabling stochastic rounding on all activation quantizers")
        for name in qlayers:
            qlayers[name].quantizer.stochastic = True
            qlayers[name].out_quantizer.stochastic = True
            qlayers[name].pre_quantizer.stochastic = True

    # --- Prepare integer GEMM with capped accumulator ---
    if args.int_gemm:
        logging.info("Preparing integer GEMM: acc_bits=%d, acc_block_k=%d, use_triton=%s, acc_wrap=%s",
                     args.acc_bits, args.acc_block_k, args.int_gemm_use_triton, args.acc_wrap)
        qlayers_ig = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        _ig_w_bits_map = getattr(args, 'w_bits_map', None)
        n_int_gemm = 0
        for name, qlayer in qlayers_ig.items():
            if 'lm_head' in name:
                continue
            if getattr(args, 'late_rot4', False) and 'down_proj' in name:
                logging.warning("late_rot4: skipping int_gemm for %s (post-rotation breaks integer grid)", name)
                qlayer.use_int_gemm = False
                continue
            if qlayer.quantizer.bits > 16:
                # Too wide for int GEMM — use fake-quant path
                qlayer.use_int_gemm = False
                continue
            if getattr(qlayer.quantizer, 'groupsize', -1) > 0:
                # Grouped activation quantization (e.g. o_per_head) is
                # incompatible with int GEMM which needs per-token scales.
                qlayer.use_int_gemm = False
                logging.info("  skipping int_gemm for %s (act groupsize=%d, need per-token)",
                             name, qlayer.quantizer.groupsize)
                continue
            # Resolve per-layer w_bits (from --imitate_gguf bit-width map
            # and --w_bits_down_proj), mirroring GPTQ's logic.
            layer_w_bits = args.w_bits
            if _ig_w_bits_map:
                layer_w_bits = _ig_w_bits_map.get(name, layer_w_bits)
            if getattr(args, 'w_bits_down_proj', None) is not None and 'down_proj' in name:
                layer_w_bits = args.w_bits_down_proj
            qlayer.prepare_int_gemm(
                w_bits=layer_w_bits,
                w_sym=not args.w_asym,
                w_group_size=args.w_groupsize,
                acc_bits=args.acc_bits,
                acc_block_k=args.acc_block_k,
                use_triton=args.int_gemm_use_triton,
                acc_wrap=args.acc_wrap,
                acc_dtype=getattr(args, 'acc_dtype', 'float'),
                gscaler_parsed=getattr(args, 'gscaler_parsed', None),
            )
            n_int_gemm += 1
        logging.info("Integer GEMM prepared for %d layers", n_int_gemm)

        if getattr(args, 'ig_compare', False):
            for name, qlayer in qlayers.items():
                if qlayer.use_int_gemm:
                    qlayer._ig_compare = True
                    qlayer.quantizer._sd_name = name
                    qlayer.quantizer._sd_norm = float('inf')
                    qlayer.quantizer._sd_check = 0  # don't trigger sd_check, ig_compare handles it
                    qlayer.quantizer._sd_logged_first = False
            logging.info("ig_compare enabled: will compare int_gemm vs float GEMM per layer")

        if getattr(args, 'semi_int_gemm', None):
            from semi_int_gemm import parse_mask, describe_mask
            _semi_mask = parse_mask(args.semi_int_gemm)
            logging.info("Semi-int GEMM enabled: mask=%s (%s)", _semi_mask, describe_mask(_semi_mask))
            for name, qlayer in qlayers_ig.items():
                if qlayer.use_int_gemm:
                    qlayer.use_int_gemm = False
                    qlayer.use_semi_int_gemm = True
                    qlayer.semi_int_mask = _semi_mask

    if args.k_bits < 16 or getattr(args, 'realint', False):
        logging.info("Add k quantization: k_bits={}, k_groupsize={}, k_sym={}, k_clip_ratio={}".format(
            args.k_bits, args.k_groupsize, not (args.k_asym), args.k_clip_ratio))

        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = model_utils.get_rope_function_name(model)
            layers = model_utils.get_layers(model)
            k_quant_config = {'k_bits': args.k_bits, "k_groupsize": args.k_groupsize,
                              "k_sym": not (args.k_asym), "k_clip_ratio": args.k_clip_ratio,
                              'use_r3': args.use_r3}
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn,
                    rope_function_name,
                    config=model.config,
                    **k_quant_config)
            if getattr(args, 'realint', False):
                for layer in layers:
                    wrapper_attr = f'{rope_function_name}_qk_rotation_wrapper'
                    if hasattr(layer.self_attn, wrapper_attr):
                        getattr(layer.self_attn, wrapper_attr).k_quantizer.realint = True

    # --- Override calibration path when semi_int_gemm bit 8 (fq_cal) is set ---
    if getattr(args, 'semi_int_gemm', None) and args.act_scales_path:
        from semi_int_gemm import parse_mask as _parse_semi_mask, MASK_FQ_CAL, _bit as _semi_bit
        _semi_mask_check = _parse_semi_mask(args.semi_int_gemm)
        if _semi_bit(_semi_mask_check, MASK_FQ_CAL):
            # Strip _intgemm... segment from the calibration filename to get
            # the fake-quant calibration path.  The int_gemm segment matches:
            #   _intgemm[_acc\d+][_bk\d+][_wrap][_t2...]
            # and sits between the base tag and subsequent suffixes like _v35, _RINT, etc.
            import re
            _orig_cal = args.act_scales_path
            _fqcal_path = re.sub(r'_intgemm(?:_acc\d+)?(?:_bk\d+)?(?:_wrap)?(?:_t2[A-Za-z0-9]+)?', '', _orig_cal)
            if _fqcal_path != _orig_cal and os.path.isfile(_fqcal_path):
                logging.warning("fq_cal: overriding calibration from int_gemm to fake_quant")
                logging.warning("  int_gemm cal: %s", _orig_cal)
                logging.warning("  fake_quant cal: %s", _fqcal_path)
                # Compare keys between the two calibration files
                _ig_scales = torch.load(_orig_cal, map_location='cpu', weights_only=True)
                _fq_scales = torch.load(_fqcal_path, map_location='cpu', weights_only=True)
                _ig_keys = set(_ig_scales.keys())
                _fq_keys = set(_fq_scales.keys())
                if _ig_keys != _fq_keys:
                    _only_ig = _ig_keys - _fq_keys
                    _only_fq = _fq_keys - _ig_keys
                    logging.warning("  KEY MISMATCH: only in int_gemm cal: %s", _only_ig or '(none)')
                    logging.warning("  KEY MISMATCH: only in fake_quant cal: %s", _only_fq or '(none)')
                else:
                    logging.info("  calibration keys match (%d entries)", len(_fq_keys))
                del _ig_scales
                args.act_scales_path = _fqcal_path
            elif _fqcal_path == _orig_cal:
                logging.warning("fq_cal: no _intgemm segment found in cal path, using as-is: %s", _orig_cal)
            else:
                logging.error("fq_cal: fake_quant calibration not found: %s", _fqcal_path)
                logging.error("  falling back to int_gemm calibration: %s", _orig_cal)

    # Load pre-calibrated static activation scales
    if args.act_scales_path:
        logging.info("Loading static activation scales from: {}".format(args.act_scales_path))
        act_scales = torch.load(args.act_scales_path, map_location='cpu', weights_only=True)

        # Apply to ActQuantWrapper quantizers (input + output/v_proj)
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        for name, qlayer in qlayers.items():
            q_key = f'{name}.quantizer'
            if q_key in act_scales and (qlayer.quantizer.bits < 16 or qlayer.quantizer.realint):
                qlayer.quantizer.scale = act_scales[q_key]['scale']
                qlayer.quantizer.zero = act_scales[q_key]['zero']
                qlayer.quantizer.static = True

            oq_key = f'{name}.out_quantizer'
            if oq_key in act_scales and (qlayer.out_quantizer.bits < 16 or qlayer.out_quantizer.realint):
                qlayer.out_quantizer.scale = act_scales[oq_key]['scale']
                qlayer.out_quantizer.zero = act_scales[oq_key]['zero']
                qlayer.out_quantizer.static = True

            pq_key = f'{name}.pre_quantizer'
            if pq_key in act_scales and (qlayer.pre_quantizer.bits < 16 or qlayer.pre_quantizer.realint):
                qlayer.pre_quantizer.scale = act_scales[pq_key]['scale']
                qlayer.pre_quantizer.zero = act_scales[pq_key]['zero']
                qlayer.pre_quantizer.static = True

        # Apply to QKRotationWrapper k_quantizers
        layers = model_utils.get_layers(model)
        for i, layer in enumerate(layers):
            rope_fn = model_utils.get_rope_function_name(model)
            wrapper_attr = f'{rope_fn}_qk_rotation_wrapper'
            if hasattr(layer.self_attn, wrapper_attr):
                wrapper = getattr(layer.self_attn, wrapper_attr)
                kq_key = f'layer.{i}.k_quantizer'
                if kq_key in act_scales and (wrapper.k_quantizer.bits < 16 or wrapper.k_quantizer.realint):
                    wrapper.k_quantizer.scale = act_scales[kq_key]['scale']
                    wrapper.k_quantizer.zero = act_scales[kq_key]['zero']
                    wrapper.k_quantizer.static = True
                qq_key = f'layer.{i}.q_quantizer'
                if qq_key in act_scales and (wrapper.q_quantizer.bits < 16 or wrapper.q_quantizer.realint):
                    wrapper.q_quantizer.scale = act_scales[qq_key]['scale']
                    wrapper.q_quantizer.zero = act_scales[qq_key]['zero']
                    wrapper.q_quantizer.static = True

        # Apply to residual quantizers
        for i, layer in enumerate(layers):
            for tag in ('_attn_res_quantizer', '_mlp_res_quantizer'):
                rq = getattr(layer, tag, None)
                if rq is not None:
                    rq_key = f'layer.{i}.{tag}'
                    if rq_key in act_scales and (rq.bits < 16 or rq.realint):
                        rq.scale = act_scales[rq_key]['scale']
                        rq.zero = act_scales[rq_key]['zero']
                        rq.static = True

        # Apply to PWLActivation quantizers (if PWL is enabled)
        if args.pwl_act:
            import pwl_utils
            pwl_modules = pwl_utils.find_pwl_activations(model)
            for name, pwl_mod in pwl_modules.items():
                iq_key = f'{name}.input_quantizer'
                if iq_key in act_scales and (pwl_mod.input_quantizer.bits < 16 or pwl_mod.input_quantizer.realint):
                    pwl_mod.input_quantizer.scale = act_scales[iq_key]['scale']
                    pwl_mod.input_quantizer.zero = act_scales[iq_key]['zero']
                    pwl_mod.input_quantizer.static = True
                oq_key = f'{name}.output_quantizer'
                if oq_key in act_scales and (pwl_mod.output_quantizer.bits < 16 or pwl_mod.output_quantizer.realint):
                    pwl_mod.output_quantizer.scale = act_scales[oq_key]['scale']
                    pwl_mod.output_quantizer.zero = act_scales[oq_key]['zero']
                    pwl_mod.output_quantizer.static = True

        logging.info("Static activation scales applied to all quantizers.")

    # --- Hardware-aligned activation scales: convert ALL static quantizers to per-group ---
    if getattr(args, 'hw_align', False) and args.act_scales_path:
        _hw_G = args.w_groupsize if args.w_groupsize > 0 else 128
        n_hw_aligned = 0

        def _try_align(q, label):
            """Align a single quantizer's scales if it's static and active."""
            nonlocal n_hw_aligned
            if not q.static or (q.bits >= 16 and not q.realint):
                return
            s, z = quant_utils.align_scales_to_groups(
                q.scale, q.zero, q.maxq, _hw_G, q.sym)
            q.scale = s
            q.zero = z
            n_hw_aligned += 1

        # 1. ActQuantWrapper quantizers (input, output, pre-R4)
        qlayers_hw = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        for name, qlayer in qlayers_hw.items():
            _try_align(qlayer.quantizer, f'{name}.quantizer')
            _try_align(qlayer.out_quantizer, f'{name}.out_quantizer')
            _try_align(qlayer.pre_quantizer, f'{name}.pre_quantizer')

        # 2. QKRotationWrapper quantizers (K-cache, Q)
        layers_hw = model_utils.get_layers(model)
        for i, layer in enumerate(layers_hw):
            rope_fn = model_utils.get_rope_function_name(model)
            wrapper_attr = f'{rope_fn}_qk_rotation_wrapper'
            if hasattr(layer.self_attn, wrapper_attr):
                wrapper = getattr(layer.self_attn, wrapper_attr)
                _try_align(wrapper.k_quantizer, f'layer.{i}.k_quantizer')
                if hasattr(wrapper, 'q_quantizer'):
                    _try_align(wrapper.q_quantizer, f'layer.{i}.q_quantizer')

        # 3. Residual quantizers
        for i, layer in enumerate(layers_hw):
            for tag in ('_attn_res_quantizer', '_mlp_res_quantizer'):
                rq = getattr(layer, tag, None)
                if rq is not None:
                    _try_align(rq, f'layer.{i}.{tag}')

        # 4. PWL quantizers
        if getattr(args, 'pwl_act', False):
            import pwl_utils
            for name, pwl_mod in pwl_utils.find_pwl_activations(model).items():
                _try_align(pwl_mod.input_quantizer, f'{name}.input_quantizer')
                _try_align(pwl_mod.output_quantizer, f'{name}.output_quantizer')

        logging.info("hw_align: converted %d static quantizers to per-group scales "
                     "(group_size=%d)", n_hw_aligned, _hw_G)

    # Convert per-column static scales to per-group for int_gemm layers.
    # Per-column scales can't factor out of a dot product.  Per-group scales
    # (one per acc_block_k columns) can — each K-block gets its own scale.
    # Applies to float32 and integer tier-2 accumulators.  Skipped for fp16/bf16
    # tier-2 where per-token scales are needed (applied after the accumulator).
    from int_acc_gemm import parse_acc_dtype
    _acc_kind, _acc_type_bits, _acc_frac_bits = parse_acc_dtype(getattr(args, 'acc_dtype', 'float'))
    _acc_dtype_is_fp32 = (_acc_kind == 'float' and _acc_type_bits == 32)
    _acc_dtype_is_int = (_acc_kind == 'int')
    _acc_dtype_needs_pergroup = _acc_dtype_is_fp32 or _acc_dtype_is_int
    if args.int_gemm and args.act_scales_path and _acc_dtype_needs_pergroup:
        G = args.acc_block_k
        qlayers_pg = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        n_converted = 0
        for name, qlayer in qlayers_pg.items():
            if not (getattr(qlayer, 'use_int_gemm', False) or getattr(qlayer, 'use_semi_int_gemm', False)):
                continue
            q = qlayer.quantizer
            if not q.static:
                continue
            col_scale = q.scale.flatten()  # [K] from calibration
            col_zero = q.zero.flatten()   # [K] from calibration
            K = col_scale.shape[0]
            assert K % G == 0, (
                f"K={K} not divisible by acc_block_k={G} for {name}")
            n_groups = K // G
            maxq = q.maxq

            if q.sym:
                # Symmetric: group scale = max of column scales within group
                q.scale = col_scale.reshape(n_groups, G).max(dim=1)[0]
                q.zero = torch.zeros(n_groups)
            else:
                # Asymmetric: derive per-group (scale, zero) from representable ranges
                col_min = -(col_zero * col_scale)            # [K]
                col_max = (maxq - col_zero) * col_scale      # [K]
                group_min = col_min.reshape(n_groups, G).min(dim=1)[0]
                group_max = col_max.reshape(n_groups, G).max(dim=1)[0]
                group_scale = (group_max - group_min) / maxq
                group_zero = torch.round(-group_min / group_scale)
                dead = (group_min == 0) & (group_max == 0)
                group_scale[dead] = 1.0
                group_zero[dead] = 0.0
                q.scale = group_scale
                q.zero = group_zero

            q.groupsize = G
            n_converted += 1
        if n_converted:
            logging.info("Converted %d int_gemm/semi_int_gemm quantizers to per-group scales "
                         "(group_size=%d)", n_converted, G)
            # Precompute static zero-point correction for asymmetric per-group mode
            for name, qlayer in qlayers_pg.items():
                if (getattr(qlayer, 'use_int_gemm', False) or getattr(qlayer, 'use_semi_int_gemm', False)) and qlayer.quantizer.static:
                    qlayer.compute_static_zp_bias()

    # --- Selective dynamic: revert matching layers from static to dynamic ---
    if args.act_scales_path and getattr(args, 'selective_dyn', None):
        patterns = [p.strip() for p in args.selective_dyn.split(',')]
        n_reverted = 0

        # ActQuantWrapper quantizers
        qlayers_sd = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        for name, qlayer in qlayers_sd.items():
            if any(p in name for p in patterns):
                if qlayer.quantizer.static:
                    qlayer.quantizer.static = False
                    qlayer.quantizer.groupsize = -1  # undo per-group conversion
                    logging.info("  selective-dyn: %s.quantizer -> dynamic", name)
                    n_reverted += 1
                if qlayer.out_quantizer.static:
                    qlayer.out_quantizer.static = False
                    logging.info("  selective-dyn: %s.out_quantizer -> dynamic", name)
                    n_reverted += 1

        # QKRotationWrapper k_quantizers
        layers_sd = model_utils.get_layers(model)
        for i, layer in enumerate(layers_sd):
            rope_fn = model_utils.get_rope_function_name(model)
            wrapper_attr = f'{rope_fn}_qk_rotation_wrapper'
            if hasattr(layer.self_attn, wrapper_attr):
                wrapper = getattr(layer.self_attn, wrapper_attr)
                kq_name = f'layer.{i}.k_quantizer'
                if any(p in kq_name for p in patterns) and wrapper.k_quantizer.static:
                    wrapper.k_quantizer.static = False
                    logging.info("  selective-dyn: %s -> dynamic", kq_name)
                    n_reverted += 1

        # PWL quantizers
        if args.pwl_act:
            import pwl_utils
            for name, pwl_mod in pwl_utils.find_pwl_activations(model).items():
                if any(p in name for p in patterns):
                    if pwl_mod.input_quantizer.static:
                        pwl_mod.input_quantizer.static = False
                        logging.info("  selective-dyn: %s.input_quantizer -> dynamic", name)
                        n_reverted += 1
                    if pwl_mod.output_quantizer.static:
                        pwl_mod.output_quantizer.static = False
                        logging.info("  selective-dyn: %s.output_quantizer -> dynamic", name)
                        n_reverted += 1

        logging.info("selective-dyn: reverted %d quantizers to dynamic (patterns: %s)",
                     n_reverted, patterns)

    # Configure sd_check on all static quantizers
    if getattr(args, 'sd_check', 0) > 0:
        _norm_map = {'1': 1, '2': 2, 'inf': float('inf')}
        _sd_norm = _norm_map[args.sd_check_norm]
        _sd_thr = args.sd_check

        def _setup_sd(quantizer, name):
            quantizer._sd_check = _sd_thr
            quantizer._sd_norm = _sd_norm
            quantizer._sd_name = name

        qlayers_sd = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        for name, qlayer in qlayers_sd.items():
            if qlayer.quantizer.static:
                _setup_sd(qlayer.quantizer, f'{name}.quantizer')
            if qlayer.out_quantizer.static:
                _setup_sd(qlayer.out_quantizer, f'{name}.out_quantizer')

        layers_sd = model_utils.get_layers(model)
        for i, layer in enumerate(layers_sd):
            rope_fn = model_utils.get_rope_function_name(model)
            wrapper_attr = f'{rope_fn}_qk_rotation_wrapper'
            if hasattr(layer.self_attn, wrapper_attr):
                wrapper = getattr(layer.self_attn, wrapper_attr)
                if wrapper.k_quantizer.static:
                    _setup_sd(wrapper.k_quantizer, f'layer.{i}.k_quantizer')

        if args.pwl_act:
            import pwl_utils
            pwl_modules = pwl_utils.find_pwl_activations(model)
            for name, pwl_mod in pwl_modules.items():
                if pwl_mod.input_quantizer.static:
                    _setup_sd(pwl_mod.input_quantizer, f'{name}.input_quantizer')
                if pwl_mod.output_quantizer.static:
                    _setup_sd(pwl_mod.output_quantizer, f'{name}.output_quantizer')

        logging.info("sd_check enabled: threshold=%.4e, norm=%s", _sd_thr, args.sd_check_norm)

    if args.distribute:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)

    # ---------- R4 stats setup ----------
    r4_collectors = None
    if getattr(args, 'r4_stats', None):
        import r4_stats
        r4_collectors = r4_stats.setup_r4_stats(
            model, max_batches=getattr(args, 'r4_stats_batches', 0))

    # ---------- Result cache ----------
    cache = None
    if args.cache_path:
        cache = ResultCache(args.cache_path, overwrite=args.overwrite)

    if args.ppl_eval:
        logging.info("Evaluating PPL on datasets: {}".format(args.ppl_eval_dataset))
        for dataset in args.ppl_eval_dataset:
            cache_key = f'ppl/{dataset}'
            if cache and cache.has(cache_key) and r4_collectors is None:
                dataset_ppl = cache.get(cache_key)
                logging.info(f'{dataset.upper()} PPL: {dataset_ppl:.2f} (cached)')
            else:
                testenc = data_utils.get_loaders(
                    dataset,
                    seed=args.seed,
                    model=args.model,
                    seqlen=model.seqlen,
                    hf_token=args.hf_token,
                    eval_mode=True)
                dataset_ppl = eval_utils.ppl_evaluator(model, testenc, utils.DEV, args)
                logging.info(f'{dataset.upper()} PPL: {dataset_ppl:.2f}')
                if cache:
                    cache.set(cache_key, dataset_ppl)
                    cache.save()

            if not args.log_to_console:
                print(f'{dataset.upper()} PPL: {dataset_ppl:.2f}')

            if args.wandb:
                wandb.log({'ppl/{}'.format(dataset.upper()): dataset_ppl})

    # ---------- R4 stats save ----------
    if r4_collectors is not None:
        import r4_stats
        r4_stats.save_r4_stats(r4_collectors, args.r4_stats, args)

    if args.lm_eval:
        logging.info("Evaluating on downstream tasks: {}".format(args.tasks))
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager   # lm_eval==0.4.3

        # Some datasets (e.g. social_i_qa) require trust_remote_code.
        # The env var is read at datasets import time, so patch the config directly.
        import datasets.config
        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

        task_manager = TaskManager()
        task_names = task_manager.match_tasks(args.tasks)

        # Determine which tasks still need to be evaluated
        tasks_to_run = []
        cached_results = {}
        for t in task_names:
            cache_key = f'lm_eval/{t}/acc'
            if cache and cache.has(cache_key):
                cached_results[t] = cache.get(cache_key)
                logging.info(f'lm_eval {t}: {cached_results[t]} (cached)')
            else:
                tasks_to_run.append(t)

        if tasks_to_run:
            logging.info(f"Running {len(tasks_to_run)} tasks (skipping {len(cached_results)} cached): {tasks_to_run}")
            tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, token=args.hf_token)
            hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

            results = lm_eval.simple_evaluate(hflm, tasks=tasks_to_run,)['results']

            for task, result in results.items():
                acc = round(result.get('acc_norm,none', result['acc,none']) * 100, 2)
                cached_results[task] = acc
                if cache:
                    cache.set(f'lm_eval/{task}/acc', acc)
            if cache:
                cache.save()
        else:
            logging.info("All lm_eval tasks found in cache — skipping evaluation.")

        metric_vals = {task: acc for task, acc in cached_results.items()}
        metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 2)
        if cache:
            cache.set('lm_eval/acc_avg', metric_vals['acc_avg'])
            cache.save()

        logging.info(metric_vals)

        if args.wandb:
            wandb.log(metric_vals)

    logging.info('--' * 30 + '\n\n')
    print("The end")


if __name__ == '__main__':
    main()
