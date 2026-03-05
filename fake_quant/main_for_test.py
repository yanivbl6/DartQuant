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
    model.eval()
    model.model_name = args.model.split('/')[-1]

    # --- kv_ex / proj_ex overrides (must be before rotate_model) ---
    if args.kv_ex != 0:
        logging.info("kv_ex=%d: disabling R3, setting k_bits=%d", args.kv_ex, args.kv_ex)
        args.use_r3 = False
        args.k_bits = args.kv_ex

    if args.proj_ex != 0:
        logging.info("proj_ex=%d: disabling R4, setting down_proj input bits=%d", args.proj_ex, args.proj_ex)
        args.use_r4 = False

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
            if args.use_r4 and 'down_proj' in name:
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

    if args.w_bits < 16:
        logging.info("Add weight quantization: w_rtn = {}, w_bits = {}, w_groupsize = {}, w_sym = {}, w_clip = {}".format(
            args.w_rtn, args.w_bits, args.w_groupsize, not (args.w_asym), args.w_clip))

        save_dict = {}

        # Resolve GPTQ checkpoint path
        _gptq_ckpt = None
        if args.gptq_checkpoint_path:
            _gptq_ckpt = os.path.join(
                args.gptq_checkpoint_path,
                f'{model.model_name}_w{args.w_bits}'
            )

        if args.load_qmodel_path:  # Load Quantized Rotated Model
            # assert args.fuse_norm, "Model should be fused to load a quantized model!"
            assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
            logging.info("Load quantized model from: {}.".format(args.load_qmodel_path))
            utils.load_model_in_parts(model, args.load_qmodel_path)
            # save_dict = torch.load(args.load_qmodel_path, map_location='cpu')
            # model.load_state_dict(save_dict["model"])

        elif _gptq_ckpt and os.path.isdir(_gptq_ckpt) and any(
                f.endswith('.pth') for f in os.listdir(_gptq_ckpt)):
            logging.info("Loading GPTQ checkpoint from: {}".format(_gptq_ckpt))
            utils.load_model_in_parts(model, _gptq_ckpt)
            logging.info("GPTQ checkpoint loaded – skipping quantization.")

        elif not args.w_rtn:  # GPTQ Weight Quantization
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
                os.makedirs(_gptq_ckpt, exist_ok=True)
                logging.info("Saving GPTQ checkpoint to: {}".format(_gptq_ckpt))
                utils.save_model_in_parts(model, _gptq_ckpt,
                                          prefix=f'{model.model_name}_part')

        else:  # RTN Weight Quantization

            if args.w_ft:  # 精度补偿：
                trainloader = data_utils.get_loaders(
                    args.cal_dataset, nsamples=args.nsamples,
                    seed=args.seed, model=args.model,
                    seqlen=model.seqlen, eval_mode=False
                )
                w_fine_tuning.w_ft(model, trainloader, utils.DEV, args)
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers

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

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
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

            if 'v_proj' in name and args.v_bits < 16:  # Set the v_proj precision
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
                if args.proj_ex != 0:
                    layer_input_bits = args.proj_ex
                elif args.a_bits_down_proj is not None:
                    layer_input_bits = args.a_bits_down_proj
                layer_groupsize = down_proj_groupsize

            qlayers[name].quantizer.configure(bits=layer_input_bits,
                                              groupsize=layer_groupsize,
                                              sym=layer_a_sym,
                                              clip_ratio=layer_a_clip,
                                              residual=residual)

    # --- Prepare integer GEMM with capped accumulator ---
    if args.int_gemm:
        logging.info("Preparing integer GEMM: acc_bits=%d, acc_block_k=%d, use_triton=%s, acc_wrap=%s",
                     args.acc_bits, args.acc_block_k, args.int_gemm_use_triton, args.acc_wrap)
        qlayers_ig = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        n_int_gemm = 0
        for name, qlayer in qlayers_ig.items():
            if 'lm_head' in name:
                continue
            if qlayer.quantizer.bits > 8:
                # Undo use_int_gemm that GPTQ propagation may have set
                # before the quantizer was reconfigured to >8 bits.
                qlayer.use_int_gemm = False
                if qlayer.quantizer.bits < 16:
                    logging.info("  skipping int_gemm for %s (act_bits=%d > 8, falling back to fake-quant)",
                                 name, qlayer.quantizer.bits)
                continue
            if getattr(qlayer.quantizer, 'groupsize', -1) > 0:
                # Grouped activation quantization (e.g. o_per_head) is
                # incompatible with int GEMM which needs per-token scales.
                qlayer.use_int_gemm = False
                logging.info("  skipping int_gemm for %s (act groupsize=%d, need per-token)",
                             name, qlayer.quantizer.groupsize)
                continue
            qlayer.prepare_int_gemm(
                w_bits=args.w_bits,
                w_sym=not args.w_asym,
                w_group_size=args.w_groupsize,
                acc_bits=args.acc_bits,
                acc_block_k=args.acc_block_k,
                use_triton=args.int_gemm_use_triton,
                acc_wrap=args.acc_wrap,
            )
            n_int_gemm += 1
        logging.info("Integer GEMM prepared for %d layers", n_int_gemm)

    if args.k_bits < 16:
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

    # Load pre-calibrated static activation scales
    if args.act_scales_path:
        logging.info("Loading static activation scales from: {}".format(args.act_scales_path))
        act_scales = torch.load(args.act_scales_path, map_location='cpu', weights_only=True)

        # Apply to ActQuantWrapper quantizers (input + output/v_proj)
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        for name, qlayer in qlayers.items():
            q_key = f'{name}.quantizer'
            if q_key in act_scales and qlayer.quantizer.bits < 16:
                qlayer.quantizer.scale = act_scales[q_key]['scale']
                qlayer.quantizer.zero = act_scales[q_key]['zero']
                qlayer.quantizer.static = True

            oq_key = f'{name}.out_quantizer'
            if oq_key in act_scales and qlayer.out_quantizer.bits < 16:
                qlayer.out_quantizer.scale = act_scales[oq_key]['scale']
                qlayer.out_quantizer.zero = act_scales[oq_key]['zero']
                qlayer.out_quantizer.static = True

        # Apply to QKRotationWrapper k_quantizers
        layers = model_utils.get_layers(model)
        for i, layer in enumerate(layers):
            rope_fn = model_utils.get_rope_function_name(model)
            wrapper_attr = f'{rope_fn}_qk_rotation_wrapper'
            if hasattr(layer.self_attn, wrapper_attr):
                wrapper = getattr(layer.self_attn, wrapper_attr)
                kq_key = f'layer.{i}.k_quantizer'
                if kq_key in act_scales and wrapper.k_quantizer.bits < 16:
                    wrapper.k_quantizer.scale = act_scales[kq_key]['scale']
                    wrapper.k_quantizer.zero = act_scales[kq_key]['zero']
                    wrapper.k_quantizer.static = True

        # Apply to PWLActivation quantizers (if PWL is enabled)
        if args.pwl_act:
            import pwl_utils
            pwl_modules = pwl_utils.find_pwl_activations(model)
            for name, pwl_mod in pwl_modules.items():
                iq_key = f'{name}.input_quantizer'
                if iq_key in act_scales and pwl_mod.input_quantizer.bits < 16:
                    pwl_mod.input_quantizer.scale = act_scales[iq_key]['scale']
                    pwl_mod.input_quantizer.zero = act_scales[iq_key]['zero']
                    pwl_mod.input_quantizer.static = True
                oq_key = f'{name}.output_quantizer'
                if oq_key in act_scales and pwl_mod.output_quantizer.bits < 16:
                    pwl_mod.output_quantizer.scale = act_scales[oq_key]['scale']
                    pwl_mod.output_quantizer.zero = act_scales[oq_key]['zero']
                    pwl_mod.output_quantizer.static = True

        logging.info("Static activation scales applied to all quantizers.")

    # Convert per-column static scales to per-group for int_gemm layers.
    # Per-column scales can't factor out of a dot product.  Per-group scales
    # (one per acc_block_k columns) can — each K-block gets its own scale.
    if args.int_gemm and args.act_scales_path:
        G = args.acc_block_k
        qlayers_pg = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        n_converted = 0
        for name, qlayer in qlayers_pg.items():
            if not getattr(qlayer, 'use_int_gemm', False):
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
            maxq = q.maxq.float()

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
            logging.info("Converted %d int_gemm quantizers to per-group scales "
                         "(group_size=%d)", n_converted, G)
            # Precompute static zero-point correction for asymmetric per-group mode
            for name, qlayer in qlayers_pg.items():
                if getattr(qlayer, 'use_int_gemm', False) and qlayer.quantizer.static:
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
            # Skip proj_ex quantizers: they use a different bit-width to bypass
            # int_gemm, so per-column vs per-token is not a valid comparison.
            if 'down_proj' in name and args.proj_ex != 0:
                continue
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

    # ---------- Result cache ----------
    cache = None
    if args.cache_path:
        cache = ResultCache(args.cache_path, overwrite=args.overwrite)

    if args.ppl_eval:
        logging.info("Evaluating PPL on datasets: {}".format(args.ppl_eval_dataset))
        for dataset in args.ppl_eval_dataset:
            cache_key = f'ppl/{dataset}'
            if cache and cache.has(cache_key):
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
