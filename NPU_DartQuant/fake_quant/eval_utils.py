import utils
import model_utils
import quant_utils
import torch
import torch_npu
import torch.nn as nn
import math
import os
import logging
from tqdm import tqdm


@torch.no_grad()
def evaluator(model, testenc, dev, args):
    model.eval()
    model = model.npu()  # Ensure the model is moved to NPU

    if 'opt' in args.model:
        opt_type = True
        llama_type = False
    elif 'meta' in args.model or "llama" in args.model:
        llama_type = True
        opt_type = False
    else:
        raise ValueError(f'Unknown model {args.model}')

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if opt_type:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.npu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.npu()
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.npu()
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.npu()

    elif llama_type:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.npu()

    layers[0] = layers[0].npu()

    # Convert the whole text of evaluation dataset into batches of sequences.
    input_ids = testenc.input_ids  # (1, text_len)
    nsamples = input_ids.numel() // model.seqlen  # The tail is truncated.
    input_ids = input_ids[:, :nsamples * model.seqlen].view(nsamples, model.seqlen).npu()  # (nsamples, seqlen)

    batch_size = 1  # args.bsz
    input_ids = [input_ids[i:i + batch_size] for i in range(0, nsamples, batch_size)]
    nbatches = len(input_ids)

    dtype = next(iter(model.parameters())).dtype
    # The input of the first decoder layer.
    inps = torch.zeros(
        (nbatches, batch_size, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    inps = [0] * nbatches
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if llama_type:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])

    # Batch inference with a safe try-except to catch the ValueError from Catcher
    for i in range(nbatches):
        batch = input_ids[i]
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    if opt_type:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif llama_type:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        position_ids = cache['position_ids']

    # Clean up memory after moving the model to CPU
    torch.npu.empty_cache()
    outs = [0] * nbatches
    attention_mask = cache['attention_mask']
    
    # npu_stream = torch.npu.Stream()
    # with torch.npu.stream(npu_stream):
    for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
        layer = layers[i].npu()

        # Dump the layer input and output if needed
        if args.capture_layer_io and args.layer_idx == i:
            captured_io = model_utils.capture_layer_io(model_utils.get_model_type(model), layer, inps)
            save_path = model_utils.get_layer_io_save_path(args)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(captured_io, save_path)
            logging.info(f'Dumped layer input and output to: {save_path}')

        for j in range(nbatches):
            bsz = inps[j].shape[0]
            if opt_type:
                outs[j] = layer(inps[j], attention_mask=attention_mask[0].repeat(bsz, 1, 1, 1))[0]
            elif llama_type:
                outs[j] = layer(
                    inps[j], attention_mask=attention_mask[0].repeat(
                        bsz, 1, 1, 1), position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        # Clean NPU cache after each batch
        torch.npu.empty_cache()
        inps, outs = outs, inps

    # Restore model to CPU after evaluation
    if opt_type:
        if model.model.decoder.final_layer_norm is not None:
            model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.npu()
        if model.model.decoder.project_out is not None:
            model.model.decoder.project_out = model.model.decoder.project_out.npu()

    elif llama_type:
        if model.model.norm is not None:
            model.model.norm = model.model.norm.npu()

    model.lm_head = model.lm_head.npu()
    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for i in range(nbatches):
        hidden_states = inps[i]
        if opt_type:
            if model.model.decoder.final_layer_norm is not None:
                hidden_states = model.model.decoder.final_layer_norm(hidden_states)
            if model.model.decoder.project_out is not None:
                hidden_states = model.model.decoder.project_out(hidden_states)
        elif llama_type:
            if model.model.norm is not None:
                hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :]
        shift_labels = input_ids[i][:, 1:]
        loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
        neg_log_likelihood = loss.float().mean(dim=1)
        nlls.append(neg_log_likelihood)
    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())
    model.config.use_cache = use_cache
    logging.info(f'\n{args.eval_dataset.upper()} PPL: {ppl.item():.3f}')
    return ppl.item()


@torch.no_grad()
def ppl_evaluator(model, testenc, dev, args):
    model.eval()

    if 'opt' in args.model:
        opt_type = True
        llama_type = False
        model.model.decoder = model.model.decoder.npu()
    elif 'meta' in args.model or "llama" in args.model:
        llama_type = True
        opt_type = False
        # model = model.npu()
    else:
        raise ValueError(f'Unknown model {args.model}')

    # Disable use_cache during evaluation to prevent unnecessary memory usage
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # Convert the input dataset into batches
    input_ids = testenc.input_ids  # (1, text_len)
    nsamples = input_ids.numel() // model.seqlen  # Tail is truncated
    input_ids = input_ids[:, :nsamples * model.seqlen].view(nsamples, model.seqlen)  # (nsamples, seqlen)

    batch_size = args.ppl_eval_batch_size
    input_ids = [input_ids[i:i + batch_size] for i in range(0, nsamples, batch_size)]
    nlls = []

    # Initialize tqdm with dynamic description to show PPL during the loop
    pbar = tqdm(range(nsamples), desc="Evaluating PPL")

    # Loop over each batch in the evaluation dataset
    for i in pbar:
        # Move batch to NPU
        # batch = input_ids[i].to(model.model.device) # origin
        
        batch = input_ids[i].to(model.model.embed_tokens.weight.device)  # for llama-2-70b
        
        # Perform the model's forward pass based on the model type
        if opt_type:
            outputs = model.model.decoder(batch)
        elif llama_type:
            # print(model.model)
            outputs = model.model(batch)
            # outputs = llama_2_70b_forward(model, batch)      

        hidden_states = outputs[0]
        # print("hidden_states shape:", hidden_states.shape)
        # print("model.lm_head.weight shape:", model.lm_head.weight.data.shape)
        # print(hidden_states.shape)
        logits = model.lm_head(hidden_states)
        # print(logits.shape)
        # exit(0)

        # Shift logits and labels for cross-entropy computation
        # shift_logits = logits[:, :-1, :]
        # shift_labels = input_ids[i][:, 1:].to(model.lm_head.weight.device)
        
        # for llama-2-70b
        shift_logits = logits[:, :-1, :].to("npu:0")
        shift_labels = input_ids[i][:, 1:].to("npu:0")

        # Compute cross-entropy loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.contiguous().view(-1, shift_logits.size(-1)),
            shift_labels.contiguous().view(-1),
        )
        neg_log_likelihood = loss.float()

        # Append the computed negative log likelihood if it's valid (not NaN)
        if not math.isnan(neg_log_likelihood):
            nlls.append(neg_log_likelihood)

        # Dynamically update the PPL in tqdm description
        if nlls:
            current_ppl = torch.exp(torch.stack(nlls).mean())
            pbar.set_description(f"PPL: {current_ppl.item():.2f}")

    # Calculate final PPL after all batches have been processed
    final_ppl = torch.exp(torch.stack(nlls).mean())
    
    # Restore the model's cache setting
    model.config.use_cache = use_cache

    return final_ppl.item()

# adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def llama_2_70b_forward(model, batch):
    batch = batch.to("npu:0")  # 限定从第一张卡开始
    inputs_embeds = model.model.embed_tokens(batch)
    
    past_seen_tokens = 0
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    )
    
    position_ids = cache_position.unsqueeze(0)

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    # if hasattr(model.model, "rotary_emb"):
    #     position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    
    for layer in model.model.layers:
        hidden_states = hidden_states.to(layer.self_attn.q_proj.weight.device)
        # position_embeddings = position_embeddings.to(layer.self_attn.q_proj.weight.device)
        position_embeddings = tuple(element.to(layer.self_attn.q_proj.weight.device) for element in position_embeddings)
        hidden_states = layer(hidden_states, position_ids=position_ids, position_embeddings=position_embeddings)[0]
    hidden_states = hidden_states.to(model.norm.weight.device)
    outputs = model.norm(hidden_states)
    return outputs