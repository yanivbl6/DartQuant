import torch
import torch_npu
import random
import numpy as np
import os
import atexit
import logging
from tqdm import tqdm
import math
# from scipy.linalg import hadamard
import torch.nn.functional as F


from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

# 启用 NPU 计算优化
torch.npu.set_option({'ACL_OP_COMPILER_CACHE_MODE': 'enable'})
torch_npu.npu.set_compile_mode(jit_compile=False)

supported_models = [
    '/home/chenrenxing/NPU_DartQuant/model/llama-2-70b',
    'meta-llama/Llama-2-7b-hf',
    'meta-llama/Llama-2-13b-hf',
    'meta-llama/Llama-2-70b-hf',
    'meta-llama/Meta-Llama-3-8B',
    'meta-llama/Meta-Llama-3-70B',
    'meta-llama/Llama-3.1-70B',
    'facebook/opt-125m'
]
supported_datasets = ['wikitext2', 'ptb', 'c4']

# These flags disable using TensorFloat-32 tensor cores (to avoid numerical issues)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# torch.npu.set_option({'ACL_OP_COMPILER_CACHE_MODE': 'disable'})  # 禁用 NPU 操作缓存优化


DEV = torch.device('npu') if torch_npu.npu.is_available() else torch.device('cpu')
# npu_id = os.getenv('NPU_VISIBLE_DEVICES', '0')  # 默认为'0'设备
# print(f'npu_id: {npu_id}')
# DEV = torch.device(f'npu:{npu_id}')  # 将模型和数据移动到指定的 NPU 设备
# print(f"device_count:{torch.npu.device_count()}")


def llama_down_proj_groupsize(model, groupsize):
    assert groupsize > 1, 'groupsize should be greater than 1!'

    if model.config.intermediate_size % groupsize == 0:
        logging.info(f'(Act.) Groupsize = Down_proj Groupsize: {groupsize}')
        return groupsize

    group_num = int(model.config.hidden_size / groupsize)
    assert groupsize * group_num == model.config.hidden_size, 'Invalid groupsize for llama!'

    down_proj_groupsize = model.config.intermediate_size // group_num
    assert down_proj_groupsize * group_num == model.config.intermediate_size, 'Invalid groupsize for down_proj!'
    logging.info(f'(Act.) Groupsize: {groupsize}, Down_proj Groupsize: {down_proj_groupsize}')
    return down_proj_groupsize


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


# Dump the log both to console and a log file.
def config_logging(log_file,
                   levels_to_log={logging.INFO, logging.ERROR},
                   to_console=True):
    
    # 确保路径存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # 清除已有 handlers，避免冲突
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 定义日志格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 自定义过滤器
    class SpecificLevelFilter(logging.Filter):
        def filter(self, record):
            return record.levelno in levels_to_log

    # 文件处理器
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    file_handler.addFilter(SpecificLevelFilter())  # 添加过滤器

    # 控制台处理器
    handlers = [file_handler]
    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.addFilter(SpecificLevelFilter())  # 添加过滤器
        handlers.append(console_handler)

    # 配置 logging
    logging.basicConfig(level=logging.DEBUG, handlers=handlers)

    # 确保缓冲区在程序退出时写入
    atexit.register(logging.shutdown)
    
    
def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.npu.memory_reserved(device=i) for i in range(torch.npu.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()
    
    if torch_npu.npu.is_available():
        torch.npu.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"NPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )



def distribute_model(model) -> None:
    """Distribute the model across available GPUs. NB: only implemented for Llama-2."""
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.utils import get_balanced_memory
    cleanup_memory()
    no_split_module_classes = ['LlamaDecoderLayer']

    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=no_split_module_classes,
    )

    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=no_split_module_classes
    )
    # print("Device Map:", device_map)
    # print(type(device_map["model.layers.1"]))  # int
    
    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )


    cleanup_memory()

def distribute_model_70b(model) -> None:
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.utils import get_balanced_memory
    cleanup_memory()


    model = model.to("cpu")
    # model = model.half()
    
    # 内存平衡配置
    no_split_module_classes = ['LlamaDecoderLayer']
    # max_memory = {
    #     0: "8GB", 1: "8GB", 2: "8GB",
    #     3: "8GB", 4: "8GB", 5: "8GB",
    #     6: "8GB", 7: '8GB', 'cpu': '200GB'
    # }
    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=no_split_module_classes,
    )
    # print(f"max_memory: {max_memory}")
    
    # 生成设备映射
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=no_split_module_classes,
    )

    # device_map["model.embed_tokens"] = "npu:0"
    # device_map["model.rotary_emb"] = "npu:0"
    # for layer_idx in range(0, 12):
    #     device_map[f"model.layers.{layer_idx}"] = "npu:0"
    # for layer_idx in range(12, 27):
    #     device_map[f"model.layers.{layer_idx}"] = "npu:1"
    # for layer_idx in range(27, 42):
    #     device_map[f"model.layers.{layer_idx}"] = "npu:2"
    # for layer_idx in range(42, 57):
    #     device_map[f"model.layers.{layer_idx}"] = "npu:3"
    # for layer_idx in range(57, 72):
    #     device_map[f"model.layers.{layer_idx}"] = "npu:4"
    # for layer_idx in range(72, 80):
    #     device_map[f"model.layers.{layer_idx}"] = "npu:5"
    # # for layer_idx in range(70, 80):
    # #     device_map[f"model.layers.{layer_idx}"] = "npu:6"
    # # for layer_idx in range(70, 80):
    # #     device_map[f"model.layers.{layer_idx}"] = 'cpu'
        
    # device_map["model.norm"] = "npu:5"
    # device_map["lm_head"] = "npu:5"
    
    device_map["model.embed_tokens"] = "npu:0"
    device_map["model.rotary_emb"] = "npu:0"
    for layer_idx in range(0, 8):
        device_map[f"model.layers.{layer_idx}"] = "npu:0"
    for layer_idx in range(8, 21):
        device_map[f"model.layers.{layer_idx}"] = "npu:1"
    for layer_idx in range(21, 34):
        device_map[f"model.layers.{layer_idx}"] = "npu:2"
    for layer_idx in range(34, 46):
        device_map[f"model.layers.{layer_idx}"] = "npu:3"
    for layer_idx in range(46, 58):
        device_map[f"model.layers.{layer_idx}"] = "npu:4"
    for layer_idx in range(58, 70):
        device_map[f"model.layers.{layer_idx}"] = "npu:5"
    for layer_idx in range(70, 80):
        device_map[f"model.layers.{layer_idx}"] = "npu:6"
    # for layer_idx in range(70, 80):
    #     device_map[f"model.layers.{layer_idx}"] = 'cpu'
        
    device_map["model.norm"] = "npu:6"
    device_map["lm_head"] = "npu:6"
    
    # print("Device Map:", device_map)
    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="npu_offload",
        state_dict=model.state_dict(),
    )

    cleanup_memory()


import torch
import os
import logging


def save_model_in_parts(model, save_qmodel_path, prefix='model_part', num_digits=5, target_file_size=10 * 1024**3):
    """
    将模型按文件大小（例如 10GB）分块保存为多个文件。
    :param model: 要保存的 PyTorch 模型
    :param save_qmodel_path: 模型保存路径
    :param target_file_size: 每个文件的目标大小（单位：字节，默认10GB）
    :param prefix: 文件名前缀（默认 'model_part'）
    :param num_digits: 文件序号的位数（默认 5 位数）
    """
    state_dict = model.state_dict()

    # 计算模型每个参数的大小（根据参数的数据类型自动计算）
    total_size = 0
    for param in state_dict.values():
        param_size = param.element_size() * param.numel()  # 获取参数的大小（单位字节）
        total_size += param_size

    # 计算分块数量
    num_parts = (total_size + target_file_size - 1) // target_file_size  # 向上取整

    logging.info(
        f"模型总大小: {total_size / (1024**3):.2f} GB, 分为 {num_parts} 部分，每个部分约 {target_file_size / (1024**3):.2f} GB")

    # 分块保存模型
    idx = 0
    current_part_size = 0  # 当前分块的实际字节大小
    part = {}

    for name, param in state_dict.items():
        param_size = param.element_size() * param.numel()  # 计算当前参数的字节数
        current_part_size += param_size  # 累加当前分块的大小
        part[name] = param  # 添加当前的参数到部分分块中

        # 如果当前块的大小超过目标大小，保存并开始新的块
        if current_part_size >= target_file_size:
            # 格式化文件名，确保序号是 5 位数
            part_filename = f"{prefix}_{str(idx).zfill(num_digits)}.pth"
            torch.save(part, os.path.join(save_qmodel_path, part_filename))
            logging.info(f"保存了模型的第 {idx + 1} 部分：{part_filename}，共 {num_parts} 部分。")
            part = {}  # 清空当前部分，开始下一个分块
            current_part_size = 0  # 重置当前分块的大小
            idx += 1

    # 最后一块
    if part:
        part_filename = f"{prefix}_{str(idx).zfill(num_digits)}.pth"
        torch.save(part, os.path.join(save_qmodel_path, part_filename))
        logging.info(f"保存了模型的第 {idx + 1} 部分：{part_filename}，共 {num_parts} 部分。")

    logging.info("模型分块保存完成。")


def load_model_in_parts(model, folder_path):
    """
    将模型的多个部分加载并逐块赋值给模型。
    :param model: 要加载的 PyTorch 模型
    :param folder_path: 存储分块模型文件的文件夹路径
    """
    # 获取文件夹中的所有 .pth 文件，按名称排序（确保加载顺序正确）
    model_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pth')])

    # 逐个加载分块
    with tqdm(total=len(model_files), desc="Loading Model Parts", unit="part") as pbar:
        for file_name in model_files:
            part = torch.load(os.path.join(folder_path, file_name), map_location='cpu')  # 加载分块

            model.load_state_dict(part, strict=False)  # 更新模型的参数（实时赋值）

            del part  # 释放已加载分块的内存
            pbar.update(1)  # 更新进度条

    logging.info("模型加载完成。")


def hadamard_torch(n, dtype=torch.float32, device="cpu"):

    # 检查 n 是否是 2 的幂
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("n must be a power of 2")

    # 初始化 H_1 = [[1]]
    H = torch.tensor([[1.0]], dtype=dtype, device=device)

    # 递归构造 Sylvester 矩阵
    for _ in range(int(torch.log2(torch.tensor(n)))):
        H = torch.cat((torch.cat((H, H), dim=1), torch.cat((H, -H), dim=1)), dim=0)

    return H

@torch.no_grad()
def hadamard_transform(X, scale=1.0):
    # n, m = X.shape
    n = X.shape[-1]
    device = X.device
    # print(device)
    if (n & (n - 1)) != 0:
        raise ValueError("Number of rows n must be a power of 2 for Hadamard transform")
    # torch.npu.synchronize()
    H = hadamard_torch(n, dtype=X.dtype).clone().detach().contiguous()
    
    H = H.to(device)
    # try:
    #     H = H.to(device)  # 模型分在多个卡的问题
    # except Exception as e:
    #     print(e)
    # X = X.to("cpu")
    
    
    X_transformed = torch.matmul(X, H)
    # X_transformed = F.linear(X, H)
    
    # H = H.to(device)  # 显存不够了
    # # X = X.to("cpu")
    # X_transformed = torch.matmul(X, H)
    # del X, H
    # torch.npu.synchronize()
    # torch.npu.empty_cache()
    X_transformed = X_transformed * scale
    return X_transformed


import torch
import torch_npu
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
from torch.utils.data.distributed import DistributedSampler

def prepare_for_wa_quant_inference(args):
    
    # Path setup for r1 and r2
    if args.r1_path and '.pt' not in args.r1_path and '.bin' not in args.r1_path:
        args.r1_path += '/' + args.r1_path.split('/')[-1] + '.pt'

    if args.r2_path and '.pt' not in args.r2_path and '.bin' not in args.r2_path:
        args.r2_path += '/' + args.r2_path.split('/')[-1] + '.pt'

    # Initialize wandb if enabled
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config.update(args)

    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token)

    model.eval()
    model.model_name = args.model.split('/')[-1]

    # Rotate the weights
    if args.fuse_norm:
        logging.info("Fuse LayerNorms")
        logging.info("Rotate the model use_r1={}, use_r2={}, use_r4={}, use_r3={}".format(
            args.use_r1, args.use_r2, args.use_r4, args.use_r3))

        rotation_utils.fuse_layer_norms(model)
        if args.use_r1 or args.use_r2 != 'none' or args.use_r4:
            rotation_utils.rotate_model(model, args)
            model = model.to("cpu")
        cleanup_memory(verbos=True)

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
        quant_utils.add_actquant(model)

    if args.w_bits < 16:
        logging.info("Add weight quantization: w_rtn = {}, w_bits = {}, w_groupsize = {}, w_sym = {}, w_clip = {}".format(
            args.w_rtn, args.w_bits, args.w_groupsize, not (args.w_asym), args.w_clip))

        save_dict = {}
        if args.load_qmodel_path:  # Load Quantized Rotated Model
            # assert args.fuse_norm, "Model should be fused to load a quantized model!"
            assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
            logging.info("Load quantized model from: {}.".format(args.load_qmodel_path))
            load_model_in_parts(model, args.load_qmodel_path)
            # save_dict = torch.load(args.load_qmodel_path, map_location='cpu')
            # model.load_state_dict(save_dict["model"])
        elif not args.w_rtn:  # GPTQ Weight Quantization
            assert "llama" in args.model, "Only llama is supported for GPTQ!"

            trainloader = data_utils.get_loaders(
                args.cal_dataset, nsamples=args.nsamples,
                seed=args.seed, model=args.model,
                seqlen=model.seqlen, eval_mode=False
            )
            if args.w_ft:
                w_fine_tuning.w_ft(model, trainloader, DEV, args)
                
            torch.npu.empty_cache() 
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, DEV, args)
            save_dict["w_quantizers"] = quantizers
        else:  # RTN Weight Quantization
            if args.w_ft:
                trainloader = data_utils.get_loaders(
                    args.cal_dataset, nsamples=args.nsamples,
                    seed=args.seed, model=args.model,
                    seqlen=model.seqlen, eval_mode=False
                )
                w_fine_tuning.w_ft(model, trainloader, DEV, args)      
            quantizers = gptq_utils.rtn_fwrd(model, args)
            save_dict["w_quantizers"] = quantizers

        if args.save_qmodel_path:
            folder_name = f'{model.model_name}_w{args.w_bits}_a{args.a_bits}_r1' if args.use_r1 else ''
            folder_name += '_r2' if args.use_r2 != 'none' else ''
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
            save_model_in_parts(model, args.save_qmodel_path, prefix=f'{model.model_name}_part')

    if args.a_bits < 16 or args.v_bits < 16:
        logging.info("Add v quantization: v_bits={}, v_groupsize={}, v_sym={}, v_clip_ratio={}".format(
            args.v_bits, args.v_groupsize, not (args.v_asym), args.v_clip_ratio))

        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0 and "llama" in args.model:
            down_proj_groupsize = llama_down_proj_groupsize(model, args.a_groupsize)

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
                if args.a_bits_down_proj is not None:
                    layer_input_bits = args.a_bits_down_proj
                layer_groupsize = down_proj_groupsize

            qlayers[name].quantizer.configure(bits=layer_input_bits,
                                              groupsize=layer_groupsize,
                                              sym=layer_a_sym,
                                              clip_ratio=layer_a_clip,
                                              residual=residual)

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
                
    torch_npu.npu.empty_cache()
    if args.distribute:
        if "70b" in args.model:
            distribute_model_70b(model)
        else:
            distribute_model(model)
    else:
        model.to(DEV)  # Ensure the model is moved to NPU for computation
    
    return model

torch.set_printoptions(precision=8)
def print_model(model):
    # print(model)
    for name, param in model.named_parameters():
        print(f"{name}: {param}")
        if len(param.shape) == 2:
            print(f"Parameter sum_0: {torch.sum(param, 0)},\nParameter sum_1: {torch.sum(param, 1)},\nParameter mean: {torch.mean(param)}")
