import datasets
import glob
import random
import transformers


HF_DATASETS_ROOT = '/data/data/huggingface/datasets'


def _find_arrow(pattern):
    """Find a cached .arrow file by glob pattern under the HF datasets cache."""
    matches = glob.glob(f'{HF_DATASETS_ROOT}/{pattern}', recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"No cached arrow file matching {pattern} under {HF_DATASETS_ROOT}. "
            f"Run: HF_DATASETS_OFFLINE=0 python scripts/download_lm_eval_datasets.py"
        )
    return sorted(matches)[0]


def get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode=False):

    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, token=hf_token)

    wikitext_cache = '/data/data/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3'

    if eval_mode:
        testdata = datasets.Dataset.from_file(f'{wikitext_cache}/wikitext-test.arrow')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.Dataset.from_file(f'{wikitext_cache}/wikitext-train.arrow')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_c4_new(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False):

    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, token=hf_token)

    if eval_mode:
        c4_cache = '/data/data/huggingface/datasets/allenai___c4/default-ad670c44f8f136e7/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2'
        valdata = datasets.concatenate_datasets([
            datasets.Dataset.from_file(f'{c4_cache}/c4-validation-00000-of-00002.arrow'),
            datasets.Dataset.from_file(f'{c4_cache}/c4-validation-00001-of-00002.arrow'),
        ])
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]

        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        valenc = TokenizerWrapper(valenc)
        return valenc
    else:
        c4_cache = '/data/data/huggingface/datasets/allenai___c4/default-b04fc8a0b8562884/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2'
        traindata = datasets.concatenate_datasets([
            datasets.Dataset.from_file(f'{c4_cache}/c4-train-00000-of-00002.arrow'),
            datasets.Dataset.from_file(f'{c4_cache}/c4-train-00001-of-00002.arrow'),
        ])

        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode=False):

    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, token=hf_token)

    if eval_mode:
        arrow = _find_arrow('ptb_text_only/penn_treebank/**/ptb_text_only-test.arrow')
        testdata = datasets.Dataset.from_file(arrow)
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
        return testenc
    else:
        arrow = _find_arrow('ptb_text_only/penn_treebank/**/ptb_text_only-train.arrow')
        traindata = datasets.Dataset.from_file(arrow)
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', hf_token=None, eval_mode=False
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if 'ptb' in name:
        return get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if 'c4' in name:
        return get_c4_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
