import argparse
import sys
sys.path.append('.')
sys.path.append('..')

import torch
import torch.nn as nn
import time
from transformers import AutoTokenizer, LlamaForCausalLM
from datautils import get_loaders


prompts = [
        "Once upon a time",
        "In a galaxy far, far away",
        "The quick brown fox jumps over the lazy dog",
        "In the middle of the night, she heard a sound",
        "The world was on the brink of change",
        "It all started with a simple mistake"
    ]

datasets = ['wikitext2', 'ptb'] 

def generate_text_args():
    """
    Craft custom parser args for text generation.
    """
    parser = argparse.ArgumentParser(description="Generate text using a quantized LLaMA model")
    
    # Model and file paths
    parser.add_argument('--model', type=str, required=True, help="Path to the base LLaMA model")
    parser.add_argument('--load', type=str, default=None, help="Path to the quantized model weights (optional)")
    
    # Generation parameters
    parser.add_argument('--max_length', type=int, default=300, help="Maximum length of generated text")
    parser.add_argument('--top_k', type=int, default=0, help="Top-k sampling (default: 0)")
    parser.add_argument('--top_p', type=float, default=0.9, help="Top-p value (default: 0.9)")
    parser.add_argument('--temperature', type=float, default=1.0, help="Sampling temperature (default: 0.9)")
    parser.add_argument('--do_sample', action='store_false', help="Use sampling for text generation (default: True)")
    
    # Device setup
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to run the model on (default: auto-detect)")
    
    # Benchmark flag
    parser.add_argument('--benchmark', action='store_true', help="Benchmark speed with original LLM")
    
    # Parse the arguments
    args = parser.parse_args()
    return args

def generate_text(model, tokenizer, prompt, max_length=50, device='cuda', top_k=0, temperature=0.7, top_p=0.9, do_sample=True, num_runs=10):
    """
    Measures tokens per second for processing input through the model.
    """
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # Warm-up run
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=1, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)
    
    # Multiple timed runs
    total_time = 0
    for _ in range(num_runs):
        start_time = time.perf_counter()
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=1, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
    
    # Calculate average time and TPS
    avg_time = total_time / num_runs
    num_tokens = input_ids.numel()
    tps = num_tokens / avg_time

    return tps
def print_tps_values(unquantized_tps, quantized_tps):
    """
    Print the Tokens Per Second (TPS) values for unquantized and quantized models.

    Args:
        unquantized_tps: List of TPS values for the unquantized model.
        quantized_tps: List of TPS values for the quantized model.
    """
    assert len(unquantized_tps) == len(unquantized_tps), "length of unquantized tps and quantized tps aren't the same"
    print("Tokens Per Second (TPS) Comparison:")
    print("Prompt                                   | Unquantized TPS | Quantized TPS")
    print("--------------------------------------------------------------------------------")
    for i in range(len(unquantized_tps)):
        print(f"{i + 1:<3} | {unquantized_tps[i]:<17.2f} | {quantized_tps[i]:<15.2f}")
    
    # Calculate and print the average TPS
    average_unquantized_tps = sum(unquantized_tps) / len(unquantized_tps) if unquantized_tps else 0
    average_quantized_tps = sum(quantized_tps) / len(quantized_tps) if quantized_tps else 0

    print("--------------------------------------------------------------------------------")
    print(f"Average TPS:                        | {average_unquantized_tps:<17.2f} | {average_quantized_tps:<15.2f}")


def print_ppl_values(unquantized_ppl_values, quantized_ppl_values):
    """
    Print the Perplexity (PPL) values for unquantized and quantized models.

    Args:
        unquantized_ppl: List of PPL values for the unquantized model.
        quantized_ppl: List of PPL values for the quantized model.
    """
    print("Perplexity (PPL) Comparison:")
    print("Dataset                                   | Unquantized PPL | Quantized PPL")
    print("--------------------------------------------------------------------------------")
    for dataset, unquantized_ppl, quantized_ppl in zip(datasets, unquantized_ppl_values, quantized_ppl_values):
        print(f"{dataset:<40} | {unquantized_ppl:<17.2f} | {quantized_ppl:<15.2f}")

def print_text_values(unquantized_texts, quantized_texts):
    """
    Print the generated text values for unquantized and quantized models.

    Args:
        unquantized_texts: List of generated texts for the unquantized model.
        quantized_texts: List of generated texts for the quantized model.
    """
    print("Generated Text Comparison:")
    print("Prompt                                   | Unquantized Text                                   | Quantized Text")
    print("----------------------------------------------------------------------------------------------------------------")
    for i in range(len(unquantized_texts)):
        print(f"{i + 1:<3} | {unquantized_texts[i][:50]:<50} | {quantized_texts[i][:50]:<50}")  # Print first 50 characters

@torch.no_grad()
def calculate_ppl(model, testenc, device):
    """
    Evaluate the LLaMA model on a test dataset by calculating perplexity.
    """
    print('Evaluating model...')

    model = model.to(device)
    testenc = testenc.to(device)
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    # Temporarily disable caching for evaluation
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # Move embedding and first layer to device
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    layers[0] = layers[0].to(device)

    # Prepare input tensors
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {'i': 0, 'attention_mask': None}

    # Use the Catcher class to capture input activations during forward pass
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = torch.as_tensor(kwargs.get('attention_mask')).to(device) if kwargs.get('attention_mask') is not None else None
            cache['position_ids'] = torch.as_tensor(kwargs.get('position_ids')).to(device) if kwargs.get('position_ids') is not None else None
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    kwargs[k] = v.to(device)
            raise ValueError

    layers[0] = Catcher(layers[0])

    # Loop through samples and feed them to the model
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(device)
        try:
            model(batch)
        except ValueError:
            pass  # Terminate forward pass to capture inputs

    # Restore original model state
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    # Process each layer sequentially during evaluation
    for i in range(len(layers)):
        layer = layers[i].to(device)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, cache_position=position_ids.squeeze())[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    # Normalize the output and compute logits for final evaluation
    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(device)
    model.lm_head = model.lm_head.to(device)

    # Calculate negative log likelihood and perplexity
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    # Calculate perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Restore original cache setting
    model.config.use_cache = use_cache

    return ppl.item()


def generate_ppl(base_model_name, model, device, seed=42):
    ppl_scores = []

    # Evaluate unquantized model
    for dataset in datasets:
        print(f"Evaluating model on dataset: {dataset}")
        dataloader, testloader = get_loaders(
            dataset, seed=seed, model=base_model_name, seqlen=model.seqlen, tokenizer=tokenizer
        )
        ppl = calculate_ppl(model, testloader, device)
        ppl_scores.append(ppl)

    return ppl_scores


def get_llama(model, model_checkpoint=None):
    """
    Load the LLaMA model with some custom initialization behavior disabled.
    """
    import torch
    def skip(*args, **kwargs):
        pass

    # Disable the default initialization behavior
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    # Load LlamaForCausalLM model
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048

    if model_checkpoint:
        model.load_state_dict(torch.load(model_checkpoint))

    model.eval()
    return model

def generate_tps_and_text(tokenizer, model, prompts, args):
    tps_values = []

    for prompt in prompts[:1]:
        tps = generate_text(
            model, 
            tokenizer, 
            prompt, 
            max_length=args.max_length, 
            device=args.device, 
            top_k=args.top_k, 
            temperature=args.temperature, 
            top_p=args.top_p,
            do_sample=args.do_sample
        )
        tps_values.append(tps)
    
    return tps_values



if __name__ == '__main__':
    # Parse arguments
    args = generate_text_args()

    torch.manual_seed(42)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Clear GPU memory
    torch.cuda.empty_cache()

    # Print the generation parameters
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Max Length: {args.max_length}")
    print(f"Top-k: {args.top_k}")
    print(f"Temperature: {args.temperature}")
    print(f"Do Sampling: {args.do_sample}")
    print("========================================")

    # Benchmarking
    if args.benchmark:
        print("==== Benchmarking unquantized model ====")
        # Load unquantized model
        unquantized_model = get_llama(args.model)
        unquantized_model = unquantized_model.to(args.device)

        # Generate text and measure TPS for unquantized model
        unquantized_tps_values, unquantized_text_values = generate_tps_and_text(tokenizer, unquantized_model)
        uquantized_ppl_values = generate_ppl(args.model, unquantized_model, args.device)

        del unquantized_model
        torch.cuda.empty_cache()

    # Load the quantized model (if a checkpoint is provided)
    if args.load:
        print("===== Running quantized model =====")
        quantized_model = get_llama(args.model, args.load)
        quantized_model = quantized_model.to(args.device)

        quantized_tps_values, quantized_text_values = generate_tps_and_text(tokenizer, quantized_model)
        quantized_ppl_values = generate_ppl(args.model, quantized_model, args.device)

        del quantized_model
        torch.cuda.empty_cache()
    else:
        print("Quantized model not loaded. No checkpoint specified.")

    print_tps_values(unquantized_tps_values, quantized_tps_values)
    # print_text_values(unquantized_text_values, quantized_text_values)
    print_ppl_values(uquantized_ppl_values, quantized_ppl_values)
