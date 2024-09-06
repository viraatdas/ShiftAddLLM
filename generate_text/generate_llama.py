import argparse
import sys
sys.path.append('.')
sys.path.append('..')

import torch
import time
from transformers import LlamaTokenizer, LlamaForCausalLM
from modelutils import *

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
    
    # Device setup
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to run the model on (default: auto-detect)")
    
    # Benchmark flag
    parser.add_argument('--benchmark', type=bool, default=False, help="Benchmark speed with original LLM")
    
    # Parse the arguments
    args = parser.parse_args()
    return args

def generate_text(model, tokenizer, prompt, max_length=50, device='cuda'):
    """
    Generates text from a prompt using the provided LLaMA model and measures tokens per second.
    """
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Start timing the generation process
    start_time = time.time()

    # Generate text from the model
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length, do_sample=True, top_k=50, top_p=0.95, temperature=0.9)

    # End timing
    end_time = time.time()

    # Get the total number of tokens (input tokens + generated tokens)
    total_tokens = output_ids.size(1)

    # Calculate the time taken
    time_taken = end_time - start_time

    # Calculate tokens per second
    tokens_per_second = total_tokens / time_taken

    # Decode the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text, tokens_per_second

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

if __name__ == '__main__':
    # Parse arguments
    args = generate_text_args()


    # Load the tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    prompts = [
        "Once upon a time",
        "In a galaxy far, far away",
        "The quick brown fox jumps over the lazy dog",
        "In the middle of the night, she heard a sound",
        "The world was on the brink of change",
        "It all started with a simple mistake"
    ]


    # Clear GPU memory
    torch.cuda.empty_cache()

    # Benchmarking
    if args.benchmark:
        print("==== Benchmarking unquantized model ====")
        # Load unquantized model
        unquantized_model = get_llama(args.model)
        unquantized_model = unquantized_model.to(args.device)

        # Generate text and measure TPS for unquantized model
        tps_values = []

        for prompt in prompts:
            unquantized_text, unquantized_tps = generate_text(unquantized_model, tokenizer, prompt, max_length=args.max_length, device=args.device)
            tps_values.append(unquantized_tps)

            print(f"Generated Text (Unquantized Model):\n{unquantized_text}")

        for tps in tps_values:
            print(f"Tokens per second (Unquantized Model): {tps:.2f} tokens/second")

        print(f"Avg TPS (Unquantized Model): {(sum(tps_values)/len(tps_values)):.2f} tokens/second")
        print("========================================\n")

        del unquantized_model
        torch.cuda.empty_cache()

    # Load the quantized model (if a checkpoint is provided)
    if args.load:
        print("===== Running quantized model =====")
        quantized_model = get_llama(args.model, args.load)
        quantized_model = quantized_model.to(args.device)

        tps_values = []

        # Generate text using the quantized model
        for prompt in prompts:
            quantized_text, quantized_tps = generate_text(quantized_model, tokenizer, prompt, max_length=args.max_length, device=args.device)
            tps_values.append(quantized_tps)

            print(f"Generated Text (Quantized Model):\n{quantized_text}")

        for tps in tps_values:
            print(f"Tokens per second (Quantized Model): {tps:.2f} tokens/second")
        
        print(f"Avg TPS (Quantized Model): {(sum(tps_values)/len(tps_values)):.2f} tokens/second")
        print("===================================")
        
        del quantized_model
        torch.cuda.empty_cache()
    else:
        print("Quantized model not loaded. No checkpoint specified.")
