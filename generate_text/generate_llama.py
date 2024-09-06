
import argparse
import sys
sys.path.append('.')
sys.path.append('..')

import torch
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
    parser.add_argument('--max_length', type=int, default=50, help="Maximum length of generated text")
    
    # Device setup
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to run the model on (default: auto-detect)")
    
    # Parse the arguments
    args = parser.parse_args()
    return args

def generate_text(model, tokenizer, prompt, max_length=50):
    """
    Generates text from a prompt using the provided LLaMA model.
    """
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEV)

    # Generate text from the model
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length, do_sample=True, top_k=50, top_p=0.95, temperature=0.9)

    # Decode the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

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

    # Load the LLaMA model and tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    model = get_llama(args.model, args.load)

    # Ensure the model is in evaluation mode
    model.eval()
    
    # Move model to device (GPU or CPU)
    model = model.to(DEV)

    # Prompt for text generation
    prompt = "Once upon a time"

    # Generate text using the quantized model
    generated_text = generate_text(model, tokenizer, prompt, max_length=args.max_length if args.max_length else 50)
    
    print(f"Generated Text:\n{generated_text}")
