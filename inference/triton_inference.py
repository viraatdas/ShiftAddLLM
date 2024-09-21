### Fix this - doesn't work 
### First fix CUDA kernel calling for the Ours (lat.) method

import sys
sys.path.append('.')
sys.path.append('..')

from parsers import parse_args
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import triton
import triton.language as tl

class TritonFusedLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, activation=torch.nn.GELU()):
        super(TritonFusedLayer, self).__init__()
        self.weight = torch.randn((output_size, input_size), device='cuda', dtype=torch.float16)  # Quantized weights
        self.bias = torch.randn(output_size, device='cuda', dtype=torch.float16)  # Bias if applicable
        self.activation = activation

    @triton.jit
    def fused_kernel(x_ptr, w_ptr, bias_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
        """
        Triton kernel to perform matrix multiplication + activation fusion.
        This fuses GEMM and the activation function (GELU here).
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        x = tl.load(x_ptr + block_start)
        w = tl.load(w_ptr + block_start)
        out = tl.dot(x, w)  # Fused matrix multiplication
        if bias_ptr:
            bias = tl.load(bias_ptr + block_start)
            out += bias  # Add bias if needed
        out = tl.libdevice.erf(out * 0.70710678118) * 0.5 * out  # GELU activation
        tl.store(out_ptr + block_start, out)

    def forward(self, x):
        BLOCK_SIZE = 256
        out = torch.empty((x.shape[0], self.weight.shape[0]), device='cuda', dtype=torch.float16)
        # Launch the fused Triton kernel
        self.fused_kernel[(x.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE](
            x, self.weight, self.bias, out, x.shape[0], BLOCK_SIZE=BLOCK_SIZE
        )
        return out


# Using the TritonFusedLayer in the LLaMA model
class LLaMATritonInference(torch.nn.Module):
    def __init__(self, llama_model):
        super(LLaMATritonInference, self).__init__()
        self.llama_model = llama_model  # Pre-loaded quantized LLaMA model
        self.fused_layer = TritonFusedLayer(input_size=4096, output_size=4096)  # Example size; adjust to match LLaMA architecture

    def forward(self, input_ids):
        # Forward pass through the quantized LLaMA model
        llama_output = self.llama_model(input_ids)

        # Access the logits from the output
        logits = llama_output.logits  # Use logits instead of llama_output directly

        # Apply the fused Triton kernel in one of the layers
        output = self.fused_layer(logits)  # Pass logits to the fused layer

        return output

def get_llama(model_name, model_checkpoint=None):
    """
    Load the LLaMA model with some custom initialization behavior disabled.
    """
    def skip(*args, **kwargs):
        pass

    # Disable the default initialization behavior
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    # Load LlamaForCausalLM model
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype='auto')
    model.seqlen = 2048

    if model_checkpoint:
        model.load_state_dict(torch.load(model_checkpoint))

    model.eval()
    return model


if __name__ == '__main__':
    args = parse_args()
    model = get_llama(args.model, args.load)

    model.to("cuda")

    # Wrap your model with Triton-optimized inference
    llama_triton_model = LLaMATritonInference(llama_model=model)

    # Run inference on some input
    input_ids = torch.randint(0, 32000, (1, 2048), device='cuda', dtype=torch.int64)  # Example input_ids for LLaMA
    output = llama_triton_model(input_ids)

    # Print output or save it as needed
    print(output)
