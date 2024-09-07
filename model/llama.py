import sys
import time
sys.path.append('.')
sys.path.append('..')

import torch
import torch.nn as nn

from quant_methods.gptq import *
from quant_methods.shiftaddllm import *
from modelutils import *
from parsers import parse_args

from quantizers.quant import *
from quant_methods.quant_model_bcq import quant_model
from quantizers.bcq_quant.quantizer import BCQuantizer
from lut_gemm.kernel import load_shiftaddllm_weight


def get_llama(model):
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
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    """
    Perform quantization-aware forward passes through the model layers sequentially.
    The input and output tensors are processed layer by layer, caching relevant attention mask
    and position id information.
    """
    print('Starting LLaMA sequential processing...')

    # Temporarily disable caching mechanism to simplify forward pass
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # Move embedding layers to device
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    for layer in layers:
        layer.to(dev)  # Move all layers to the specified device

    # Prepare inputs and cache
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    # Define a catcher class to catch the input activations and store them in the cache
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1

            # Capture attention mask and position ids
            cache['attention_mask'] = torch.as_tensor(kwargs.get('attention_mask')).to(dev) if kwargs.get('attention_mask') is not None else None
            cache['position_ids'] = torch.as_tensor(kwargs.get('position_ids')).to(dev) if kwargs.get('position_ids') is not None else None

            # Move other kwargs to device
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    kwargs[k] = v.to(dev)

            raise ValueError  # Force forward pass to terminate and cache data

    # Catch the activations in the first layer
    layers[0] = Catcher(layers[0])
    
    # Loop through the dataloader to gather input data
    for batch in dataloader:
        try:
            model(batch[0].to(dev))  # Feed input to model
        except ValueError:
            pass  # Catcher terminates here on purpose

    # Restore original model state
    layers[0] = layers[0].module
    for layer in layers:
        layer.cpu()  # Move layers back to CPU after processing
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    # Prepare outputs tensor and attention mask
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    # Load quantization configuration if provided
    quant_config_dict = None
    if args.quant_config:
        import json
        with open(args.quant_config, "r") as f:
            quant_config_dict = json.load(f)
        print(f"Loaded quantization config: {quant_config_dict}")

    print('Ready for quantization.')

    # Dictionary to store quantizers for each layer
    quantizers = {}

    # Process layers sequentially
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            # Specify a sequence of operations for layerwise processing
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]  # Process all layer operations together
        
        for names in sequential:
            subset = {n: full[n] for n in names}

        quant_method = {}
        for name in subset:
            # Decide quantization method based on arguments
            if args.gptq or args.lut_eval:
                quant_method[name] = GPTQ(subset[name])
            else:
                quant_method[name] = ShiftAddLLM(subset[name])

            # Use the quantization configuration if available
            if quant_config_dict is not None:
                wbits = quant_config_dict['model.layers.%d.%s' % (i, name)]["bits"]
            else:
                wbits = args.wbits

            # Setup quantizer
            if args.gptq:
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(
                    wbits, perchannel=True, sym=args.sym, mse=False, trits=args.trits
                )
            else:
                quant_method[name].quantizer = BCQuantizer(subset[name].weight.data.size(),
                                                    groupsize=args.groupsize, 
                                                    wbits=wbits,
                                                    rounds=args.bcq_round,
                                                    use_bst=args.use_bst, 
                                                    apot_nums=args.apot_nums)

        # Hook to track input/output tensors for each layer during the forward pass
        def add_batch(name):
            def tmp(_, inp, out):
                quant_method[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # Run forward pass to collect statistics for quantization
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, cache_position=position_ids.squeeze())[0]
        
        # Remove hooks after processing
        for h in handles:
            h.remove()
        
        # Post-processing after batch collection
        for name in subset:
            quant_method[name].post_batch()

        # Quantize the layer using the collected data
        for name in subset:
            print(f" ====== Processing Layer {i}, Operation {name} ====== ")
            quant_method[name].preproc(
                preproc_gptqH=args.pre_gptqH, percdamp=args.percdamp,
                preproc_rescale=args.pre_rescale, 
                preproc_proj=args.pre_proj, preproc_proj_extra=args.pre_proj_extra
            )

            # Perform faster quantization
            quant_method[name].fasterquant(
                args, model_name=str(args.model).split("/")[-1], layer_name=f"{i}.{name}"
            )

            # Store quantizer for this layer operation
            quantizers['model.layers.%d.%s' % (i, name)] = quant_method[name].quantizer
            quant_method[name].free()

        # Final forward pass with quantized weights
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, cache_position=position_ids.squeeze())[0]

        # Move layer back to CPU after processing
        layers[i] = layer.cpu()
        del layer
        del quant_method
        torch.cuda.empty_cache()

        # Swap inputs and outputs
        inps, outs = outs, inps

    # Restore original cache configuration
    model.config.use_cache = use_cache
    
    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, dev):
    """
    Evaluate the LLaMA model on a test dataset by calculating perplexity.
    """
    print('Evaluating model...')

    model = model.to(dev)
    testenc = testenc.to(dev)
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    # Temporarily disable caching for evaluation
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # Move embedding and first layer to device
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    # Prepare input tensors
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
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
            cache['attention_mask'] = torch.as_tensor(kwargs.get('attention_mask')).to(dev) if kwargs.get('attention_mask') is not None else None
            cache['position_ids'] = torch.as_tensor(kwargs.get('position_ids')).to(dev) if kwargs.get('position_ids') is not None else None
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    kwargs[k] = v.to(dev)
            raise ValueError

    layers[0] = Catcher(layers[0])

    # Loop through samples and feed them to the model
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
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
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, cache_position=position_ids.squeeze())[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    # Normalize the output and compute logits for final evaluation
    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

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
    print(f"Perplexity (PPL): {ppl.item()}")

    # Restore original cache setting
    model.config.use_cache = use_cache


def llama_pack3(model, quantizers):
    """
    Pack the quantized layers back into the model and finalize the quantization.
    """
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing quantized layers...')

    for name in qlayers:
        print(f"Packing layer: {name}")
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)

    print('Packing completed.')
    return model


if __name__ == '__main__':
    from datautils import *
    args = parse_args()

    # Create temporary storage if needed
    if args.temp_storage is not None:
        os.makedirs(args.temp_storage, exist_ok=True)

    # Load the LLaMA model
    model = get_llama(args.model)
    if args.load:
        model.load_state_dict(torch.load(args.load))
    model.eval()

    print(f"Model: {args.model}")
    print(model)

    # Load quantization weights if specified
    if args.load_temp_storage is not None:
        assert args.block_quant, "temp_storage only works for blockwise quantization"
        load_shiftaddllm_weight(model, args.load_temp_storage, model_name=str(args.model).split("/")[-1],
                                wbits=args.wbits, groupsize=args.groupsize)

    # Load dataset and prepare dataloaders
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    
    # Perform quantization if model is not already quantized
    if args.wbits < 16 and not args.nearest and not args.load:
        tick = time.time()
        if args.bcq:
            print("Quantizing model with BCQ...")
            model = quant_model(model, qbits=args.wbits, group_size=args.groupsize)
        else:
            quantizers = llama_sequential(model, dataloader, DEV)
        print(f"Full quantization time: {time.time() - tick:.2f} seconds")
    
    # Save the quantized model if requested
    if args.save:
        torch.save(model.state_dict(), args.save)
    
    # Evaluate model on different datasets
    datasets = ['wikitext2', 'ptb'] 
    if args.new_eval:
        datasets = ['wikitext2', 'ptb-new', 'c4-new']

    for dataset in datasets:
        print(f"Evaluating on dataset: {dataset}")
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        llama_eval(model, testloader, DEV)
