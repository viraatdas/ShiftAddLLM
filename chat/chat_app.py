import streamlit as st
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM


SYSTEM_PROMPT = "You are a helpful AI assistant. Answer the user's questions accurately and concisely."


# Function to load the quantized model
@st.cache_resource
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


# Function to generate a response using the model
def generate_response(model, tokenizer, system_prompt, user_prompt, max_length=100, top_k=0, top_p=0.9, temperature=0.7):
    full_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_length=input_ids.shape[1] + max_length,
            do_sample=True, 
            top_k=top_k, 
            top_p=top_p, 
            temperature=temperature
        )
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# Streamlit App
st.title("Chat with a Quantized LLaMA Model")

# Sidebar for model and sampling configuration
with st.sidebar:
    st.header("Model Configuration")
    model_path = st.text_input("Model Path", value="meta-llama/Llama-2-13b-hf")
    checkpoint_path = st.text_input("Checkpoint Path (Optional)", value="BCQ_ACC_13b_HF")
    max_length = st.slider("Max Length", min_value=50, max_value=500, value=300)
    top_k = st.slider("Top-k Sampling", min_value=0, max_value=100, value=0)  # Default: 0
    top_p = st.slider("Top-p (Nucleus) Sampling", min_value=0.0, max_value=1.0, value=0.9)  # Default: 0.9
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=0.7)  # Default: 0.7

    system_prompt = st.text_area("System Prompt", value=SYSTEM_PROMPT)


# Load model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_path)
torch.cuda.empty_cache()
model = get_llama(model_path, checkpoint_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Chat interface
st.header("Chat with the Model")
if "history" not in st.session_state:
    st.session_state.history = []

# Input prompt
user_input = st.text_input("You: ", value="", placeholder="Type your message here...")

# Generate response when the user submits input
if st.button("Send"):
    if user_input:
        st.session_state.history.append({"role": "user", "content": user_input})
        
        with st.spinner("Generating response..."):
            model_response = generate_response(
                model, 
                tokenizer, 
                system_prompt,
                user_input, 
                max_length=max_length, 
                top_k=top_k, 
                top_p=top_p, 
                temperature=temperature
            )
        
        st.session_state.history.append({"role": "assistant", "content": model_response})


# Display chat history
for chat in st.session_state.history:
    if chat["role"] == "user":
        st.write(f"**You**: {chat['content']}")
    else:
        st.write(f"**Assistant**: {chat['content']}")
