from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
import time

MODEL_NAME_OR_PATH = "TheBloke/CodeLlama-34B-Python-GPTQ"
DEVICE = "cuda:0"




def main():
    """Initialize the application with the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)

    model = AutoGPTQForCausalLM.from_quantized(MODEL_NAME_OR_PATH,
            use_safetensors=True,
            trust_remote_code=False,
            device="cuda:0",
            use_triton=False,
            quantize_config=None,
            inject_fused_attention=False)

    max_new_tokens = 512
    temperature = 0.7
    prompt = "Write a python function to print from 0 to 100."
    start_time = time.perf_counter()

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()

    output = model.generate(inputs=input_ids, temperature=temperature, max_new_tokens=max_new_tokens)
    
    end_time = time.perf_counter()
    latency = end_time - start_time

    result = tokenizer.decode(output[0])
    
    print('Latency:{latency}')
    print(result)