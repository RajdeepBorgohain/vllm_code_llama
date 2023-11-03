import os
from vllm import SamplingParams
from vllm import LLM
from huggingface_hub import snapshot_download


class InferlessPythonModel:
    def initialize(self):
        snapshot_download(
            "codellama/CodeLlama-34b-Python-hf",
            local_dir="/model",
            token="<<your_token>>",
        )
        self.llm = LLM("/model")
    
    def infer(self, inputs):
        print("inputs[prompt] -->", inputs["prompt"], flush=True)
        prompts = [inputs["prompt"]]
        print("Prompts -->", prompts, flush=True)
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=10,
            max_tokens=256,
            presence_penalty=1.15,
        )
        result = self.llm.generate(prompts, sampling_params)
        result_output = [output.outputs[0].text for output in result]

        return {"result": result_output[0]}

    def finalize(self, args):
        pass