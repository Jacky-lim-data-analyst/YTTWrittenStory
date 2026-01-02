from ollama import Client
from typing import Optional, Union, Literal
from pydantic.json_schema import JsonSchemaValue

class OllamaClient:
    class OptionsDefaultValues:
        # derived from ollama default values (https://github.com/ollama/ollama/blob/main/docs/modelfile.mdx#valid-parameters-and-values)
        REPEAT_PENALTY = 1.1
        TEMPERATURE = 0.8
        TOP_P = 0.9

    def __init__(self, model_name: str, system_prompt: str):
        self.client = Client()
        self.system_prompt = system_prompt
        self.model_name = model_name

    def generate(self, 
                 prompt: str, 
                 format: Optional[Union[Literal['', 'json'], JsonSchemaValue]] = None, 
                 thinking: Optional[bool] = None, 
                 **kwargs) -> str:
        """Generate model response
        kwargs:
        * num_ctx (int): sets the size of the context window used to generate next token (Default: 2048)
        * repeat_last_n (int): sets how far back for the model to look back to prevent repetition (Default: 64, 0=disabled, -1=num_ctx)
        * repeat_penalty (float): sets how strongly to penalize repetitions. Default: 1.1
        * temperature (float): Default=0.8
        * num_predict (int): Maximum number of tokens to predict when generating text (Default=-1, infinite generation)
        * top_k (int): A higher value gives more diverse answers.
        * top_p (float): Work with top_k
        * min_p (float): with p=0.05 and the most likely token having a probability of 0.9, logits with a value less than 0.045 are filtered out. (Default: 0.0)"""
        ALLOWED_KEYS = {"repeat_penalty", "temperature", "top_p"}
        
        # check for unexpected keys
        extra_keys = set(kwargs.keys()) - ALLOWED_KEYS
        if extra_keys:
            raise ValueError(f"Unexpected arguments: {', '.join(extra_keys)}")
        # define the options arguments
        options = {
            "repeat_penalty": kwargs.get("repeat_penalty", self.OptionsDefaultValues.REPEAT_PENALTY),
            "temperature": kwargs.get("temperature", self.OptionsDefaultValues.TEMPERATURE),
            "top_p": kwargs.get("top_p", self.OptionsDefaultValues.TOP_P)
        }
        
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            system=self.system_prompt,
            format=format,
            think=thinking,
            options=options
        )

        return response['response']


if __name__ == "__main__":
    import yaml
    sample_input = """YAML is a portable and widely used data serialization format. 
    Unlike the more compact JSON or verbose XML formats, 
    YAML emphasizes human readability with block indentation, 
    which should be familiar to most Python programmers."""

    with open('./config/llm_settings.yaml', 'r', encoding='utf-8') as f:
        llm_settings = yaml.safe_load(f)
        # print(llm_settings)
    
    translate_setting = llm_settings["translate_agent"]
    client = OllamaClient(model_name="qwen3:4b", system_prompt=translate_setting["system_prompt"])
    # print(client.system_prompt)
    
    repeat_penalty = translate_setting["options"]['repeat_penalty']
    temperature = translate_setting["options"]["temperature"]
    top_p = translate_setting["options"]["top_p"]
    # print(gen_settings)
    res = client.generate(sample_input, repeat_penalty=repeat_penalty, temperature=temperature, top_p=top_p)

    print(res)
