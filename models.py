import transformers
from torch import cuda, bfloat16
import torch

class LLama2:
    def __init__(self, model_id):
        """
        Initializes an instance of the LLama2 class.

        Args:
            model_id (str): The identifier of the LLama2 model to be used.

        Attributes:
            model_id (str): The identifier of the LLama2 model.
            device (str): The device to be used for model inference (e.g., 'cuda:0' for GPU or 'cpu' for CPU).
            tokenizer (transformers.LlamaTokenizer): The tokenizer for the LLama2 model.
            model_causal_lm (transformers.AutoModelForCausalLM): The LLama2 model for causal language modeling.
        """
        self.model_id = model_id
        self.device = f'cuda:{cuda.current_device()}' #if cuda.is_available() else 'cpu'

        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(model_id)

        self.model_causal_lm = transformers.AutoModelForCausalLM.from_pretrained(model_id,
                                                    device_map='auto',
                                                    torch_dtype=torch.bfloat16,
                                                    use_auth_token=True,
                                                    load_in_4bit=True
                                                    )