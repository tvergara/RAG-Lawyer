from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun

class BaseLLM(LLM):
    model_name: str
    max_new_tokens = 20
    min_new_tokens = 10
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForCausalLM] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_models()

    def init_models(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        tokens = self.tokenizer(prompt, return_tensors="pt")
        generated_ids = self.model.generate(
            **tokens,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens
        )
        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return response.removeprefix(prompt)


if __name__ == "__main__":
    model_name = "llmware/bling-sheared-llama-1.3b-0.1"
    llm = BaseLLM(model_name=model_name)
    print(llm("What is the capital of France?"))
