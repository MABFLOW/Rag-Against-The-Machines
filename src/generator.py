from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def build_prompt(question: str, contexts: list[str]) -> str:
    joined_context = "\n\n---\n\n".join(contexts)

    return f"""
    You are answering a question about a codebase.

    Use only the provided context.
    Do not invent information.
    If the context is insufficient, say that the answer cannot be determined.

    Context:
    {joined_context}

    Question:
    {question}

    Answer:
    """.strip()

class Generator:
    def __init__(self) -> None:
        model_name = "Qwen/Qwen3-4B-Instruct-2507"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

    def generate(
        self,
        question: str,
        contexts: list[str],
    ) -> str:
        prompt = build_prompt(question, contexts)

       

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
        ).to(self.model.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = output[0][inputs["input_ids"].shape[1]:]

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()
    



   