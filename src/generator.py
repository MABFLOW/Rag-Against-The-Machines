from transformers import pipeline, logging as hf_logging
import torch 

hf_logging.set_verbosity_error()

SYSTEM_PROMPT = (
    "You are answering questions about a codebase. "
    "Use only the provided context to answer. "
    "Do not invent information. "
    "If the context is insufficient to answer, say so explicitly."
)

def build_prompt(question: str, contexts: list[str]) -> str:
    joined_context = "\n\n---\n\n".join(contexts)
    return f"""Context:
{joined_context}

Question:
{question}

Answer:"""

class Generator:
    def __init__(self) -> None:
        self.pipe = pipeline(
            "text-generation",
            model="Qwen/Qwen3-0.6B",
            dtype="float32",
            device=-1,
        )        


    def generate(self, question: str, contexts: list[str]) -> str:
        prompt = build_prompt(question, contexts)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        text = self.pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        output = self.pipe(
            text,
            return_full_text=False,
        )

        return output[0]["generated_text"].strip()

