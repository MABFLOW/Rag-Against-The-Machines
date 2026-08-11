from transformers import pipeline, logging as hf_logging


hf_logging.set_verbosity_error()

SYSTEM_PROMPT = (
    "You are answering questions about a codebase. "
    "Use only the provided context to answer. "
    "Do not invent information. "
    "If the context is insufficient to answer, say so explicitly."
)


def build_prompt(question: str, contexts: list[str]) -> str:
    """Builds a prompt combining retrieved contexts and a question.

    Args:
        question: The user's question.
        contexts: Retrieved context passages to ground the answer in.

    Returns:
        The formatted prompt string.
    """
    joined_context = "\n\n---\n\n".join(contexts)
    return f"""Context:
{joined_context}

Question:
{question}

Answer:"""


class Generator:
    """Generates answers to questions using a local text-generation model."""

    def __init__(self) -> None:
        """Initializes the Generator with a text-generation pipeline."""
        self.pipe = pipeline(
            "text-generation",
            model="Qwen/Qwen3-0.6B",
            dtype="float32",
            device=-1,
        )

    def generate(self, question: str, contexts: list[str]) -> str:
        """Generates an answer to a question grounded in the given contexts.

        Args:
            question: The user's question.
            contexts: Retrieved context passages to ground the answer in.

        Returns:
            The generated answer text.
        """
        prompt = build_prompt(question, contexts)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        tokenizer = self.pipe.tokenizer
        if tokenizer is None:
            raise RuntimeError("Pipeline has no tokenizer configured.")

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(text, str):
            raise RuntimeError(
                "Expected apply_chat_template to return a string prompt.")

        output = self.pipe(
            text,
            return_full_text=False,
        )

        return output[0]["generated_text"].strip()
