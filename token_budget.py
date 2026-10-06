import tiktoken


TOKEN_ENCODING = tiktoken.get_encoding("o200k_harmony")
MAX_QUERY_TOKEN_BUDGET = 7500
MAX_PLANNER_PROMPT_TOKENS = 1200
MAX_PLANNER_COMPLETION_TOKENS = 128
MAX_RESPONSE_PROMPT_TOKENS = 5000
MAX_RESPONSE_COMPLETION_TOKENS = 1000

if (
    MAX_PLANNER_PROMPT_TOKENS
    + MAX_PLANNER_COMPLETION_TOKENS
    + MAX_RESPONSE_PROMPT_TOKENS
    + MAX_RESPONSE_COMPLETION_TOKENS
    > MAX_QUERY_TOKEN_BUDGET
):
    raise ValueError("Configured model token limits exceed the per-query budget.")


def count_tokens(text: str) -> int:
    """Count text with the tokenizer used by GPT-OSS models."""
    return len(TOKEN_ENCODING.encode(text, disallowed_special=()))


def build_bounded_prompt(
    template: str,
    parts: dict[str, str],
    token_limit: int,
    trim_order: tuple[str, ...],
) -> tuple[str, int]:
    """Build a prompt under its token limit by trimming low-priority fields."""
    bounded_parts = parts.copy()
    prompt = template.format_map(bounded_parts)
    token_count = count_tokens(prompt)

    while token_count > token_limit:
        trimmed = False
        for name in trim_order:
            field_tokens = TOKEN_ENCODING.encode(
                bounded_parts[name],
                disallowed_special=(),
            )
            if not field_tokens:
                continue

            excess = token_count - token_limit
            new_size = max(0, len(field_tokens) - excess - 32)
            if new_size >= len(field_tokens):
                new_size = len(field_tokens) - 1

            bounded_parts[name] = TOKEN_ENCODING.decode(
                field_tokens[:new_size]
            )
            prompt = template.format_map(bounded_parts)
            token_count = count_tokens(prompt)
            trimmed = True
            if token_count <= token_limit:
                break

        if not trimmed:
            raise ValueError(
                f"Prompt's fixed text exceeds the {token_limit}-token limit."
            )

    return prompt, token_count
