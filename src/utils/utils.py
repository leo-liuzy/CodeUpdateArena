import tiktoken

OPENAI_MODEL, MAX_LENGTH = "gpt-4", 8191 # read from playground
# assuming one interaction (system_prompt + 1 * user prompt) with {gpt-4, gpt-3.5-turbo}
NUM_OPENAI_HIDDEN_TOKENS = 11 

def set_random_seed(seed):
    import torch
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)


def call_openai_chat(client, sys_prompt: str, user_prompt: str, model: str = OPENAI_MODEL):
    max_generatable_tokens = MAX_LENGTH - NUM_OPENAI_HIDDEN_TOKENS
    tokenizer =  tiktoken.encoding_for_model(OPENAI_MODEL)
    tok_sys_prompt = tokenizer.encode(sys_prompt)
    tok_user_prompt = tokenizer.encode(user_prompt)
    max_generatable_tokens -= len(tok_user_prompt) + len(tok_sys_prompt)
    # raise Exception("Should not call OpenAI-API again.")
    return client.chat.completions.create(
        model=model,
        # TODO: consider using response_format={ "type": "json_object" },
        # response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": user_prompt, 
            }
        ],
        max_tokens=max_generatable_tokens,
    )