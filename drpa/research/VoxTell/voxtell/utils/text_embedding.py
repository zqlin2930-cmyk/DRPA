import torch

def last_token_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def wrap_with_instruction(text_prompts):
    instruct = 'Given an anatomical term query, retrieve the precise anatomical entity and location it represents'

    instruct_text_prompts = []
    for text in text_prompts:
        instruct_text_prompts.append(f'Instruct: {instruct}\nQuery: {text}')
    return instruct_text_prompts