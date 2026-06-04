"""Forward pass wiring: embed → N×transformer_block → final_norm → lm_head."""


class LlamaModel:
    def __init__(self, weights: dict, config: dict):
        raise NotImplementedError

    def forward(self, token_ids):
        raise NotImplementedError
