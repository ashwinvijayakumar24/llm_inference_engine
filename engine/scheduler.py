"""Generation loop and request scheduler (single-request first, batched in Phase 4)."""


class Scheduler:
    def __init__(self, model, cache, sampler):
        raise NotImplementedError

    def generate(self, token_ids: list, max_tokens: int, **sample_kwargs):
        raise NotImplementedError
