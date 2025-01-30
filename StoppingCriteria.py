# implementation of a StoppingCriteria (inspired from https://github.com/huggingface/transformers/issues/22436)
# with the modification that we don't stop at the first occuring stoppingCriteria (###) but at a certain counter of the occuring stoppingCriteria 
# since we use (###) already a few times in the examples in the prompt


import torch
import transformers

class TokenStoppingCriteria(transformers.StoppingCriteria):

    def __init__(self, sentinel_token_ids: torch.LongTensor,
                 starting_idx: int, counter: int, stop_counter: int):
        transformers.StoppingCriteria.__init__(self)
        self.sentinel_token_ids = sentinel_token_ids
        self.starting_idx = starting_idx
        self.counter = counter
        self.stop_counter = stop_counter

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor) -> bool:
        self.counter = 0
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx:]
            # Can't unfold, output is still too tiny. Skip.
            if trimmed_sample.shape[-1] < self.sentinel_token_ids.shape[-1]:
                continue

            for window in trimmed_sample.unfold(0, self.sentinel_token_ids.shape[-1], 1):
                if torch.all(torch.eq(self.sentinel_token_ids, window)):
                    if self.counter == self.stop_counter:
                        return True
                    else:
                        self.counter += 1
        return False