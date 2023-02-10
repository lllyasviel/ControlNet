import torch
import einops

import ldm.modules.encoders.modules

from transformers import logging


def disable_verbosity():
    logging.set_verbosity_error()


def hack_everything(clip_skip=0):
    disable_verbosity()
    ldm.modules.encoders.modules.FrozenCLIPEmbedder.forward = _hacked_clip_forward
    ldm.modules.encoders.modules.FrozenCLIPEmbedder.clip_skip = clip_skip
    return


def _hacked_clip_forward(self, text):
    PAD = self.tokenizer.pad_token_id
    EOS = self.tokenizer.eos_token_id
    BOS = self.tokenizer.bos_token_id

    def tokenize(t):
        return self.tokenizer(t, truncation=False, add_special_tokens=False)["input_ids"]

    def transformer_encode(t):
        if self.clip_skip > 1:
            rt = self.transformer(input_ids=t, output_hidden_states=True)
            return self.transformer.text_model.final_layer_norm(rt.hidden_states[-self.clip_skip])
        else:
            return self.transformer(input_ids=t, output_hidden_states=False).last_hidden_state

    def split(x):
        return x[75 * 0: 75 * 1], x[75 * 1: 75 * 2], x[75 * 2: 75 * 3]

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    raw_tokens_list = tokenize(text)
    tokens_list = []

    for raw_tokens in raw_tokens_list:
        raw_tokens_123 = split(raw_tokens)
        raw_tokens_123 = [[BOS] + raw_tokens_i + [EOS] for raw_tokens_i in raw_tokens_123]
        raw_tokens_123 = [pad(raw_tokens_i, PAD, 77) for raw_tokens_i in raw_tokens_123]
        tokens_list.append(raw_tokens_123)

    tokens_list = torch.IntTensor(tokens_list).to(self.device)

    feed = einops.rearrange(tokens_list, 'b f i -> (b f) i')
    y = transformer_encode(feed)
    z = einops.rearrange(y, '(b f) i c -> b (f i) c', f=3)

    return z
