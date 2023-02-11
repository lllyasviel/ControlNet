import torch
import einops

import ldm.modules.encoders.modules
import ldm.modules.attention

from transformers import logging
from ldm.modules.attention import default


def disable_verbosity():
    logging.set_verbosity_error()
    print('logging improved.')
    return


def enable_sliced_attention():
    ldm.modules.attention.CrossAttention.forward = _hacked_sliced_attentin_forward
    print('Enabled sliced_attention.')
    return


def hack_everything(clip_skip=0):
    disable_verbosity()
    ldm.modules.encoders.modules.FrozenCLIPEmbedder.forward = _hacked_clip_forward
    ldm.modules.encoders.modules.FrozenCLIPEmbedder.clip_skip = clip_skip
    print('Enabled clip hacks.')
    return


# Written by Lvmin
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


# Stolen from https://github.com/basujindal/stable-diffusion/blob/main/optimizedSD/splitAttention.py
def _hacked_sliced_attentin_forward(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)
    del context, x

    q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    limit = k.shape[0]
    att_step = 1
    q_chunks = list(torch.tensor_split(q, limit // att_step, dim=0))
    k_chunks = list(torch.tensor_split(k, limit // att_step, dim=0))
    v_chunks = list(torch.tensor_split(v, limit // att_step, dim=0))

    q_chunks.reverse()
    k_chunks.reverse()
    v_chunks.reverse()
    sim = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device)
    del k, q, v
    for i in range(0, limit, att_step):
        q_buffer = q_chunks.pop()
        k_buffer = k_chunks.pop()
        v_buffer = v_chunks.pop()
        sim_buffer = torch.einsum('b i d, b j d -> b i j', q_buffer, k_buffer) * self.scale

        del k_buffer, q_buffer
        # attention, what we cannot get enough of, by chunks

        sim_buffer = sim_buffer.softmax(dim=-1)

        sim_buffer = torch.einsum('b i j, b j d -> b i d', sim_buffer, v_buffer)
        del v_buffer
        sim[i:i + att_step, :, :] = sim_buffer

        del sim_buffer
    sim = einops.rearrange(sim, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(sim)
