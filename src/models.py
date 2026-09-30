"""The two required models, plus the Luong-attention ablation.

All share the same encoder (2-layer bidirectional LSTM) and the same decoder LSTM,
so the only architectural difference is how the decoder sees the source:

* Seq2Seq          — only through the encoder's final hidden/cell states
                     (a fixed-size "thought vector").
* BahdanauSeq2Seq  — the Attention-LSTM: before each target word, the previous
                     decoder state attends over all encoder states (additive
                     attention, Bahdanau et al., 2015).
* AttnSeq2Seq      — ablation: Luong "general" attention computed after the decoder
                     step, with input feeding (Luong et al., 2015). Its attention
                     collapsed onto the sentence-final token (see REPORT §4.3).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


class Encoder(nn.Module):
    def __init__(self, vocab, emb, hidden, layers, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=C.PAD)
        self.rnn = nn.LSTM(emb, hidden // 2, layers, batch_first=True,
                           bidirectional=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.layers = layers

    def forward(self, src, src_len):
        x = self.drop(self.emb(src))
        packed = nn.utils.rnn.pack_padded_sequence(x, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, c) = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=src.size(1))

        # (layers*2, B, H/2) → (layers, B, H): concatenate the two directions
        def merge(s):
            s = s.view(self.layers, 2, s.size(1), s.size(2))
            return torch.cat([s[:, 0], s[:, 1]], dim=-1)

        return out, (merge(h), merge(c))


class Seq2Seq(nn.Module):
    """Basic encoder–decoder LSTM (Sutskever et al., 2014; Cho et al., 2014)."""
    uses_attention = False

    def __init__(self, src_vocab, tgt_vocab, emb=C.EMB_DIM, hidden=C.HIDDEN,
                 layers=C.LAYERS, dropout=C.DROPOUT):
        super().__init__()
        self.config = dict(src_vocab=src_vocab, tgt_vocab=tgt_vocab, emb=emb,
                           hidden=hidden, layers=layers, dropout=dropout)
        self.encoder = Encoder(src_vocab, emb, hidden, layers, dropout)
        self.tgt_emb = nn.Embedding(tgt_vocab, emb, padding_idx=C.PAD)
        self.decoder = nn.LSTM(self._dec_input(emb, hidden), hidden, layers,
                               batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, tgt_vocab)

    def _dec_input(self, emb, hidden):
        return emb

    def encode(self, src, src_len):
        return self.encoder(src, src_len)

    def forward(self, src, src_len, tgt_in):
        """Teacher-forced training pass. tgt_in = [BOS, y1, ..., y_{n-1}]."""
        _, state = self.encode(src, src_len)
        y, _ = self.decoder(self.drop(self.tgt_emb(tgt_in)), state)
        return self.out(self.drop(y)), None

    # -- incremental decoding (inference) ------------------------------------
    def init_decode(self, src, src_len):
        enc_out, state = self.encode(src, src_len)
        return {"state": state}

    def decode_step(self, tok, cache):
        y, cache["state"] = self.decoder(self.tgt_emb(tok).unsqueeze(1), cache["state"])
        return self.out(y.squeeze(1)), None

    @staticmethod
    def reorder_cache(cache, idx):
        out = {}
        for k, v in cache.items():
            if k == "state":
                out[k] = tuple(s.index_select(1, idx) for s in v)
            else:
                out[k] = v.index_select(0, idx)
        return out


class LuongAttention(nn.Module):
    """score(h_t, h_s) = h_t^T W_a h_s  (Luong "general")."""

    def __init__(self, hidden):
        super().__init__()
        self.W = nn.Linear(hidden, hidden, bias=False)

    def forward(self, query, keys, mask):
        # query (B, T, H); keys (B, S, H); mask (B, S) True = real token
        scores = torch.bmm(query, self.W(keys).transpose(1, 2))           # (B, T, S)
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, keys), weights


class AttnSeq2Seq(Seq2Seq):
    """Encoder–decoder LSTM with global attention and input feeding."""
    uses_attention = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = self.config["hidden"]
        self.attn = LuongAttention(hidden)
        self.combine = nn.Linear(2 * hidden, hidden)   # h̃_t = tanh(W_c [c_t; h_t])

    def _dec_input(self, emb, hidden):
        return emb + hidden                               # input feeding: [y_{t-1}; h̃_{t-1}]

    def _step(self, emb_t, feed, state, enc_out, mask):
        y, state = self.decoder(torch.cat([emb_t, feed], -1).unsqueeze(1), state)
        ctx, w = self.attn(y, enc_out, mask)
        h_tilde = torch.tanh(self.combine(torch.cat([ctx, y], -1))).squeeze(1)
        return h_tilde, state, w.squeeze(1)

    def forward(self, src, src_len, tgt_in):
        enc_out, state = self.encode(src, src_len)
        mask = src != C.PAD
        emb = self.drop(self.tgt_emb(tgt_in))
        feed = emb.new_zeros(src.size(0), self.config["hidden"])
        outs, attns = [], []
        for t in range(tgt_in.size(1)):
            feed, state, w = self._step(emb[:, t], feed, state, enc_out, mask)
            feed = self.drop(feed)
            outs.append(feed)
            attns.append(w)
        return self.out(torch.stack(outs, 1)), torch.stack(attns, 1)

    def init_decode(self, src, src_len):
        enc_out, state = self.encode(src, src_len)
        return {"state": state, "enc_out": enc_out, "mask": src != C.PAD,
                "feed": enc_out.new_zeros(src.size(0), self.config["hidden"])}

    def decode_step(self, tok, cache):
        h, cache["state"], w = self._step(self.tgt_emb(tok), cache["feed"], cache["state"],
                                          cache["enc_out"], cache["mask"])
        cache["feed"] = h
        return self.out(h), w


class BahdanauAttention(nn.Module):
    """score(s_{t-1}, h_j) = v^T tanh(W_q s_{t-1} + W_k h_j)  (Bahdanau et al., 2015)."""

    def __init__(self, hidden):
        super().__init__()
        self.W_q = nn.Linear(hidden, hidden, bias=False)
        self.W_k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, query, keys, keys_proj, mask):
        # query (B, H); keys / keys_proj (B, S, H) — keys_proj = W_k keys, computed once per sentence
        scores = self.v(torch.tanh(keys_proj + self.W_q(query).unsqueeze(1))).squeeze(-1)   # (B, S)
        weights = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        return torch.bmm(weights.unsqueeze(1), keys).squeeze(1), weights


class BahdanauSeq2Seq(Seq2Seq):
    """Encoder–decoder LSTM with additive attention: the previous decoder state
    chooses where to look *before* the next word is generated, and the context
    vector is an input to the decoder LSTM."""
    uses_attention = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden, emb = self.config["hidden"], self.config["emb"]
        self.attn = BahdanauAttention(hidden)
        self.pre_out = nn.Linear(hidden + hidden + emb, hidden)   # [s_t; c_t; y_{t-1}] → output layer

    def _dec_input(self, emb, hidden):
        return emb + hidden                                        # [y_{t-1}; c_t]

    def _step(self, emb_t, state, enc_out, keys_proj, mask):
        ctx, w = self.attn(state[0][-1], enc_out, keys_proj, mask)
        y, state = self.decoder(torch.cat([emb_t, ctx], -1).unsqueeze(1), state)
        h = torch.tanh(self.pre_out(torch.cat([y.squeeze(1), ctx, emb_t], -1)))
        return h, state, w

    def forward(self, src, src_len, tgt_in):
        enc_out, state = self.encode(src, src_len)
        mask, keys_proj = src != C.PAD, self.attn.W_k(enc_out)
        emb = self.drop(self.tgt_emb(tgt_in))
        outs, attns = [], []
        for t in range(tgt_in.size(1)):
            h, state, w = self._step(emb[:, t], state, enc_out, keys_proj, mask)
            outs.append(h)
            attns.append(w)
        return self.out(self.drop(torch.stack(outs, 1))), torch.stack(attns, 1)

    def init_decode(self, src, src_len):
        enc_out, state = self.encode(src, src_len)
        return {"state": state, "enc_out": enc_out, "keys_proj": self.attn.W_k(enc_out), "mask": src != C.PAD}

    def decode_step(self, tok, cache):
        h, cache["state"], w = self._step(self.tgt_emb(tok), cache["state"], cache["enc_out"],
                                          cache["keys_proj"], cache["mask"])
        return self.out(h), w


MODELS = {"seq2seq": Seq2Seq, "attention": AttnSeq2Seq, "bahdanau": BahdanauSeq2Seq}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -- decoding -----------------------------------------------------------------
def blocked_tokens(toks, n=C.NO_REPEAT_NGRAM):
    """Tokens that may not come next: the previous token (no immediate repeats) and any token
    that would complete an n-gram already present in `toks` (as in no_repeat_ngram_size)."""
    if not toks:
        return set()
    banned = {toks[-1]}
    if n and len(toks) >= n - 1:
        prefix = tuple(toks[len(toks) - n + 1:])
        for i in range(len(toks) - n + 1):
            if tuple(toks[i:i + n - 1]) == prefix:
                banned.add(toks[i + n - 1])
    banned.discard(C.EOS)
    return banned


def remember(toks, seen, t, n=C.NO_REPEAT_NGRAM):
    """Append token t and index the n-gram it completes (prefix → possible next tokens)."""
    toks.append(t)
    if n and len(toks) >= n:
        seen.setdefault(tuple(toks[-n:-1]), set()).add(t)


def blocked_next(toks, seen, n=C.NO_REPEAT_NGRAM):
    """Incremental version of blocked_tokens for batched greedy decoding."""
    banned = {toks[-1]} | seen.get(tuple(toks[-(n - 1):]), set()) if n and len(toks) >= n - 1 else {toks[-1]}
    banned.discard(C.EOS)
    return banned


@torch.no_grad()
def greedy_decode(model, src, src_len, max_len=C.MAX_DECODE_LEN, block_repeats=C.BLOCK_REPEATS):
    """Batched greedy decoding. Returns token ids (B, T) and, for the attention
    model, attention weights (B, T, S)."""
    model.eval()
    cache = model.init_decode(src, src_len)
    B = src.size(0)
    tok = torch.full((B,), C.BOS, dtype=torch.long, device=src.device)
    done = torch.zeros(B, dtype=torch.bool, device=src.device)
    out, attns = [], []
    hist, seen = [[] for _ in range(B)], [{} for _ in range(B)]   # per-sentence tokens and n-gram index
    limit = min(max_len, int(src_len.max()) * 2 + 10)
    for _ in range(limit):
        logits, w = model.decode_step(tok, cache)
        if block_repeats and out:
            rows, cols = [], []
            for b in range(B):
                if hist[b]:
                    for t in blocked_next(hist[b], seen[b]):
                        rows.append(b)
                        cols.append(t)
            if rows:
                logits[rows, cols] = float("-inf")
        tok = logits.argmax(-1).masked_fill(done, C.PAD)
        out.append(tok)
        if block_repeats:
            for b, t in enumerate(tok.tolist()):
                if t != C.PAD:
                    remember(hist[b], seen[b], t)
        if w is not None:
            attns.append(w)
        done |= tok == C.EOS
        if done.all():
            break
    ids = torch.stack(out, 1)
    return ids, (torch.stack(attns, 1) if attns else None)


@torch.no_grad()
def beam_decode(model, src, src_len, beam=5, max_len=C.MAX_DECODE_LEN, alpha=0.7, block_repeats=C.BLOCK_REPEATS):
    """Beam search for one sentence (src is (1, S)). Length-normalised with
    GNMT penalty ((5+len)/6)^alpha. Returns (ids list, attention (T, S) or None)."""
    model.eval()
    cache = model.init_decode(src, src_len)
    device = src.device
    beams = [([], 0.0, [])]                # (tokens, logprob, attention rows)
    finished = []
    limit = min(max_len, int(src_len.max()) * 2 + 10)
    for _ in range(limit):
        last = torch.tensor([b[0][-1] if b[0] else C.BOS for b in beams], device=device)
        logits, w = model.decode_step(last, cache)
        logp = F.log_softmax(logits, -1)
        cand = []
        for i, (toks, score, att) in enumerate(beams):
            if block_repeats:
                ban = blocked_tokens(toks)
                if ban:
                    logp[i, list(ban)] = float("-inf")
            top = logp[i].topk(beam)
            for lp, t in zip(top.values.tolist(), top.indices.tolist()):
                cand.append((toks + [t], score + lp, att + ([w[i].cpu()] if w is not None else []), i))
        cand.sort(key=lambda x: x[1], reverse=True)
        beams, parents = [], []
        for toks, score, att, parent in cand:
            if toks[-1] == C.EOS:
                finished.append((toks, score / (((5 + len(toks)) / 6) ** alpha), att))
            else:
                beams.append((toks, score, att))
                parents.append(parent)
            if len(beams) == beam:
                break
        if len(finished) >= beam or not beams:
            break
        cache = model.reorder_cache(cache, torch.tensor(parents, device=device))
    if not finished:
        finished = [(t, s / (((5 + len(t)) / 6) ** alpha), a) for t, s, a in beams]
    toks, _, att = max(finished, key=lambda x: x[1])
    return toks, (torch.stack(att) if att else None)
