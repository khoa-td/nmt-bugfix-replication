import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD, SOS, EOS = 0, 1, 2


class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_units):
        super().__init__()

        self.W_k = nn.Linear(enc_dim, attn_units)
        self.W_q = nn.Linear(dec_dim, attn_units)
        self.v = nn.Parameter(torch.randn(attn_units) * 0.1)

    def precompute(self, enc_out):  # Precompute  once for the entire source sentence to save time during decoding.              
        return self.W_k(enc_out)

    def forward(self, query, keys, enc_out, mask):
        query_proj = self.W_q(query).unsqueeze(1) # query: [B, dec_dim] -> query_proj: [B, 1, attn_units]

        scores = (self.v * torch.tanh(keys + query_proj)).sum(dim = -1) # Bahdanau: score = v^T * tanh(W_k * h_enc + W_q * s_dec)

        scores = scores.masked_fill(~mask, float("-inf")) # use -inf to skip <pad>
        weights = torch.softmax(scores, dim=1)

        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)
        return context, weights


class Seq2Seq(nn.Module):
    def __init__(self, cfg, src_vocab, tgt_vocab):
        super().__init__()
        self.cfg = cfg
        rnn = nn.LSTM if cfg["cell"] == "lstm" else nn.GRU
        U = cfg["units"]
        E = cfg["emb"]

        self.src_emb = nn.Embedding(src_vocab, E, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_vocab, E, padding_idx=PAD)
        self.drop = nn.Dropout(cfg["dropout"])

        enc_drop = cfg["dropout"] if cfg["enc_layers"] > 1 else 0.0
        self.encoder = rnn(E, U, num_layers= cfg["enc_layers"], batch_first= True, bidirectional= True, dropout= enc_drop)

        self.attn = BahdanauAttention(2 * U, U, cfg["attn_units"])

        dec_drop = cfg["dropout"] if cfg["dec_layers"] > 1 else 0.0
        self.decoder = rnn(E + 2 * U, U, num_layers=cfg["dec_layers"], batch_first=True, dropout=dec_drop)

        self.mix = nn.Linear(U + 2 * U, U)
        self.out = nn.Linear(U, tgt_vocab)

        for p in self.parameters():
            nn.init.uniform_(p, -0.04, 0.04)
        
        with torch.no_grad():
            self.src_emb.weight[PAD].zero_()
            self.tgt_emb.weight[PAD].zero_()

    def encode(self, src, src_len): # Encode the source sentence (buggy code) into a sequence of hidden states
        x = self.drop(self.src_emb(src))
        packed = pack_padded_sequence(x, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.encoder(packed)
        enc_out, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))

        mask = (src != PAD)
        keys = self.attn.precompute(enc_out)

        return enc_out, keys, mask

    def init_state(self, batch_size, device):               
        h = torch.zeros(self.cfg["dec_layers"], batch_size, self.cfg["units"], device=device)
        if self.cfg["cell"] == "lstm":
            return (h, h.clone())
        return h

    def decode_step(self, prev_tok, state, prev_ctx, enc_out, keys, mask): # Perform a single-step decoding (1 token)
        emb = self.drop(self.tgt_emb(prev_tok))
        x = self.drop(torch.cat([self.tgt_emb(prev_tok), prev_ctx], dim=-1)).unsqueeze(1)

        out, state = self.decoder(x, state)
        cell_out = out[:, 0]

        ctx, _ = self.attn(cell_out, keys, enc_out, mask)

        mixed = torch.tanh(self.mix(torch.cat([cell_out, ctx], dim=-1)))
        logits = self.out(mixed)

        return logits, state, ctx

    def forward(self, src, src_len, tgt_in):      
        enc_out, keys, mask = self.encode(src, src_len)
        B, T = tgt_in.shape

        state = self.init_state(B, src.device)
        ctx = torch.zeros(B, 2 * self.cfg["units"], device=src.device)

        logits = []
        for t in range(T):
            lg, state, ctx = self.decode_step(tgt_in[:, t], state, ctx, enc_out, keys, mask)
            logits.append(lg)
        
        return torch.stack(logits, 1) 


