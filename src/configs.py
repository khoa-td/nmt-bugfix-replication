# List of 10 experimental configurations for the Seq2Seq model (based on the paper by Tufano et al.)
# - emb: size of embedding vector
# - enc_layers: number of RNN layers in the Encoder (Bidirectional)
# - dec_layers: number of RNN layers in the Decoder
# - units: hidden state dimension of an RNN
# - cell: the type of RNN used (GRU or LSTM)
CONFIGS = {
    1:  {"emb": 256, "enc_layers": 1, "dec_layers": 2, "units": 256, "cell": "gru"},
    2:  {"emb": 256, "enc_layers": 1, "dec_layers": 2, "units": 256, "cell": "lstm"},
    3:  {"emb": 256, "enc_layers": 2, "dec_layers": 4, "units": 256, "cell": "gru"},
    4:  {"emb": 256, "enc_layers": 2, "dec_layers": 4, "units": 256, "cell": "lstm"},
    5:  {"emb": 256, "enc_layers": 2, "dec_layers": 4, "units": 512, "cell": "gru"},
    6:  {"emb": 256, "enc_layers": 2, "dec_layers": 4, "units": 512, "cell": "lstm"},
    7:  {"emb": 512, "enc_layers": 2, "dec_layers": 4, "units": 512, "cell": "gru"},
    8:  {"emb": 512, "enc_layers": 2, "dec_layers": 4, "units": 512, "cell": "lstm"},
    9:  {"emb": 512, "enc_layers": 1, "dec_layers": 2, "units": 256, "cell": "gru"},
    10: {"emb": 512, "enc_layers": 1, "dec_layers": 2, "units": 256, "cell": "lstm"},
}

BEST_CONFIG = 10

#Hyperparemters
COMMON = {
    "lr": 1e-4,
    "adam_eps": 8e-7,
    "dropout": 0.2,
    "batch_size": 32,
    "attn_units": 256,
}