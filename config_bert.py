#config_bertpy
class BERT_config:
    def __init__(self,vocab_size=30522, d_model=768, num_layers=12, n_heads=12, max_position_embeddings=512, type_vocab_size=2):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size