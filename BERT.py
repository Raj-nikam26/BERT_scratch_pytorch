#BERT.py
from model_parts import *
from config_bert import *
import torch.nn as nn
import torch
import torch.nn.functional as F

class BERTModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.word_embd=nn.Embedding(config.vocab_size,config.d_model)
        self.positional_embd=nn.Embedding(config.max_position_embeddings,config.d_model)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.d_model)
        self.layernorm = LayerNorm(config.d_model)
        self.encoder = EncoderStack(config.num_layers, config.d_model, config.n_heads)
        self.pooler=nn.Sequential(nn.Linear(config.d_model,config.d_model),nn.Tanh())
        self.mlm_dense=nn.Linear(config.d_model, config.d_model)
        self.mlm_act = nn.GELU()
        self.mlm_norm = LayerNorm(config.d_model)
        self.mlm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.nsp_classifier = nn.Linear(config.d_model, 2)
        self.dropout=nn.Dropout(0.1)
    
    def embed_in(self,x,token_type_ids=None):
        batch_size, seq_len = x.size()
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(x)
        words = self.word_embd(x)
        position_ids = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos = self.positional_embd(position_ids)
        type_emb = self.token_type_embeddings(token_type_ids)
        x = words + pos + type_emb
        return self.dropout(self.layernorm(x))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, mlm_labels=None, nsp_label=None):
        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)
        attn_mask = attention_mask.unsqueeze(1)
        embeddings = self.embed_in(input_ids, token_type_ids)
        encoder_output = self.encoder(embeddings, attn_mask)
        pooled_output = self.pooler(encoder_output[:, 0, :])
        mlm_hidden = self.mlm_norm(self.mlm_act(self.mlm_dense(encoder_output)))
        mlm_logits = F.linear(mlm_hidden, self.word_embd.weight, bias=self.mlm_bias)
        nsp_logits = self.nsp_classifier(pooled_output)
        loss = None
        if mlm_labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
        if nsp_label is not None:
            nsp_loss = nn.CrossEntropyLoss()(nsp_logits.view(-1, 2), nsp_label.view(-1))
            loss = loss + nsp_loss if loss is not None else nsp_loss
        return loss, mlm_logits, nsp_logits, encoder_output, pooled_output


cfg = BERT_config()
model = BERTModel(cfg)
input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
token_types = torch.zeros_like(input_ids)
attention_mask = torch.ones_like(input_ids)
mlm_labels = torch.full((2, 16), -100)
mlm_labels[0, 5] = input_ids[0, 5]
mlm_labels[1, 7] = input_ids[1, 7]
nsp_label = torch.tensor([0, 1])
out = model(input_ids, token_types, attention_mask, mlm_labels=mlm_labels, nsp_label=nsp_label)
print(out[0], out[1].shape, out[2].shape)