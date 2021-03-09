"""Sequence to sequence"""
import torch 
from torch import nn 
from torch.optim import Adam

from argparse import ArgumentParser
from frtorch import FRModel
from frtorch import torch_model_utils as tmu
from seq_models import LSTMEncoder, LSTMDecoder

class Seq2seqModel(nn.Module):
  def __init__(self, 
               pad_id,
               start_id,
               max_dec_len,
               src_vocab_size, 
               tgt_vocab_size, 
               embedding_size, 
               state_size,
               dropout,
               device
               ):
    """Pytorch seq2seq model"""
    super().__init__()
    self.pad_id = pad_id

    self.src_embeddings = nn.Embedding(
      src_vocab_size, embedding_size)
    self.tgt_embeddings = nn.Embedding(
      tgt_vocab_size, embedding_size)
    self.encoder = LSTMEncoder(state_size=state_size, 
                               dropout=dropout, 
                               embedding_size=embedding_size,
                               device=device
                               )
    self.decoder = LSTMDecoder(pad_id=pad_id,
                               start_id=start_id,
                               vocab_size=tgt_vocab_size,
                               max_dec_len=max_dec_len,
                               embedding_size=embedding_size,
                               state_size=state_size,
                               dropout=dropout
                               )

    return 

  def forward(self, src, tgt):
    enc_lens = tmu.seq_to_lens(src, self.pad_id)
    src = self.src_embeddings(src)
    enc_outputs, enc_state = self.encoder(src, enc_lens)
    dec_inputs = self.tgt_embeddings(tgt[:, :-1])
    dec_targets = tgt[:, 1:]
    log_prob, predictions = self.decoder.decode_train(
      enc_state, dec_inputs, dec_targets)
    dec_mask = dec_targets != self.pad_id
    acc = ((predictions == dec_targets) * dec_mask).sum().float() 
    acc /= dec_mask.sum().float()
    loss = -log_prob
    return loss, acc

  def predict(self, src):
    enc_lens = tmu.seq_to_lens(src, self.pad_id)
    src = self.src_embeddings(src)
    enc_outputs, enc_state = self.encoder(src, enc_lens)
    predictions = self.decoder.decode_predict(enc_state, self.tgt_embeddings)
    return predictions

class Seq2seq(FRModel):

  def __init__(self, 
               pad_id,
               start_id,
               max_dec_len,
               src_vocab_size, 
               tgt_vocab_size, 
               embedding_size, 
               state_size, 
               dropout, 
               learning_rate,
               device
               ):
    """FRTorch seq2seq wrapper"""
    super().__init__()
    self.pad_id = pad_id
    self.start_id = start_id
    self.max_dec_len = max_dec_len
    self.src_vocab_size = src_vocab_size
    self.tgt_vocab_size = tgt_vocab_size
    self.embedding_size = embedding_size
    self.state_size = state_size
    self.dropout = dropout
    self.learning_rate = learning_rate
    self.device = device
    return 

  def build(self):
    """"""
    self.model = Seq2seqModel(self.pad_id, 
                              self.start_id,
                              self.max_dec_len,
                              self.src_vocab_size, 
                              self.tgt_vocab_size, 
                              self.embedding_size, 
                              self.state_size,
                              self.dropout,
                              self.device
                              )

    self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

    self.validation_scores = ['exact_match', 'loss', 'acc']
    self.log_info = ['loss', 'acc']
    self.validation_criteria = 'exact_match'
    return 

  def train_step(self, batch, n_iter, ei, bi):
    self.model.zero_grad()
    # print(self.model.device)
    loss, acc = self.model(batch['src'].to(self.device), 
                           batch['tgt'].to(self.device))
    loss.backward()
    self.optimizer.step()

    out_dict = {'loss': loss.item(), 'acc': acc.item()}
    return out_dict

  def val_step(self, batch, n_iter, ei, bi):
    with torch.no_grad():
      loss, acc = self.model(batch['src'].to(self.device), 
                             batch['tgt'].to(self.device))
      predictions = self.model.predict(batch['src'].to(self.device))
      tgt_lens = tmu.seq_to_lens(batch['tgt'])
      tgt_mask = batch['tgt'] != self.pad_id
      max_tgt_len = batch['tgt'].size(1)
      exact_match = predictions[:, :max_tgt_len] == batch['tgt']
      exact_match = (exact_match * tgt_mask).sum(1)
      exact_match = (exact_match == tgt_lens).sum()
      batch_size = tgt_mask.size(0)
      exact_match = float(exact_match) / float(batch_size)
    out_dict = {'exact_match': exact_match, 
                'loss': loss,
                'acc': acc,
                'predictions': predictions}
    return out_dict

  @staticmethod
  def add_model_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    return parser
