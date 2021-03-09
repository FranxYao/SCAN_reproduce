"""Sequence to sequence"""
import torch 
from torch import nn 
from torch.optim import Adam

from argparse import ArgumentParser
from frtorch import FRModel, str2bool
from frtorch import torch_model_utils as tmu
from seq_models import LSTMEncoder, LSTMDecoder

class Seq2seqPosModel(nn.Module):
  def __init__(self, 
               word_dropout, 
               use_attention,
               pad_id,
               start_id,
               max_dec_len,
               src_vocab_size, 
               pos_size,
               tgt_vocab_size, 
               embedding_size, 
               state_size,
               lstm_layers,
               dropout,
               device
               ):
    """Pytorch seq2seq model"""
    super().__init__()
    self.pad_id = pad_id
    self.word_dropout = word_dropout
    self.use_attention = use_attention
    self.device = device

    self.src_embeddings = nn.Embedding(
      src_vocab_size, embedding_size)
    self.pos_embeddings = nn.Embedding(pos_size, embedding_size)
    self.tgt_embeddings = nn.Embedding(
      tgt_vocab_size, embedding_size)
    self.encoder = LSTMEncoder(state_size=state_size, 
                               lstm_layers=lstm_layers,
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
                               lstm_layers=lstm_layers,
                               dropout=dropout
                               )

    return 

  def forward(self, src, pos, tgt, ei):
    """"""
    # word dropout
    if(self.word_dropout):
      word_dropout_ratio = 1 - (ei + 1) % 0.1
      if(ei >= 10): word_dropout_ratio = 0.
      dropout_mask = torch.rand(src.size(), device=self.device) 
      dropout_mask = (dropout_mask > word_dropout_ratio).float().unsqueeze(-1)

    enc_lens = tmu.seq_to_lens(src, self.pad_id).to('cpu')
    enc_mask = (src != self.pad_id).to(self.device)
    src = self.src_embeddings(src) 

    if(self.word_dropout): src = src * dropout_mask

    pos = self.pos_embeddings(pos)
    enc_outputs, enc_state = self.encoder(src + pos, enc_lens)
    max_enc_size = enc_outputs.size(1)
    enc_mask = enc_mask[:, :max_enc_size]
    dec_inputs = self.tgt_embeddings(tgt[:, :-1])
    dec_targets = tgt[:, 1:]
    if(self.use_attention):
      log_prob, predictions = self.decoder.decode_train(
        enc_state, dec_inputs, dec_targets, None, enc_outputs, enc_mask)
    else:
      log_prob, predictions = self.decoder.decode_train(
        enc_state, dec_inputs, dec_targets)
    dec_mask = dec_targets != self.pad_id
    acc = ((predictions == dec_targets) * dec_mask).sum().float() 
    acc /= dec_mask.sum().float()
    loss = -log_prob
    return loss, acc

  def predict(self, src, pos):
    """"""
    # During prediction we do not use word dropout
    enc_lens = tmu.seq_to_lens(src, self.pad_id).to('cpu')
    enc_mask = (src != self.pad_id).to(self.device)
    src = self.src_embeddings(src)
    pos = self.pos_embeddings(pos)
    enc_outputs, enc_state = self.encoder(src, enc_lens)
    max_enc_size = enc_outputs.size(1)
    enc_mask = enc_mask[:, :max_enc_size]
    if(self.use_attention):
      predictions = self.decoder.decode_predict(
        enc_state, self.tgt_embeddings, None, enc_outputs, enc_mask)
    else: 
      predictions = self.decoder.decode_predict(enc_state, self.tgt_embeddings)
    return predictions

class Seq2seqPos(FRModel):

  def __init__(self, 
               learning_rate,
               device,
               pad_id,
               tgt_id2word
               ):
    """FRTorch seq2seq wrapper"""
    super().__init__()
    self.learning_rate = learning_rate
    self.device = device
    self.pad_id = pad_id
    self.tgt_id2word = tgt_id2word
    return 

  def build(self, model):
    """"""
    self.model = model

    self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

    self.validation_scores = ['exact_match', 'loss', 'acc']
    self.log_info = ['loss', 'acc']
    self.validation_criteria = 'exact_match'
    return 

  def train_step(self, batch, n_iter, ei, bi):
    """
    Returns
    """
    self.model.zero_grad()
    loss, acc = self.model(batch['src'],
                           batch['pos'],
                           batch['tgt'],
                           ei
                           )
    loss.backward()
    self.optimizer.step()

    out_dict = {'loss': loss.item(), 'acc': acc.item()}
    return out_dict

  def val_step(self, batch, n_iter, ei, bi):
    """
    Returns:
      exact_match: Need update
      loss:
      acc:
      predictions:
    """
    with torch.no_grad():
      loss, acc = self.model(batch['src'], 
                             batch['pos'],
                             batch['tgt'],
                             10
                             )
      predictions = self.model.predict(batch['src'], batch['pos'])
      tgt_lens = tmu.seq_to_lens(batch['tgt'])
      tgt_mask = (batch['tgt'] != self.pad_id)
      max_tgt_len = batch['tgt'].size(1)

    batch_size = batch['src'].size(0)
    tgt_str, pred_str = [], []
    em = 0
    for bi in range(batch_size):
      tgt = []
      for idx in tmu.to_np(batch['tgt'][bi][1:]):
        w = self.tgt_id2word[idx]
        if(w == '<END>'): break
        tgt.append(w)
      tgt = ' '.join(tgt)
      tgt_str.append(tgt)

      pred = []
      for idx in tmu.to_np(predictions[bi]):
        w = self.tgt_id2word[idx]
        if(w == '<END>'): break
        pred.append(w)
      pred = ' '.join(pred)
      pred_str.append(pred)
      if(tgt == pred): em += 1
    exact_match = float(em) / batch_size

    out_dict = {'exact_match': exact_match, 
                'loss': loss.item(),
                'acc': acc.item(),
                'tgt_str': tgt_str, 
                'predictions': pred_str}
    return out_dict

  @staticmethod
  def add_model_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--lstm_layers", default=1, type=int)
    parser.add_argument("--lstm_bidirectional", default=True, type=str2bool)
    parser.add_argument("--word_dropout", default=False, type=str2bool)
    parser.add_argument("--use_attention", default=False, type=str2bool)
    return parser
