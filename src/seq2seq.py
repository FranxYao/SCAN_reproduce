                                              """Sequence to sequence"""
import torch 

import numpy as np 

from torch import nn 
from torch.optim import Adam

from argparse import ArgumentParser
from frtorch import FRModel, str2bool
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
               lstm_layers,
               dropout,
               lambda_align=1.0, 
               device='cpu',
               ):
    """Pytorch seq2seq model"""
    super().__init__()
    self.pad_id = pad_id
    self.device = device
    self.lambda_align = lambda_align

    self.src_embeddings = nn.Embedding(
      src_vocab_size, embedding_size)
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

  def forward(self, src, tgt, alignment=None):
    enc_lens = tmu.seq_to_lens(src, self.pad_id).to('cpu')
    src = self.src_embeddings(src)
    enc_outputs, enc_state = self.encoder(src, enc_lens)
    enc_max_len = enc_outputs.size(1)
    dec_inputs = self.tgt_embeddings(tgt[:, :-1])
    dec_targets = tgt[:, 1:]
    enc_mask = tmu.length_to_mask(enc_lens, enc_max_len).to(self.device)

    log_prob, predictions, attn_dist, pred_dist = self.decoder.decode_train(
      enc_state, dec_inputs, dec_targets, 
      mem_emb=enc_outputs, mem_mask=enc_mask, return_attn=True)
    dec_mask = dec_targets != self.pad_id
    acc = ((predictions == dec_targets) * dec_mask).sum().float() 
    acc /= dec_mask.sum().float()
    loss_lm = -log_prob
    loss = loss_lm

    if(alignment is not None):
      loss_align = alignment[:, :, :enc_max_len] * (attn_dist + 1e-5).log()
      loss_align = loss_align.sum() / dec_mask.sum()
      loss += self.lambda_align * loss_align
    else: loss_align = 0.
    return loss, loss_lm, loss_align, acc, attn_dist, pred_dist

  def predict(self, src):
    enc_lens = tmu.seq_to_lens(src, self.pad_id).to('cpu')
    src = self.src_embeddings(src)
    enc_outputs, enc_state = self.encoder(src, enc_lens)
    max_enc_len = enc_outputs.size(1)
    enc_mask = tmu.length_to_mask(enc_lens, max_enc_len).to(self.device)
    predictions, attn_dist, pred_dist = self.decoder.decode_predict(
      enc_state, self.tgt_embeddings, 
      mem_emb=enc_outputs, mem_mask=enc_mask, return_attn=True)
    return predictions, attn_dist, pred_dist

class Seq2seq(FRModel):

  def __init__(self, 
               learning_rate,
               device,
               pad_id,
               tgt_id2word,
               model, 
               output_path_fig,
               write_fig_after_epoch
               ):
    """FRTorch seq2seq wrapper"""
    super().__init__()
    self.learning_rate = learning_rate
    self.device = device
    self.pad_id = pad_id
    self.tgt_id2word = tgt_id2word
    self.model = model
    self.output_path_fig = output_path_fig
    self.write_fig_after_epoch = write_fig_after_epoch
    
    self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

    self.validation_scores = ['exact_match', 'loss_lm', 'acc']
    self.log_info = ['loss', 'acc', 'loss_lm', 'loss_align']
    self.validation_criteria = 'exact_match'
    return 

  def train_step(self, batch, n_iter, ei, bi):
    """
    Returns
    """
    self.model.zero_grad()
    if('alignment' in batch): alignment = batch['alignment']
    else: alignment = None
    loss, loss_lm, loss_align, acc, attn_dist, pred_dist = self.model(batch['src'], 
                           batch['tgt'],
                           alignment
                           )
    loss.backward()
    self.optimizer.step()

    out_dict = {'loss': loss.item(), 
                'loss_lm': loss_lm.item(),
                'loss_align': loss_align.item(),
                'acc': acc.item(), 
                'attn_dist': tmu.to_np(attn_dist)}
    return out_dict

  def inspect_step(self, batch, out_dict, n_iter, ei, bi, dataset):
    # write random 3 src-tgt pair 
    attn_dist = out_dict['attn_dist']
    batch_size = attn_dist.shape[0]
    visualized_index = np.random.choice(batch_size, 3, False)
    src = tmu.to_np(batch['src'])
    tgt = tmu.to_np(batch['tgt'])
    if(ei >= self.write_fig_after_epoch):
      for bi in visualized_index:
        slen_src = tmu.seq_to_lens(batch['src'][bi])
        # print(bi, slen_src)
        # print(src.shape)
        src_s = [dataset.src_id2word[i] for i in src[bi][: slen_src]]
        slen_tgt = tmu.seq_to_lens(batch['tgt'][bi])
        tgt_s = [dataset.tgt_id2word[i] for i in tgt[bi][1: slen_tgt]]

        attn_dist_bi = attn_dist[bi, : slen_tgt - 1, : slen_src]

        # print(len(src), len(tgt), attn_dist_bi.shape)
        fpath = self.output_path_fig + 'train_e%d/%d'  % (ei, batch['idx'][bi])
        # print(len(src_s), len(tgt_s), attn_dist_bi.shape)
        tmu.save_attn_figure(src_s, tgt_s, attn_dist_bi, fpath)
    return 

  def val_step(self, batch, n_iter, ei, bi):
    """
    Returns:
      exact_match: Need update
      loss:
      acc:
      predictions:
    """
    with torch.no_grad():
      loss, loss_lm, loss_align, acc, attn_dist_ref, pred_dist_ref =\
        self.model(batch['src'], batch['tgt'])
      predictions, attn_dist, pred_dist = self.model.predict(batch['src'])
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
                'loss_lm': loss_lm.item(),
                'loss_align': loss_align.item(),
                'acc': acc.item(),
                'tgt_str': tgt_str, 
                'predictions': pred_str, 
                'attn_dist_ref': tmu.to_np(attn_dist_ref), 
                'attn_dist_pred': tmu.to_np(attn_dist), 
                'pred_dist_ref': tmu.to_np(pred_dist_ref), 
                'pred_dist': tmu.to_np(pred_dist)}
    return out_dict

  # @staticmethod
  # def add_model_specific_args(parent_parser):
  #   parser = ArgumentParser(parents=[parent_parser], add_help=False)
  #   parser.add_argument(
  #     "--lstm_layers", default=1, type=int)
  #   parser.add_argument(
  #     "--lstm_bidirectional", type=str2bool, 
  #     nargs='?', const=True, default=True)
  #   return parser
