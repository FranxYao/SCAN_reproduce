import torch
import numpy as np 

from torch import nn 
from torch.optim import Adam
from transformers import (
  BertModel, BertConfig, AdamW, get_constant_schedule_with_warmup)

from frtorch import FRModel
from frtorch import torch_model_utils as tmu
from frtorch import LSTMEncoder, LSTMDecoder

class Seq2seqParseModel(nn.Module):
  def __init__(self, 
               pad_id,
               start_id,
               max_dec_len, 
               tgt_vocab_size, 
               embedding_size, 
               state_size,
               lstm_layers=1,
               dropout=0.,
               device='cpu'
               ):
    """Pytorch seq2seq model with Bert encoder"""
    super(Seq2seqParseModel, self).__init__()
    self.pad_id = pad_id
    self.device = device
    self.lstm_layers = lstm_layers

    self.encoder = BertModel.from_pretrained('bert-base-uncased')
    self.enc_config = BertConfig.from_pretrained('bert-base-uncased')

    self.tgt_embeddings = nn.Embedding(
      tgt_vocab_size, embedding_size)
    self.dec_state_init_proj = nn.Linear(
      self.enc_config.hidden_size, lstm_layers * 2 * state_size)
    self.decoder = LSTMDecoder(pad_id=pad_id,
                               start_id=start_id,
                               vocab_size=tgt_vocab_size,
                               max_dec_len=max_dec_len,
                               embedding_size=embedding_size,
                               state_size=state_size,
                               mem_state_size=self.enc_config.hidden_size,
                               lstm_layers=lstm_layers,
                               dropout=dropout
                               )
    return 

  def bridge(self, enc_state):
    """Map encoder sentence representation to decoder initial state
    """
    dec_state = self.dec_state_init_proj(enc_state)
    batch_size = enc_state.size(0)
    dec_state_size = dec_state.size(-1) // 2
    dec_state = dec_state.view(batch_size, self.lstm_layers, 2, dec_state_size)
    dec_state = (
      dec_state[:, :, 0, :].view(self.lstm_layers, batch_size, dec_state_size), 
      dec_state[:, :, 1, :].view(self.lstm_layers, batch_size, dec_state_size))
    return dec_state

  def forward(self, src, src_attn_mask, tgt):
    """
    Args:
      src: size=[batch, max_src_len]
      src_attn_mask: size=[batch, max_src_len]
      tgt: size=[batch, max_tgt_len]

    Returns:
      loss
      loss_lm
      acc
    """
    # enc_outputs: [B, T, S], enc_state: [B, S]
    enc_outputs_ = self.encoder(src, src_attn_mask)
    enc_outputs = enc_outputs_[0]
    enc_state = enc_outputs_[1]

    dec_inputs = self.tgt_embeddings(tgt[:, :-1])
    dec_targets = tgt[:, 1:]
    dec_init_state = self.bridge(enc_state)

    log_prob, predictions = self.decoder.decode_train(
      dec_init_state, dec_inputs, dec_targets, 
      mem_emb=enc_outputs, mem_mask=src_attn_mask, return_attn=False)

    dec_mask = dec_targets != self.pad_id
    acc = ((predictions == dec_targets) * dec_mask).sum().float() 
    acc /= dec_mask.sum().float()

    loss_lm = -log_prob
    loss = loss_lm
    return loss, loss_lm, acc

  def predict(self, src, src_attn_mask):
    """
    Args:
      src: size=[batch, max_src_len]
      src_attn_mask: size=[batch, max_src_len]

    Returns:
      predictions: size=[batch, max_dec_len]
    """
    enc_outputs_ = self.encoder(src, src_attn_mask)
    enc_outputs = enc_outputs_[0]
    enc_state = enc_outputs_[1]
    dec_init_state = self.bridge(enc_state)

    predictions = self.decoder.decode_predict(
      dec_init_state, self.tgt_embeddings, 
      mem_emb=enc_outputs, mem_mask=src_attn_mask, return_attn=False)
    return predictions

class Seq2seqParse(FRModel):
  """"""
  def __init__(self, 
               pad_id,
               end_id,
               model, 
               enc_learning_rate=1e-4,
               dec_learning_rate=1e-3,
               device='cpu',
               validation_criteria='exact_match'
               ):
    """FRTorch seq2seq wrapper"""
    super(Seq2seqParse, self).__init__()
    self.device = device
    self.pad_id = pad_id
    self.end_id = end_id
    self.model = model
    
    # self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
    enc_params, dec_params = [], [] 
    for name, param in self.model.named_parameters():
      if('encoder' in name):
        print('encoder param %s' % name)
        enc_params.append(param)
      else: 
        print('decoder param %s' % name)
        dec_params.append(param)
    self.enc_optim = AdamW(enc_params, lr=enc_learning_rate)
    self.dec_optim = Adam(dec_params, lr=dec_learning_rate)

    self.validation_scores = ['exact_match', 'loss_lm', 'acc']
    self.log_info = ['loss', 'acc', 'loss_lm',]
    self.validation_criteria = validation_criteria
    return 

  def train_step(self, batch, n_iter, ei, bi):
    """Single step training
    """
    self.model.zero_grad()

    loss, loss_lm, acc = self.model(batch['questions'].to(self.device),
                                    batch['attention_masks'].to(self.device), 
                                    batch['queries'].to(self.device)
                                    )
    loss.backward()
    self.enc_optim.step()
    self.dec_optim.step()

    out_dict = {'loss': loss.item(), 
                'loss_lm': loss_lm.item(),
                'acc': acc.item()
                }
    return out_dict

  def inspect_step(self, batch, out_dict, n_iter, ei, bi, dataset):
    """Single step inspection during training. 
    """
    return 

  def val_step(self, batch, n_iter, ei, bi, dataset):
    """Single step validation
    """
    with torch.no_grad():
      loss, loss_lm, acc = self.model(batch['questions'].to(self.device),
                                      batch['attention_masks'].to(self.device), 
                                      batch['queries'].to(self.device)
                                      )
      predictions = self.model.predict(batch['questions'].to(self.device),
                                       batch['attention_masks'].to(self.device)
                                       )
      em = 0
      predictions = tmu.to_np(predictions)
      targets = tmu.to_np(batch['queries'])
      for pi, ti in zip(predictions, targets):
        pi_ = dataset.convert_ids_to_string(pi)
        ti_ = dataset.convert_ids_to_string(ti)
        if(pi_ == ti_): em += 1
      batch_size = predictions.shape[0]
    exact_match = float(em) / batch_size

    out_dict = {'exact_match': exact_match, 
                'loss': loss.item(),
                'loss_lm': loss_lm.item(),
                'acc': acc.item(),
                }
    return out_dict

  def val_end(self, outputs, n_iter, ei, bi, dataset, mode, output_path_base):
    return 