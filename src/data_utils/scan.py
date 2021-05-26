import numpy as np 

from scan_parser import parse_unit_command
from argparse import ArgumentParser
from collections import Counter
from torch.utils.data import Dataset, DataLoader, random_split
from frtorch import torch_model_utils as tmu
from frtorch import str2bool

def pipeline(data: list, 
             word2id: dict, 
             add_start_end: bool = False, 
             dataset_type: str = ''):
  """
  sentence to list of index, pad or truncate sentence length, return np array

  NOTE: since there is no UNK, this pipeline will assume all words are seen in 
  the training set

  Args:
    data: a list of words, a word is a str

  Returns:
    data: np.ndarray, shape = [dataset_size, max_len]
  """
  print('Pipeline for %s' % dataset_type)
  max_len = max([len(d) for d in data])
  print('max_len = %d' % max_len)
  if(add_start_end): max_len += 2

  data_ = []
  for d in data:
    d_ = [word2id[di] for di in d]
    if(add_start_end): d_ = [word2id['<GOO>']] + d_ + [word2id['<END>']]
    d_ = tmu.pad_or_trunc_seq(d_, max_len)
    assert(len(d_) == max_len)
    data_.append(d_)

  data = np.array(data_)
  return data, max_len

def get_pos(data):
  pos = []
  for d in data:
    p = []
    for di in d:
      if(di in ['jump', 'run', 'look', 'turn', 'walk']): p.append('V')
      elif(di in ['around', 'opposite']): p.append('VC')
      elif(di in ['left', 'right']): p.append('Dir')
      elif(di in ['twice', 'thrice']): p.append('Num')
      elif(di in ['and', 'after']): p.append('Conj')
      else:
        raise NameError('%s not recognized' % di)
    pos.append(p)
  return pos

def process_alignment(alignment, src_max_len, tgt_max_len):
  """pad alignment to max length"""
  dataset_size = len(alignment)
  alignment_ = np.zeros(shape=[dataset_size, tgt_max_len, src_max_len])
  for i, a in enumerate(alignment):
    t_len, s_len = a.shape
    # print(a.shape)
    alignment_[i, :t_len, :s_len] = a
  return alignment_

def get_symbol_cnt(src, tgt):
  if('and' in src):
    src_ = src.split(' and ')
    assert(len(src_) == 2)

    tgt_ = []
    tgt_0 = parse_unit_command(src_[0].split(), add_cnt_symbol=True)
    tgt_.extend(tgt_0)
    tgt_.append('AND')
    tgt_1 = parse_unit_command(src_[1].split(), add_cnt_symbol=True)
    tgt_.extend(tgt_1)

  elif('after' in src):
    src_ = src.split(' after ')
    assert(len(src_) == 2)

    tgt_ = []
    tgt_0 = parse_unit_command(src_[1].split(), add_cnt_symbol=True)
    tgt_.extend(tgt_0)
    tgt_.append('AFTER')
    tgt_1 = parse_unit_command(src_[0].split(), add_cnt_symbol=True)
    tgt_.extend(tgt_1)

  else: # do not need seperator 
    src_ = src.split()
    tgt_ = parse_unit_command(src_, add_cnt_symbol=True)

  tgt_ret = tgt_
  return tgt_ret

def get_sep_align(src, tgt):
  """Add chunk seperator and construct alignment matrix for supervised attention

  Args:
    src: type=string. source command
    tgt: type=string. target command
  """ 
  if('and' in src):
    src_ = src.split(' and ')
    src_len = len(src.split())
    src_and_id = len(src_[0].split())
    assert(len(src_) == 2)

    tgt_ = []
    tgt_0 = parse_unit_command(src_[0].split())
    tgt_.extend(tgt_0)
    tgt_and_id = len(tgt_)
    tgt_.append('AND')
    tgt_1 = parse_unit_command(src_[1].split())
    tgt_.extend(tgt_1)
    tgt_len = len(tgt_)
    assert(' '.join(tgt_0 + tgt_1) == tgt)

    alignment = np.zeros(shape=[tgt_len, src_len])
    alignment[tgt_and_id, src_and_id] = 1
    alignment[tgt_and_id + 1:, src_and_id + 1:] = 1
    alignment[: tgt_and_id, :src_and_id] = 1
  elif('after' in src):
    src_ = src.split(' after ')
    src_after_id = len(src_[0].split())
    src_len = len(src.split())
    assert(len(src_) == 2)

    tgt_ = []
    tgt_0 = parse_unit_command(src_[1].split())
    tgt_.extend(tgt_0)
    tgt_after_id = len(tgt_)
    tgt_.append('AFTER')
    tgt_1 = parse_unit_command(src_[0].split())
    tgt_.extend(tgt_1)
    tgt_len = len(tgt_)
    assert(' '.join(tgt_0 + tgt_1) == tgt)

    alignment = np.zeros(shape=[tgt_len, src_len])
    alignment[tgt_after_id, src_after_id] = 1
    alignment[tgt_after_id + 1: , :src_after_id] = 1
    alignment[: tgt_after_id, src_after_id + 1: ] = 1
  else: # do not need seperator 
    src_ = src.split()
    tgt_ = parse_unit_command(src_)
    assert(' '.join(tgt_) == tgt)

    alignment = np.zeros(shape=[len(tgt_), len(src_)]) + 1.

  tgt_sep = tgt_
  return tgt_sep, alignment


class SCANDataset(Dataset):
  def __init__(self, 
               src, 
               pos, 
               tgt, 
               tgt_sep=None, 
               tgt_symbol_cnt=None, 
               alignment=None, 
               require_pos=False, 
               add_sep=False
               ):
    """
    Args:
      src: source sentences. size = [dataset_size, max_src_len]
      tgt: target sentences. size = [dataset_size, max_src_len]
    """
    super().__init__()
    self.src = src
    self.pos = pos
    self.tgt = tgt
    self.tgt_sep = tgt_sep
    self.tgt_symbol_cnt = tgt_symbol_cnt
    self.alignment = alignment
    self.require_pos = require_pos
    self.add_sep = add_sep
    return 

  def __len__(self):
    return len(self.src)

  def __getitem__(self, idx):
    instance = {'src': self.src[idx], 
                'pos': self.pos[idx], 
                'tgt': self.tgt[idx],
                'idx': idx}
    if(self.alignment is not None):
      instance['alignment'] = self.alignment[idx]
    if(self.tgt_sep is not None):
      instance['tgt_sep'] = self.tgt_sep[idx]
    if(self.tgt_symbol_cnt is not None):
      instance['tgt_symbol_cnt'] = self.tgt_symbol_cnt[idx]
    return instance

class SCANData(object):

  def __init__(self,
               split_name='random',
               batch_size=64,
               num_workers=0,
               require_pos=False,
               output_path_fig='',
               write_fig_after_epoch=10,
               add_sep=False,
               change_counter_to_symbol=False
               ):
    """
    Args:
      split_name: "random", "length", "length_no_new_command"
    """
    super().__init__()

    self.split_name = split_name
    self.num_workers = num_workers
    self.batch_size = batch_size
    self.require_pos = require_pos
    self.output_path_fig = output_path_fig
    self.write_fig_after_epoch = write_fig_after_epoch
    self.add_sep = add_sep
    self.change_counter_to_symbol = change_counter_to_symbol
    
    self.src_word2id = {'<PAD>': 0}
    self.src_id2word = {0: '<PAD>'}
    self.pos_word2id = {'<PAD>': 0, 'V': 1, 'VC': 2, 'Dir': 3, 'Num': 4, 'Conj': 5}
    self.pos_id2word = {}
    for w in self.pos_word2id:
      self.pos_id2word[self.pos_word2id[w]] = w
    self.tgt_word2id = {'<PAD>': 0, '<GOO>': 1, '<END>': 2}
    self.tgt_id2word = {0: '<PAD>', 1: '<GOO>', 2: '<END>'}
    return 

  def build(self):
    print('Preparing the data module ... ')
    if(self.split_name == 'random'):
      train_path = '../data/scan/simple_split/tasks_train_simple.txt'
      test_path = '../data/scan/simple_split/tasks_test_simple.txt'
    elif(self.split_name == 'length'):
      train_path = '../data/scan/length_split/tasks_train_length.txt'
      test_path = '../data/scan/length_split/tasks_test_length.txt'
    elif(self.split_name == 'length_trunc25'):
      train_path = '../data/scan/scan_new_splits/tasks_train_length_trunc25.txt'
      test_path = '../data/scan/scan_new_splits/tasks_test_length_trunc25.txt'
    elif(self.split_name == 'length_trunc27'):
      train_path = '../data/scan/scan_new_splits/tasks_train_length_trunc27.txt'
      test_path = '../data/scan/scan_new_splits/tasks_test_length_trunc27.txt'
    elif(self.split_name == 'length_trunc30'):
      train_path = '../data/scan/scan_new_splits/tasks_train_length_trunc30.txt'
      test_path = '../data/scan/scan_new_splits/tasks_test_length_trunc30.txt'
    elif(self.split_name == 'length_1shot'):
      train_path = '../data/scan/scan_new_splits/tasks_train_length_1shot.txt'
      test_path = '../data/scan/scan_new_splits/tasks_test_length_1shot.txt'
    elif(self.split_name == 'length_new_command'):
      train_path = '../data/scan/scan_new_splits/tasks_train_length_new_command.txt'
      test_path = '../data/scan/scan_new_splits/tasks_test_length_new_command.txt'
    else:
      raise NotImplementedError(
        'data split %s not implemented' % self.split_name)

    # split
    train_data = open(train_path).readlines()
    train_len = int(len(train_data) * 0.8)
    dev_len = len(train_data) - train_len
    train_data, dev_data = random_split(train_data, [train_len, dev_len])
    test_data = open(test_path).readlines()

    ## train
    train_src = [d.split(' OUT: ')[0][4:].split() for d in train_data] 
    train_pos = get_pos(train_src)
    train_tgt = [d.split(' OUT: ')[1].split() for d in train_data]

    if(self.add_sep):
      train_tgt_sep = []
      train_alignment = []
      for src, tgt in zip(train_src, train_tgt):
        tgt_sep, alignment = get_sep_align(' '.join(src), ' '.join(tgt))
        train_tgt_sep.append(tgt_sep)
        train_alignment.append(alignment)

    if(self.change_counter_to_symbol):
      train_tgt_symbol_cnt = []
      for src, tgt in zip(train_src, train_tgt):
        tgt_symbol_cnt = get_symbol_cnt(' '.join(src), ' '.join(tgt))
        train_tgt_symbol_cnt.append(tgt_symbol_cnt)

    src_word2id, src_id2word = tmu.build_vocab(
      train_src, start_id=len(self.src_word2id)) 
    self.src_word2id.update(src_word2id)
    self.src_id2word.update(src_id2word)
    self.src_vocab_size = len(self.src_word2id)

    # TODO: update the solution for the dictionary, currently on too messy
    if(self.add_sep):
      tgt_word2id, tgt_id2word = tmu.build_vocab(
        train_tgt_sep, start_id=len(self.tgt_word2id))
    elif(self.change_counter_to_symbol):
      tgt_word2id, tgt_id2word = tmu.build_vocab(
        train_tgt_symbol_cnt, start_id=len(self.tgt_word2id))
    else:
      tgt_word2id, tgt_id2word = tmu.build_vocab(
        train_tgt, start_id=len(self.tgt_word2id))

    self.tgt_word2id.update(tgt_word2id)
    self.tgt_id2word.update(tgt_id2word)
    self.tgt_vocab_size = len(self.tgt_word2id)

    train_src, train_src_max_len =\
      pipeline(train_src, self.src_word2id, False, 'train_src')
    train_pos, _ = pipeline(train_pos, self.pos_word2id, False, 'train_pos')
    train_tgt, train_tgt_max_len =\
      pipeline(train_tgt, self.tgt_word2id, True, 'train_tgt')

    if(self.add_sep):
      train_tgt_sep, train_tgt_sep_max_len =\
        pipeline(train_tgt_sep, self.tgt_word2id, True, 'train_tgt_sep')
      train_alignment = process_alignment(
        train_alignment, train_src_max_len, train_tgt_sep_max_len)
    else: 
      train_tgt_sep = None
      train_alignment = None

    if(self.change_counter_to_symbol):
      train_tgt_symbol_cnt, train_tgt_symbol_cnt_max_len =\
        pipeline(train_tgt_symbol_cnt, self.tgt_word2id, True, 'train_tgt_symbol_cnt')
    else: train_tgt_symbol_cnt = None

    self.train_dataset = SCANDataset(train_src, 
                                     train_pos, 
                                     train_tgt, 
                                     train_tgt_sep, 
                                     train_tgt_symbol_cnt, 
                                     train_alignment, 
                                     self.require_pos, 
                                     self.add_sep)

    ## dev 
    dev_src = [d.split(' OUT: ')[0][4:].split() for d in dev_data] 
    dev_pos = get_pos(dev_src)
    dev_tgt = [d.split(' OUT: ')[1].split() for d in dev_data]

    if(self.add_sep):
      dev_tgt_sep = []
      for src, tgt in zip(dev_src, dev_tgt):
        tgt_sep, _ = get_sep_align(' '.join(src), ' '.join(tgt))
        dev_tgt_sep.append(tgt_sep)

    if(self.change_counter_to_symbol):
      dev_tgt_symbol_cnt = []
      for src, tgt in zip(dev_src, dev_tgt):
        tgt_symbol_cnt = get_symbol_cnt(' '.join(src), ' '.join(tgt))
        dev_tgt_symbol_cnt.append(tgt_symbol_cnt)

    dev_src, _ = pipeline(dev_src, self.src_word2id, False, 'dev_src')
    dev_pos, _ = pipeline(dev_pos, self.pos_word2id, False, 'dev_pos')
    dev_tgt, dev_tgt_max_len =\
      pipeline(dev_tgt, self.tgt_word2id, True, 'dev_tgt')

    if(self.add_sep):
      dev_tgt_sep, dev_tgt_max_len =\
        pipeline(dev_tgt_sep, self.tgt_word2id, True, 'dev_tgt_sep')
      dev_alignment = None
    else: 
      dev_tgt_sep, dev_alignment = None, None

    if(self.change_counter_to_symbol):
      dev_tgt_symbol_cnt, dev_tgt_max_len = pipeline(
        dev_tgt_symbol_cnt, self.tgt_word2id, True, 'dev_tgt_symbol_cnt')
    else: dev_tgt_symbol_cnt = None
    
    self.dev_dataset = SCANDataset(dev_src, 
                                   dev_pos, 
                                   dev_tgt, 
                                   dev_tgt_sep,
                                   dev_tgt_symbol_cnt, 
                                   dev_alignment, 
                                   self.require_pos,
                                   self.add_sep)

    ## test
    test_src = [d.split(' OUT: ')[0][4:].split() for d in test_data] 
    test_pos = get_pos(test_src)
    test_tgt = [d.split(' OUT: ')[1].split() for d in test_data]

    if(self.add_sep):
      test_tgt_sep = []
      for src, tgt in zip(test_src, test_tgt):
        tgt_sep, _ = get_sep_align(' '.join(src), ' '.join(tgt))
        test_tgt_sep.append(tgt_sep)

    if(self.change_counter_to_symbol):
      test_tgt_symbol_cnt = []
      for src, tgt in zip(test_src, test_tgt):
        tgt_symbol_cnt = get_symbol_cnt(' '.join(src), ' '.join(tgt))
        test_tgt_symbol_cnt.append(tgt_symbol_cnt)

    test_src, _ = pipeline(test_src, self.src_word2id, False, 'test_src')
    test_pos, _ = pipeline(test_pos, self.pos_word2id, False, 'test_pos')
    test_tgt, test_tgt_max_len =\
      pipeline(test_tgt, self.tgt_word2id, True, 'test_tgt')

    if(self.add_sep):
      test_tgt_sep, test_tgt_max_len =\
        pipeline(test_tgt_sep, self.tgt_word2id, True, 'test_tgt_sep')
      test_alignment = None
    else:
      test_tgt_sep, test_alignment = None, None

    if(self.change_counter_to_symbol):
      test_tgt_symbol_cnt, test_tgt_max_len = pipeline(
        test_tgt_symbol_cnt, self.tgt_word2id, True, 'test_tgt_symbol_cnt')
    else: test_tgt_symbol_cnt = None
    
    self.test_dataset = SCANDataset(test_src, 
                                    test_pos, 
                                    test_tgt, 
                                    test_tgt_sep, 
                                    test_tgt_symbol_cnt, 
                                    test_alignment,
                                    self.require_pos,
                                    self.add_sep)

    self.max_dec_len =\
      max([train_tgt_max_len, dev_tgt_max_len, test_tgt_max_len]) + 1
    print('... Finished!')
    return 

  def train_dataloader(self):
    loader = DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=self.num_workers,
      drop_last=False,
      pin_memory=False
      )
    return loader

  def val_dataloader(self):
    loader = DataLoader(
      self.dev_dataset,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=self.num_workers,
      drop_last=False,
      pin_memory=True
      )
    return loader

  def test_dataloader(self):
    loader = DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=self.num_workers,
      drop_last=False,
      pin_memory=True
      )
    return loader

  def write_output_batch(self, fd, batch, out_dict, mode, e_id, b_id):
    batch_size = batch['src'].size(0)
    visualized_index = np.random.choice(batch_size, 1, False)
    vocab = [self.tgt_id2word[i].split('_')[-1] for i in range(len(self.tgt_id2word))]
    for bi in range(batch_size):
      slen = tmu.seq_to_lens(batch['src'][bi])
      src = [self.src_id2word[idx.item()] for idx in batch['src'][bi][:slen]]
      s = 'case id %d, src: %s' % (batch['idx'][bi], ' '.join(src)) + '\n'

      tgt = out_dict['tgt_str'][bi].split()
      tgt = [str(i) + ': ' + t for i, t in enumerate(tgt)]
      s += 'tgt: ' + out_dict['tgt_str'][bi] + '\n'
      s += 'pred: ' + out_dict['predictions'][bi] + '\n'

      pred = out_dict['predictions'][bi].split()
      pred = [str(i) + ': ' + t for i, t in enumerate(pred)]

      # slen = tmu.seq_to_lens(batch['tgt'][bi])
      # s += 'tgt: ' + ' '.join(
      #   self.tgt_id2word[idx.item()] for idx in batch['tgt'][bi][1:slen]) + '\n'
      # slen = tmu.seq_to_lens(out_dict['predictions'][bi])
      # s += 'pred: ' + ' '.join(
      #   self.tgt_id2word[idx.item()] for idx in out_dict['predictions'][bi][1:slen]) + '\n'
      fd.write(s)

      attn_pred = out_dict['attn_dist_pred'][bi, :len(pred), :slen]
      attn_ref = out_dict['attn_dist_ref'][bi, :len(tgt), :slen]
      pred_dist = out_dict['pred_dist'][bi, :len(tgt)]
      pred_dist_ref = out_dict['pred_dist_ref'][bi, :len(tgt)]
      # print(slen, len(tgt), attn.shape)
      # fpath_ref = self.output_path_fig + mode + '_e%d/ref_%d' % (e_id, batch['idx'][bi])
      # fpath_pred = self.output_path_fig + mode + '_e%d/pred_%d' % (e_id, batch['idx'][bi])
      fpath = self.output_path_fig + mode + '_e%d/ref_pred_%d' % (e_id, batch['idx'][bi])
      if(bi in visualized_index and e_id >= self.write_fig_after_epoch):
        # tmu.save_attn_figure(src, tgt, attn_ref, fpath_ref)
        # tmu.save_attn_figure(src, pred, attn_pred, fpath_pred)
        # tmu.save_two_attn_figure(src, pred, tgt, attn_pred, attn_ref, fpath)
        if(np.random.uniform() > 0.6):
          tmu.save_attn_pred_figure(src, pred, tgt, attn_pred, attn_ref, 
            pred_dist, pred_dist_ref, vocab, fpath)
    return 

  def write_output_full(self, batches, outputs, mode, e_id):
    fpath_src = self.output_path_fig + mode + '_e%d/src.pkl'
    fpath_tgt = self.output_path_fig + mode + '_e%d/tgt.pkl'
    fpath_pred = self.output_path_fig + mode + '_e%d/pred.pkl'
    fpath_attn_ref = self.output_path_fig + mode + '_e%d/attn_ref.pkl'
    fpath_attn_pred = self.output_path_fig + mode + '_e%d/attn_pred.pkl'
    for batch, out_dict in zip(batches, outputs):
      batch_size = batch['src'].size(0)
      for bi in range(batch_size):
        slen = tmu.seq_to_lens(batch['src'][bi])
        src = [self.src_id2word[idx.item()] for idx in batch['src'][bi][:slen]]
        
        tgt = out_dict['tgt_str'][bi].split()
        tgt = [str(i) + ': ' + t for i, t in enumerate(tgt)]

        pred = out_dict['predictions'][bi].split()
        pred = [str(i) + ': ' + t for i, t in enumerate(pred)]

        attn_pred = out_dict['attn_dist_pred'][bi, :len(pred), :slen]
        attn_ref = out_dict['attn_dist_ref'][bi, :len(tgt), :slen]
        pred_dist = out_dict['pred_dist'][bi, :len(tgt)]
    return 
