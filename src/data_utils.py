import numpy as np 

from argparse import ArgumentParser
from collections import Counter
from torch.utils.data import Dataset, DataLoader, random_split
from frtorch import torch_model_utils as tmu

def pipeline(data: list, 
             word2id: dict, 
             add_start_end: bool = False, 
             dataset_type: str = '') -> np.ndarray:
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

class SCANDataset(Dataset):
  def __init__(self, src: np.ndarray, tgt: np.ndarray):
    """
    Args:
      src: source sentences. size = [dataset_size, max_src_len]
      tgt: target sentences. size = [dataset_size, max_src_len]
    """
    super().__init__()
    self.src = src
    self.tgt = tgt
    return 

  def __len__(self):
    return len(self.src)

  def __getitem__(self, idx):
    instance = {'src': self.src[idx], 'tgt': self.tgt[idx]}
    return instance

class SCANData(object):

  def __init__(self,
      split_name: str = 'random',
      batch_size: int = 64,
      seed: int = 15213,
      num_workers: int = 0,
    ):
    """
    Args:
      split_name: "random", "length", "length_no_new_command"
    """
    super().__init__()

    self.split_name = split_name
    self.num_workers = num_workers
    self.seed = seed
    self.batch_size = batch_size
    
    self.src_word2id = {'<PAD>': 0}
    self.src_id2word = {0: '<PAD>'}
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
    else:
      raise NotImplementedError(
        'data split %s not implemented' % self.split_name)

    # split
    train_data = open(train_path).readlines()
    train_len = int(len(train_data) * 0.8)
    dev_len = len(train_data) - train_len
    train_data, dev_data = random_split(train_data, [train_len, dev_len])
    test_data = open(test_path).readlines()

    # train
    train_src = [d.split(' OUT: ')[0][4:].split() for d in train_data] 
    train_tgt = [d.split(' OUT: ')[1].split() for d in train_data]

    src_word2id, src_id2word = tmu.build_vocab(
      train_src, start_id=len(self.src_word2id)) 
    self.src_word2id.update(src_word2id)
    self.src_id2word.update(src_id2word)
    self.src_vocab_size = len(self.src_word2id)

    tgt_word2id, tgt_id2word = tmu.build_vocab(
      train_tgt, start_id=len(self.tgt_word2id))
    self.tgt_word2id.update(tgt_word2id)
    self.tgt_id2word.update(tgt_id2word)
    self.tgt_vocab_size = len(self.tgt_word2id)

    train_src, _ = pipeline(train_src, self.src_word2id, False, 'train_src')
    train_tgt, train_tgt_max_len =\
      pipeline(train_tgt, self.tgt_word2id, True, 'train_tgt')
    self.train_dataset = SCANDataset(train_src, train_tgt)

    # dev 
    dev_src = [d.split(' OUT: ')[0][4:].split() for d in dev_data] 
    dev_tgt = [d.split(' OUT: ')[1].split() for d in dev_data]
    dev_src, _ = pipeline(dev_src, self.src_word2id, False, 'dev_src')
    dev_tgt, dev_tgt_max_len =\
      pipeline(dev_tgt, self.tgt_word2id, True, 'dev_tgt')
    self.dev_dataset = SCANDataset(dev_src, dev_tgt)

    # test
    test_src = [d.split(' OUT: ')[0][4:].split() for d in test_data] 
    test_tgt = [d.split(' OUT: ')[1].split() for d in test_data]
    test_src, _ = pipeline(test_src, self.src_word2id, False, 'test_src')
    test_tgt, test_tgt_max_len =\
      pipeline(test_tgt, self.tgt_word2id, True, 'test_tgt')
    self.test_dataset = SCANDataset(test_src, test_tgt)

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

  def write_output(self, fd, batch, out_dict):
    batch_size = batch['src'].size(0)
    for bi in range(batch_size):
      slen = tmu.seq_to_lens(batch['src'][bi])
      s = 'src: ' + ' '.join(
        self.src_id2word[idx.item()] for idx in batch['src'][bi][:slen]) + '\n'

      s += 'tgt: ' + out_dict['tgt_str'][bi] + '\n'
      s += 'pred: ' + out_dict['predictions'][bi] + '\n'

      # slen = tmu.seq_to_lens(batch['tgt'][bi])
      # s += 'tgt: ' + ' '.join(
      #   self.tgt_id2word[idx.item()] for idx in batch['tgt'][bi][1:slen]) + '\n'
      # slen = tmu.seq_to_lens(out_dict['predictions'][bi])
      # s += 'pred: ' + ' '.join(
      #   self.tgt_id2word[idx.item()] for idx in out_dict['predictions'][bi][1:slen]) + '\n'
      fd.write(s)
    return 

  @staticmethod
  def add_data_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--split_name', type=str, default='random')
    return parser