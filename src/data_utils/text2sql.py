"""Text to SQL data processing

Yao Fu. ILCC, University of Edinburgh.
yao.fu@ed.ac.uk
Mon Apr 26th 2021
"""

import numpy as np 
import json 

from tqdm import tqdm 
from transformers import BertTokenizer
from .text2sql_utils import print_sql_as_tree, simplify_sql, sql_width_depth
from torch.utils.data import Dataset, DataLoader, random_split
from frtorch import torch_model_utils as tmu


def pipeline(questions, 
             queries, 
             query_original, 
             tokenizer, 
             setname, 
             word2id, 
             max_q_len=0, 
             max_s_len=0):
  """
  Args:

  Returns:
  """
  print('processing %s' % setname)
  if(max_q_len == 0):
    max_q_len = 0
    max_s_len = 0
    for q, s in tqdm(zip(questions, queries)):
      q = tokenizer(q)['input_ids']
      if(len(q) > max_q_len): max_q_len = len(q)
      s = s.split()
      if(len(s) > max_s_len): max_s_len = len(s)
    max_s_len += 2

  questions_tokenized = []
  attention_masks = []
  queries_tokenized = []
  for q, s, so in tqdm(zip(questions, queries, query_original)):
    q = tokenizer(q)['input_ids']
    a = [1] * len(q)
    q = tmu.pad_or_trunc_seq(q, max_q_len, 0)
    a = tmu.pad_or_trunc_seq(a, max_q_len, 0)
    questions_tokenized.append(q)
    attention_masks.append(a)

    s_ = s.split(' ')[:-1]
    if('S' in s_): 
      print(so)
      print_sql_as_tree(so)
      print('----')
      print_sql_as_tree(s)
      print('----')
      print('[' + ' '.join(s_) + ']')

    s = [word2id[w] for w in s_]
    s = [1] + s + [2]
    s = tmu.pad_or_trunc_seq(s, max_s_len, 0)
    queries_tokenized.append(s)
  return (questions_tokenized, queries_tokenized, 
    attention_masks, max_q_len, max_s_len)

def read_vocab(path):
  """Read target vocabulary """
  word2id = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2}
  id2word = {0: '[PAD]', 1: '[BOS]', 2: '[EOS]'}
  i = 3
  with open(path) as fd:
    for l in fd.readlines():
      w = l[:-1]
      word2id[w] = i
      id2word[i] = w
      i += 1
  return word2id, id2word

def preprocess(sql_data):
  """Preprocess sql data, change the folowing tokens """
  sql_data_ = []
  for s in sql_data:
    if('MIN' in s or 'MAX' in s or 'COUNT' in s): 
      s_ = s.replace('MIN(', 'MIN (').replace('MAX(', 'MAX (')\
        .replace('COUNT(', 'COUNT (').replace('IS NOT NULL', 'IS-NOT-NULL')
      sql_data_.append(s_)
    else:
      sql_data_.append(s)
  return sql_data_

def simplify_dataset(questions, queries, setname):
  """
  NOTE: currently do not expand the OR conditions
  TODO: expand the OR conditions

  Args:
    questions: 
    queries: 

  Returns: 
    questions_simple: 
    queries_simple: 
  """
  questions_simple = [] 
  queries_simple = []
  query_original = []
  complex_query_cnt = 0
  for q, s in zip(questions, queries):
    d, w = sql_width_depth(s)
    if(d == 1):
      ret_code, sql_simple, _ = simplify_sql(s)
      if(ret_code in [1, 2]):
        for qi in q:
          questions_simple.append(qi)
          queries_simple.append(sql_simple)
          query_original.append(s)
      else: complex_query_cnt += 1
    else: complex_query_cnt += 1

  print('%s set, %d query ignored' % (setname, complex_query_cnt))
  return questions_simple, queries_simple, query_original

class AtisDataset(Dataset):
  """"""
  def __init__(self, questions, queries, attention_masks):
    super().__init__()
    self.questions = np.array(questions)
    self.queries = np.array(queries)
    self.attention_masks = np.array(attention_masks)
    return 

  def __len__(self):
    return len(self.questions)

  def __getitem__(self, idx):
    instance = {'question': self.questions[idx], 
                'attention_masks': self.attention_masks[idx], 
                'queries': self.queries[idx]}
    return instance

class AtisDataSQL(object):
  """Atis dataset, simplified version, basically remove all the unnecessary 
  parenthesis 

  TODO: tree representation and poset decoding for Atis 
  """
  def __init__(self, batch_size):
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    self.batch_size = batch_size
    self.tgt_word2id, self.tgt_id2word = read_vocab(
      '../data/text2sql/atis_train.vocab')

    data_path = '../data/text2sql/atis.json'
    data = json.load(open(data_path))
    train_questions = [] 
    dev_questions = []
    test_questions = []
    for d in data:
      q = []
      for qi in d['sentences']:
        # if(test_contain_all_var_w_sql(qi['text'], sql, qi['variables'])):
        q.append(qi['text'])
      if(d['query-split'] == 'train'): train_questions.append(q)
      elif(d['query-split'] == 'dev'): dev_questions.append(q)
      else: test_questions.append(q)

    train_sql = preprocess([d['sql'][0] for d in data if d['query-split'] == 'train'])
    dev_sql = preprocess([d['sql'][0] for d in data if d['query-split'] == 'dev'])
    test_sql = preprocess([d['sql'][0] for d in data if d['query-split'] == 'test'])

    # simplify sql 
    train_questions, train_sql, train_sql_original = simplify_dataset(
      train_questions, train_sql, 'train')
    dev_questions, dev_sql, dev_sql_original = simplify_dataset(
      dev_questions, dev_sql, 'dev')
    test_questions, test_sql, test_sql_original = simplify_dataset(
      test_questions, test_sql, 'test')

    # data processing pipeline 
    train_questions, train_sql, train_attn, max_q_len, max_s_len = pipeline(
      train_questions, train_sql, train_sql_original, self.tokenizer, 'train', self.tgt_word2id)
    print('max question length %d, max query length %d' % (max_q_len, max_s_len))
    dev_questions, dev_sql, dev_attn, _, _ = pipeline(dev_questions, dev_sql, dev_sql_original,
      self.tokenizer, 'dev', self.tgt_word2id, max_q_len, max_s_len)
    test_questions, test_sql, test_attn, _, _ = pipeline(test_questions, test_sql, test_sql_original, 
      self.tokenizer, 'dev', self.tgt_word2id, max_q_len, max_s_len)

    # init dataset
    self.train_data = AtisDataset(train_questions, train_sql, train_attn)
    self.dev_data = AtisDataset(dev_questions, dev_sql, dev_attn)
    self.test_data = AtisDataset(test_questions, test_sql, test_attn)
    return 

  def convert_ids_to_string(self, ids, remove_pad=True):
    return 

  def train_dataloader(self):
    loader = DataLoader(self.train_data, 
                        batch_size=self.batch_size, 
                        shuffle=True, 
                        num_workers=0,
                        drop_last=False,
                        pin_memory=False)
    return loader

  def val_dataloader(self):
    loader = DataLoader(self.dev_data, 
                        batch_size=self.batch_size, 
                        shuffle=False, 
                        num_workers=0,
                        drop_last=False,
                        pin_memory=False)
    return loader

  def test_dataloader(self):
    loader = DataLoader(self.test_data, 
                        batch_size=self.batch_size, 
                        shuffle=False, 
                        num_workers=0,
                        drop_last=False,
                        pin_memory=False)
    return loader
  