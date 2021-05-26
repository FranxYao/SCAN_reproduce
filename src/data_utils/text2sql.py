"""Text to SQL data processing

Yao Fu. ILCC, University of Edinburgh.
yao.fu@ed.ac.uk
Mon Apr 26th 2021
"""

import numpy as np 
import json 

from text2sql_utils import simplify_sql
from torch.utils.data import Dataset, DataLoader, random_split
from frtorch import torch_model_utils as tmu


def preprocess(sql_data):
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
  complex_query_cnt = 0
  for q, s in zip(questions, queries):
    ret_code, sql_simple, _ = simplify_sql(s)
    if(ret_code in [1, 2]):
      for qi in q:
        questions_simple.append(qi)
        queries_simple.append(sql_simple)
    else: 
      complex_query_cnt += 1

  print('%s set, %d query ignored' % setname, complex_query_cnt)
  return questions_simple, queries_simple

class AtisDataset(Dataset):
  """"""
  def __init__():
    return 

  def __len__(self):
    return 

  def __getitem__(self, idx):
    return 

class AtisData(object):
  """"""
  def __init__(self):
    data_path = '../data/text2sql/atis.json'
    data = json.load(open(data_path + '/atis.json'))
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
    train_questions, train_sql = simplify_dataset(
      train_questions, train_sql, 'train')
    dev_questions, dev_sql = simplify_dataset(
      dev_questions, dev_sql, 'dev')
    test_questions, test_sql = simplify_dataset(
      test_questions, test_sql, 'test')

    return 

  def train_dataloader(self):
    return 

  def val_dataloader(self):
    return 

  def test_dataloader(self):
    return 
  