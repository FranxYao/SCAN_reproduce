import sys
import re
import copy 

def print_lf(s):
  """Could be problematic, need update"""
  indent = -1
  new_line = False
  for i, si in enumerate(s):
    if(si == '('):
      indent += 1
      if(indent >= 1 and i >= 1): 
        if(s[i - 1] != ')'): print()
      print('  ' * indent + '(')
      new_line = True
    elif(si == ')'):
      if(s[i - 1] != '(' and s[i - 1] != ')'): print() 
      print('  ' * indent + ')')
      indent -= 1
      new_line = True
    else:
      if(new_line):
        print('  ' * (indent + 1) + si, end=' ')
        new_line = False
      else:
        print(si, end=' ')
  return 

def update_in_quote(in_quote, token):
    if '"' in token and len(token.split('"')) % 2 == 0:
        in_quote[0] = not in_quote[0]
    if "'" in token and len(token.split("'")) % 2 == 0:
        in_quote[1] = not in_quote[1]

def sql_width_depth(sql):
  """Adapted from 
  https://github.com/jkkummerfeld/text2sql-data/blob/master/tools/corpus_stats.py 
  """ 
  max_depth = 0
  max_breadth = 1
  depth = 0
  prev = None
  other_bracket = []
  breadth = [0]
  in_quote = [False, False]
  for token in sql.split():
    if in_quote[0] or in_quote[1]:
      update_in_quote(in_quote, token)
    elif token == 'SELECT':
      depth += 1
      max_depth = max(max_depth, depth)
      other_bracket.append(0)
      breadth[-1] += 1
      breadth.append(0)
    elif '(' in prev:
      other_bracket[-1] += 1
      update_in_quote(in_quote, token)
    elif token == ')':
      if other_bracket[-1] == 0:
        depth -= 1
        other_bracket.pop()
        possible = breadth.pop()
        max_breadth = max(max_breadth, possible)
      else:
        other_bracket[-1] -= 1
    else:
      update_in_quote(in_quote, token)
    
    if '(' in token and ')' in token:
      prev = "SQL_FUNCTION"
    else:
      prev = token
  assert len(other_bracket) == 1 and other_bracket[0] == 0, sql
  assert depth == 1, sql
  return max_depth, max_breadth

def insert_variables(sql, sql_variables, sent, sent_variables):
  """Copied from:
  https://github.com/jkkummerfeld/text2sql-data/blob/master/systems/baseline-template/text2sql-template-baseline.py
  """
  tokens = []
  tags = []
  seen_sent_variables = set()
  for token in sent.strip().split():
      # if (token not in sent_variables) or args.no_vars:
      if (token not in sent_variables):
          tokens.append(token)
          tags.append("O")
      else:
          assert len(sent_variables[token]) > 0
          seen_sent_variables.add(token)
          for word in sent_variables[token].split():
              tokens.append(word)
              tags.append(token)

  sql_tokens = []
  for token in sql.strip().split():
      if token.startswith('"%') or token.startswith("'%"):
          sql_tokens.append(token[:2])
          token = token[2:]
      elif token.startswith('"') or token.startswith("'"):
          sql_tokens.append(token[0])
          token = token[1:]

      if token.endswith('%"') or token.endswith("%'"):
          sql_tokens.append(token[:-2])
          sql_tokens.append(token[-2:])
      elif token.endswith('"') or token.endswith("'"):
          sql_tokens.append(token[:-1])
          sql_tokens.append(token[-1])
      else:
          sql_tokens.append(token)

  template = []
  complete = []
  case = 0
  for token in sql_tokens:
      # Do the template
      if token in seen_sent_variables:
          # The token is a variable name that will be copied from the sentence
          template.append(token)
      elif (token not in sent_variables) and (token not in sql_variables):
          # The token is an SQL keyword
          template.append(token)
      elif token in sent_variables and sent_variables[token] != '':
          # The token is a variable whose value is unique to this questions,
          # but is not explicitly given
          template.append(sent_variables[token])
          case = '1'
      else:
          # The token is a variable whose value is not unique to this
          # question and not explicitly given
          template.append(sql_variables[token])
          case = '2'

      # Do the complete case
      if token in sent_variables and sent_variables[token] != '':
          complete.append(sent_variables[token])
      elif token in sql_variables:
          complete.append(sql_variables[token])
      else:
          complete.append(token)

  template = ' '.join(template)
  complete = ' '.join(complete)
  # assert(template == complete)
  return (tokens, tags, template, complete, case)

def print_indent(level):
  for _ in level:
    print('  ')
  return 

def print_sql_as_tree(sql, fd = None, relax=False):
  level = 0
  if(not isinstance(sql, list)):
    sql = sql.split()
  else:
    if(sql[-1] != ';'):
      sql.append(';')
  buffer = []
  prev_parenthesis_level = [] 
  if(fd is None):
    fd = sys.stdout
  for ti, token in enumerate(sql):
    if(token == 'SELECT'):
      if(sql[ti + 1] == 'DISTINCT'):
        print('..' * level + 'SELECT DISTINCT', file=fd)
      else: 
        print('..' * level + 'SELECT', file=fd)
      level += 1
    elif(token == 'DISTINCT'): pass
    elif(token == ','):
      if(len(buffer) > 0):
        print('..' * level + ' '.join(buffer), file=fd)
        buffer = []
    elif(token == 'FROM'):
      if(len(buffer) > 0):
        print('..' * level + ' '.join(buffer), file=fd)
        buffer = []
      print('..' * (level - 1) + 'FROM', file=fd)
    elif(token == 'WHERE'):
      if(len(buffer) > 0):
        print('..' * level + ' '.join(buffer), file=fd)
        buffer = []
      print('..' * (level - 1) + 'WHERE', file=fd)
    elif(token == 'AND'):
      if(len(buffer) > 0):
        print('..' * level + ' '.join(buffer), file=fd)
        buffer = []
      print('..' * level + 'AND', file=fd)
    elif(token == 'OR'):
      if(len(buffer) > 0):
        print('..' * level + ' '.join(buffer), file=fd)
        buffer = []
      print('..' * level + 'OR', file=fd)
    elif(token == '('):
      if(len(buffer) > 0):
        print('..' * level + ' '.join(buffer), file=fd)
        buffer = []
      print('..' * (level) + '(', file=fd)
      buffer = []
      prev_parenthesis_level.append(level)
      level += 1
      # if(sql[ti - 1] in ['MIN', 'MAX', 'COUNT']):
        # prev_parenthesis_level.append(level)
        # level += 1
      # else: 
      #   prev_parenthesis_level.append(level - 1)
    elif(token == ')'):
      if(len(buffer) > 0):
        print('..' * level + ' '.join(buffer), file=fd)
        buffer = []
      level = prev_parenthesis_level.pop()
      print('..' * (level) + ')', file=fd)
    elif(token == ';'):
      if(len(buffer) > 0):
        print('..' * level + ' '.join(buffer), file=fd)
        buffer = []
    elif(token == 'IN'):
      if(len(buffer) > 0):
        print('..' * level + ' '.join(buffer), file=fd)
        print('..' * level + 'IN', file=fd)
        buffer = []
    else:
      buffer.append(token)
  if(relax == True):
    print(buffer)
  else:
    assert(len(buffer) == 0)
  return 

def process_all_or_cond(cond):
  cond = copy.deepcopy(cond)
  indices = [i for i, x in enumerate(cond) if x == "OR"]

  prev_right = ''
  cond_simple = cond
  or_states = []
  # print('\n\nprocess condition with OR, %d OR statements' % len(indices))
  # print(cond)

  chained_or = False
  nested_or = False
  finished_process = False 
  next_ind = -2
  num_or = 0

  while(not finished_process):
    # try to process next OR statement
    or_left, or_right, cond_simple, next_ind, left_nest =\
      process_single_or(cond_simple, next_ind, num_or)
    if(next_ind == -1): 
      finished_process = True 
      break

    # succeed
    if(next_ind == 0):
      or_states.append({'or_id': num_or,
                        'or_left': or_left,
                        'or_right': or_right})
      num_or += 1

    if(next_ind > 0):
      nested_or = True
    if(left_nest): nested_or = True

  for i, ci in enumerate(cond_simple):
    if(i >= 2 and ci[-3:] == 'OR]' and cond_simple[i - 2][-3:] == 'OR]'):
      chained_or = True
  return cond_simple, or_states, chained_or, nested_or

def process_single_or(cond, next_ind, num_or):
  """Find the OR clauses 
  Algorithm:

  Args:

  Returns:
    ret_code: 0 = success, 1 = fail
    tuple: (or_left, left_boundary, or_right, right_boundary)
  """
  cond = copy.deepcopy(cond)
  # print('single pass processing OR')
  # print_sql_as_tree(cond)
  # print(cond)
  def test_cond_boundary(ci):
    boundary_keywords = ['AND', 'OR', '(', ')', '', 'NOT', 'SELECT', 'WHERE', '[OR]']
    if(ci in boundary_keywords): 
      # print('D1', ci)
      return True
    if(ci[-3:] == 'OR]'): 
      # print('D2', ci)
      return True 
    return False

  def contain_or_left(left):
    for ci in left:
      if(ci[-3:] == 'OR]'): return True
    return False

  cond_len = len(cond)

  # if assigned index, directly try the assigned index 

  for i in range(next_ind, len(cond)):
    x = cond[i]
    if(x == 'OR'): break 
  ind = i
  if(ind == len(cond) - 1): 
    next_ind = -1
    return None, None, cond, next_ind, False

  or_left = []
  i = ind - 1
  if(cond[i] != ')'):
    while(i >= 0 and not test_cond_boundary(cond[i])):
      or_left.append(cond[i])
      i -= 1
  else:
    # print(cond)
    bracket = 1
    or_left.append(cond[i])
    i -= 1
    while(bracket > 0):
      or_left.append(cond[i])
      if(cond[i] == '('): bracket -= 1
      if(cond[i] == ')'): bracket += 1
      i -= 1
  if(cond[i] == 'NOT'):
    or_left.append(cond[i])
    i -= 1
  left_boundary = i + 1
  
  or_right = [] 
  i = ind + 1
  # print(ind)
  # print(cond[ind])
  # print_sql_as_tree(cond, relax=True)
  if(cond[i] == 'NOT'):
    or_right.append(cond[i])
    i += 1
  if(cond[i] != '('):
    while(i < cond_len and not test_cond_boundary(cond[i])):
      or_right.append(cond[i])
      i += 1
    right_boundary = i - 1
  else:
    bracket = 1
    or_right.append(cond[i])
    i += 1
    while(bracket > 0):
      # print(i)
      # print(cond[i])
      or_right.append(cond[i])
      if(cond[i] == '('): bracket += 1
      if(cond[i] == ')'): bracket -= 1
      i += 1
    right_boundary = i - 1

  # trial success
  if('OR' not in or_right):
    if(cond[left_boundary - 1] == '(' and cond[right_boundary + 1] == ')'):
      left_boundary -= 1
      right_boundary += 1
    cond_simple = cond[: left_boundary]\
                  + ['S1', '[%d:OR]' % num_or, 'S2']\
                  + cond[right_boundary + 1: ]
    or_left.reverse()

    if(contain_or_left(or_left)): left_nest = True
    else: left_nest = False

    if('NOT' not in or_left): or_left = remove_parenthesis(or_left).split()
    if('NOT' not in or_right): or_right = remove_parenthesis(or_right).split()
    return or_left, or_right, cond_simple, 0, left_nest
  else: 
    cond_simple = cond
    return None, None, cond_simple, ind + 1, False

def remove_parenthesis(cond):
  cond = copy.deepcopy(cond)
  if(isinstance(cond, str)):
    cond = cond.split(' ')
  
  cond_simple = re.sub(' +', ' ', 
      ' '.join(cond).replace('(', '').replace(')', ''))

  if(cond_simple[0] == ' '): cond_simple = cond_simple[1:]
  if(cond_simple[-1] == ' '): cond_simple = cond_simple[:-1]
  return cond_simple

def sort_simple_condition(cond):
  cond = copy.deepcopy(cond)
  cond = cond.split(' ')
  if('[0:OR]' in cond):
    if(len(cond) > 3):
      all_cond = ' '.join(cond).split(' AND ')
      or_cond = [ci for ci in all_cond if 'OR]' in ci]
      or_cond = [''.join(ci.split(' ')) for ci in or_cond]
      general_cond = [ci for ci in all_cond if 'OR]' not in ci]

      cond = ' AND '.join(or_cond)
      if(len(general_cond) > 0):
        cond = cond + ' AND ' + ' AND '.join(general_cond) 
    else: 
      cond = 'S1[0:OR]S2'
  else: 
    cond = ' '.join(cond).split(' AND ')
    cond.sort()
    cond = ' AND '.join(cond)
  
  return cond

def get_condition(sql):
  """Find the first WHERE, and the rest of the sequence is the condition"""
  cond_ind = sql.find('WHERE ')
  cond = sql[cond_ind + 6:]
  return cond_ind, cond

def simplify_condition(cond):
  """Simplifiy the SQL conditions, currently no nest
  
  Args:
    cond: condition terms in WHERE clauses

  Returns:
    ret_code: 0 = success, 1 = fail
    cond_simple: simplified condition
  """
  # TODO: handle the case of NOT(cond)
  if(('OR' not in cond) and ('NOT' not in cond)):
    # simplify AND statements
    cond_simple = remove_parenthesis(cond)
    cond_simple = sort_simple_condition(cond_simple)
    ret_code = 1
  elif('OR' in cond): 
    # simplify OR statements
    cond_simple, or_states, chained_or, nested_or = process_all_or_cond(cond)
    # simplify AND statements
    cond_simple = remove_parenthesis(cond_simple)
    if((not chained_or) and (not nested_or)):
      cond_simple = sort_simple_condition(cond_simple)
    cond_simple = {'cond_simple': cond_simple, 'or_states': or_states}
    ret_code = 2

    if(chained_or or nested_or): ret_code = 3
    # print('!!!')
    # print(ret_code)
  else: 
    ret_code = 0
    cond_simple = cond
  return ret_code, cond_simple

def simplify_sql(sql):
  if('WHERE' not in sql):
    return 1, sql, sql
  cond_ind, cond = get_condition(sql)

  cond = cond.replace('IS NOT NULL', 'IS-NOT-NULL')
  cond = cond[:-2].split(' ')
  ret_code, cond_simple = simplify_condition(cond)

  if(ret_code == 1):
    sql_simple = sql[: cond_ind] + 'WHERE ' + cond_simple
  elif(ret_code in [2, 3]):
    sql_simple = sql[: cond_ind] + 'WHERE ' + cond_simple['cond_simple']
  else: 
    sql_simple = sql

  sql_simple += ' ;'
  return ret_code, sql_simple, cond_simple