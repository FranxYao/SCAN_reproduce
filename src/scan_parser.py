CMD_SRC2TGT = {'jump': 'I_JUMP', 
               'look': 'I_LOOK',
               'run': 'I_RUN', 
               'walk': 'I_WALK'}

def parse_unit_len_2(unit):
  if(unit[0] == 'turn'):
    if(unit[1] == 'left'):
      parse = ['I_TURN_LEFT']
    elif(unit[1] == 'right'):
      parse = ['I_TURN_RIGHT']
  else: 
    if(unit[1] == 'left'):
      parse = ['I_TURN_LEFT', CMD_SRC2TGT[unit[0]]]
    elif(unit[1] == 'right'):
      parse = ['I_TURN_RIGHT', CMD_SRC2TGT[unit[0]]]
    elif(unit[1] == 'twice'):
      parse = [CMD_SRC2TGT[unit[0]]] * 2
    elif(unit[1] == 'thrice'):
      parse = [CMD_SRC2TGT[unit[0]]] * 3
    else:
      raise NameError('%s' % ' '.join(unit))
  return parse

def parse_unit_len_3(unit):
  if(unit[1] == 'around'):
    parse = parse_unit_len_2([unit[0], unit[2]])
    parse += parse_unit_len_2([unit[0], unit[2]])
    parse += parse_unit_len_2([unit[0], unit[2]])
    parse += parse_unit_len_2([unit[0], unit[2]])
  elif(unit[1] == 'opposite'):
    if(unit[2] == 'left'):
      if(unit[0] == 'turn'):
        parse = ['I_TURN_LEFT', 'I_TURN_LEFT']
      else:
        parse = ['I_TURN_LEFT', 'I_TURN_LEFT', CMD_SRC2TGT[unit[0]]]
    elif(unit[2] == 'right'):
      if(unit[0] == 'turn'):
        parse = ['I_TURN_RIGHT', 'I_TURN_RIGHT']
      else:
        parse = ['I_TURN_RIGHT', 'I_TURN_RIGHT', CMD_SRC2TGT[unit[0]]]
    else:
      raise NameError('%s' % ' '.join(unit))  
  else:
    raise NameError('%s' % ' '.join(unit))
  return parse

def parse_unit_command(unit):
  if(len(unit) == 1):
    parse = [CMD_SRC2TGT[unit[0]]]
  elif(len(unit) == 2):
    parse = parse_unit_len_2(unit)
  elif(len(unit) == 3):
    if(unit[-1] == 'twice'):
      parse = parse_unit_len_2(unit[:2])
      parse += parse_unit_len_2(unit[:2])
    elif(unit[-1] == 'thrice'):
      parse = parse_unit_len_2(unit[:2])
      parse += parse_unit_len_2(unit[:2])
      parse += parse_unit_len_2(unit[:2])
    else:
      parse = parse_unit_len_3(unit)
  elif(len(unit) == 4):
    if(unit[-1] == 'twice'):
      parse = parse_unit_len_3(unit[:3])
      parse += parse_unit_len_3(unit[:3])
    elif(unit[-1] == 'thrice'):
      parse = parse_unit_len_3(unit[:3])
      parse += parse_unit_len_3(unit[:3])
      parse += parse_unit_len_3(unit[:3])
    else: 
      raise NameError('%s' % ' '.join(unit))
  return parse