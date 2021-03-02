
import argparse
import os 
import sys 
import shutil
from datetime import datetime
from frtorch import torch_model_utils as tmu


def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def set_arguments(args):
  """Set the commandline argument

  Because I see many different convensions of passing arguments (w. commandline,
  .py file, .json file etc.) Here I try to find a clean command line convension
  
  Argument convension:
    * All default value of commandline arguments are from args.py 
    * So instead of using commandline arguments, you can also modify args.py 
    * Instead of using commandline switching, all boolean values are explicitly
      set as 'True' or 'False'
    * The arguments passed through commandline will overwrite the default 
      arguments from args.py 
    * The final arguments are printed out
  """
  ## build model saving path 
  model = args.model_name + "_" + args.model_version
  output_path = args.output_path + model 
  model_path = args.model_path + model
  tensorboard_path = args.tensorboard_path + model + '_'
  tensorboard_path += datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
  if(not args.is_test):
    # Training mode, create directory for storing model and outputs
    print('model path: %s' % model_path)
    if(os.path.exists(model_path)):
      print("model %s already existed" % model)
      print('removing existing cache directories')
      print('removing %s' % model_path)
      shutil.rmtree(model_path)
      os.mkdir(model_path)
      if(os.path.exists(output_path)): 
        print('removing %s' % output_path)
        shutil.rmtree(output_path)
      os.mkdir(output_path)
      if(args.use_tensorboard):
        for p in os.listdir(args.tensorboard_path):
          if(p.startswith(model)): 
            try:
              shutil.rmtree(args.tensorboard_path + p)
            except:
              print('cannot remove %s, pass' % (args.tensorboard_path + p))
        os.mkdir(tensorboard_path)
    else:
      os.mkdir(model_path)
      os.mkdir(output_path)
  else: 
    pass  # test mode, do not create any directory 

  args.model_path = model_path + '/'
  args.output_path = output_path + '/'
  args.tensorboard_path = tensorboard_path + '/'

  # set gpu 
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

  # print out the final argsuration
  tmu.print_args(args)
  return args