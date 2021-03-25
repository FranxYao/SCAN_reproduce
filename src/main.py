import argparse
import torch
import sys

from data_utils import SCANData
from seq2seq import Seq2seq, Seq2seqModel
from seq2seq_pos import Seq2seqPos, Seq2seqPosModel

from frtorch import torch_model_utils as tmu
from frtorch import str2bool, set_arguments, PrintLog
from controller import Controller


def define_argument():
  ## add commandline arguments
  parser = argparse.ArgumentParser()

  # general 
  parser.add_argument(
    "--model_name", default='seq2seq', type=str)
  parser.add_argument(
    "--model_version", default='0.1.0.0', type=str)
  parser.add_argument(
    "--dataset", default='', type=str)
  parser.add_argument(
    "--task", default='', type=str)
  parser.add_argument(
    "--model_path", default='../models/', type=str)
  parser.add_argument(
    "--output_path", default='../outputs/', type=str)
  parser.add_argument(
    "--output_path_fig", default='', type=str)
  parser.add_argument(
    "--tensorboard_path", default='../tensorboard/', type=str)

  # hardware
  parser.add_argument(
    "--device", default='cpu', type=str)
  parser.add_argument(
    "--gpu_id", default='0', type=str)

  # batch, epoch 
  parser.add_argument(
    "--num_epoch", default=10, type=int)
  parser.add_argument(
    "--batch_size", default=64, type=int)
  parser.add_argument(
    "--start_epoch", default=0, type=int)

  # saving, logging
  parser.add_argument(
    "--load_ckpt", type=str2bool, 
    nargs='?', const=True, default=False)
  parser.add_argument(
    "--pretrained_model_path", default='', type=str)
  parser.add_argument(
    "--print_log_per_nbatch", default=50, type=int)
  parser.add_argument(
    "--print_var", type=str2bool, 
    nargs='?', const=True, default=False)
  parser.add_argument(
    "--use_tensorboard", type=str2bool, 
    nargs='?', const=True, default=False)
  parser.add_argument(
    "--save_checkpoints", type=str2bool, 
    nargs='?', const=True, default=False)
  parser.add_argument(
    "--inspect_model", type=str2bool, 
    nargs='?', const=True, default=False)
  parser.add_argument(
    "--inspect_grad", type=str2bool, 
    nargs='?', const=True, default=False)
  parser.add_argument(
    "--log_print_to_file", type=str2bool, 
    nargs='?', const=True, default=False)
  parser.add_argument(
    "--write_fig_after_epoch", default=10, type=int)



  # Validation, Test
  parser.add_argument(
    "--is_test", type=str2bool, 
    nargs='?', const=True, default=False)
  parser.add_argument(
    "--fast_test_pipeline", type=str2bool, 
    nargs='?', const=True, default=False)
  parser.add_argument(
    "--validate_start_epoch", default=0, type=int)
  parser.add_argument(
    "--validation_criteria", default='val_loss', type=str)
  parser.add_argument(
    "--write_output", type=str2bool, 
    nargs='?', const=True, default=False)
  parser.add_argument(
    "--write_output_full_log", type=str2bool, 
    nargs='?', const=True, default=False)
  

  # optimization
  parser.add_argument(
    "--optimizer", default='Adam', type=str)
  parser.add_argument(
    "--learning_rate", default=1e-4, type=float)

  # model 
  parser.add_argument(
    "--state_size", type=int, default=100)
  parser.add_argument(
    "--embedding_size", type=int, default=100)
  parser.add_argument(
    "--dropout", type=float, default=0.0)
  parser.add_argument(
    "--src_vocab_size", type=int, default=-1)
  parser.add_argument(
    "--tgt_vocab_size", type=int, default=-1)

  return parser


def main():
  # arguments
  parser = define_argument()
  parser = SCANData.add_data_specific_args(parser)
  # parser = Seq2seq.add_model_specific_args(parser)
  parser = Seq2seqPos.add_model_specific_args(parser)
  args = parser.parse_args()
  args = set_arguments(args)

  if(args.log_print_to_file): 
    print('All printed log also written in: %s' % 
      args.output_path + 'train_log.txt')
    sys.stdout = PrintLog(args.output_path + 'train_log.txt')

  # dataset
  if(args.dataset == 'scan'):
    if(args.model_name == 'seq2seq_pos'): require_pos = True
    else: require_pos = False
    dataset = SCANData(split_name=args.split_name,
                       batch_size=args.batch_size,
                       require_pos=require_pos,
                       output_path_fig=args.output_path_fig,
                       write_fig_after_epoch=args.write_fig_after_epoch
                       )
    dataset.build()
  else: 
    raise NotImplementedError('dataset %s not implemented' % args.dataset)

  # model 
  if(args.model_name == 'seq2seq'):
    model_ = Seq2seqModel(pad_id=dataset.tgt_word2id['<PAD>'],
                          start_id=dataset.tgt_word2id['<GOO>'],
                          max_dec_len=dataset.max_dec_len,
                          src_vocab_size=dataset.src_vocab_size, 
                          tgt_vocab_size=dataset.tgt_vocab_size,
                          embedding_size=args.embedding_size,
                          state_size=args.state_size,
                          lstm_layers=args.lstm_layers,
                          dropout=args.dropout,
                          device=args.device
                          )
    model = Seq2seq(args.learning_rate, 
                    args.device, 
                    dataset.tgt_word2id['<PAD>'], 
                    dataset.tgt_id2word, 
                    model_, 
                    args.output_path_fig,
                    args.write_fig_after_epoch)
  elif(args.model_name == 'seq2seq_pos'):
    model_ = Seq2seqPosModel(word_dropout=args.word_dropout,
                            use_attention=args.use_attention,
                            pad_id=dataset.tgt_word2id['<PAD>'],
                            start_id=dataset.tgt_word2id['<GOO>'],
                            max_dec_len=dataset.max_dec_len,
                            src_vocab_size=dataset.src_vocab_size, 
                            pos_size=len(dataset.pos_word2id),
                            tgt_vocab_size=dataset.tgt_vocab_size,
                            embedding_size=args.embedding_size,
                            state_size=args.state_size,
                            lstm_layers=args.lstm_layers,
                            dropout=args.dropout,
                            device=args.device
                            )
    model = Seq2seqPos(args.learning_rate, args.device, 
      dataset.tgt_word2id['<PAD>'], dataset.tgt_id2word)
    model.build(model_)
  else: 
    raise NotImplementedError('model %s not implemented!' % args.model_name)  
  tmu.print_params(model)

  # controller
  controller = Controller(args, model, dataset)

  if(not args.is_test):
    if(args.load_ckpt):
      print('Loading model from: %s' % args.pretrained_model_path)
      model.load_state_dict(torch.load(args.pretrained_model_path))
    model.to(args.device)
    controller.train(model, dataset)
  else:
    print('Loading model from: %s' % args.pretrained_model_path)
    checkpoint = torch.load(args.pretrained_model_path)
    # tmu.load_partial_state_dict(model, checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    controller.test_model(model, dataset, ckpt_e)
  return 


if __name__ == '__main__':
  main()
