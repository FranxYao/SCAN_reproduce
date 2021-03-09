# Reproducing SCAN results

Simple seq2seq example implementation on SCAN dataset for fast prototyping

Usage: 
```
mkdir models
mkdir outputs
mkdir tensorboard

cd src
# Original length split
python main.py --dataset=scan --split_name=length --model_name=seq2seq --model_version=0.1.2.0 --batch_size=64 --embedding_size=200 --state_size=200 --device=cuda --write_output=true --lstm_layers=2 --lstm_bidirectional=False --dropout=0.5  --learning_rate=1e-3 --num_epoch=20 --validate_start_epoch=0

# Change length split point from 22 to 25. i.e. move test cases with target length <= 25 back to training
python main.py --dataset=scan --split_name=length_trunc25 --model_name=seq2seq --model_version=0.1.4.0 --batch_size=64 --embedding_size=200 --state_size=200 --device=cuda --write_output=true --lstm_layers=2 --lstm_bidirectional=False --dropout=0.5  --learning_rate=1e-3 --num_epoch=20 --validate_start_epoch=0 --log_print_to_file=true

# Change length split point from 22 to 27. i.e. move test cases with target length <= 27 back to training
python main.py --dataset=scan --split_name=length_trunc27 --model_name=seq2seq --model_version=0.1.5.0 --batch_size=64 --embedding_size=200 --state_size=200 --device=cuda --write_output=true --lstm_layers=2 --lstm_bidirectional=False --dropout=0.5  --learning_rate=1e-3 --num_epoch=20 --validate_start_epoch=0 --log_print_to_file=true

# Change length split point from 22 to 30. i.e. move test cases with target length <= 30 back to training
python main.py --dataset=scan --split_name=length_trunc30 --model_name=seq2seq --model_version=0.1.3.0 --batch_size=64 --embedding_size=200 --state_size=200 --device=cuda --write_output=true --lstm_layers=2 --lstm_bidirectional=False --dropout=0.5  --learning_rate=1e-3 --num_epoch=20 --validate_start_epoch=0
```