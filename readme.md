# Fast Research with PyTorch, FRTorch

Simple seq2seq example implementation on SCAN dataset for fast prototyping
Most of the engineering code is reusable, only need to focus on research code

Usage: 
```
mkdir models
mkdir outputs
mkdir tensorboard
python main.py --dataset=scan --model_name=seq2seq --batch_size=64 --embedding_size=200 --state_size=200 --device=cpu --write_output=true
```