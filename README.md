# punctuation-restoration
Punctuation Restoration using deep neural network for English and Bangla.
## Model Architecture
We fine-tune a Transformer architecture based language model (e.g., BERT) for the punctuation restoration task. 
Transformer encoder is followed by a bidirectional LSTM and linear layer that predicts target punctuation token at 
each sequence position. 
![](./assets/model_architectue.png) 


## Dependencies
Install PyTorch following instructions from [PyTorch website](https://pytorch.org/get-started/locally/). Remaining 
dependencies can be installed with the following command
```bash
pip install -r requirements.txt
```


## Training
To train punctuation restoration model with optimal parameter settings for English run the following command
```
python src/train.py --cuda=True --pretrained-model=roberta-large --freeze-bert=False --lstm-dim=-1 --language=english --seed=1 
--lr=5e-6 --epoch=10 --use-crf=False --augment-type=all  --augment-rate=0.15 --alpha-sub=0.4 --alpha-del=0.4 
--data-path=data --save-path=out
```
To train for Bangla the corresponding command is
```
python src/train.py --cuda=True --pretrained-model=xlm-roberta-large --freeze-bert=False --lstm-dim=-1 --language=bangla --seed=1 
--lr=5e-6 --epoch=10 --use-crf=False --augment-type=all  --augment-rate=0.15 --alpha-sub=0.4 --alpha-del=0.4 
--data-path=data --save-path=out
```

Following models are supported
```
bert-base-uncased
bert-large-uncased
bert-base-multilingual-cased
bert-base-multilingual-uncased
xlm-mlm-en-2048
xlm-mlm-100-1280
roberta-base
roberta-large
distilbert-base-uncased
distilbert-base-multilingual-cased
xlm-roberta-base
xlm-roberta-large
albert-base-v1
albert-base-v2
albert-large-v2
```
Any of these models can be used for English. However, for Bangla, only multilingual models should be used. These are
```
bert-base-multilingual-cased
bert-base-multilingual-uncased
xlm-mlm-100-1280
distilbert-base-multilingual-cased
xlm-roberta-base
xlm-roberta-large
```

## Test 
To test pretrained models on processed test datasets run run following command
```bash
python src/test.py --pretrained-model=roberta-large --lstm-dim=-1 --use-crf=False --data-path=data/test 
--weight-path=weights/roberta-large.pt --sequence-length=256 --save-path=out
```
Please provide corresponding arguments for `pretrained-model`, `lstm-dim`, `use-crf` that were used during training the 
model. This will run test for all data available in `data-path` directory. 

## Known Issues
Please install the Transformers version mentioned in `requirements.txt` i.e, `v2.11.0`. There has been some change in 
`tokenization` in  `v3.0.0` and there might be some mismatch between reported result if it is used.

## TO-DO
- [ ] Add data pre-processing code.
- [ ] Add inference code that takes punctuated or unpunctuated string and adds punctuation from model prediction. If the
 text is already punctuated those will be stripped beforehand.