# punctuation-restoration
Punctuation Restoration using deep neural network for English and Bangla.
## Model Architecture
![](./assets/model_architecture.png) 


## Dependencies
Need to install PyTorch. You can install download [PyTorch](https://pytorch.org/get-started/locally/) by selecting your preferences and run the install command. 
After downloading PyTorch you can run the following command to install other dependencies.  
```bash
pip install -r requirements.txt
```


## Train Punctuation Restoration Model
To train punctuation restoration model run the following command.
1. Run
```bash
python src/train.py --pretrained-model=distilbert-base-multilingual-cased --freeze-bert=False --lstm-dim=-1 --language=english --seed=1 --lr=5e-6 --epoch=10 --use-crf=False --augment-type=delete  --augment-rate=0.1 --alpha-sub=0.5 --alpha-del=0.4 --gradient-clip=-1  --data-path=data --save-path=out
```


## Test Punctuation Restoration Model
To test punctuation restoration model run the following command.
1. Run
```bash
!python src/test.py --pretrained-model=xlm-roberta-large --language=bangla --sequence-length=256 --batch-size=8 --augment-rate=0.05 --data-path=data --save-path=out
```
