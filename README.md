# simple-pretraining
심플 한국어 언어모델 사전학습

ref: [https://huggingface.co/transformers/examples.html](https://huggingface.co/transformers/examples.html)

## Tokenizers
1. Exobrain + Wordpiece: exobrain_wordpiece.json

## Models
1. [BERT](https://github.com/deepvine/simple-pretraining/blob/main/simple-bert-training.py): simple BERT pre-training.
2. BART: simple BART pre-training.
3. ALBERT: simple ALBERT pre-training
4. GPT-2: simple GPT-2 pre-training

## Test
```
python test.py
```
ex) 요새 [MASK]에 앉자 있으면

[{'sequence': '요새 자리 에 앉아 있으면.', 'score': 0.13420788943767548, 'token': 7132,}, {'sequence': '요새 의자 에 앉아 있으면.', 'score': 0. ...08035723119974136,


## TO-DO list
