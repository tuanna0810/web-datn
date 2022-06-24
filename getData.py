from ctypes import Union
from fastapi import FastAPI, Depends
from transformers import RobertaForSequenceClassification, RobertaConfig, AdamW
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
import argparse

app = FastAPI()

@app.get("/submit/{sentence}")
async def submit(sentence: str):
    path_config  = '/home/tuannguyenanh/Desktop/DATN-WEB-PYTHON/datn_web_python/datn/model/config (2).json'
    path_model = '/home/tuannguyenanh/Desktop/DATN-WEB-PYTHON/datn_web_python/datn/model/pytorch_model (2).bin'
    path_bpe = '/home/tuannguyenanh/Desktop/DATN-WEB-PYTHON/datn_web_python/datn/model/bpe.codes'
    path_vocab = '/home/tuannguyenanh/Desktop/DATN-WEB-PYTHON/datn_web_python/datn/model/dict.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpe-codes', 
        default="/home/tuannguyenanh/Desktop/DATN-WEB-PYTHON/datn_web_python/datn/model/bpe.codes",
        required=False,
        type=str,
        help='path to fastBPE BPE'
    )
    args, unknown = parser.parse_known_args()
    bpe = fastBPE(args)

    # Load the dictionary
    vocab = Dictionary()
    vocab.add_from_file("/home/tuannguyenanh/Desktop/DATN-WEB-PYTHON/datn_web_python/datn/model/dict.txt")
    def get_model(path_model= None, path_config = None, path_bpe = None, path_vocab = None):
      config = RobertaConfig.from_pretrained(
          path_config, from_tf=False, num_labels = 8, output_hidden_states=False
      )
      BERT_SA_NEW = RobertaForSequenceClassification.from_pretrained(
          path_model,
          config=config
      )
    
    
      try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--bpe-codes', 
            default=path_bpe,
            required=False,
            type=str,
            help='path to fastBPE BPE'
        )
        args, unknown = parser.parse_known_args()
        bpe = fastBPE(args)
      except:
        bpe = None
        print("load bpe fail")
      try:
        vocab = Dictionary()
        vocab.add_from_file(path_vocab)
      except:
        vocab=None
        print('load vocab fail')
      return BERT_SA_NEW, bpe, vocab
    
    model, bpe, vocab = get_model(path_model, path_config, path_bpe, path_vocab)

    def predict(model, bpe, sense, vocab):
      subwords = '<s> ' + bpe.encode(sense) + ' </s>'
      encoded_sent = vocab.encode_line(subwords, append_eos=True, add_if_not_exist=False).long().tolist()
      encoded_sent = pad_sequences([encoded_sent], maxlen=195, dtype="long", value=0, truncating="post", padding="post")
      mask = [int(token_id > 0) for token_id in encoded_sent[0]]


      encoded_sent = torch.tensor(encoded_sent)
      mask = torch.tensor(mask)
      encoded_sent = torch.reshape(encoded_sent, (1, 195))
      mask = torch.reshape(mask, (1, 195))

      with torch.no_grad():
        outputs = model(encoded_sent, 
          token_type_ids=None, 
          attention_mask=mask)
        print(outputs)
        logits = outputs[0]
      return int(torch.argmax(logits))
    emotionId = predict(model, bpe, sentence, vocab)
    print(emotionId)
    if(emotionId == 0):
      emotionName = 'Disgust'
    elif(emotionId == 1):
      emotionName = 'Enjoyment'
    elif(emotionId == 2):
      emotionName = 'Sadness'
    elif(emotionId == 3):
      emotionName = 'Fear'
    elif(emotionId == 4):
      emotionName = 'Anger'
    else:
      emotionName = 'Other'
    
  
    return {"emotion": emotionName}




































