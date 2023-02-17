import torch
import types
import transformers
import json
import joblib
import numpy as np
from transformers import AutoModel,AutoTokenizer
import faiss
import os

def load_data(path):
  index = faiss.read_index(path+"/data/custom_index") # data라는 폴더 안에 아래 4가지 데이터들을 넣어두시면 됩니다
  total_passage = joblib.load(path+"/data/custom_passages")
  total_text = joblib.load(path+"/data/custom_text")
  id_to_document = joblib.load(path+"/data/id_to_document")
  return index, total_passage, total_text, id_to_document

def search(query, index, total_passage, total_text, id_to_document, model, tokenizer, k=20):
    inputs = tokenizer(query, padding=True, truncation=True, return_tensors='pt')
    embeddings, _ = model(**inputs.to('cuda'), return_dict=False)
    query_vector = embeddings[:, 0, :]
    top_k = index.search(query_vector.detach().cpu().numpy(), k)
    passage_ids = top_k[1].tolist()[0]

    output = dict()
    doc_ids = []
    raw_texts = []
    for passage_id in passage_ids:
        doc_ids.append(id_to_document[passage_id])
    for doc_id in doc_ids:
        raw_text = " ".join(total_text[doc_id]['text'])
        if total_text[doc_id].get("name"):
            title = total_text[doc_id].get("name")
        else:
            title = total_text[doc_id].get("titles")[1:-1]
        raw_texts.append((title, raw_text)) # (title, raw_text) 튜플 형태입니다

    output['query'] = query
    output['passage'] = [total_passage[_id] for _id in passage_ids]
    output['article'] = raw_texts
    return output # 딕셔너리 형태로 아웃풋을 보냅니다

if __name__=="__main__":
  cur_path = os.getcwd()
  index, total_passage, total_text, id_to_document = load_data(cur_path)
  q_model = AutoModel.from_pretrained(cur_path+"/my_model_second_q_128_lowlr").to('cuda') # 모델은 프로그램이 실행되는 현재 경로에 폴더를 두시면 됩니다.
  q_tokenizer = AutoTokenizer.from_pretrained("BM-K/KoSimCSE-roberta-multitask")
  q_model.eval()
  
  query = "삼성전자의 경쟁사는" # 질문으로 들어가는 쿼리
  with torch.no_grad():
    output = search(query,index,total_passage, total_text, id_to_document, q_model, q_tokenizer, k=10)
    print(output['query'])
    print(output['passage'][0])
    print(output['article'][0])