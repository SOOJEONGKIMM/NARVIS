import torch
from torch import tensor as T
import pickle
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import joblib
from indexers import DenseFlatIndexer
from transformers import AutoModel, AutoTokenizer
from dpr_data import KorQuadDataset, KorQuadSampler, korquad_collator
from utils import get_passage_file
from typing import List

class FiDDataSet(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.questions = self.data['question']
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(self.questions[idx],
                                    return_attention_mask=True, 
                                    padding="max_length",
                                    max_length = 168,
                                    truncation=True,
                                    return_tensors='pt')
        return encoding


class KorDPRRetriever:
    def __init__(self, model, index, tokenizer, train_dataset, batch_size, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.train_loader = torch.utils.data.DataLoader(
            batch_size=self.batch_size,
            dataset=dataset,
            num_workers=4
        )
        self.index = index

    def retrieve(self, k: int = 100):
        """주어진 쿼리에 대해 가장 유사도가 높은 passage를 반환합니다."""
        self.model.eval()  # 평가 모드
        passages = []
        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="train"):
                batch_size = batch['input_ids'].size(0)
                out, _ = self.model(batch['input_ids'].view(batch_size,-1).to(self.device), batch['attention_mask'].view(batch_size,-1).to(self.device), return_dict=False)
                results = self.index.search_knn(query_vectors=out[:, 0, :].cpu().numpy(), top_docs=k)
                # 원문 가져오기
                for result in results:
                    for idx, sim in zip(*result):
                        path = get_passage_file([idx])
                        if not path:
                            print(f"No single passage path for {idx}")
                            continue
                        with open(path, "rb") as f:
                            passage_dict = pickle.load(f)
                        #print(f"passage : {passage_dict[idx]}, sim : {sim}")
                        passages.append((passage_dict[idx], sim))
        return passages


if __name__ == "__main__":
    # load index
    index = DenseFlatIndexer()
    index.deserialize(path="2050iter_flat_128_lowlr")
    # load model & tokenizer
    model = AutoModel.from_pretrained("/home/seongilpark/fid/my_model_second_q_128_lowlr")
    tokenizer = AutoTokenizer.from_pretrained("BM-K/KoSimCSE-roberta-multitask")
    # load data
    a = joblib.load("dump")
    dataset = FiDDataSet(a, tokenizer=tokenizer)
    # load retriever
    k = 40
    retriever = KorDPRRetriever(train_dataset=dataset,
                                model=model,
                                index=index,
                                batch_size=128, 
                                tokenizer=tokenizer)
    passages = retriever.retrieve(k=k)
    a['passages'] = passages
    joblib.dump(a, "retrieved_dump")