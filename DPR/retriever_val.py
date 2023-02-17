import torch
from torch import tensor as T
import pickle
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from indexers import DenseFlatIndexer
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from dpr_data import KorQuadDataset, KorQuadSampler, korquad_collator
from utils import get_passage_file
from typing import List


class KorDPRRetriever:
    def __init__(self, model, valid_dataset, index, val_batch_size: int = 64, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = valid_dataset.tokenizer
        self.val_batch_size = val_batch_size
        self.valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset.dataset,
            batch_sampler=KorQuadSampler(
                valid_dataset.dataset, batch_size=val_batch_size, drop_last=True
            ),
            collate_fn=lambda x: korquad_collator(
                x, padding_value=valid_dataset.pad_token_id
            ),
            num_workers=4,
        )
        self.index = index

    def val_top_k_acc(self, k:List[int]=[5] + list(range(10,101,10))):
        '''validation set에서 top k 정확도를 계산합니다.'''
        
        self.model.eval()  # 평가 모드
        k_max = max(k)
        sample_cnt = 0
        retr_cnt = defaultdict(int)
        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc='valid'):
                q, q_mask, p_id, a, a_mask = batch # p_id랑 a 사이에 p, p_mask 지움
                q, q_mask = (
                    q.to(self.device),
                    q_mask.to(self.device),
                )
                output, _ = self.model(q, q_mask, return_dict=False) # bsz x bert_dim
               # print("output ------>",output.shape)
                result = self.index.search_knn(query_vectors=output[:, 0, :].cpu().numpy(), top_docs=k_max)
               # print("length of result ---> ", len(result))
                for ((pred_idx_lst, _), true_idx, _a , _a_mask) in zip(result, p_id, a, a_mask):
                    a_len = _a_mask.sum()
                    _a = _a[:a_len]
                    _a = _a[1:-1]
                    #print("Pred_idx_lst ---> ", pred_idx_lst)
                    _a_txt = self.tokenizer.decode(_a).strip()
                    docs = [pickle.load(open(get_passage_file([idx]),'rb'))[idx] for idx in pred_idx_lst]

                    for _k in k:
                        if _a_txt in ' '.join(docs[:_k]):
                            retr_cnt[_k] += 1

                bsz = q.size(0)
                sample_cnt += bsz
                batch_retr_acc = {_k:round(float(v) / float(sample_cnt),2) for _k,v in retr_cnt.items()}
                print("batch--->", batch_retr_acc)
        retr_acc = {_k:float(v) / float(sample_cnt) for _k,v in retr_cnt.items()}
        return retr_acc


    def retrieve(self, query: str, k: int = 100):
        """주어진 쿼리에 대해 가장 유사도가 높은 passage를 반환합니다."""
        self.model.eval()  # 평가 모드
        tok = self.tokenizer.batch_encode_plus([query])
        with torch.no_grad():
            out = self.model(T(tok["input_ids"]).to(self.device), T(tok["attention_mask"]).to(self.device))['pooler_output']
        result = self.index.search_knn(query_vectors=out.cpu().numpy(), top_docs=k)

        # 원문 가져오기
        passages = []
        for idx, sim in zip(*result[0]):
            path = get_passage_file([idx])
            if not path:
                print(f"No single passage path for {idx}")
                continue
            with open(path, "rb") as f:
                passage_dict = pickle.load(f)
            print(f"passage : {passage_dict[idx]}, sim : {sim}")
            passages.append((passage_dict[idx], sim))
        return passages


if __name__ == "__main__":

    model = AutoModel.from_pretrained("/home/seongilpark/fid/my_model_second_q_128_lowlr")
    model.eval()
    valid_dataset = KorQuadDataset("dataset/KorQuAD_v1.0_dev.json")
    index = DenseFlatIndexer()
    index.deserialize(path="2050iter_flat_128_lr")
    retriever = KorDPRRetriever(model=model, valid_dataset=valid_dataset, index=index)
    # retriever.retrieve(query=args.query, k=args.k)
    retr_acc = retriever.val_top_k_acc()
    print(retr_acc)
   # print(retriever.retrieve(query="네이버가 설립된 날은?", k=20))
