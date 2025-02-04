import torch
import torch.nn as nn
import logging
import os
import wandb
import numpy as np
import transformers
from typing import Tuple
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from dpr_data import KorQuadDataset, KorQuadSampler, korquad_collator

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/log.log",
    level=logging.DEBUG,
    format="[%(asctime)s | %(funcName)s @ %(pathname)s] %(message)s",
)
logger = logging.getLogger()  # get root logger


class Trainer:
    """basic trainer"""

    def __init__(
        self,
        train_dataset,
        valid_dataset,
        num_epoch: int,
        batch_size: int,
        lr: float,
        betas: Tuple[float],
        num_warmup_steps: int,
        num_training_steps: int,
        valid_every: int,
        best_val_ckpt_path: str,
    ):
        NGPU = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.question_model = AutoModel.from_pretrained("BM-K/KoSimCSE-roberta-multitask")
        self.passage_model = AutoModel.from_pretrained("BM-K/KoSimCSE-roberta-multitask")
        if NGPU > 1:
            self.question_model = torch.nn.DataParallel(self.question_model, device_ids=list(range(NGPU)))
            self.passage_model = torch.nn.DataParallel(self.passage_model, device_ids=list(range(NGPU)))
        self.question_model.to(self.device)
        self.passage_model.to(self.device)
        params = list(self.question_model.parameters()) + list(self.passage_model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr, betas=betas)
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps, num_training_steps
        )
        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset.dataset,
            batch_sampler=KorQuadSampler(
                train_dataset.dataset, batch_size=batch_size, drop_last=True
            ),
            collate_fn=lambda x: korquad_collator(
                x, padding_value=train_dataset.pad_token_id
            ),
            num_workers=4
        )
        self.valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset.dataset,
            batch_sampler=KorQuadSampler(
                valid_dataset.dataset, batch_size=batch_size, drop_last=True
            ),
            collate_fn=lambda x: korquad_collator(
                x, padding_value=valid_dataset.pad_token_id
            ),
            num_workers=4
        )

        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.valid_every = valid_every
        self.lr = lr 
        self.betas = betas 
        self.num_warmup_steps = num_warmup_steps 
        self.num_training_steps = num_training_steps
        self.best_val_ckpt_path = best_val_ckpt_path
        self.best_val_optim_path = best_val_ckpt_path.split(".pt")[0] + "_optim.pt"

        self.start_ep = 1
        self.start_step = 1

    def ibn_loss(self, pred: torch.FloatTensor):
        """in-batch negative를 활용한 batch의 loss를 계산합니다.
        pred : bsz x bsz 또는 bsz x bsz*2의 logit 값을 가짐. 후자는 hard negative를 포함하는 경우.
        """
        bsz = pred.size(0)
        target = torch.arange(bsz).to(self.device)  # 주대각선이 answer
        return torch.nn.functional.cross_entropy(pred, target)

    def batch_acc(self, pred: torch.FloatTensor):
        """batch 내의 accuracy를 계산합니다."""
        bsz = pred.size(0)
        target = torch.arange(bsz)  # 주대각선이 answer
        return (pred.detach().cpu().max(1).indices == target).sum().float() / bsz

    def fit(self):
        """모델을 학습합니다."""
        wandb.init(
            project="kordpr",
            config={
                "batch_size": self.batch_size,
                "lr": self.lr,
                "betas": self.betas,
                "num_warmup_steps": self.num_warmup_steps,
                "num_training_steps": self.num_training_steps,
                "valid_every": self.valid_every,
            }
        )
        logger.debug("start training")
        #self.question_model.train()  # 학습모드
        self.passage_model.train()
        global_step_cnt = 0
        prev_best = None
        for ep in range(self.start_ep, self.num_epoch + 1):
            for step, batch in enumerate(
                tqdm(self.train_loader, desc=f"epoch {ep} batch"), 1
            ):

                if ep == self.start_ep and step < self.start_step:
                    continue  # 중간부터 학습시키는 경우 해당 지점까지 복원
                #self.question_model.train()  # 학습모드
                self.passage_model.train()
                global_step_cnt += 1
                q, q_mask, _, p, p_mask = batch
                q, q_mask, p, p_mask = (
                    q.to(self.device),
                    q_mask.to(self.device),
                    p.to(self.device),
                    p_mask.to(self.device),
                )
                try:
                    q_emb, _ = self.passage_model(q, q_mask, return_dict=False)
                except:
                    print("********************Fail********************")
                    print(q.shape)
                    continue    # bsz x bert_dim
                #p_emb, _ = self.passage_model(p, p_mask, return_dict=False)  # bsz x bert_dim
                p_emb, _ = self.passage_model(p, p_mask, return_dict=False)
                pred = torch.matmul(q_emb[:, 0, :], p_emb[:, 0, :].T)  # bsz x bsz
                loss = self.ibn_loss(pred)
                acc = self.batch_acc(pred)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                log = {
                    "epoch": ep,
                    "step": step,
                    "global_step": global_step_cnt,
                    "train_step_loss": loss.cpu().item(),
                    "current_lr": float(
                        self.scheduler.get_last_lr()[0]
                    ),  # parameter group 1개이므로
                    "step_acc": acc,
                }
                if global_step_cnt % self.valid_every == 0:
                    eval_dict = self.evaluate()
                    if eval_dict == None:
                        continue
                    log.update(eval_dict)
                    if (
                        prev_best is None or eval_dict["valid_loss"] < prev_best
                    ):  # best val loss인 경우 저장
                        # self.model.checkpoint(self.best_val_ckpt_path)
                        self.save_training_state(log)
                wandb.log(log)

    def evaluate(self):
        """모델을 평가합니다."""
        #self.question_model.eval()  # 평가 모드
        self.passage_model.eval()
        loss_list = []
        sample_cnt = 0
        valid_acc = 0
        with torch.no_grad():
            for batch in self.valid_loader:
                q, q_mask, _, p, p_mask = batch
                q, q_mask, p, p_mask = (
                    q.to(self.device),
                    q_mask.to(self.device),
                    p.to(self.device),
                    p_mask.to(self.device),
                )
                try:
                    q_emb, _ = self.question_model(q, q_mask, return_dict=False)  # bsz x bert_dim
                    #q_emb = self.passage_model(q, q_mask, output_hidden_states=True)['hidden_states'][-1]
                except:
                    print("******Fail*********************")
                    print(q.shape)
                    return None
                p_emb, _ = self.passage_model(p, p_mask, return_dict=False)  # bsz x bert_dim
                #p_emb = self.passage_model(p, p_mask, output_hidden_states=True)['hidden_states'][-1]
                pred = torch.matmul(q_emb[:, 0, :], p_emb[:, 0, :].T)  # bsz x bsz
                loss = self.ibn_loss(pred)
                step_acc = self.batch_acc(pred)

                bsz = q.size(0)
                sample_cnt += bsz
                valid_acc += step_acc * bsz
                loss_list.append(loss.cpu().item() * bsz)
        return {
            "valid_loss": np.array(loss_list).sum() / float(sample_cnt),
            "valid_acc": valid_acc / float(sample_cnt),
        }

    def save_training_state(self, log_dict: dict) -> None:
        """모델, optimizer와 기타 정보를 저장합니다"""
        #self.question_model.save_pretrained(self.best_val_ckpt_path+"_roberta")
        self.passage_model.module.save_pretrained(self.best_val_ckpt_path+"_roberta")
        training_state = {
            "optimizer_state": deepcopy(self.optimizer.state_dict()),
            "scheduler_state": deepcopy(self.scheduler.state_dict()),
        }
        training_state.update(log_dict)
        torch.save(training_state, self.best_val_optim_path+"_roberta")
        logger.debug(f"saved optimizer/scheduler state into {self.best_val_optim_path}")

    def load_training_state(self) -> None:
        """모델, optimizer와 기타 정보를 로드합니다"""
        self.model = AutoModel.from_pretrained(self.best_val_ckpt_path)
        training_state = torch.load(self.best_val_optim_path)
        logger.debug(
            f"loaded optimizer/scheduler state from {self.best_val_optim_path}"
        )
        self.optimizer.load_state_dict(training_state["optimizer_state"])
        self.scheduler.load_state_dict(training_state["scheduler_state"])
        self.start_ep = training_state["epoch"]
        self.start_step = training_state["step"]
        logger.debug(
            f"resume training from epoch {self.start_ep} / step {self.start_step}"
        )


if __name__ == "__main__":
    train_dataset = KorQuadDataset("dataset/KorQuAD_v1.0_train.json")
    valid_dataset = KorQuadDataset("dataset/KorQuAD_v1.0_dev.json")
    my_trainer = Trainer(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        num_epoch=40,
        batch_size=128,
        lr=0.00001,
        betas=(0.9, 0.999),
        num_warmup_steps=500,
        num_training_steps=100000,
        valid_every=100,
        best_val_ckpt_path="my_model_roberta",
    )
    #my_trainer.load_training_state()
    my_trainer.fit()
    eval_dict = my_trainer.evaluate()
    print(eval_dict)
