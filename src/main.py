import sys
import json
import os
import time
from datetime import datetime
import math
import sys
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import colorama
from colorama import Back, Fore, Style
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
import wandb
import torch.multiprocessing as mp

from utils.core_utils import (
    get_data_collator,
    get_dataset,
    load_model,
    set_seed,
    set_worker_seed,
    get_optimizer,
    get_scheduler,
    BinaryClassificationMetric,
    EarlyStopping
)


log_path = Path(f'log/{datetime.now().strftime("%m%d-%H%M%S")}')


class Trainer():
    def __init__(self,
                 cfg: DictConfig):
        self.cfg = cfg
        
        self.device = 'cuda'
        self.evaluator = BinaryClassificationMetric(self.device)
        self.type = cfg.type
        self.model_name = cfg.model
        self.dataset_name = cfg.dataset
        self.batch_size = cfg.batch_size
        self.num_epoch = cfg.num_epoch
        self.generator = torch.Generator().manual_seed(cfg.seed)
        self.save_path = log_path
        
        if cfg.type == '5-fold':
            self.dataset_range = [1, 2, 3, 4, 5]
        elif cfg.type == 'temporal':
            self.dataset_range = ['temporal']
        else:
            raise ValueError('experiment type not supported')
        
        self.collator = get_data_collator(cfg.model, cfg.dataset, **cfg.data)
    
    def _reset(self, cfg, fold, type):
        drop_last = True if cfg.model == 'CAFE' else False
        train_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='train', **cfg.data)
        test_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='test', **cfg.data)
        test_collator = self.collator
        self.train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=min(32, cfg.batch_size//2), shuffle=True, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed), drop_last=drop_last)
        self.test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, collate_fn=test_collator, num_workers=min(32, cfg.batch_size//2), shuffle=False, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed), drop_last=drop_last)
        if type == 'temporal':
            valid_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='valid', **cfg.data)
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=min(32, cfg.batch_size//2), shuffle=False, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed), drop_last=drop_last)

        steps_per_epoch = math.ceil(len(train_dataset) / cfg.batch_size)
        self.model = load_model(cfg.model, **dict(cfg.para))
        self.model.to(self.device)
        self.optimizer = get_optimizer(self.model, **dict(cfg.opt))
        num_epoch = cfg.num_epoch
        self.scheduler = get_scheduler(self.optimizer, steps_per_epoch=steps_per_epoch, num_epoch=num_epoch, **dict(cfg.sche))
        self.earlystopping = EarlyStopping(patience=cfg.patience, path=self.save_path/'best_model.pth')
        
    def run(self):
        acc_list, f1_list, prec_list, rec_list = [], [], [], []
        for fold in self.dataset_range:
            self._reset(self.cfg, fold, self.type)
            logger.info(f'Current fold: {fold}')
            for epoch in range(self.num_epoch):
                logger.info(f'Current Epoch: {epoch}')
                self._train()
                if self.type == 'temporal':
                    self._valid(split='valid', use_earlystop=True)
                elif self.type == '5-fold':
                    self._valid(split='test', use_earlystop=True)
                if self.earlystopping.early_stop:
                    logger.info(f"{Fore.GREEN}Early stopping at epoch {epoch}")
                    break
                if self.type == 'temporal':
                    self._valid(split='test')
            logger.info(f'{Fore.RED}Best of Acc in fold {fold}:')
            self.model.load_state_dict(torch.load(self.save_path/'best_model.pth', weights_only=False))
            best_metrics = self._valid(split='test', final=True)
            acc_list.append(best_metrics['acc'])
            f1_list.append(best_metrics['f1'])
            prec_list.append(best_metrics['prec'])
            rec_list.append(best_metrics['rec'])
            
        logger.info(f'Best of Acc in all fold: {np.mean(acc_list)}, Best F1: {np.mean(f1_list)}, Best Precision: {np.mean(prec_list)}, Best Recall: {np.mean(rec_list)}')
        
    def _train(self):
        loss_list =  []
        self.model.train()
        pbar = tqdm(self.train_dataloader, bar_format=f"{Fore.BLUE}{{l_bar}}{{bar}}{{r_bar}}")
        for batch in pbar:
            _ = batch.pop('vids')
            inputs = {key: value.to(self.device) for key, value in batch.items()}
            labels = inputs.pop('labels')
            
            if hasattr(self.model, 'name') and self.model.name == 'CAFE':
                output = self.model(**inputs, task='similarity')
                loss = self.model.calculate_loss(*output)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            output = self.model(**inputs)
            if type(output) is dict:
                pred = output['pred']
            else:
                pred = output
            
            loss = None
            if hasattr(self.model, 'name'):
                match self.model.name:
                    case _:
                        loss = F.cross_entropy(pred, labels)
            else:
                loss = F.cross_entropy(pred, labels)

            _, preds = torch.max(pred, 1)
            self.evaluator.update(preds, labels)
            loss_list.append(loss.item())

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
        metrics = self.evaluator.compute()
        # print
        logger.info(f"{Fore.BLUE}Train: Loss: {np.mean(loss_list)}")
        logger.info(f'{Fore.BLUE}Train: Acc: {metrics["acc"]}, F1: {metrics["f1"]}, Precision: {metrics["prec"]}, Recall: {metrics["rec"]}')
    
    def _valid(self, split, use_earlystop=False, final=False):
        loss_list = []
        self.model.eval()
        if split == 'valid' and final:
            raise ValueError('print_wrong only support test split')
        if split == 'valid':
            dataloader = self.valid_dataloader
            split_name = 'Valid'
            fcolor = Fore.YELLOW
        elif split == 'test':
            dataloader = self.test_dataloader
            split_name = 'Test'
            fcolor = Fore.RED
        else:
            raise ValueError('split not supported')
        for batch in tqdm(dataloader, bar_format=f"{fcolor}{{l_bar}}{{bar}}{{r_bar}}"):
            vids = batch.pop('vids')
            inputs = {key: value.to(self.device) for key, value in batch.items()}
            labels = inputs.pop('labels')
        
            with torch.no_grad():
                output = self.model(**inputs)
                if type(output) is dict:
                    pred = output['pred']
                else:
                    pred = output
                loss = F.cross_entropy(pred, labels)
            
            _, preds = torch.max(pred, 1)

            self.evaluator.update(preds, labels)
            loss_list.append(loss.item())
        metrics = self.evaluator.compute()
        
        logger.info(f"{fcolor}{split_name}: Loss: {np.mean(loss_list)}")
        logger.info(f"{fcolor}{split_name}: Acc: {metrics['acc']}, F1: {metrics['f1']}, Precision: {metrics['prec']}, Recall: {metrics['rec']}")
        
        if use_earlystop:
            self.earlystopping(metrics['acc'], self.model)
        return metrics

@hydra.main(version_base=None, config_path="config", config_name="ExMRD_FakeSV")
def main(cfg: DictConfig):
    logger.remove()
    logger.add(log_path / 'log.log', retention="10 days", level="DEBUG")
    logger.add(sys.stdout, level="INFO")
    logger.info(OmegaConf.to_yaml(cfg))
    colorama.init()
    set_seed(cfg.seed)
    # mp.set_sharing_strategy('file_system')
    
    trainer = Trainer(cfg)
    trainer.run()

if  __name__ == '__main__':
    main()