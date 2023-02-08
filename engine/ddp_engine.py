import os
import sys
import glob
import torch
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from yacs.config import CfgNode
from typing import Dict, Union, Any,List
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from model import BaseMonoDetector
from dataset.mono_dataset import MonoDataset
from solver import CyclicScheduler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.visualizer import Visualizer
from utils.decorators import decorator_timer
from utils.engine_utils import tprint, export_cfg, load_cfg, count_trainable_params, move_ddp_data_device, reduce_loss_dict, progress_to_string_bar


class DDPEngine:
    def __init__(self, 
                 cfg: Union[str, CfgNode], 
                 auto_resume: bool = True,
                 is_test: bool = False):
        
        # Configuration
        if isinstance(cfg, str):
            self.cfg = load_cfg(cfg_file=cfg)
        elif isinstance(cfg, CfgNode):
            self.cfg = cfg
        else:
            raise Exception("Argument 'cfg' must be either a string or a CfgNode.")
        
        self.version = (cfg.VERSION)
        self.description = (cfg.DESCRIPTION)
        
        # Counter Params (1-based Numbering)
        self.epochs = 1
        
        target_epochs = (cfg.SOLVER.OPTIM.NUM_EPOCHS)
        assert (self.epochs <= target_epochs), \
            f"Argument 'target_epochs'({target_epochs}) must be equal to or greater than 'epochs'({self.epochs})."
        self.target_epochs = target_epochs
        self.global_iters = 1
        
        # Period Params
        self.log_period = (cfg.PERIOD.LOG_PERIOD)
        self.val_period = (cfg.PERIOD.EVAL_PERIOD)
        
        # Dataset and Data-Loader
        self.train_dataset, self.train_loader, self.train_sampler = \
            self.build_loader(is_train=True) if not is_test else (None, None)
        self.test_dataset, self.test_loader, _ = self.build_loader(is_train=False, is_ddp=False)
        
        # Model, Optimizer, Schduler
        self.model = self.build_model()
        self.optimizer, self.scheduler = \
            self.build_solver() if not is_test else (None, None)
        
        # Directory
        self.root = (cfg.OUTPUT_DIR)
        self.writer_dir = os.path.join(self.root, 'tf_logs')
        self.weight_dir = os.path.join(self.root, 'checkpoints')
        if not is_test and self.cfg.GPU_ID == 0:
            exist = False
            if os.path.isdir(self.weight_dir) and auto_resume:
                pth_files = sorted(glob.glob(os.path.join(self.weight_dir, r'*.pth')))
                if len(pth_files) > 0:
                    exist = True
                    latest_weight = pth_files[-1]
                    self.load_checkpoint(latest_weight)
                    tprint(f"Existing checkpoint '{latest_weight}' is found and loaded automatically.")
            
            if not exist:
                for dir_ in [self.writer_dir, self.weight_dir]:
                    os.makedirs(dir_, exist_ok=True)
        
            # Writer
            self.writer = SummaryWriter(self.writer_dir)
        
        # Storage
        self.epoch_times = []
        self.entire_losses = []
        
    def build_model(self):
        detector = BaseMonoDetector(
            num_dla_layers=self.cfg.MODEL.BACKBONE.NUM_LAYERS,
            pretrained_backbone=self.cfg.MODEL.BACKBONE.IMAGENET_PRETRAINED,
            dense_heads=self.cfg.DETECTOR)
        detector.cuda(self.cfg.GPU_ID)
        detector = torch.nn.SyncBatchNorm.convert_sync_batchnorm(detector).to(self.cfg.GPU_ID)
        detector = torch.nn.parallel.DistributedDataParallel(detector, device_ids=[self.cfg.GPU_ID],
                                                            output_device=self.cfg.GPU_ID,
                                                            find_unused_parameters=True)

        return detector
    
    def build_solver(self):
        assert (self.model is not None)
        assert (self.train_loader is not None)
        
        optimizer = optim.AdamW(
            params=self.model.parameters(),
            lr=self.cfg.SOLVER.OPTIM.LR,
            weight_decay=self.cfg.SOLVER.OPTIM.WEIGHT_DECAY,
            betas=(0.95, 0.99))
        
        scheduler = None
        if self.cfg.SOLVER.SCHEDULER.ENABLE:
            total_steps = (len(self.train_loader) * self.cfg.SOLVER.OPTIM.NUM_EPOCHS)
            scheduler = CyclicScheduler(
                optimizer,
                total_steps=total_steps,
                target_lr_ratio=self.cfg.SOLVER.SCHEDULER.TARGET_LR_RATIO,
                target_momentum_ratio=(0.85 / 0.95, 1.0),
                period_up=0.4)
            
        return optimizer, scheduler
    
    def build_loader(self, is_train: bool = True, is_ddp: bool = True):
        dataset = MonoDataset(
            base_root=self.cfg.DATA.ROOT,
            split=self.cfg.DATA.TRAIN_SPLIT if is_train else self.cfg.DATA.TEST_SPLIT,
            max_objs=self.cfg.MODEL.HEAD.MAX_OBJS,
            filter_configs={k.lower(): v for k, v in dict(self.cfg.DATA.FILTER).items()})
        
        sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_ddp else None

        loader = DataLoader(
            dataset,
            batch_size=self.cfg.DATA.BATCH_SIZE,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            # shuffle=True if is_train else False,
            collate_fn=dataset.collate_fn,
            drop_last=False,
            sampler=sampler)
        return dataset, loader, sampler
        
    def train(self, resume_from: str = None) -> None:
        assert torch.cuda.is_available(), "CUDA is not available."
        assert (self.epochs < self.target_epochs), \
            "Argument 'target_epochs' must be equal to or greater than 'epochs'."
            
        # Print Info and Export Current Configuration
        if self.cfg.GPU_ID == 0:
            self._print_engine_info()
            export_cfg(self.cfg, os.path.join(self.root, 'config.yaml'))
        
        # Resume Training if 'resume_from' is specified.
        if (resume_from is not None):
            self.load_checkpoint(resume_from)
            tprint(f"Training resumes from '{resume_from}'. (Start Epoch: {self.epochs})")
        
        # Start Training
        if self.cfg.GPU_ID == 0:
            tprint(f"Training will be proceeded from epoch {self.epochs} to epoch {self.target_epochs}.")
            tprint(f"Result files will be saved to '{self.root}'.")
        for epoch in range(self.epochs, self.target_epochs + 1):
            if self.cfg.GPU_ID == 0:
                print(f" Epoch {self.epochs:3d} / {self.target_epochs:3d} ".center(90, "="))
            np.random.seed(np.random.get_state()[1][0] + self.epochs)
            self.train_sampler.set_epoch(self.epochs)

            avg_loss = self.train_one_epoch()
            
            if self.cfg.GPU_ID == 0:
                print(f"\n- Average Loss: {avg_loss:.3f}")

            
            # Validation
            if (self.cfg.GPU_ID == 0) and (self.val_period > 0) and (epoch % self.val_period == 0):
                self.model.eval()
                
                tprint(f"Evaluating on Epoch {epoch}...", indent=True)
                eval_dict = self.evaluate()

                # Write evaluation results to tensorboard.
                self._update_dict_to_writer(eval_dict, tag='eval')
                
                self.model.train()
                
                # Save Checkpoint (.pth)
                self.save_checkpoint(post_fix=None)
        
        # Save Final Checkpoint (.pth)
        if self.cfg.GPU_ID == 0:
            self.save_checkpoint(post_fix='final')
    
    def train_one_epoch(self, weights=None):

        epoch_losses = []
        for batch_idx, data_dict in enumerate(self.train_loader):
            
            self.optimizer.zero_grad()
            
            # Forward
            data_dict = move_ddp_data_device(data_dict, self.current_device)
            _, loss_dict = self.model(data_dict)
            torch.distributed.barrier()

            total_loss = reduce_loss_dict(loss_dict)
            total_loss.backward()
            
            # Save Losses
            step_loss = total_loss.detach().item()
            epoch_losses.append(step_loss)
            self.entire_losses.append(step_loss)
            
            # Clip Gradient (Option)
            if self.cfg.SOLVER.CLIP_GRAD.ENABLE:
                clip_args = {k.lower(): v for k, v in dict(self.cfg.SOLVER.CLIP_GRAD).items()
                             if k not in ['ENABLE']}
                clip_grad_norm_(self.model.parameters(), **clip_args)
            
            # Step
            self.optimizer.step()
            if (self.scheduler is not None):
                self.scheduler.step()
            
            # Update and Log
            if (self.cfg.GPU_ID == 0) and (self.global_iters % self.log_period == 0):
                one_epoch_steps = len(self.train_loader)
                prog_bar = progress_to_string_bar((batch_idx + 1), one_epoch_steps, bins=20)
                recent_loss = sum(self.entire_losses[-100:]) / len(self.entire_losses[-100:])
                print(f"| Progress {prog_bar} | LR {self.current_lr:.6f} | Loss {total_loss.item():8.4f} ({recent_loss:8.4f}) |")
                
                self._update_dict_to_writer(loss_dict, tag='loss')
                
            self._iter_update()
        self._epoch_update()

        # Return Average Loss
        epoch_loss = (sum(epoch_losses) / len(epoch_losses))
        return epoch_loss
                
    @torch.no_grad()
    def evaluate(self):
        cvt_flag = False
        if self.model.training:
            self.model.eval()
            cvt_flag = True
            tprint("Model is converted to eval mode.")
            
        eval_container = {
            'img_bbox': []}
        
        for test_data in tqdm(self.test_loader, desc="Collecting Results..."):
            test_data = move_ddp_data_device(test_data, self.current_device)
            eval_results = self.model.module.batch_eval(test_data)
            
            for field in ['img_bbox']:
                eval_container[field].extend(eval_results[field])
        
        eval_dict = self.test_dataset.evaluate(eval_container,
                                               eval_classes=['Pedestrian', 'Cyclist', 'Car'],
                                               verbose=True)
        
        if cvt_flag:
            self.model.train()
            tprint("Model is converted to train mode.")
        return eval_dict
            
    
    def save_checkpoint(self, 
                        post_fix: str = None,
                        save_after_update: bool = True,
                        verbose: bool = True) -> None:
        
        save_epoch = self.epochs
        if save_after_update:
            save_epoch -= 1
        
        if (post_fix is None):
            file_name = f'epoch_{save_epoch:03d}.pth'
        else:
            file_name = f'epoch_{save_epoch:03d}_{post_fix}.pth'
        file_path = os.path.join(self.weight_dir, file_name)
        
        # Hard-coded
        attr_except = ['cfg', 'writer', 'train_loader', 'test_loader', 'train_dataset', 'test_dataset'
                       'model', 'optimizer', 'scheduler',]
        attrs = {k: v for k, v in self.__dict__.items() \
            if not callable(getattr(self, k)) and (k not in attr_except)}
        engine_dict = {
            'engine_attrs': attrs,
            'state_dict': {
                'model': self.model.module.state_dict() \
                    if (self.model is not None) else None,
                'optimizer': self.optimizer.state_dict() \
                    if (self.optimizer is not None) else None,
                'scheduler': self.scheduler.state_dict() \
                    if (self.scheduler is not None) else None,
            }
        }
        
        torch.save(engine_dict, file_path)
        if verbose:
            tprint(f"Checkpoint is saved to '{file_path}'.")
    
    def load_checkpoint(self, 
                        ckpt_file: str, 
                        verbose: bool = False) -> None:
        
        engine_dict = torch.load(ckpt_file)
        
        
        # Load Engine Attributes
        attrs = engine_dict['engine_attrs']
        for attr_k, attr_v in attrs.items():
            setattr(self, attr_k, attr_v)
        
            
        state_dict = engine_dict['state_dict']
        
        # Load Model
        if (state_dict['model'] is not None) and (self.model is not None):
            self.model.module.load_state_dict(state_dict['model'])
        
        # Load Optimizer
        if (state_dict['optimizer'] is not None) and (self.optimizer is not None):
            self.optimizer.load_state_dict(state_dict['optimizer'])
        
        # Load Scheduler
        if (state_dict['scheduler'] is not None) and (self.scheduler is not None):
            self.scheduler.load_state_dict(state_dict['scheduler'])
        
        if verbose:
            tprint(f"Checkpoint is loaded from '{ckpt_file}'.")
            
    def _epoch_update(self):
        self.epochs += 1
    
    def _iter_update(self):
        self.global_iters += 1
        
    def _update_dict_to_writer(self, data: Dict[str, Union[torch.Tensor, float]], tag: str):
        for k, v in data.items():
            self.writer.add_scalar(f'{tag}/{k}',
                                   scalar_value=v if isinstance(v, float) else v.detach().item(),
                                   global_step=self.global_iters)
       
        
    def _print_engine_info(self):
        print(f"\n==================== Engine Info ====================")
        print(f"- Root: {self.root}")
        print(f"- Version: {self.version}")
        print(f"- Description: {self.description}")
        
        print(f"\n- Seed: {self.cfg.SEED}")
        print(f"- Using {torch.cuda.device_count()} GPUs!")
        print(f"- Master Device: GPU {self.cfg.GPU_ID} ({torch.cuda.get_device_name(self.cfg.GPU_ID)})")
        
        print(f"\n- Model: {self.model.__class__.__name__} (# Params: {count_trainable_params(self.model)})")
        print(f"- Optimizer: {self.optimizer.__class__.__name__}")
        print(f"- Scheduler: {self.scheduler.__class__.__name__}\n")
        
        print(f"- Epoch Progress: {self.epochs}/{self.target_epochs}")
        print(f"- # Train Samples: {len(self.train_dataset)}")
        print(f"- # Test Samples: {len(self.test_dataset)}")
        print(f"=====================================================\n")
        

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']
    
    @property
    def current_device(self) -> torch.device:
        return torch.device(f'cuda:{self.cfg.GPU_ID}')