# if __name__ == "__main__":
#     import sys
#     import os
#     import pathlib

#     ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
#     sys.path.append(ROOT_DIR)
#     os.chdir(ROOT_DIR)

# import os
# import hydra
# import torch
# from omegaconf import OmegaConf
# import pathlib
# from torch.utils.data import DataLoader
# import copy
# import random
# import wandb
# import tqdm
# import numpy as np
# import shutil
# from diffusion_policy.workspace.base_workspace import BaseWorkspace
# from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
# from diffusion_policy.dataset.base_dataset import BaseImageDataset
# from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
# from diffusion_policy.common.json_logger import JsonLogger
# from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
# from diffusion_policy.model.diffusion.ema_model import EMAModel
# from diffusion_policy.model.common.lr_scheduler import get_scheduler
# from accelerate import Accelerator


# OmegaConf.register_new_resolver("eval", eval, replace=True)

# class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
#     include_keys = ['global_step', 'epoch']

#     def __init__(self, cfg: OmegaConf, output_dir=None):
#         super().__init__(cfg, output_dir=output_dir)

#         # set seed
#         seed = cfg.training.seed
#         torch.manual_seed(seed)
#         np.random.seed(seed)
#         random.seed(seed)

#         # configure model
#         self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

#         self.ema_model: DiffusionUnetImagePolicy = None
#         if cfg.training.use_ema:
#             self.ema_model = copy.deepcopy(self.model)

#         # configure training state
#         self.optimizer = hydra.utils.instantiate(
#             cfg.optimizer, params=self.model.parameters())

#         # configure training state
#         self.global_step = 0
#         self.epoch = 0

#     def run(self):
#         cfg = copy.deepcopy(self.cfg)

#         # configure accelerator
#         accelerator = Accelerator(log_with='wandb')
#         wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
#         wandb_cfg.pop('project')
#         accelerator.init_trackers(
#             project_name=cfg.logging.project,
#             config=OmegaConf.to_container(cfg, resolve=True),
#             init_kwargs={"wandb": wandb_cfg}
#         )

#         # resume training
#         if cfg.training.resume:
#             lastest_ckpt_path = self.get_checkpoint_path()
#             if lastest_ckpt_path.is_file():
#                 accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
#                 self.load_checkpoint(path=lastest_ckpt_path)

#         # configure dataset
#         dataset: BaseImageDataset
#         dataset = hydra.utils.instantiate(cfg.task.dataset)
#         assert isinstance(dataset, BaseImageDataset)
#         train_dataloader = DataLoader(dataset, **cfg.dataloader)
#         normalizer = dataset.get_normalizer()

#         # configure validation dataset
#         val_dataset = dataset.get_validation_dataset()
#         val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

#         self.model.set_normalizer(normalizer)
#         if cfg.training.use_ema:
#             self.ema_model.set_normalizer(normalizer)

#         # configure lr scheduler
#         lr_scheduler = get_scheduler(
#             cfg.training.lr_scheduler,
#             optimizer=self.optimizer,
#             num_warmup_steps=cfg.training.lr_warmup_steps,
#             num_training_steps=(
#                 len(train_dataloader) * cfg.training.num_epochs) \
#                     // cfg.training.gradient_accumulate_every,
#             # pytorch assumes stepping LRScheduler every epoch
#             # however huggingface diffusers steps it every batch
#             last_epoch=self.global_step-1
#         )

#         # configure ema
#         ema: EMAModel = None
#         if cfg.training.use_ema:
#             ema = hydra.utils.instantiate(
#                 cfg.ema,
#                 model=self.ema_model)

#         # configure logging
#         wandb_run = wandb.init(
#             dir=str(self.output_dir),
#             config=OmegaConf.to_container(cfg, resolve=True),
#             **cfg.logging
#         )
#         wandb.config.update(
#             {
#                 "output_dir": self.output_dir,
#             }
#         )

#         # configure checkpoint
#         topk_manager = TopKCheckpointManager(
#             save_dir=os.path.join(self.output_dir, 'checkpoints'),
#             **cfg.checkpoint.topk
#         )

#         # accelerator - prepare everything including EMA model
#         if cfg.training.use_ema:
#             train_dataloader, val_dataloader, self.model, self.ema_model, self.optimizer, lr_scheduler = accelerator.prepare(
#                 train_dataloader, val_dataloader, self.model, self.ema_model, self.optimizer, lr_scheduler
#             )
#         else:
#             train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
#                 train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
#             )

#         # save batch for sampling
#         train_sampling_batch = None

#         if cfg.training.debug:
#             cfg.training.num_epochs = 2
#             cfg.training.max_train_steps = 3
#             cfg.training.max_val_steps = 3
#             cfg.training.rollout_every = 1
#             cfg.training.checkpoint_every = 1
#             cfg.training.val_every = 1
#             cfg.training.sample_every = 1

#         # training loop
#         log_path = os.path.join(self.output_dir, 'logs.json.txt')
#         with JsonLogger(log_path) as json_logger:
#             for local_epoch_idx in range(cfg.training.num_epochs):
#                 step_log = dict()
#                 # ========= train for this epoch ==========
#                 if cfg.training.freeze_encoder:
#                     self.model.obs_encoder.eval()
#                     self.model.obs_encoder.requires_grad_(False)

#                 train_losses = list()
#                 with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
#                         leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
#                     for batch_idx, batch in enumerate(tepoch):
#                         # device transfer
#                         # batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
#                         if train_sampling_batch is None:
#                             train_sampling_batch = batch

#                         # compute loss
#                         raw_loss = self.model.compute_loss(batch)
#                         loss = raw_loss / cfg.training.gradient_accumulate_every
#                         loss.backward()

#                         # step optimizer
#                         if self.global_step % cfg.training.gradient_accumulate_every == 0:
#                             self.optimizer.step()
#                             self.optimizer.zero_grad()
#                             lr_scheduler.step()
                        
#                         # update ema
#                         if cfg.training.use_ema:
#                             ema.step(self.model)  # No need to unwrap since we prepared EMA model with accelerator

#                         # logging
#                         raw_loss_cpu = raw_loss.item()
#                         tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
#                         train_losses.append(raw_loss_cpu)
#                         step_log = {
#                             'train_loss': raw_loss_cpu,
#                             'global_step': self.global_step,
#                             'epoch': self.epoch,
#                             'lr': lr_scheduler.get_last_lr()[0]
#                         }

#                         is_last_batch = (batch_idx == (len(train_dataloader)-1))
#                         if not is_last_batch:
#                             # log of last step is combined with validation and rollout
#                             # wandb_run.log(step_log, step=self.global_step)
#                             accelerator.log(step_log, step=self.global_step)
#                             json_logger.log(step_log)
#                             self.global_step += 1

#                         if (cfg.training.max_train_steps is not None) \
#                             and batch_idx >= (cfg.training.max_train_steps-1):
#                             break

#                 # at the end of each epoch
#                 # replace train_loss with epoch average
#                 train_loss = np.mean(train_losses)
#                 step_log['train_loss'] = train_loss

#                 # ========= eval for this epoch ==========
#                 policy = self.model
#                 if cfg.training.use_ema:
#                     policy = self.ema_model
#                 policy.eval()

#                 # run validation
#                 if (self.epoch % cfg.training.val_every) == 0 and accelerator.is_main_process:
#                     with torch.no_grad():
#                         val_losses = list()
#                         with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
#                                 leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
#                             for batch_idx, batch in enumerate(tepoch):
#                                 batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
#                                 loss = self.model.compute_loss(batch)
#                                 val_losses.append(loss)
#                                 if (cfg.training.max_val_steps is not None) \
#                                     and batch_idx >= (cfg.training.max_val_steps-1):
#                                     break
#                         if len(val_losses) > 0:
#                             val_loss = torch.mean(torch.tensor(val_losses)).item()
#                             # log epoch average validation loss
#                             step_log['val_loss'] = val_loss

#                 # run diffusion sampling on a training batch
#                 if (self.epoch % cfg.training.sample_every) == 0 and accelerator.is_main_process:
#                     with torch.no_grad():
#                         # sample trajectory from training set, and evaluate difference
#                         batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
#                         obs_dict = batch['obs']
#                         gt_action = batch['action']
                        
#                         result = policy.predict_action(obs_dict)
#                         pred_action = result['action_pred']
#                         mse = torch.nn.functional.mse_loss(pred_action, gt_action)
#                         step_log['train_action_mse_error'] = mse.item()
#                         del batch
#                         del obs_dict
#                         del gt_action
#                         del result
#                         del pred_action
#                         del mse
                
#                 # checkpoint
#                 if (self.epoch % cfg.training.checkpoint_every) == 0 and accelerator.is_main_process:
                    
#                     # unwrap the model to save ckpt
#                     model_ddp = self.model
#                     self.model = accelerator.unwrap_model(self.model)
                    
#                     # checkpointing
#                     if cfg.checkpoint.save_last_ckpt:
#                         self.save_checkpoint()
#                     if cfg.checkpoint.save_last_snapshot:
#                         self.save_snapshot()

#                     # sanitize metric names
#                     metric_dict = dict()
#                     for key, value in step_log.items():
#                         new_key = key.replace('/', '_')
#                         metric_dict[new_key] = value
                    
#                     # We can't copy the last checkpoint here
#                     # since save_checkpoint uses threads.
#                     # therefore at this point the file might have been empty!
#                     topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

#                     if topk_ckpt_path is not None:
#                         self.save_checkpoint(path=topk_ckpt_path)

#                     # recover the DDP model
#                     self.model = model_ddp
#                 # ========= eval end for this epoch ==========
#                 policy.train()

#                 # end of epoch
#                 # log of last step is combined with validation and rollout
#                 # wandb_run.log(step_log, step=self.global_step)
#                 accelerator.log(step_log, step=self.global_step)
#                 json_logger.log(step_log)
#                 self.global_step += 1
#                 self.epoch += 1

#         accelerator.end_training()

# @hydra.main(
#     version_base=None,
#     config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
#     config_name=pathlib.Path(__file__).stem)
# def main(cfg):
#     workspace = TrainDiffusionUnetImageWorkspace(cfg)
#     workspace.run()

# if __name__ == "__main__":
#     main()if __name__ == "__main__":

import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from accelerate import Accelerator


OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # configure accelerator
        accelerator = Accelerator(
            log_with='wandb' if cfg.logging.mode == 'online' else None
        )
        
        # Only initialize wandb trackers if not already initialized
        if accelerator.is_main_process and cfg.logging.mode == 'online':
            wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            project_name = wandb_cfg.pop('project')
            accelerator.init_trackers(
                project_name=project_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                init_kwargs={"wandb": wandb_cfg}
            )

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        accelerator.print(f"Training dataset size: {len(dataset)}")
        accelerator.print(f"Training dataloader config: batch_size={cfg.dataloader.batch_size}, " 
                         f"num_workers={cfg.dataloader.num_workers}")
        
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        
        # Check if validation is disabled in config
        disable_validation = cfg.training.get('disable_validation', False)
        
        # Check if validation dataset exists and has data
        val_dataloader = None
        self.has_validation = False
        
        if disable_validation:
            accelerator.print("Validation is disabled in config")
        elif val_dataset is not None:
            try:
                val_dataset_len = len(val_dataset)
                accelerator.print(f"Validation dataset size: {val_dataset_len}")
                
                if val_dataset_len > 0:
                    # Check if batch size is appropriate
                    val_batch_size = cfg.val_dataloader.get('batch_size', 1)
                    if val_batch_size > val_dataset_len:
                        accelerator.print(f"Warning: Validation batch size ({val_batch_size}) > dataset size ({val_dataset_len})")
                        accelerator.print(f"Setting validation batch size to {val_dataset_len}")
                        cfg.val_dataloader.batch_size = val_dataset_len
                    
                    # Create validation dataloader
                    temp_val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
                    
                    # Test if we can get at least one batch
                    try:
                        temp_iter = iter(temp_val_dataloader)
                        first_batch = next(temp_iter)
                        if first_batch is not None and len(first_batch) > 0:
                            val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)  # Create fresh dataloader
                            self.has_validation = True
                            accelerator.print(f"Validation dataloader verified with batch_size={cfg.val_dataloader.batch_size}")
                        else:
                            accelerator.print("Warning: Validation dataloader returns empty/None batches")
                    except StopIteration:
                        accelerator.print("Warning: Validation dataloader has no batches")
                    except Exception as e:
                        accelerator.print(f"Warning: Could not verify validation dataloader: {e}")
                else:
                    accelerator.print("Warning: Validation dataset is empty")
            except Exception as e:
                accelerator.print(f"Warning: Error setting up validation dataset: {e}")
        else:
            accelerator.print("Warning: No validation dataset available")

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure logging
        if accelerator.is_main_process:
            # Update wandb config with output directory
            if cfg.logging.mode == 'online':
                wandb.config.update(
                    {
                        "output_dir": self.output_dir,
                    }
                )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # accelerator - prepare everything including EMA model
        # Create a list of objects to prepare, excluding None values
        to_prepare = [train_dataloader]
        
        if val_dataloader is not None and self.has_validation:
            to_prepare.append(val_dataloader)
        
        to_prepare.extend([self.model, self.optimizer, lr_scheduler])
        
        if cfg.training.use_ema:
            to_prepare.insert(-2, self.ema_model)  # Insert before optimizer
        
        # Prepare all objects
        prepared = accelerator.prepare(*to_prepare)
        
        # Unpack prepared objects
        idx = 0
        train_dataloader = prepared[idx]
        idx += 1
        
        if val_dataloader is not None and self.has_validation:
            val_dataloader = prepared[idx]
            idx += 1
        
        self.model = prepared[idx]
        idx += 1
        
        if cfg.training.use_ema:
            self.ema_model = prepared[idx]
            idx += 1
        
        self.optimizer = prepared[idx]
        lr_scheduler = prepared[idx + 1]

        # Save some info about dataloaders for debugging
        accelerator.print(f"Train dataloader batches: {len(train_dataloader)}")
        if val_dataloader is not None:
            accelerator.print(f"Val dataloader batches: {len(val_dataloader)}")
        
        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec, 
                        disable=not accelerator.is_local_main_process) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # Check if batch is valid
                        if batch is None:
                            accelerator.print(f"Warning: Training batch {batch_idx} is None, skipping")
                            continue
                            
                        # No need for manual device transfer with accelerate
                        if train_sampling_batch is None:
                            # Make a copy of the batch for sampling later
                            train_sampling_batch = {k: v.clone() if torch.is_tensor(v) else v 
                                                  for k, v in batch.items()}

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        
                        # Use accelerator.backward instead of loss.backward()
                        accelerator.backward(loss)

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            if accelerator.is_main_process:
                                json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run validation
                if (self.epoch % cfg.training.val_every) == 0 and self.has_validation and val_dataloader is not None:
                    try:
                        with torch.no_grad():
                            val_losses = list()
                            val_batch_count = 0
                            with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                    leave=False, mininterval=cfg.training.tqdm_interval_sec,
                                    disable=not accelerator.is_local_main_process) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    # Check if batch is None
                                    if batch is None:
                                        accelerator.print(f"Warning: Validation batch {batch_idx} is None, skipping")
                                        continue
                                    
                                    val_batch_count += 1
                                    # No need for manual device transfer with accelerate
                                    loss = policy.compute_loss(batch)
                                    val_losses.append(loss)
                                    if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps-1):
                                        break
                            
                            if len(val_losses) > 0:
                                val_loss = torch.mean(torch.stack(val_losses)).item()
                                # log epoch average validation loss
                                step_log['val_loss'] = val_loss
                                accelerator.print(f"Validation complete: {val_batch_count} batches, avg loss: {val_loss:.4f}")
                            else:
                                accelerator.print("Warning: No valid validation batches found")
                    except Exception as e:
                        accelerator.print(f"Error during validation: {e}")
                        accelerator.print("Skipping validation for this epoch")
                elif (self.epoch % cfg.training.val_every) == 0:
                    accelerator.print("Skipping validation - no validation dataloader available")

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    if train_sampling_batch is not None:
                        with torch.no_grad():
                            # sample trajectory from training set, and evaluate difference
                            # No need for manual device transfer
                            obs_dict = train_sampling_batch['obs']
                            gt_action = train_sampling_batch['action']
                            
                            result = policy.predict_action(obs_dict)
                            pred_action = result['action_pred']
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                            step_log['train_action_mse_error'] = mse.item()
                            del obs_dict
                            del gt_action
                            del result
                            del pred_action
                            del mse
                    else:
                        accelerator.print("Warning: train_sampling_batch is None, skipping sampling")
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and accelerator.is_main_process:
                    
                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)
                    if cfg.training.use_ema:
                        ema_model_ddp = self.ema_model
                        self.ema_model = accelerator.unwrap_model(self.ema_model)
                    
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    # recover the DDP model
                    self.model = model_ddp
                    if cfg.training.use_ema:
                        self.ema_model = ema_model_ddp
                        
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                if accelerator.is_main_process:
                    json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()