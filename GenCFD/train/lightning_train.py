# Copyright 2024 The CAM Lab at ETH Zurich.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main File to run Training for GenCFD."""
# import netCDF4
import wandb
import time
import os
import math
from pytorch_lightning.loggers import WandbLogger
# Set the cache size and debugging for torch.compile before importing torch
# os.environ["TORCH_LOGS"] = "all"  # or any of the valid log settings
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from torch.distributed import is_initialized
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from GenCFD.train import training_loop
from GenCFD.utils.dataloader_builder import get_dataset_loader
from GenCFD.utils.gencfd_builder import (
    create_denoiser,
    create_callbacks,
    save_json_file,
    get_model
)
from GenCFD.utils.diffusion_utils import (
    get_noise_sampling,
    get_noise_weighting
)
from GenCFD.utils.parser_utils import train_args
from GenCFD.model.probabilistic_diffusion.lightning_model import LightningDenoisingModel
from pytorch_lightning.callbacks.progress import TQDMProgressBar

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# netCDF4.set_chunk_cache(0,0)

torch.set_float32_matmul_precision("high")  # Better performance on newer GPUs!
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0

# Setting global seed for reproducibility
torch.manual_seed(SEED)  # For CPU operations
torch.cuda.manual_seed(SEED)  # For GPU operations
torch.cuda.manual_seed_all(SEED)  # Ensure all GPUs (if multi-GPU) are set


from pytorch_lightning.callbacks.progress import TQDMProgressBar

class ProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate: int = 100):
        super().__init__()
        self._refresh_rate = refresh_rate  # Set the refresh rate

    @property
    def refresh_rate(self):
        """Override property to return the custom refresh rate"""
        return self._refresh_rate

class ModelCheckpointWithEMA(pl.Callback):
    def __init__(self, save_dir: str, every_n_steps: int = 1000, resume: bool = True):
        """
        Callback to save model checkpoints, including EMA model and optimizer state.

        Args:
            save_dir (str): Directory where checkpoints will be saved.
            every_n_steps (int): Frequency (in steps) to save checkpoints.
        """
        super().__init__()
        self.save_dir = save_dir
        self.every_n_steps = every_n_steps
        os.makedirs(save_dir, exist_ok=True)
        self.resume = resume
 

    # def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
    #     """Loads checkpoint at the start of training if resume=True, ensuring all ranks receive the update."""
    #     if not self.resume:
    #         return  # Don't resume if not needed

    #     latest_checkpoint = None
    #     if trainer.is_global_zero:
    #         latest_checkpoint = self._get_latest_checkpoint()

    #     # Broadcast checkpoint path to all ranks
    #     latest_checkpoint = [latest_checkpoint]
    #     dist.broadcast_object_list(latest_checkpoint, src=0)
    #     latest_checkpoint = latest_checkpoint[0]

    #     if latest_checkpoint is not None:
    #         if trainer.is_global_zero:
    #             print(f"Resuming from checkpoint: {latest_checkpoint}")

    #         # Load checkpoint only on rank 0 and broadcast the state
    #         checkpoint = None
    #         if trainer.is_global_zero:
    #             checkpoint = torch.load(latest_checkpoint, map_location=pl_module.device)

    #         # Broadcast checkpoint to all ranks
    #         checkpoint_data = [checkpoint]  # Wrap in a list for PyTorch broadcasting
    #         dist.broadcast_object_list(checkpoint_data, src=0)
    #         checkpoint = checkpoint_data[0]

    #         if checkpoint:
    #             # Determine model state (compiled/DDP)
    #             model_compiled = isinstance(pl_module, torch._dynamo.eval_frame.OptimizedModule)
    #             model_ddp = isinstance(pl_module, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel))
    #             checkpoint_compiled = checkpoint.get("is_compiled", False)
    #             checkpoint_ddp = checkpoint.get("is_parallelized", False)

    #             # Key transformations for compiled and DDP models
    #             keyword_compiled = "_orig_mod."
    #             keyword_ddp = "module."

    #             if not model_compiled and checkpoint_compiled:
    #                 checkpoint["model_state_dict"] = {
    #                     key.replace(keyword_compiled, ""): value
    #                     for key, value in checkpoint["model_state_dict"].items()
    #                 }

    #             if model_compiled and not checkpoint_compiled:
    #                 checkpoint["model_state_dict"] = {
    #                     keyword_compiled + key: value
    #                     for key, value in checkpoint["model_state_dict"].items()
    #                 }

    #             if not model_ddp and checkpoint_ddp:
    #                 checkpoint["model_state_dict"] = {
    #                     key.replace(keyword_ddp, ""): value
    #                     for key, value in checkpoint["model_state_dict"].items()
    #                 }

    #             if model_ddp and not checkpoint_ddp:
    #                 checkpoint["model_state_dict"] = {
    #                     keyword_ddp + key: value
    #                     for key, value in checkpoint["model_state_dict"].items()
    #                 }

    #             # Load model state on all ranks
    #             pl_module.load_state_dict(checkpoint["model_state_dict"])

    #             # Load optimizer state on all ranks
    #             if "optimizer_state_dict" in checkpoint:
    #                 optimizer = trainer.optimizers[0] if trainer.optimizers else None
    #                 if optimizer:
    #                     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #                     if trainer.is_global_zero:
    #                         print(f"Optimizer state restored successfully")

    #             # Load EMA parameters if available
    #             if "ema_param" in checkpoint and checkpoint["ema_param"] is not None:
    #                 pl_module.ema_model.shadow_params = checkpoint["ema_param"]

    #             # Restore global step on all ranks
    #             trainer.fit_loop.epoch_loop._batches_that_stepped = checkpoint.get("step", 0)
    #             trainer.fit_loop.epoch_progress.current.completed = checkpoint.get("step", 0)

    #             if trainer.is_global_zero:
    #                 print(f"Training resumed from step {checkpoint.get('step', 0)}")


    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        """Saves checkpoint every n steps"""
        global_step = trainer.global_step  # Current training step

        if global_step % self.every_n_steps == 0 and trainer.is_global_zero: 
            checkpoint = {
                "model_state_dict": pl_module.state_dict(),
                "optimizer_state_dict": trainer.optimizers[0].state_dict(),  # Get optimizer state
                "ema_param": pl_module.ema_model.module.state_dict() if pl_module.store_ema else None,
                "step": global_step,
                "is_compiled": isinstance(pl_module.denoiser, torch._dynamo.eval_frame.OptimizedModule),
                "is_parallelized": isinstance(pl_module.denoiser, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel))
            }

            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_{global_step}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
    
    def _get_latest_checkpoint(self):
        """Finds the latest checkpoint file in the directory"""
        checkpoints = [f for f in os.listdir(self.save_dir) if f.startswith("checkpoint_") and f.endswith(".pth")]
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))  # Sort by step number
        return os.path.join(self.save_dir, checkpoints[-1])


def init_distributed_mode(args):
    """Initialize a Distributed Data Parallel Environment"""

    args.local_rank = int(os.getenv("LOCAL_RANK", -1))  # Get from environment variable

    if args.local_rank == -1:
        raise ValueError(
            "--local_rank was not set. Ensure torchrun is used to launch the script."
        )

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend="nccl", rank=args.local_rank, world_size=args.world_size
    )

    device = torch.device(f"cuda:{args.local_rank}")
    print(" ")
    print(f"DDP initialized with rank {args.local_rank} and device {device}.")

    return args, device


if __name__ == "__main__":

    # get arguments for training
    args = train_args()

    #Initialize distributed mode (if multi-GPU)
    # if args.world_size > 1:
    #     args, device = init_distributed_mode(args)
    # else:
    #     print(" ")
    #     print(f"Used device: {device}")


    cwd = os.getcwd()
    if args.save_dir is None:
        raise ValueError("Save directory not specified in arguments!")
    savedir = os.path.join(cwd, args.save_dir)
    if not os.path.exists(savedir):
        if (args.world_size > 1 and args.local_rank == 0) or args.world_size == 1:
            os.makedirs(savedir)
            print(f"Created a directory to store metrics and models: {savedir}")

    train_dataset, eval_dataset, dataset, time_cond = get_dataset_loader(
        args=args,
        name=args.dataset,
        split=True,
        split_ratio=0.96,
        only_dataset=True
    )

    # if dist.is_initialized():
    #     dist.destroy_process_group()

    # print(args.local_rank)

    # wandb_logger = WandbLogger(
    #     project=f"{args.dataset}",
    #     save_dir=savedir
    # )

    # breakpoint()

    denoiser = get_model(
        args=args,
        # Here corresponds to the channel dimension with the noise
        in_channels=dataset.input_channel + dataset.output_channel,
        out_channels=dataset.output_channel,
        spatial_resolution=dataset.spatial_resolution,
        time_cond=time_cond,
        device=None,
        dtype=torch.float32
    )

    if args.compile:
        denoiser = torch.compile(denoiser)

    denoising_model = LightningDenoisingModel(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        input_shape=dataset.input_shape,
        output_shape=dataset.output_shape,
        spatial_resolution=dataset.spatial_resolution,
        denoiser=denoiser,
        ema_decay=args.ema_decay,
        store_ema=True,
        time_cond=time_cond,
        world_size=args.world_size,
        args=args
    )

    trainer = pl.Trainer(
        # logger=wandb_logger,
        default_root_dir=savedir,
        devices=args.world_size,
        # accelerator='gpu',
        # strategy='ddp',
        max_epochs=40,
        precision="bf16-mixed",
        log_every_n_steps=20,
        callbacks=[
            ProgressBar(refresh_rate=1), 
            ModelCheckpointWithEMA(
                save_dir=os.path.join(savedir,"checkpoints"),
                every_n_steps=10000
            )
        ],
    )

    trainer.fit(denoising_model)


    # if (args.world_size > 1 and args.local_rank == 0) or args.world_size == 1:
    #     # Save parameters in a JSON File
    #     save_json_file(
    #         args=args,
    #         time_cond=time_cond,
    #         split_ratio=0.96,
    #         out_shape=dataset.output_shape,  # output shape of the prediction
    #         input_channel=dataset.input_channel,
    #         output_channel=dataset.output_channel,
    #         spatial_resolution=dataset.spatial_resolution,
    #         device=device,
    #         seed=SEED,
    #     )

    # denoising_model = create_denoiser(
    #     args=args,
    #     input_channels=dataset.input_channel,
    #     out_channels=dataset.output_channel,
    #     spatial_resolution=dataset.spatial_resolution,
    #     time_cond=time_cond,
    #     device=device,
    #     dtype=args.dtype,
    #     use_ddp_wrapper=True,
    # )

    # if (args.world_size > 1 and args.local_rank == 0) or args.world_size == 1:
    #     # Print number of Parameters:
    #     model_params = sum(
    #         p.numel() for p in denoising_model.denoiser.parameters() if p.requires_grad
    #     )
    #     print(" ")
    #     print(f"Total number of model parameters: {model_params}")
    #     print(" ")

    # Initialize optimizer
    # optimizer = optim.AdamW(
    #     denoising_model.denoiser.parameters(),
    #     lr=args.peak_lr,
    #     weight_decay=args.weight_decay,
    # )

    # trainer = training_loop.trainers.DenoisingTrainer(
    #     model=denoising_model,
    #     optimizer=optimizer,
    #     device=device,
    #     ema_decay=args.ema_decay,
    #     store_ema=True,  # Store ema model as well
    #     track_memory=args.track_memory,
    #     use_mixed_precision=args.use_mixed_precision,
    #     is_compiled=args.compile,
    #     world_size=args.world_size,
    #     local_rank=args.local_rank,
    # )

    # start_train = time.time()

    # if (
    #     ((args.world_size > 1 and args.local_rank == 0) or args.world_size == 1)
    #     and args.log_training
    # ):
    #     # initialize wandb to synch tensorboard results
    #     wandb.tensorboard.patch(root_logdir=savedir)
    #     wandb.init(
    #         project=f"{args.dataset}",
    #         name=f"initial_run",
    #         dir=savedir,
    #         sync_tensorboard=True,
    #         config=vars(args),
    #         resume="allow"
    #     )

    # # Initialize the metric writer
    # metric_writer = (
    #     SummaryWriter(log_dir=savedir) 
    #     if args.local_rank in {0, -1} 
    #     else None
    # )

    # Calculate iteration steps required for 1 epoch
    # steps_per_epoch = int(len(dataset) / (args.batch_size * args.world_size))

    # training_loop.run(
    #     train_dataloader=train_dataloader,
    #     trainer=trainer,
    #     workdir=savedir,
    #     # DDP configs
    #     world_size=args.world_size,
    #     local_rank=args.local_rank,
    #     # Training configs
    #     total_train_steps=args.num_train_steps,
    #     metric_writer=metric_writer,
    #     metric_aggregation_steps=args.metric_aggregation_steps,
    #     # Evaluation configs
    #     eval_dataloader=eval_dataloader,
    #     eval_every_steps=args.eval_every_steps,
    #     # Visualization configs
    #     visualize_every_epoch=True,
    #     visualize_every_steps=steps_per_epoch,
    #     dataset=dataset,
    #     args=args,
    #     # Other configs
    #     num_batches_per_eval=args.num_batches_per_eval,
    #     callbacks=create_callbacks(args, savedir),
    #     device=device
    # )

    # end_train = time.time()
    # elapsed_train = end_train - start_train
    # if (args.world_size > 1 and args.local_rank == 0) or args.world_size == 1:
    #     print(f"Done training. Elapsed time {elapsed_train / 3600} h")

    # if args.world_size > 1:
    #     dist.destroy_process_group()
