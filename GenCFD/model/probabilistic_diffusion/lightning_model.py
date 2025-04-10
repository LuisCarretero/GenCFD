import torch.distributed as dist
import wandb
import copy
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from typing import Any, Mapping, Callable, Union, Sequence, TypeVar
from torchmetrics import MeanMetric
from argparse import ArgumentParser
from torch.utils.data import Dataset
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

import GenCFD.diffusion as dfn_lib
from GenCFD.train import train_states
from GenCFD.utils.train_utils import StdMetric
from GenCFD.utils.gencfd_builder import create_sampler
from GenCFD.utils.dataloader_builder import get_dataloader
from GenCFD.utils.visualization_utils import visualize_training_samples
from GenCFD.utils.diffusion_utils import get_noise_weighting, get_noise_sampling

Tensor = torch.Tensor
Metrics = dict  # Placeholder for metrics


M = TypeVar("M")  # Model
SD = TypeVar("SD", bound=train_states.DenoisingModelTrainState)


class LightningDenoisingModel(pl.LightningModule):
    """PyTorch Lightning wrapper for the DenoisingModel"""

    def __init__(
        self, 
        train_dataset: Dataset,
        eval_dataset: Dataset,
        input_shape: tuple,
        output_shape: tuple,
        spatial_resolution: Sequence[int], 
        denoiser: nn.Module, 
        ema_decay: float,
        store_ema: bool,
        noise_sampling: dfn_lib.NoiseLevelSampling | None = None, 
        noise_weighting: dfn_lib.NoiseLossWeighting | None = None, 
        num_eval_noise_levels: int = 5, 
        num_eval_cases_per_lvl: int = 1, 
        min_eval_noise_lvl: float = 1e-3, 
        max_eval_noise_lvl: float = 50.0,
        consistent_weight: float = 0.0, 
        time_cond: bool = False,
        world_size: int = 0,
        args: ArgumentParser | None = None
    ):
        super(LightningDenoisingModel, self).__init__()
        
        # All Arguments used for training and evaluation
        self.args = args # TODO: Replace with sampler args!
        # Datasets
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.spatial_resolution = spatial_resolution
        # NN Model
        self.denoiser = denoiser
        self.ema_decay = ema_decay
        self.store_ema = store_ema
        # Training Specific Settings
        self.noise_sampling = noise_sampling
        self.noise_weighting = noise_weighting
        self.num_eval_noise_levels = num_eval_noise_levels
        self.num_eval_cases_per_lvl = num_eval_cases_per_lvl
        self.min_eval_noise_lvl = min_eval_noise_lvl
        self.max_eval_noise_lvl = max_eval_noise_lvl
        self.consistent_weight = consistent_weight
        self.time_cond = time_cond
        self.world_size = world_size
        self.curr_step = 0
        # Setup Train State that keeps track of the training
        # self.train_state = self.initialize_train_state() if self.global_rank == 0 else None

        # self.ema_model = self.EMAWrapper(self.denoiser, decay=self.ema_decay)
        if store_ema and (not dist.is_initialized() or dist.get_rank() == 0):
            self.ema_model = AveragedModel(self.denoiser, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
            for param in self.ema_model.parameters():
                param.requires_grad=False
        else:
            self.ema_model = None

        # Setup Train Metrics that keeps track of training metrics like losses
        # self.train_metrics = None
        self.train_metrics = {
            'loss': 0.0,
            'samples': 0
        }

    class TrainMetrics(Metrics):
        """Train metrics including mean and std of loss and if required
        computes the mean of the memory profiler."""

        def __init__(self, device: torch.device = None, world_size: int = 1):
            train_metrics = {
                "loss": MeanMetric(
                    sync_on_compute=True if world_size > 1 else False
                ).to(device),
                "loss_std": StdMetric().to(device),  
            }
            super().__init__(metrics=train_metrics)

    
    # class EMAWrapper:
    #     def __init__(self, model: nn.Module, decay: float = 0.999):
    #         self.decay = decay
    #         self.shadow_params = {n: p.clone().detach().to(p.device) for n, p in model.named_parameters()}
            
    #     @torch.no_grad()
    #     def update(self, model: nn.Module):
    #         for name, param in model.named_parameters():
    #             if param.requires_grad:
    #                 self.shadow_params[name] = self.shadow_params[name].to(param.device)
    #                 # self.shadow_params[name].lerp_(param, 1 - self.decay)  # In-place EMA update
    #                 self.shadow_params[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)


    @torch.no_grad()
    def update_ema(self):
        """Manually updates the EMA model in-place."""
        if self.ema_model is None:
            return
        
        for ema_param, model_param in zip(self.ema_model.parameters(), self.denoiser.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)

        
    # @torch.no_grad()
    # def update_ema(self):
    #     """Manually updates the EMA model in-place, only on rank 0."""
    #     if self.ema_model is None or (dist.is_initialized() and dist.get_rank() != 0):
    #         return  # Skip EMA update for non-zero ranks in DDP

    #     for ema_param, model_param in zip(self.ema_model.parameters(), self.denoiser.parameters()):
    #         ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)


    def setup(self, stage: str):
        """Ensure all custom objects are on the correct device"""

        device = self.device
        self.noise_sampling = get_noise_sampling(args=self.args, device=device)
        self.noise_weighting = get_noise_weighting(args=self.args, device=device)
        # self.train_metrics = self.TrainMetrics(device=self.device, world_size=self.world_size)
    
    def log_uniform_sampling(
        self, shape: tuple[int, ...]
        # TODO: ADD to object members
        # clip_min: float = 1e-4,
        # uniform_grid: bool = False,
        # device: torch.device | None = None
    ) -> Tensor:

        s0 = torch.rand((), dtype=torch.float32, device=self.device)
        num_elements = int(np.prod(shape))
        step_size = 1 / num_elements
        grid = torch.linspace(
            0, 1 - step_size, num_elements, dtype=torch.float32, device=self.device
        )
        samples = torch.remainder(grid + s0, 1).reshape(shape)

        log_min = torch.log(torch.as_tensor(clip_min, dtype=samples.dtype, device=device))


        

#     scheme: Diffusion,
#     clip_min: float = 1e-4,
#     uniform_grid: bool = False,
#     device: th.device = None,
# ) -> NoiseLevelSampling:
#     """Samples noise whose natural log follows a uniform distribution."""

#     def _noise_sampling(shape: tuple[int, ...]) -> Tensor:
#         samples = _uniform_samples(shape, uniform_grid, device)
#         log_min = th.log(th.as_tensor(clip_min, dtype=samples.dtype, device=device))
#         log_max = th.log(
#             th.as_tensor(scheme.sigma_max, dtype=samples.dtype, device=device)
#         )
#         samples = (log_max - log_min) * samples + log_min
#         return th.exp(samples)

    def edm_weighting(self):
        pass

    
    def on_load_checkpoint(self, checkpoint):
        """Ensures noise_sampling and noise_weighting are on the correct device after loading"""

        device = self.device
        # self.noise_sampling = self.noise_sampling.to(device)
        # self.noise_weighting = self.noise_weighting.to(device)
        self.noise_sampling = get_noise_sampling(args=self.args, device=device)
        self.noise_weighting = get_noise_weighting(args=self.args, device=device)
        # self.train_metrics = self.TrainMetrics(device=self.device, world_size=self.world_size)


    # def on_after_batch_transfer(self, batch, dataloader_idx):
    #     """Move batch data to the correct device after transfer."""
    #     batch = batch.to(self.device)
    #     return batch
    

    def configure_optimizers(self):

        optimizer = optim.AdamW(
            self.denoiser.parameters(),
            lr=self.args.peak_lr,
            weight_decay=self.args.weight_decay
        )

        return optimizer
    

    # def initialize_train_state(self) -> SD:
    #     """Initializes the training state with EMA and model params"""
    #     return train_states.DenoisingModelTrainState(
    #         # Further parameters can be added here to track
    #         model=self.denoiser if self.store_ema else None,
    #         step=0,
    #         ema_decay=self.ema_decay,
    #         store_ema=self.store_ema,
    #     )

    def forward(
        self, 
        x: Tensor, 
        y: Tensor, 
        sigma: Tensor, 
        time: Tensor | None = None
    ) -> Tensor:
        """Forward pass through the denoising model"""

        if self.time_cond:
            return self.denoiser(x=x, y=y, sigma=sigma, time=time)
        else:
            return self.denoiser(x=x, y=y, sigma=sigma)


    def training_step(self, batch: dict, batch_idx: int) -> float:
        """Override the training step"""
        self.denoiser.train()

        y = batch["initial_cond"]
        x = batch["target_cond"]
        time = batch["lead_time"] if self.time_cond else None
        # pixel_mask = batch.get("pixel_mask", None)
        pixel_mask = None

        batch_size = len(x)

        x_squared = x ** 2
        sigma = self.noise_sampling(shape=(batch_size,))
        weights = self.noise_weighting(sigma)

        if weights.ndim != x.ndim:
            weights = weights.view(-1, *([1] * (x.ndim - 1)))

        noise = torch.randn(x.shape).to(self.device)

        if sigma.ndim != x.ndim:
            noised = x + noise * sigma.view(-1, *([1] * (x.ndim - 1)))
        else:
            noised = x + noise * sigma

        denoised = self(noised, y, sigma, time)

        if pixel_mask is not None:
            mask_expanded = pixel_mask.unsqueeze(-1).unsqueeze(-1)
            mask_expanded = mask_expanded.expand_as(denoised)
            denoised[mask_expanded] = x[mask_expanded] = 0.0

        denoised_squared = denoised ** 2

        if pixel_mask is not None:
            mask_weight = (~mask_expanded).float()
            mean_sq_sq = torch.square(x_squared).sum() / mask_weight.sum()
            mean_sq = torch.square(x).sum() / mask_weight.sum()
            rel_norm = mean_sq / mean_sq_sq

            loss = (weights * torch.square(denoised - x)).sum() / mask_weight.sum()
            loss += (
                self.consistent_weight
                * rel_norm
                * ((weights * torch.square(denoised_squared - x_squared)).sum() / mask_weight.sum())
            )
        else:
            rel_norm = torch.mean(x ** 2 / torch.mean(x_squared ** 2))
            loss = torch.mean(weights * torch.square(denoised - x))
            loss += (
                self.consistent_weight
                * rel_norm
                * torch.mean(weights * torch.square(denoised_squared - x_squared))
            )
        
        self.update_train_metrics(loss=loss.detach().cpu().item(), samples=x.shape[0])

        return loss


    def optimizer_step(
        self, 
        epoch: int, 
        batch_idx: int, 
        optimizer: optim.AdamW, 
        optimizer_closure: Any
    ):
        optimizer.zero_grad() 
        optimizer.step(closure=optimizer_closure)

        self.curr_step += 1

        # next_step = self.train_state.step + 1
        # if isinstance(next_step, Tensor):
        #     next_step= next_step.item()

        if self.store_ema:
            # self.train_state.ema_model.update_parameters(self.denoiser)
            # ema_params = self.train_state.ema_parameters
            # self.ema_model.update(self.denoiser)
            self.update_ema()

        # self.train_state.replace(
        #     step=next_step,
        #     ema=ema_params if self.store_ema else None
        # )


    # def on_train_batch_end(self, outputs: dict, batch: dict, batch_idx: int):
    #     """"Called at the end of every training batch"""
    #     # Update metrics
    #     self.train_metrics['metrics']['loss'].update(outputs['loss'])
    #     self.train_metrics['metrics']['loss_std'].update(outputs['loss'])

    #     # if self.train_state.step % 10 == 0:
    #     if self.curr_step % 100 == 0:
    #         self.log(
    #             'train_loss', 
    #             self.train_metrics['metrics']['loss'].compute(), 
    #             on_step=True, 
    #             on_epoch=False,
    #             prog_bar=True,
    #             logger=True,
    #             sync_dist=True if self.world_size > 1 else False
    #         )
    #         self.log(
    #             'train_loss_std',
    #             self.train_metrics['metrics']['loss_std'].compute(),
    #             on_step=True,
    #             on_epoch=False,
    #             prog_bar=True,
    #             logger=True,
    #             sync_dist=True if self.world_size > 1 else False
    #         )
    #         # Reinitialize metric aggregation
    #         self.train_metrics["metrics"]["loss"].reset()
    #         self.train_metrics["metrics"]["loss_std"].reset()
    
    #     del batch, outputs
    #     torch.cuda.empty_cache()


    def update_train_metrics(self, loss: float, samples: int):
        """"Called at the end of every training batch"""
        # Update metrics
        self.train_metrics['loss'] += loss * samples
        self.train_metrics['samples'] += samples

        # if self.train_state.step % 10 == 0:
        if self.curr_step % 100 == 0:
            self.log(
                'train_loss', 
                self.train_metrics['loss'] / self.train_metrics['samples'], 
                on_step=True, 
                on_epoch=False,
                prog_bar=True,
                logger=True,
                sync_dist=True if self.world_size > 1 else False
            )
            # Reinitialize metric aggregation
            self.train_metrics['loss'] = 0.0
            self.train_metrics['samples'] = 0


    def on_train_epoch_end(self):
        # Ensure only the main process (rank 0) logs the image
        if self.trainer.is_global_zero:
            dataloader = self.val_dataloader()
            batch = next(iter(dataloader))
            batch = {
                k: v.to(self.device, non_blocking=True) for k, v in batch.items()
            }
            u0 = batch['initial_cond']
            u = batch['target_cond']
            lead_time = batch.get('lead_time', None)

            # Initialize sampler:
            denoise_fn = self.inference_fn(
                denoiser=self.denoiser, 
                lead_time=self.time_cond
            )

            sampler = create_sampler(
                args=self.args,
                input_shape=self.output_shape,
                denoise_fn=denoise_fn,
                device=self.device
            )

            gen_samples = sampler.generate(
                num_samples=dataloader.batch_size,
                y=u0,
                lead_time=lead_time
            )

            pil_img = visualize_training_samples(
                gen_samples=gen_samples,
                gt_samples=u,
                batch_size=self.args.batch_size,
                return_img=True
            )

            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log({"training_samples": wandb.Image(pil_img)})

            elif isinstance(self.logger, pl.loggers.TensorBoardLogger):
                transform = transforms.ToTensor()
                img_tensor = transform(pil_img)  # Convert PIL to Tensor
                self.logger.experiment.add_image("training_samples", img_tensor, self.current_epoch)

            del dataloader, batch, u0, u, lead_time, sampler, denoise_fn, gen_samples
            torch.cuda.empty_cache()


    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """Override validation step for metrics"""

        initial_cond = batch["initial_cond"]
        target_cond = batch["target_cond"]
        time = batch["lead_time"] if self.time_cond else None

        rand_idx_set = torch.randint(
            0,
            initial_cond.shape[0],
            (self.num_eval_noise_levels, self.num_eval_cases_per_lvl),
        )

        y = initial_cond[rand_idx_set]
        x = target_cond[rand_idx_set]

        if time is not None:
            time_inputs = time[rand_idx_set]

        sigma = torch.exp(
            torch.linspace(
                np.log(self.min_eval_noise_lvl),
                np.log(self.max_eval_noise_lvl),
                self.num_eval_noise_levels,
            )
        ).to(x.device)

        noise = torch.randn(x.shape).to(x.device)

        if sigma.ndim != x.ndim:
            noised = x + noise * sigma.view(-1, *([1] * (x.ndim - 1)))
        else:
            noised = x + noise * sigma

        denoise_fn = self.inference_fn(self.denoiser, lead_time=False if time is None else True)

        if time is not None:
            denoised = torch.stack(
                [
                    denoise_fn(x=noised[i], y=y[i], sigma=sigma[i].unsqueeze(0), time=time_inputs[i]).detach()
                    for i in range(self.num_eval_noise_levels)
                ]
            )
        else:
            denoised = torch.stack(
                [
                    denoise_fn(x=noised[i], y=y[i], sigma=sigma[i])
                    for i in range(self.num_eval_noise_levels)
                ]
            )

        ema_losses = torch.mean(
            torch.square(denoised - x), dim=[i for i in range(1, x.ndim)]
        )
        eval_losses = {
            f"denoise_lvl{i}": loss.cpu().item() for i, loss in enumerate(ema_losses)
        }

        for i, loss in enumerate(ema_losses):
            self.log(
                f'denoise_lvl{i}', 
                loss.cpu().item(), 
                on_step=True, 
                on_epoch=False,
                logger=True,
                sync_dist=True if self.world_size > 1 else False
            )

        return eval_losses

    # Not useful after every epoch of the evaluation dataset
    # def on_validation_epoch_end(self, batch: dict):
    #     if self.curr_step == 100_000:
    #         # implement visualization pipeline
    #         pass


    def configure_optimizers(self) -> optim.AdamW:
        """Override to specify the optimizer"""
        optimizer = optim.AdamW(self.denoiser.parameters(), lr=1e-3)  # Example optimizer
        return optimizer


    @staticmethod
    def inference_fn(
        denoiser: nn.Module, lead_time: bool = False
    ) -> Tensor:
        """Returns the inference denoising function.
        Args:
          denoiser: Neural Network (NN) Module for the forward pass
          lead_time: If set to True it can be used for datasets which have time
            included. This time value can then be used for conditioning. Commonly
            done for an All2All training strategy.

        Return:
          _denoise: corresponding denoise function
        """
        denoiser.eval()

        if lead_time == False:

            @torch.no_grad()
            def _denoise(
                x: Tensor,
                sigma: float | Tensor,
                y: Tensor,
                cond: Mapping[str, Tensor] | None = None,
            ) -> Tensor:

                if not torch.is_tensor(sigma):
                    sigma = sigma * torch.ones((x.shape[0],))

                return denoiser.forward(x=x, sigma=sigma, y=y)

        elif lead_time == True:
            
            @torch.no_grad()
            def _denoise(
                x: Tensor,
                sigma: float | Tensor,
                y: Tensor,
                time: float | Tensor,
                cond: Mapping[str, Tensor] | None = None,
            ) -> Tensor:

                if not torch.is_tensor(sigma):
                    sigma = sigma * torch.ones((x.shape[0],))

                if not torch.is_tensor(time):
                    time = time * torch.ones((x.shape[0],))

                return denoiser.forward(x=x, sigma=sigma, y=y, time=time)

        else:
            raise ValueError(
                "Lead Time needs to be a boolean, if a time condition is required"
            )

        return _denoise



    def train_dataloader(self):
        return get_dataloader(
            dataset=self.train_dataset,
            args=self.args,
            name=self.args.dataset,
            batch_size=self.args.batch_size,
            num_worker=self.args.worker,
            prefetch_factor=self.args.prefetch_factor if self.args.worker > 0 else None
        )


    def val_dataloader(self):
        return get_dataloader(
            dataset=self.eval_dataset,
            args=self.args,
            name=self.args.dataset,
            batch_size=self.args.batch_size,
            num_worker=self.args.worker,
            prefetch_factor=self.args.prefetch_factor if self.args.worker > 0 else None
        )