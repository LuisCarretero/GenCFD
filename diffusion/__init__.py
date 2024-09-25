# Copyright 2024 The swirl_dynamics Authors.
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

"""Diffusion library."""

# pylint: disable=g-importing-member
# pylint: disable=g-multiple-import

from diffusion.diffusion import (
    Diffusion,
    InvertibleSchedule,
    NoiseLevelSampling,
    NoiseLossWeighting,
    create_variance_exploding_scheme,
    create_variance_preserving_scheme,
    edm_weighting,
    exponential_noise_schedule,
    inverse_squared_weighting,
    log_uniform_sampling,
    normal_sampling,
    power_noise_schedule,
    tangent_noise_schedule,
    time_uniform_sampling,
)
from diffusion.guidance import (
    ClassifierFreeHybrid,
    InfillFromSlices,
    Transform as GuidanceTransform,
)
# from diffusion.samplers import (
#     DenoiseFn,
#     OdeSampler,
#     Sampler,
#     ScoreFn,
#     SdeSampler,
#     TimeStepScheduler,
#     edm_noise_decay,
#     exponential_noise_decay,
#     uniform_time,
# )
from model.building_blocks.unets.unets import (
    # AxialMLPInterpConvMerge,
    # InterpConvMerge,
    UNet,
)
from model.building_blocks.unets.unets import PreconditionedConditionalDenoiser as PreconditionedDenoiserUNet
# from model.building_blocks.unets.unets3d import PreconditionedDenoiser3d as PreconditionedDenoiserUNet3d
# from model.building_blocks.unets.unets3d import UNet3d
# from diffusion.vivit import ViViT
# from diffusion.vivit_diffusion import PreconditionedDenoiser as PreconditionedDenoiserViViT
# from diffusion.vivit_diffusion import ViViTDiffusion

from model.probabilistic_diffusion.denoising_model import DenoisingModel
