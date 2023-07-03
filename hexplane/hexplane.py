import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import (
    PDFSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc

from hexplane.hexplane_encoding import HexplaneEncoding
from hexplane.hexplane_field import EmptyGridMask, HexPlaneField

"""
    We heuristically set initial time_grid and final time grid. 
    A common strategy to set these two parameters for DNeRF dataset is:
    We empirically set time_grid_final = int(0.24 * N_frames), and time_grid_init = int(0.5 * time_grid_final)
    We show N_frames for each video 
    "standup": 150; "jumpingjacks": 200; "hook"   : 100; "bouncingballs": 150
    "lego"   :  50; "hellwarrior" : 100; "mutant" : 150; "trex"         : 200
"""


@dataclass
class HexPlaneModelConfig(ModelConfig):
    """HexPlane Model Config"""

    _target: Type = field(default_factory=lambda: HexPlaneModel)
    model_name: str = "HexPlane"
    N_voxel_init: int = 32 * 32 * 32
    """Initial voxel number"""
    N_voxel_final: int = 200 * 200 * 200
    """Final voxel number"""
    step_ratio: float = 0.5
    nonsquare_voxel: bool = True
    """If True, voxel numbers along each axis depend on scene length along each axis"""
    time_grid_init: int = 6
    """Initial grid size of time axis"""
    time_grid_final: int = 12
    """Final grid size of time axis"""
    upsample_list: List[int] = field(default_factory=lambda: [3000, 6000, 9000])
    """Upsampling grid resolution step"""
    update_emptymask_list: List[int] = field(default_factory=lambda: [4000, 10000])
    """Updating empty grid step"""
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss": 1.0,})
    """Loss specific weights"""

    # Plane Initialization
    density_n_comp: List[int] = field(default_factory=lambda: [24, 24, 24])
    """Density hexplane R1, R2, R3"""
    app_n_comp: List[int] = field(default_factory=lambda: [48, 48, 48])
    """Appearance hexplane R1, R2, R3"""
    app_dim: int = 27
    """Dimension of appearance feature (Input of MLP)"""
    init_scale: float = 0.1
    init_shift: float = 0.0

    # Fusion Methods
    fusion_one: str = "multiply"
    fusion_two: str = "concat"

    # Plane Index
    mat_mode: List[List[int]] = field(default_factory=lambda: [[0, 1], [0, 2], [1, 2]])  # xy(zt), xz(yt), yz(xt)
    vec_mode: List[int] = field(default_factory=lambda: [2, 1, 0])

    # Density/Appearance Feature Settings
    density_shift: float = -10.0
    mlp_layer_width: int = 128
    mlp_num_layers: int = 3

    # Empty mask settings
    emptymask_thres: float = 0.001
    """Threshold whether the grid is empty or not"""
    raymarch_weight_thres: float = 0.0001

    # Sampling
    align_corners: bool = True
    upsampling_type: str = "unaligned"  # choose from "aligned", "unaligned".
    """
    There are two types of upsampling: aligned and unaligned.
    Aligned upsampling: N_t = 2 * N_t-1 - 1. Like: |--|--| ->upsampling -> |-|-|-|-|, where | represents sampling points and - represents distance.
    Unaligned upsampling: We use linear_interpolation to get the new sampling points without considering the above rule.
    using "unaligned" upsampling will essentially double the grid sizes at each time, ignoring N_voxel_final.
    """
    n_samples: int = 1000
    """
    Maximum number of samples per ray
    self.n_samples = min(self.config.n_samples, int(np.linalg.norm(self.reso_cur) / self.config.step_ratio))
    """
    use_emptymask: bool = False
    """Whether to use empty mask or not"""
    sampling: str = "pdf" # choose from "uniform", "pdf"
    """Ray sampling method"""


class HexPlaneModel(Model):
    """Base HexPlane model

    Args:
        config: Base HexPlane configuration to instantiate model
    """
    def __init__(self, config: HexPlaneModelConfig, **kwargs,) -> None:
        super().__init__(config=config, **kwargs)

        self.config = config
        self.N_voxel_init = config.N_voxel_init
        self.N_voxel_final = config.N_voxel_final
        self.time_grid_init = config.time_grid_init
        self.time_grid_final = config.time_grid_final
        self.upsample_list = config.upsample_list
        self.update_emptymask_list = config.update_emptymask_list
        self.upsampling_type = config.upsampling_type

        self.get_voxel_upsample_list()

        self.n_samples = min(self.config.n_samples, int(np.linalg.norm(self.reso_cur) / self.config.step_ratio))

    def get_voxel_upsample_list(self):
        """Precompute spatial and temporal grid upsampling sizes"""
        # Logaritmic upsampling
        N_voxel_list = []
        if self.upsampling_type == "unaligned":
            N_voxel_list = (
                torch.round(
                    torch.exp(
                        torch.linspace(
                            np.log(self.N_voxel_init), np.log(self.N_voxel_final), len(self.upsample_list) + 1
                        )
                    )
                ).long()
            ).tolist()[1:]
        # Aligned upsampling doesn't need to precompute N_voxel_list
        elif self.upsampling_type == "aligned":
            pass
        # Logaritmic upsampling for time grid.
        Time_grid_list = (
            torch.round(
                torch.exp(
                    torch.linspace(
                        np.log(self.time_grid_init), np.log(self.time_grid_final), len(self.upsample_list) + 1
                    )
                )
            ).long()
        ).tolist()[1:]
        self.N_voxel_list = N_voxel_list
        self.Time_grid_list = Time_grid_list

    def get_training_callbacks(  # pylint:disable=no-self-use
        self, training_callback_attributes: TrainingCallbackAttributes  # pylint: disable=unused-argument
    ) -> List[TrainingCallback]:
        # Update the emptiness voxel
        def update_emptiness_voxel(self, training_callback_attributes: TrainingCallbackAttributes, step: int):
            if self.config.use_emptymask:
                self.empty_mask = self.update_empty_mask(tuple(self.reso_cur), self.time_grid)
                self.field.empty_mask = self.empty_mask

        # Upsample the volume grid
        def upsample_volume_grid(self, training_callback_attributes: TrainingCallbackAttributes, step: int):
            if self.upsampling_type == "aligned":
                self.reso_cur = [self.reso_cur[i] * 2 - 1 for i in range(len(self.reso_cur))]
            else:
                N_voxel = self.N_voxel_list.pop(0)
                self.reso_cur = self.N_to_reso(N_voxel, self.scene_box.aabb, self.config.nonsquare_voxel)
            self.time_grid = self.Time_grid_list.pop(0)
            self.n_samples = min(self.config.n_samples, int(np.linalg.norm(self.reso_cur) / self.config.step_ratio))
            self.field.density_encoding.up_sampling_planes(self.reso_cur, self.time_grid)
            self.field.color_encoding.up_sampling_planes(self.reso_cur, self.time_grid)
            self.update_stepSize()
            # reinitialize optimizers
            optimizers_config = training_callback_attributes.optimizers.config
            for param_group_name, params in optimizers_config.items():
                opt_params = training_callback_attributes.pipeline.get_param_groups()[param_group_name]
                lr_init = optimizers_config[param_group_name]["optimizer"].lr
                training_callback_attributes.optimizers.optimizers[param_group_name] = optimizers_config[
                    param_group_name
                ]["optimizer"].setup(params=opt_params)
                if optimizers_config[param_group_name]["scheduler"]:
                    training_callback_attributes.optimizers.schedulers[param_group_name] = (
                        optimizers_config[param_group_name]["scheduler"]
                        .setup()
                        .get_scheduler(
                            optimizer=training_callback_attributes.optimizers.optimizers[param_group_name],
                            lr_init=lr_init,
                        )
                    )

        callbacks = [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=tuple(self.update_emptymask_list),
                func=update_emptiness_voxel,
                args=[self, training_callback_attributes],
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=tuple(self.upsample_list),
                func=upsample_volume_grid,
                args=[self, training_callback_attributes],
            ),
        ]
        return callbacks

    def update_stepSize(self):
        print("aabb", self.scene_box.aabb.view(-1))
        print("grid size", self.reso_cur)
        device = self.scene_box.aabb.device
        self.aabbSize = self.scene_box.aabb[1] - self.scene_box.aabb[0]
        self.invaabbSize = 2.0 / self.aabbSize
        self.gridSize = torch.LongTensor(self.reso_cur).to(device)
        if self.field.empty_mask is not None:
            self.field.empty_mask.gridSize = self.gridSize
        self.units = self.aabbSize / (self.gridSize - 1)
        self.stepSize = torch.mean(self.units) * self.config.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.n_samples = int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.n_samples)

    def update_to_step(self, step: int) -> None:
        if step < self.upsample_list[0]:
            return
        new_iters = list(self.upsample_list) + [step + 1]
        new_iters.sort()
        index = new_iters.index(step + 1)
        if self.upsampling_type == "unaligned":
            for _ in range(index):
                N_voxel = self.N_voxel_list.pop(0)
            self.reso_cur = self.N_to_reso(N_voxel, self.scene_box.aabb, self.config.nonsquare_voxel)
        elif self.upsampling_type == "aligned":
            for _ in range(index):
                self.reso_cur = [self.reso_cur[i] * 2 - 1 for i in range(len(self.reso_cur))]
        else:
            raise NotImplementedError(f"No such upsampling_type: '{self.upsampling_type}'")
        new_time_res = self.Time_grid_list[index - 1]
        for _ in range(index):
            self.time_grid = self.Time_grid_list.pop(0)

        print("STEP:", step)
        print("new grid resolution:", *self.reso_cur)
        print("new time resolution:", new_time_res)
        self.field.density_encoding.up_sampling_planes(self.reso_cur, self.time_grid)
        self.field.color_encoding.up_sampling_planes(self.reso_cur, self.time_grid)
        self.update_stepSize()

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        # Grid resolution
        self.reso_cur = self.N_to_reso(self.config.N_voxel_init, self.scene_box.aabb, self.config.nonsquare_voxel)
        self.time_grid = self.config.time_grid_init
        # Input encoding
        density_encoding = HexplaneEncoding(
            gridSize=self.reso_cur,
            time_grid=self.config.time_grid_init,
            num_components=self.config.density_n_comp,
            init_scale=self.config.init_scale,
            init_shift=self.config.init_shift,
            mat_mode=self.config.mat_mode,
            vec_mode=self.config.vec_mode,
            fusion_one=self.config.fusion_one,
            fusion_two=self.config.fusion_two,
        )
        color_encoding = HexplaneEncoding(
            gridSize=self.reso_cur,
            time_grid=self.config.time_grid_init,
            num_components=self.config.app_n_comp,
            init_scale=self.config.init_scale,
            init_shift=self.config.init_shift,
            mat_mode=self.config.mat_mode,
            vec_mode=self.config.vec_mode,
            fusion_one=self.config.fusion_one,
            fusion_two=self.config.fusion_two,
        )
        feature_encoding = NeRFEncoding(in_dim=self.config.app_dim, num_frequencies=2, min_freq_exp=0, max_freq_exp=2)
        direction_encoding = NeRFEncoding(in_dim=3, num_frequencies=2, min_freq_exp=0, max_freq_exp=2)

        # HexPlane Field
        self.field = HexPlaneField(
            self.scene_box.aabb,
            feature_encoding=feature_encoding,
            direction_encoding=direction_encoding,
            density_encoding=density_encoding,
            color_encoding=color_encoding,
            appearance_dim=self.config.app_dim,
            mlp_num_layers=self.config.mlp_num_layers,
            mlp_layer_width=self.config.mlp_layer_width,
        )

        self.update_stepSize()

        # EmptyMaskGrid
        self.empty_mask = None

        # Ray samplers
        self.sampler_uniform = UniformSampler(num_samples=self.n_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.n_samples, include_original=False)

        # Colliders
        if self.config.enable_collider:
            self.collider = AABBBoxCollider(scene_box=self.scene_box)

        # Renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # Losses & Regularizations
        self.rgb_loss = MSELoss()

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Returns the parameter groups needed to optimize your model components."""
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = (
            list(self.field.mlp_head.parameters())
            + list(self.field.density_basis_mat.parameters())
            + list(self.field.color_basis_mat.parameters())
            + list(self.field.field_output_rgb.parameters())
        )
        param_groups["encodings"] = list(self.field.color_encoding.parameters()) + list(
            self.field.density_encoding.parameters()
        )

        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        """Process a RayBundle object and return RayOutputs describing quanties for each ray."""
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")

        ray_samples_uniform: RaySamples = self.sampler_uniform(ray_bundle)
        empty_mask = None
        if self.empty_mask:
            positions = SceneBox.get_normalized_positions(ray_samples_uniform.frustums.get_positions(), self.field.aabb)
            positions = positions * 2 - 1
            emptiness = self.empty_mask.sample_empty(positions).reshape(*ray_samples_uniform.shape)
            empty_mask = (emptiness > 0).to(self.device)  # emptiness = 1: Exist, = 0: Non-exist
        
        density = self.field.get_density(ray_samples_uniform, mask=empty_mask, density_shift=self.config.density_shift)
        weights = ray_samples_uniform.get_weights(density)

        if self.config.sampling == "uniform":
            coarse_accumulation = self.renderer_accumulation(weights)
            acc_mask = torch.where(coarse_accumulation < self.config.raymarch_weight_thres, False, True).reshape(-1)
            rgb = self.field.get_outputs(ray_samples_uniform, mask=acc_mask, bg_color=colors.WHITE.to(self.device))
        elif self.config.sampling == "pdf":
            ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights)
            if self.empty_mask:
                positions = SceneBox.get_normalized_positions(ray_samples_pdf.frustums.get_positions(), self.field.aabb)
                positions = positions * 2 - 1
                emptiness = self.empty_mask.sample_empty(positions).reshape(*ray_samples_pdf.shape)
                empty_mask = (emptiness > 0).to(self.device)  # emptiness = 1: Exist, = 0: Non-exist
            density = self.field.get_density(ray_samples_pdf, mask=empty_mask)
            weights = ray_samples_pdf.get_weights(density)
            coarse_accumulation = self.renderer_accumulation(weights)
            acc_mask = torch.where(coarse_accumulation < self.config.raymarch_weight_thres, False, True).reshape(-1)
            rgb = self.field.get_outputs(ray_samples_pdf, mask=acc_mask, bg_color=colors.WHITE.to(self.device))
        else:
            raise NotImplementedError(f"No such sampling method: '{self.config.sampling}'")

        acc_map = torch.clamp(coarse_accumulation, min=0)
        rgb_map = self.renderer_rgb(rgb, weights)
        rgb_map = torch.where(acc_map < 0, colors.WHITE.to(rgb_map.device), rgb_map)
        depth_map = self.renderer_depth(weights, ray_samples_uniform)

        outputs = {"rgb": rgb_map, "accumulation": acc_map, "depth": depth_map}
        return outputs

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Any]:
        """Returns metrics dictionary which will be plotted with wandb or tensorboard."""
        return super().get_metrics_dict(outputs, batch)

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, Any]:
        """Returns a dictionary of losses to be summed which will be your loss."""
        device = outputs["rgb"].device
        image = batch["image"].to(device)

        # RGB loss
        rgb_loss = self.rgb_loss(image, outputs["rgb"])
        loss_dict = {
            "rgb_loss": rgb_loss,
        }
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, Any], batch: Dict[str, Any]
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Returns a dictionary of images and metrics to plot. Here you can apply your colormaps."""
        image = batch["image"].to(outputs["rgb"].device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)

        # Switch image from (H, W, C) to (1, C, H, W) for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        dssim = 1 - ssim / 2
        lpips = self.lpips(image, rgb)

        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
            "dssim": float(dssim.item()),
            "lpips": float(lpips.item()),
        }
        images_dict = {"img": combined_rgb, "accumulation": acc, "depth": depth}
        return metrics_dict, images_dict

    def N_to_reso(self, n_voxels, bbox, adjusted_grid=True):
        """
        Args:
            n_voxels: N_voxel_init
            bbox: aabb
            adjusted_grid: nonsquare_voxel (True for dnerf dataset)
        """
        if adjusted_grid:
            # scene size
            xyz_min, xyz_max = bbox
            # root(xyz, 1/3) = unit voxel size
            voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / 3)
            # voxel length at each axis
            return ((xyz_max - xyz_min) / voxel_size).long().tolist()
        else:
            # grid_each = n_voxels.pow(1 / 3)
            grid_each = math.pow(n_voxels, 1 / 3)
            return [int(grid_each), int(grid_each), int(grid_each)]

    @torch.no_grad()
    def update_empty_mask(self, gridSize, time_grid):
        emptiness, dense_xyz = self.get_dense_empty(gridSize, time_grid)
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        emptiness = emptiness.clamp(0, 1).transpose(0, 2).contiguous()[None, None]

        ks = 3
        if not isinstance(gridSize, tuple):
            gridSize = tuple(gridSize)
        emptiness = F.max_pool3d(emptiness, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        emptiness[emptiness >= self.config.emptymask_thres] = 1
        emptiness[emptiness < self.config.emptymask_thres] = 0

        return EmptyGridMask(self.scene_box.aabb, emptiness, self.device)

    @torch.no_grad()
    def get_dense_empty(self, gridSize, time_grid):
        samples = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, gridSize[0]), torch.linspace(0, 1, gridSize[1]), torch.linspace(0, 1, gridSize[2]),
            ),
            -1,
        ).to(self.device) 
        dense_xyz = samples * 2.0 - 1.0
        emptiness = torch.zeros_like(dense_xyz[..., 0])
        for i in range(gridSize[0]):
            emptiness[i] = self.compute_emptiness(dense_xyz[i].view(-1, 3).contiguous(), time_grid, self.stepSize).view(
                (gridSize[1], gridSize[2])
            )
        return emptiness, dense_xyz

    def compute_emptiness(self, xyz_locs, time_grid, length):
        if self.empty_mask is not None:
            emptiness = self.empty_mask.sample_empty(xyz_locs)
            empty_mask = emptiness > 0
        else:
            empty_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)
        if empty_mask.any():
            xyz_sampled = xyz_locs[empty_mask]
            time_samples = torch.linspace(-1, 1, time_grid).to(xyz_sampled.device)
            N, T = xyz_sampled.shape[0], time_samples.shape[0]
            xyz_sampled = xyz_sampled.unsqueeze(1).expand(-1, T, -1).contiguous().view(-1, 3)
            time_samples = time_samples.unsqueeze(0).expand(N, -1).contiguous().view(-1, 1)

            density = self.field.density_encoding(xyz_sampled, time_samples)
            density = torch.sum(density, dim=-1)  # (N, T)
            density = torch.amax(density, -1)
            sigma[empty_mask] = density

        emptiness = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])

        return emptiness