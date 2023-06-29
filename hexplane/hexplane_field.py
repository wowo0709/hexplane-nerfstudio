from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import Encoding, Identity, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames, RGBFieldHead
from nerfstudio.field_components.mlp import MLP
from nerfstudio.fields.base_field import Field, FieldConfig


class EmptyGridMask(torch.nn.Module):
    def __init__(self, aabb: torch.Tensor, empty_volume: torch.Tensor, device):
        super().__init__()
        self.device = device 
        self.aabb = aabb
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0 / self.aabbSize * 2
        self.empty_volume = empty_volume.view(1, 1, *empty_volume.shape[-3:])
        self.gridSize = torch.LongTensor(  
            [empty_volume.shape[-1], empty_volume.shape[-2], empty_volume.shape[-3]]
        ).to(self.device)

    def sample_empty(self, xyz_sampled):
        empty_vals = F.grid_sample(
            self.empty_volume, xyz_sampled.view(1, -1, 1, 1, 3).to(self.device), align_corners=True
        ).view(-1) 
        return empty_vals


class HexPlaneField(Field):
    """HexPlane Field"""

    def __init__(
        self,
        aabb: TensorType,
        # the aabb bounding box of the dataset
        feature_encoding: Encoding = Identity(in_dim=3),
        # the encoding method used for appearance encoding outputs
        direction_encoding: Encoding = Identity(in_dim=3),
        # the encoding method used for ray direction
        density_encoding: Encoding = Identity(in_dim=3),
        # the tensor encoding method used for scene density
        color_encoding: Encoding = Identity(in_dim=3),
        # the tensor encoding method used for scene color
        appearance_dim: int = 27,
        # the number of dimensions for the appearance embedding
        mlp_num_layers: int = 2,
        # number of layers for the MLP
        mlp_layer_width: int = 128,
        # layer width for the MLP
        use_sh: bool = False,
        # whether to use spherical harmonics as the feature decoding function
        sh_levels: int = 2,
        # number of levels to use for spherical harmonics
    ) -> None:
        super().__init__()

        self.aabb = Parameter(aabb, requires_grad=False)
        self.feature_encoding = feature_encoding
        self.direction_encoding = direction_encoding
        self.density_encoding = density_encoding
        self.color_encoding = color_encoding
        self.mlp_head = MLP(
            in_dim=appearance_dim + self.feature_encoding.get_out_dim() + 3 + self.direction_encoding.get_out_dim(),
            num_layers=mlp_num_layers,
            layer_width=mlp_layer_width,
            activation=nn.ReLU(),
            out_activation=nn.ReLU(),
        )

        self.density_basis_mat = nn.Linear(in_features=self.density_encoding.get_out_dim(), out_features=1, bias=False)
        with torch.no_grad():
            self.density_basis_mat.weight.copy_(torch.ones_like(self.density_basis_mat.weight))

        self.use_sh = use_sh
        if self.use_sh:
            self.sh_encoding = SHEncoding(sh_levels)
            self.color_basis_mat = nn.Linear(
                in_features=self.color_encoding.get_out_dim(),
                out_features=3 * self.sh_encoding.get_out_dim(),
                bias=False,
            )
        else:
            self.color_basis_mat = nn.Linear(
                in_features=self.color_encoding.get_out_dim(), out_features=appearance_dim, bias=False
            )

        self.field_output_rgb = RGBFieldHead(in_dim=self.mlp_head.get_out_dim(), activation=nn.Sigmoid())
        self.empty_mask = None

    def get_density(
        self, 
        ray_samples: RaySamples, 
        mask: Optional[TensorType] = None, 
        density_shift: float = -10.0
    ) -> TensorType:
        original_shape = ray_samples.shape
        if mask is not None:
            base_density = torch.zeros(ray_samples.shape)[:, :, None].to(mask.device)
            if mask.any():
                ray_samples = ray_samples[mask]
                input_shape = ray_samples.shape
            else:
                return base_density
        ray_samples = ray_samples.reshape(-1)
        
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1
        density_enc = self.density_encoding(positions, torch.Tensor(ray_samples.times))
        density = self.density_basis_mat(density_enc)
        softplus = nn.Softplus()
        density = softplus(density + density_shift)

        if mask is not None:
            base_density[mask] = density.view(*input_shape, 1)
            base_density.requires_grad_()
            density = base_density
        else:
            density = density.view(*original_shape, 1)
        return density

    def get_outputs(
        self, 
        ray_samples: RaySamples, 
        mask: Optional[TensorType] = None, 
        bg_color: Optional[TensorType] = None, 
        density_embedding: Optional[TensorType] = None
    ) -> TensorType:
        original_shape = ray_samples.shape
        if mask is not None:
            base_rgb = bg_color.repeat(ray_samples[:, :, None].shape)
            if mask.any():
                ray_samples = ray_samples[mask]
                input_shape = ray_samples.shape
            else:
                return base_rgb
        ray_samples = ray_samples.reshape(-1)
        
        d = ray_samples.frustums.directions
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1
        rgb_features = self.color_encoding(positions, torch.Tensor(ray_samples.times))
        rgb_features = self.color_basis_mat(rgb_features)
        d_encoded = self.direction_encoding(d)
        rgb_features_encoded = self.feature_encoding(rgb_features)
        if self.use_sh:
            sh_mult = self.sh_encoding(d)[:, :, None]
            rgb_sh = rgb_features.view(sh_mult.shape[0], sh_mult.shape[1], 3, sh_mult.shape[-1])
            rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
        else:
            out = self.mlp_head(torch.cat([rgb_features, rgb_features_encoded, d, d_encoded], dim=-1))
            rgb = self.field_output_rgb(out)

        if mask is not None:
            base_rgb[mask] = rgb.view(*input_shape, 3)
            base_rgb.requires_grad_()
            rgb = base_rgb
        else:
            rgb = rgb.view(*original_shape, 3)
        return rgb
