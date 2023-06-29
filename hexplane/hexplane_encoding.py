import torch
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType


class HexplaneEncoding(Encoding):
    """Learned vector-matrix encoding proposed by HexPlane
    """
    def __init__(
        self, 
        gridSize: list = [32, 32, 32], 
        time_grid: int = 6,
        num_components: list = [24, 24, 24], 
        init_scale: float = 0.1,
        init_shift: float = 0.0, 
        mat_mode: list = [[0, 1], [0, 2], [1, 2]], # xy(zt), xz(yt), yz(xt)
        vec_mode: list = [2, 1, 0],
        fusion_one: str = "multiply", 
        fusion_two: str = "concat",
    ) -> None:
        super().__init__(in_dim=3)

        self.gridSize = gridSize
        self.time_grid = time_grid
        self.num_components = num_components
        self.init_scale = init_scale
        self.init_shift = init_shift
        self.mat_mode = mat_mode
        self.vec_mode = vec_mode
        self.fusion_one = fusion_one
        self.fusion_two = fusion_two

        self.init_one_hexplane()

    def init_one_hexplane(self):
        plane_coef, line_time_coef = [], []

        for i in range(len(self.vec_mode)):
            vec_id = self.vec_mode[i]
            mat_id_0, mat_id_1 = self.mat_mode[i]

            plane_coef.append(
                nn.Parameter(
                    self.init_scale 
                    * torch.randn(
                        (1, self.num_components[i], self.gridSize[mat_id_1], self.gridSize[mat_id_0])
                    )
                    + self.init_shift
                )
            )
            line_time_coef.append(
                nn.Parameter(
                    self.init_scale
                    * torch.randn(
                        (1, self.num_components[i], self.gridSize[vec_id], self.time_grid)
                    )
                    + self.init_shift
                )
            )

        self.plane_coef = nn.ParameterList(plane_coef)
        self.line_time_coef = nn.ParameterList(line_time_coef)

    def get_out_dim(self) -> int:
        return sum(self.num_components)

    def forward(self, xyz_sampled, frame_time) -> TensorType:
        """
        Args:
            xyz_sampled: sampled points' xyz coordinates.
            frame_time: sampled points' frame time.
        """
        # Prepare coordinates for grid sampling. (Stop gradients from going to sampler)
        # plane_coord: (3, B, 1, 2), coordinates for spatial planes, where plane_coord[:, 0, 0, :] = [[x, y], [x,z], [y,z]].
        plane_coord = (
            torch.stack(
                (
                    xyz_sampled[..., self.mat_mode[0]],
                    xyz_sampled[..., self.mat_mode[1]],
                    xyz_sampled[..., self.mat_mode[2]],
                )
            )
            .detach()
            .view(3, -1, 1, 2)
        )
        # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t, z], [t, y], [t, x]].
        line_time_coord = torch.stack(
            (
                xyz_sampled[..., self.vec_mode[0]], 
                xyz_sampled[..., self.vec_mode[1]], 
                xyz_sampled[..., self.vec_mode[2]],
            )
        )
        line_time_coord = (
            torch.stack(
                (frame_time.expand(3, -1, -1).squeeze(-1), line_time_coord), dim=-1
            )
            .detach()
            .view(3, -1, 1, 2)
        )

        # Extract features from six feature planes
        plane_feat, line_time_feat = [], []
        for idx_plane in range(len(self.plane_coef)):
            # Spatial Plane Feature: Grid sampling on spatial plane[idx_plane] given coordinates plane_coord[idx_plane].
            plane_feat.append( # [3, Components, -1, 1]
                F.grid_sample(
                    self.plane_coef[idx_plane], 
                    plane_coord[[idx_plane]], 
                    align_corners=True
                ).view(-1, *xyz_sampled.shape[:1])
            )
            # Spatial-Temoral Feature: Grid sampling on line-time plane[idx_plane] given coordinates line_time_coord[idx_plane].
            line_time_feat.append(
                F.grid_sample(
                    self.line_time_coef[idx_plane], 
                    line_time_coord[[idx_plane]], 
                    align_corners=True, 
                ).view(-1, *xyz_sampled.shape[:1])
            )
        plane_feat, line_time_feat = torch.stack(plane_feat, dim=0), torch.stack(line_time_feat, dim=0)

        # Fusion One
        if self.fusion_one == "multiply":
            features = plane_feat * line_time_feat
        elif self.fusion_one == "sum":
            features = plane_feat + line_time_feat
        elif self.fusion_one == "concat":
            features = torch.cat([plane_feat, line_time_feat], dim=0)
        else:
            raise NotImplementedError("No such fusion type (fusion one)")

        # Fusion Two
        if self.fusion_two == "multiply":
            features = torch.prod(features, dim=0)
        elif self.fusion_two == "sum":
            features = torch.sum(features, dim=0)
        elif self.fusion_two == "concat":
            features = features.view(-1, features.shape[-1])
        else:
            raise NotImplementedError("No such fusion type (fusion two)")

        features = torch.moveaxis(features.view(sum(self.num_components), *xyz_sampled.shape[:-1]), 0, -1)
        return features # [N, T, 3 * Components]

    @torch.no_grad()
    def up_sampling_planes(self, res_target, time_grid):
        for i in range(len(self.vec_mode)):
            vec_id = self.vec_mode[i]
            mat_id_0, mat_id_1 = self.mat_mode[i]
            self.plane_coef[i] = nn.Parameter(
                F.interpolate(
                    self.plane_coef[i].data, 
                    size=(res_target[mat_id_1], res_target[mat_id_0]), 
                    mode="bilinear", 
                    align_corners=True,
                )
            )
            self.line_time_coef[i] = nn.Parameter(
                F.interpolate(
                    self.line_time_coef[i].data, 
                    size=(res_target[vec_id], time_grid), 
                    mode="bilinear", 
                    align_corners=True,
                )
            )
        self.gridSize = res_target
        self.time_grid = time_grid