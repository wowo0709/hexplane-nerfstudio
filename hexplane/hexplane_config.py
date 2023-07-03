from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from hexplane.hexplane import HexPlaneModelConfig

hexplane_method = MethodSpecification(
    config = TrainerConfig(
        method_name="hexplane", 
        steps_per_eval_batch=25000, # eval loss 
        steps_per_eval_image=500, # eval image
        steps_per_eval_all_images= 24999, # eval all images
        steps_per_save=2000, 
        max_num_iterations=25000,
        mixed_precision=False, 
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=DNeRFDataParserConfig(),
                train_num_rays_per_batch=1024, 
                eval_num_rays_per_batch=1024, 
            ),
            model=HexPlaneModelConfig(eval_num_rays_per_chunk=1 << 12),
        ),
        optimizers={
            "fields": { # For V^RF and NN
                "optimizer": AdamOptimizerConfig(lr=0.001),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=25000)
            },
            "encodings": { # For feature planes
                "optimizer": AdamOptimizerConfig(lr=0.02),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.002, max_steps=25000)
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
        vis="viewer"
    ), 
    description="HexPlane: A Fast Representation for Dynamic Scenes"
)