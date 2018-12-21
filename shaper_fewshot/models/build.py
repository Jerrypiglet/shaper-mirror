from shaper.models.build import build_model, register_model_builder
from .pointnet_cls import build_pointnet_fewshot

register_model_builder("POINTNET_FEWSHOT", build_pointnet_fewshot)
