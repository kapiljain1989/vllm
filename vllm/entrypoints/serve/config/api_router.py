# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, FastAPI, Request

from vllm.config import VllmConfig
from vllm.logger import init_logger

from .protocol import (
    ChunkedLocalAttentionGroupSpec,
    CrossAttentionGroupSpec,
    FullAttentionGroupSpec,
    HMAInfo,
    InferenceConfigResponse,
    KVCacheGroupInfo,
    KVCacheInfo,
    MambaGroupSpec,
    MLAAttentionGroupSpec,
    ParallelismInfo,
    SinkFullAttentionGroupSpec,
    SlidingWindowGroupSpec,
    UniformTypeGroupSpec,
)

logger = init_logger(__name__)

router = APIRouter()

_SPEC_TYPE_TO_MODEL: dict[str, type] = {
    "FullAttentionSpec": FullAttentionGroupSpec,
    "MLAAttentionSpec": MLAAttentionGroupSpec,
    "SlidingWindowSpec": SlidingWindowGroupSpec,
    "ChunkedLocalAttentionSpec": ChunkedLocalAttentionGroupSpec,
    "MambaSpec": MambaGroupSpec,
    "CrossAttentionSpec": CrossAttentionGroupSpec,
    "SinkFullAttentionSpec": SinkFullAttentionGroupSpec,
    "UniformTypeKVCacheSpecs": UniformTypeGroupSpec,
}


def _build_group_spec(group: dict) -> KVCacheGroupInfo:
    """Map one serialized group dict to its API protocol counterpart."""
    spec_type = group["spec_type"]
    model_cls = _SPEC_TYPE_TO_MODEL.get(spec_type)
    if model_cls is None:
        raise ValueError(f"Unhandled KVCacheSpec type: {spec_type!r}")
    return model_cls(**group)


def _build_response(
    vllm_config: VllmConfig,
    kv_cache_groups: list[dict] | None,
) -> InferenceConfigResponse:
    cache_cfg = vllm_config.cache_config
    parallel_cfg = vllm_config.parallel_config
    scheduler_cfg = vllm_config.scheduler_config

    groups: list[KVCacheGroupInfo] = (
        [_build_group_spec(g) for g in kv_cache_groups]
        if kv_cache_groups is not None
        else []
    )

    return InferenceConfigResponse(
        kv_cache=KVCacheInfo(
            num_gpu_blocks=cache_cfg.num_gpu_blocks,
            num_cpu_blocks=cache_cfg.num_cpu_blocks,
            gpu_memory_utilization=cache_cfg.gpu_memory_utilization,
            enable_prefix_caching=cache_cfg.enable_prefix_caching,
            kv_offloading_enabled=cache_cfg.kv_offloading_size is not None,
            kv_offloading_backend=(
                cache_cfg.kv_offloading_backend
                if cache_cfg.kv_offloading_size is not None
                else None
            ),
            kv_offloading_size_gib=cache_cfg.kv_offloading_size,
            groups=groups,
        ),
        parallelism=ParallelismInfo(
            tensor_parallel_size=parallel_cfg.tensor_parallel_size,
            pipeline_parallel_size=parallel_cfg.pipeline_parallel_size,
            data_parallel_size=parallel_cfg.data_parallel_size,
            data_parallel_rank=parallel_cfg.data_parallel_rank,
            data_parallel_master_ip=parallel_cfg.data_parallel_master_ip,
            data_parallel_master_port=parallel_cfg.data_parallel_master_port,
            data_parallel_rpc_port=parallel_cfg.data_parallel_rpc_port,
        ),
        hma=HMAInfo(
            enabled=not bool(scheduler_cfg.disable_hybrid_kv_cache_manager),
        ),
    )


@router.get("/inference/v1/config")
async def get_inference_config(raw_request: Request) -> InferenceConfigResponse:
    vllm_config: VllmConfig = raw_request.app.state.vllm_config
    kv_cache_groups: list[dict] | None = raw_request.app.state.kv_cache_config
    return _build_response(vllm_config, kv_cache_groups)


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
