# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import Response

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/reset_prefix_cache")
async def reset_prefix_cache(
    raw_request: Request,
    reset_running_requests: bool = Query(default=False),
    reset_external: bool = Query(default=False),
):
    """
    Reset the local prefix cache.

    Optionally, if the query parameter `reset_external=true`
    also resets the external (connector-managed) prefix cache.

    Note that we currently do not check if the prefix cache
    is successfully reset in the API server.

    Example:
       POST /reset_prefix_cache?reset_external=true
    """
    logger.info("Resetting prefix cache...")

    await engine_client(raw_request).reset_prefix_cache(
        reset_running_requests, reset_external
    )
    return Response(status_code=200)


@router.post("/reset_mm_cache")
async def reset_mm_cache(raw_request: Request):
    """
    Reset the multi-modal cache. Note that we currently do not check if the
    multi-modal cache is successfully reset in the API server.
    """
    logger.info("Resetting multi-modal cache...")
    await engine_client(raw_request).reset_mm_cache()
    return Response(status_code=200)


@router.post("/reset_encoder_cache")
async def reset_encoder_cache(raw_request: Request):
    """
    Reset the encoder cache. Note that we currently do not check if the
    encoder cache is successfully reset in the API server.
    """
    logger.info("Resetting encoder cache...")
    await engine_client(raw_request).reset_encoder_cache()
    return Response(status_code=200)


@router.post("/evict_block")
async def evict_block(
    raw_request: Request,
    block_hash: str = Query(..., description="Block hash (hex string)"),
    group_idx: int = Query(default=0, description="KV cache group index"),
    pod_id: str | None = Query(default=None, description="Pod ID to target"),
):
    """
    Send a BlockRemoved event for a specific block from a specific group on a specific pod.
    This is a dev API for testing KV offload eviction events.

    Args:
        block_hash: Block hash as a hex string
        group_idx: KV cache group index (default 0)
        pod_id: Optional pod ID to target (if None, broadcasts to all)

    Example:
        POST /evict_block?block_hash=deadbeef&group_idx=0&pod_id=pod-123
    """
    logger.info(
        "Sending BlockRemoved event for block_hash=%s, group_idx=%d, pod_id=%s",
        block_hash,
        group_idx,
        pod_id,
    )

    await engine_client(raw_request).evict_offload_block(block_hash, group_idx, pod_id)
    return Response(status_code=200)


@router.post("/store_block")
async def store_block(
    raw_request: Request,
    block_hash: str = Query(..., description="Block hash (hex string)"),
    group_idx: int = Query(default=0, description="KV cache group index"),
    pod_id: str | None = Query(default=None, description="Pod ID to target"),
    block_size: int = Query(default=16, description="Block size in tokens"),
    parent_block_hash: str | None = Query(
        default=None, description="Parent block hash (hex string)"
    ),
    medium: str = Query(default="CPU", description="Storage medium (CPU/GPU)"),
):
    """
    Send a BlockStored event for a specific block to a specific group on a specific pod.
    This is a dev API for testing KV offload storage events.

    Args:
        block_hash: Block hash as a hex string
        group_idx: KV cache group index (default 0)
        pod_id: Optional pod ID to target (if None, broadcasts to all)
        block_size: Block size in tokens (default 16)
        parent_block_hash: Optional parent block hash as hex string
        medium: Storage medium (default "CPU")

    Example:
        POST /store_block?block_hash=deadbeef&group_idx=0&pod_id=pod-123&block_size=16
    """
    logger.info(
        "Sending BlockStored event for block_hash=%s, group_idx=%d, pod_id=%s, "
        "block_size=%d, medium=%s",
        block_hash,
        group_idx,
        pod_id,
        block_size,
        medium,
    )

    await engine_client(raw_request).store_offload_block(
        block_hash, group_idx, pod_id, block_size, parent_block_hash, medium
    )
    return Response(status_code=200)


def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return
    app.include_router(router)
