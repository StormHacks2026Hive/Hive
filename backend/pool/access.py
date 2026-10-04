"""Authorize pool resources without changing the legacy wire protocol."""

import os

from fastapi import HTTPException

from ..auth import require_user
from ..database import member


def legacy_enabled():
    return os.getenv("HIVE_ALLOW_LEGACY_POOL", "false").lower() == "true"


def network_context(request, network_id=None):
    if legacy_enabled() and not network_id:
        return None, None
    user = require_user(request)
    network_id = network_id or request.query_params.get("network_id")
    if not network_id or not member(network_id, user.id):
        raise HTTPException(403, "Choose a network you have joined.")
    return user, network_id


def job_access(request, job, *, control=False):
    if not job.network_id and legacy_enabled():
        return None
    user = require_user(request)
    network = member(job.network_id, user.id) if job.network_id else None
    if not network:
        raise HTTPException(404, "Unknown job")
    if control and user.id not in (job.owner_id, network["owner_id"]):
        raise HTTPException(
            403, "Only the sender or network owner can cancel this job."
        )
    return user
