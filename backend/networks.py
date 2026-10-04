"""Authenticated network membership and persistent node controls."""

import asyncio
import json
import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from . import database as db
from .auth import require_user
from .models import Model

router = APIRouter(prefix="/api", tags=["networks"])


class NetworkCreate(Model):
    name: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=8, max_length=128)


class NetworkJoin(Model):
    network_id: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=8, max_length=128)


class NodeEnroll(Model):
    device_key: str = Field(min_length=16, max_length=128)
    label: str = Field(min_length=1, max_length=80)


class NodeControl(Model):
    action: str = Field(pattern="^(pause|resume|kill|cancel)$")


def access(request, network_id):
    user = require_user(request)
    network = db.member(network_id, user.id)
    if not network:
        raise HTTPException(403, "Join this network first.")
    return user, network


def public_network(network, user):
    return {k: network[k] for k in ("id", "name")} | {
        "owner": network["owner_id"] == user.id
    }


@router.get("/networks")
def networks(request: Request):
    user = require_user(request)
    return [
        public_network(n, user)
        for n in db.query(
            "SELECT n.* FROM networks n JOIN memberships m ON n.id=m.network_id WHERE m.user_id=? ORDER BY n.created DESC",
            (user.id,),
        )
    ]


@router.post("/networks", status_code=201)
def create(payload: NetworkCreate, request: Request):
    user = require_user(request)
    name = payload.name.strip()
    if not name:
        raise HTTPException(422, "Enter a network name.")
    network_id = "HIVE-" + uuid4().hex[:12].upper()
    db.upsert_user(user)
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO networks VALUES(?,?,?,?,?)",
            (
                network_id,
                name,
                db.password_hash(payload.password),
                user.id,
                time.time(),
            ),
        )
        conn.execute("INSERT INTO memberships VALUES(?,?)", (network_id, user.id))
    return {"id": network_id, "name": name, "owner": True}


@router.post("/networks/join")
def join(payload: NetworkJoin, request: Request):
    user = require_user(request)
    network = db.query(
        "SELECT * FROM networks WHERE id=?",
        (payload.network_id.strip().upper(),),
        one=True,
    )
    hashed = (
        network["password_hash"] if network else db.password_hash("unknown-network")
    )
    if not db.password_matches(payload.password, hashed) or not network:
        raise HTTPException(403, "Network ID or password does not match.")
    db.upsert_user(user)
    db.execute(
        "INSERT OR IGNORE INTO memberships VALUES(?,?)", (network["id"], user.id)
    )
    return public_network(network, user)


@router.post("/networks/{network_id}/nodes")
def enroll(network_id: str, payload: NodeEnroll, request: Request):
    user, _ = access(request, network_id)
    node = db.enroll(
        network_id, user, payload.device_key, payload.label.strip() or user.name
    )
    return {"id": node["id"], "mode": node["mode"], "label": node["label"]}


def can_control(user, network, node, worker, pool):
    return user.id in (network["owner_id"], node["user_id"]) or bool(
        worker
        and worker.busy
        and pool.jobs.get(worker.busy[0])
        and pool.jobs[worker.busy[0]].owner_id == user.id
    )


@router.get("/networks/{network_id}")
async def snapshot(network_id: str, request: Request):
    from .pool.routes import pool

    user, network = access(request, network_id)
    live = {w.node_id: w for w in pool.workers.values() if w.network_id == network_id}
    weights = pool.weights(network_id)
    nodes = []
    for node in db.query(
        "SELECT * FROM nodes WHERE network_id=? ORDER BY last_seen DESC", (network_id,)
    ):
        w = live.get(node["id"])
        caps = json.loads(node["capabilities"])
        state = (
            "offline"
            if not w
            else "paused"
            if not w.active or not w.visible
            else w.phase
            if w.busy
            else "idle"
        )
        assignment = None
        if w and w.busy:
            job = pool.jobs.get(w.busy[0])
            chunk = (
                next((c for c in job.chunks if c.chunk_id == w.busy[1]), None)
                if job
                else None
            )
            if chunk:
                assignment = {
                    "job_id": job.job_id,
                    "chunk_id": chunk.chunk_id,
                    "kind": chunk.assignment.get("kind", "image_tile"),
                    "offset": chunk.offset,
                    "count": chunk.count,
                    "frame_index": chunk.frame_index,
                    "attempt": chunk.attempts,
                }
        nodes.append(
            {
                "id": node["id"],
                "name": node["label"],
                "user_id": node["user_id"],
                "status": state,
                "mode": node["mode"],
                "capabilities": caps,
                "device": caps.get("device_type", "browser"),
                "last_seen": node["last_seen"],
                "completed": node["completed"],
                "received": node["received"],
                "sent": node["sent"],
                "weight": weights.get(w.worker_id, 0) if w else 0,
                "cpu_weight": pool.weights(network_id, cpu=True).get(w.worker_id, 0)
                if w
                else 0,
                "work": assignment,
                "can_control": can_control(user, network, node, w, pool),
            }
        )
    jobs = [
        pool.status(j).model_dump() | {"owner_id": j.owner_id}
        for j in pool.jobs.values()
        if j.network_id == network_id
    ]
    return {"network": public_network(network, user), "nodes": nodes, "jobs": jobs}


@router.post("/networks/{network_id}/nodes/{node_id}/control")
async def control(
    network_id: str, node_id: str, payload: NodeControl, request: Request
):
    from .pool.routes import pool

    user, network = access(request, network_id)
    node = db.query(
        "SELECT * FROM nodes WHERE id=? AND network_id=?",
        (node_id, network_id),
        one=True,
    )
    if not node:
        raise HTTPException(404, "Unknown node.")
    worker = next(
        (
            w
            for w in pool.workers.values()
            if w.node_id == node_id and w.network_id == network_id
        ),
        None,
    )
    if not can_control(user, network, node, worker, pool):
        raise HTTPException(
            403,
            "Only this node’s owner, the network owner, or its current job’s sender can control it.",
        )
    mode = {"kill": "off", "pause": "paused", "resume": "running", "cancel": "paused"}[
        payload.action
    ]
    db.execute("UPDATE nodes SET mode=? WHERE id=?", (mode, node_id))
    if worker:
        if payload.action in ("kill", "cancel"):
            pool.disconnect(worker.worker_id, "Stopped from network map")
            try:
                await asyncio.wait_for(
                    worker.socket.close(code=4001, reason="Node stopped"), 1
                )
            except Exception:
                pass
        else:
            worker.active = mode == "running"
            try:
                await asyncio.wait_for(
                    worker.socket.send_json(
                        {"v": 1, "type": "node_control", "mode": mode}
                    ),
                    2,
                )
            except (RuntimeError, asyncio.TimeoutError):
                pool.disconnect(worker.worker_id, "Node disconnected during control")
            pool.publish_workers()
    return {"id": node_id, "mode": mode}


@router.post("/networks/{network_id}/leave")
async def leave(network_id: str, request: Request):
    from .pool.routes import pool

    user, _ = access(request, network_id)
    for w in list(pool.workers.values()):
        if w.network_id == network_id and w.user_id == user.id:
            pool.disconnect(w.worker_id)
            try:
                await w.socket.close(code=4001, reason="Left network")
            except Exception:
                pass
    # Membership remains saved so the user can return without losing node history.
    return {"left": True}
