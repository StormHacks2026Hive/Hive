import asyncio
import time
from dataclasses import dataclass
from uuid import uuid4
from .chunker import capacity, make_chunks
from .aggregator import aggregate, equivalent
from .models import Registered

@dataclass
class Node:
    node_id: str
    socket: object
    limits: object
    last_heartbeat: float
    busy: str | None = None

class Scheduler:
    def __init__(self, store, timeout=15, heartbeat_timeout=30, max_attempts=4):
        self.store = store
        self.nodes = {}
        self.timeout = timeout
        self.heartbeat_timeout = heartbeat_timeout
        self.max_attempts = max_attempts
        self.lock = asyncio.Lock()

    async def register(self, socket, message):
        if not message.webgpu:
            raise ValueError('WebGPU is required')
        node = Node(uuid4().hex, socket, message.limits, time.monotonic())
        await socket.send_json(Registered(node_id=node.node_id).model_dump())
        self.nodes[node.node_id] = node
        return node

    def disconnect(self, node_id):
        node = self.nodes.pop(node_id, None)
        if node:
            for job in self.store.jobs.values():
                for chunk in job.chunks:
                    if chunk.assigned == node_id and not chunk.done:
                        chunk.assigned = None

    def fail_attempt(self, node, chunk, error):
        node.busy = None
        chunk.assigned = None
        if chunk.attempts >= self.max_attempts:
            job = self.store.get(chunk.message.job_id)
            job.status, job.error = 'failed', error

    def find(self, node, message):
        for job in self.store.jobs.values():
            for chunk in job.chunks:
                if chunk.message.chunk_id == message.chunk_id and chunk.assigned == node.node_id and chunk.message.attempt_id == message.attempt_id and job.status == 'running':
                    return job, chunk
        return None, None

    def result(self, node, result):
        job, chunk = self.find(node, result)
        if not chunk:
            return  # stale attempt, never release the node's current work
        try:
            dtype = next(b.element_type for b in job.kernel.buffers if b.access == 'read_write')
            if job.request.reduce:
                if result.summary is None or result.summary.count != chunk.message.count:
                    raise ValueError('Invalid reduction summary/count')
                if job.request.reduce == 'count' and not 0 <= result.summary.value <= result.summary.count:
                    raise ValueError('Invalid nonzero count')
            elif result.output is None or result.output.dtype != dtype or len(result.output.decode()) != chunk.message.count:
                raise ValueError('Invalid output length or dtype')
            if job.request.verify:
                if chunk.candidates and not equivalent(chunk.candidates[0], result):
                    job.status, job.error = 'failed', 'Verification mismatch between nodes'
                    node.busy, chunk.assigned = None, None
                    return
                chunk.candidates.append(result)
                chunk.verifier_nodes.add(node.node_id)
                if len(chunk.candidates) < 2:
                    node.busy, chunk.assigned = None, None
                    return
            chunk.done = True
            chunk.assigned = None
            node.busy = None
            job.results[chunk.message.chunk_id] = result
            if len(job.results) == len(job.chunks):
                job.result = aggregate(job.chunks, job.results, job.request.reduce, dtype)
                job.status = 'done'
                job.values = None
        except (ValueError, OverflowError) as exc:
            self.fail_attempt(node, chunk, str(exc))

    async def tick(self):
        async with self.lock:
            now = time.monotonic()
            for node in list(self.nodes.values()):
                timed_out = any(c.assigned == node.node_id and c.deadline < now for j in self.store.jobs.values() for c in j.chunks if not c.done)
                if now - node.last_heartbeat > self.heartbeat_timeout or timed_out:
                    self.disconnect(node.node_id)
                    try:
                        await asyncio.wait_for(node.socket.close(code=1013), 1)
                    except Exception:
                        pass
            for job in self.store.jobs.values():
                if job.status in ('done', 'failed'):
                    continue
                if not job.chunks:
                    job.chunks = make_chunks(job.job_id, job.request, job.kernel, job.values, list(self.nodes.values()))
                for chunk in job.chunks:
                    if chunk.done or chunk.assigned:
                        continue
                    if chunk.attempts >= self.max_attempts:
                        job.status, job.error = 'failed', 'Chunk retry limit exceeded'
                        break
                    for node in list(self.nodes.values()):
                        if node.busy or node.node_id in chunk.verifier_nodes or capacity(node.limits, job.request.mode) < chunk.message.count:
                            continue
                        chunk.attempts += 1
                        chunk.assigned, node.busy = node.node_id, chunk.message.chunk_id
                        chunk.deadline = now + self.timeout
                        chunk.message.attempt_id = uuid4().hex
                        chunk.message.timeout_ms = int(self.timeout * 1000)
                        job.status = 'running'
                        try:
                            await asyncio.wait_for(node.socket.send_json(chunk.message.model_dump()), 2)
                        except Exception:
                            self.disconnect(node.node_id)
                        break

    async def run(self):
        while True:
            await self.tick()
            await asyncio.sleep(.25)
