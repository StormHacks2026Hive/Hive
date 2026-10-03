from dataclasses import dataclass, field
from .models import JobStatus

@dataclass
class Job:
    job_id: str
    request: object
    kernel: object
    values: list | None
    chunks: list = field(default_factory=list)
    results: dict = field(default_factory=dict)
    status: str = 'queued'
    result: object = None
    error: str | None = None

class MemoryStore:
    def __init__(self):
        self.jobs = {}

    def add(self, job):
        self.jobs[job.job_id] = job

    def get(self, job_id):
        return self.jobs.get(job_id)

    def status(self, job):
        total = len(job.chunks)
        return JobStatus(job_id=job.job_id, status=job.status,
            progress=len(job.results)/total if total else 0,
            completed_chunks=len(job.results), total_chunks=total, result=job.result, error=job.error)
