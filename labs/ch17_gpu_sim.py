from dataclasses import dataclass
from heapq import heappush, heappop

@dataclass
class Node:
    name: str
    gpus: int
    domain: str
    used: int = 0

@dataclass
class Job:
    name: str
    tenant: str
    gpus: int
    steps: int
    gang: bool
    submit: int
    start: int | None = None
    finish: int | None = None

nodes = [Node("a", 4, "rack-a"), Node("b", 4, "rack-b")]
queue = [
    Job("train-a", "team-a", 4, 5, True, 0),
    Job("serve-b", "team-b", 1, 3, False, 0),
    Job("train-c", "team-a", 4, 4, True, 1),
    Job("serve-d", "team-b", 1, 2, False, 2),
]
quota = {"team-a": 4, "team-b": 2}
used_quota = {tenant: 0 for tenant in quota}
active = []
time = 0
sequence = 0
completed = []

def place(job):
    if used_quota[job.tenant] + job.gpus > quota[job.tenant]:
        return False
    free = [n for n in nodes if n.gpus - n.used > 0]
    if job.gang and sum(n.gpus - n.used for n in free) < job.gpus:
        return False
    # 简化：先选择剩余容量最大的单一 domain，失败时再跨域
    by_domain = {}
    for n in free:
        by_domain.setdefault(n.domain, []).append(n)
    domain_order = sorted(
        by_domain,
        key=lambda d: sum(n.gpus - n.used for n in by_domain[d]),
        reverse=True,
    )
    chosen = []
    for domain in domain_order:
        for n in sorted(by_domain[domain], key=lambda n: n.gpus - n.used, reverse=True):
            take = min(job.gpus - len(chosen), n.gpus - n.used)
            chosen += [n] * take
            if len(chosen) == job.gpus:
                break
        if len(chosen) == job.gpus:
            break
    if len(chosen) != job.gpus:
        return False
    for n in chosen:
        n.used += 1
    used_quota[job.tenant] += job.gpus
    job.start = time
    global sequence
    sequence += 1
    heappush(active, (time + job.steps, sequence, job.name, job, chosen))
    return True

pending = sorted(queue, key=lambda j: (j.submit, -j.gang, j.name))
while pending or active:
    for job in list(pending):
        if job.submit <= time and place(job):
            pending.remove(job)
    if active:
        end, _, _, job, chosen = heappop(active)
        time = end
        for n in chosen:
            n.used -= 1
        used_quota[job.tenant] -= job.gpus
        job.finish = time
        completed.append(job)
    elif pending:
        time = min(j.submit for j in pending)

for job in completed:
    wait = job.start - job.submit
    print(job.name, "wait", wait, "runtime", job.finish - job.start)
