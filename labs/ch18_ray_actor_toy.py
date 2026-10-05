from __future__ import annotations
from dataclasses import dataclass, field
from queue import Queue, Empty
from threading import Thread, Lock
from time import monotonic, sleep
from typing import Callable, Any

@dataclass
class Call:
    name: str
    fn: Callable[..., Any]
    args: tuple
    kwargs: dict
    resources: int = 1
    retries: int = 0
    submitted: float = field(default_factory=monotonic)

class Actor:
    def __init__(self, name: str, initial: int = 0, fail_once: bool = False):
        self.name = name
        self.value = initial
        self.fail_once = fail_once
        self._lock = Lock()

    def add(self, x: int) -> int:
        with self._lock:
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("simulated actor crash before commit")
            self.value += x
            return self.value

    def snapshot(self) -> int:
        with self._lock:
            return self.value

class CpuRuntime:
    def __init__(self, workers: int = 2):
        self.queue: Queue[Call] = Queue()
        self.capacity = workers
        self.results: Queue[tuple[str, Any, float, float]] = Queue()
        self.threads = [Thread(target=self._loop, daemon=True) for _ in range(workers)]
        for t in self.threads:
            t.start()

    def submit(self, call: Call):
        self.queue.put(call)

    def _loop(self):
        while True:
            call = self.queue.get()
            start = monotonic()
            try:
                value = call.fn(*call.args, **call.kwargs)
                self.results.put((call.name, value,
                                  start - call.submitted,
                                  monotonic() - start))
            except Exception as exc:
                if call.retries > 0:
                    call.retries -= 1
                    self.queue.put(call)
                else:
                    self.results.put((call.name, exc,
                                      start - call.submitted,
                                      monotonic() - start))
            finally:
                self.queue.task_done()

def square(x: int) -> int:
    sleep(0.01)
    return x * x

def main() -> None:
    rt = CpuRuntime(workers=2)
    actor = Actor("counter", initial=10, fail_once=True)
    for i in range(6):
        rt.submit(Call(f"square-{i}", square, (i,), {}, retries=1))
    rt.submit(Call("actor-add", actor.add, (5,), {}, retries=1))
    rt.queue.join()
    rows = []
    while True:
        try:
            rows.append(rt.results.get_nowait())
        except Empty:
            break
    for row in sorted(rows):
        print(row)
    print("actor_value", actor.snapshot())

if __name__ == "__main__":
    main()
