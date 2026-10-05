---
id: ch18-ray-distributed-application-runtimes
title: Ray 与分布式应用运行时：actors、tasks、placement、Serve 与 KubeRay
slug: /chapters/18-ray-distributed-application-runtimes
description: 从 task 和 actor 的生命周期出发，建立资源声明、placement group、Ray Serve、KubeRay、状态与故障恢复的可测量模型，并与纯 Kubernetes 部署比较
sidebar_position: 18
level: systems
prerequisites:
  - ch02-linux-process-files-observability
  - ch03-networking
  - ch08-distributed-collectives
  - ch10-training-ops
  - ch15-inference-execution
  - ch16-model-serving-system
  - ch17-kubernetes-gpu-orchestration
learning_objectives:
  - 能区分 Ray task、actor、对象引用和 placement group 的语义与生命周期
  - 能把 CPU、GPU、自定义资源、对象存储和拓扑约束写成可检查的声明
  - 能解释 Ray Serve 的副本、路由、批处理、autoscaling 与故障恢复路径
  - 能读懂 KubeRay RayCluster、RayJob、RayService 的控制流和升级边界
  - 能用标准库 CPU actor 模拟器测量调度等待、背压、重试和 actor 重启
  - 能在 Ray 与纯 Kubernetes 之间按状态、扩缩容、可观测性和安全边界做决策
  - 能为长任务、服务和训练作业设计幂等、检查点和降级策略
estimated_hours: 26
hardware: CPU-only simulation; Kubernetes and GPU optional
risk_level: L4
last_verified: 2026-10-05
---

# 第18章　Ray 与分布式应用运行时：actors、tasks、placement、Serve 与 KubeRay

> Kubernetes 解决的是“哪些容器应该存在、它们应该落在哪些节点、何时替换不健康容器”。Ray 解决的是“一个 Python 分布式程序如何把函数和有状态对象拆成可调度的执行单元、如何传递结果、如何在资源变化和进程失败时继续推进”。两者不是同一层的替代品。Ray 的 task 和 actor 让应用代码拥有运行时语义，placement group 让一组资源以原子形状被放置，Ray Serve 在运行时内部处理模型副本与请求路由，KubeRay 则把 Ray 集群的生命周期交给 Kubernetes 控制器。本章从一次远程调用的字节流开始，追踪到 worker、对象存储、raylet、Kubernetes Pod 和用户请求，并把每个层的承诺与盲点写成实验可验证的假设。

本章延续全书的证据标记：官方文档或源码直接保证的内容标为“事实”；根据多个组件之间的调用重构出的顺序标为“机制”；实验脚本运行得到的数值标为“测量”；由测量计算的容量、等待时间和成本标为“推断”；为某种可靠性或安全目标提出的规则标为“设计判断”。Ray Dashboard 显示一个 actor 为 `ALIVE` 只能证明控制面仍持有进程记录，不能证明它的业务状态已经落盘；Kubernetes 显示一个 Pod 为 `Running` 也不能证明 Serve 副本可以处理请求。任何 SLO 都必须绑定到用户可观察的成功、延迟和数据完整性。

## 18.1 为什么需要分布式应用运行时

### 18.1.1 容器编排的抽象与应用编程的鸿沟

纯 Kubernetes 以 Pod、Service、Deployment、Job 等对象表达进程集合。它擅长声明副本数、滚动升级、节点约束和网络入口，但应用开发者仍需要处理进程间的序列化、对象生命周期、异步依赖、重试、结果缓存、节点亲和、长任务检查点和缩放信号。一个多阶段推理流水线若直接用 Kubernetes，常见做法是为每个阶段写一个服务，再用消息队列或数据库粘合；每一次中间结果都要经过网络协议和外部序列化，作业状态被拆到多个系统。

Ray 的目标是让 Python 或其他支持语言的程序把普通函数声明为 task，把持久进程声明为 actor，把结果放进分布式对象存储，并让运行时根据资源标签选择 worker。这样，控制流可以仍然写成函数调用的形状，数据流则由对象引用表达。这个抽象减少了“胶水服务”的数量，却把更多语义交给 Ray：对象引用何时解析、任务失败是否重试、actor 状态是否可恢复、资源声明是否精确，都会影响生产正确性。

### 18.1.2 任务、状态和服务是三种不同问题

批量任务通常是有限次调用，输入和输出可以重放，最重要的是吞吐、成本和恢复。Actor 是带身份的长期进程，方法调用依赖它的内存状态、串行顺序或内部并发，最重要的是状态一致性和重启语义。在线服务是持续请求流，要求 admission control、尾延迟、隔离和渐进发布。Ray 把三者放在一个运行时中，但不会自动把批任务变成可恢复服务，也不会把 actor 内存变成持久数据库。架构设计第一步是明确状态、时限和重放边界，再选 task、actor、Serve 或外部存储。

### 18.1.3 什么时候不该引入 Ray

如果工作负载只是少量无状态 HTTP 副本、节点数量很小、扩缩容与滚动更新是主要需求，Kubernetes Deployment 加一个成熟网关通常更简单。若团队只需要 DAG 编排，已有工作流系统且无需进程内共享对象，新增 Ray 会增加镜像、版本和可观测性栈。若任务必须严格遵守容器级安全隔离、网络策略和合规审计，先验证 Ray worker 的端口、对象存储和身份模型能否满足要求。Ray 的优势出现在“需要应用级调度和状态，但又不想为每一条数据流编写独立分布式协议”的区域，而不是所有容器的默认替代方案。

## 18.2 前置知识、学习结果与一句话心智模型

本章假定读者已经理解 Linux 进程和信号、TCP 基础、Kubernetes Pod 与资源请求、模型服务的批处理和 SLO。读完后应该能够画出一次 `remote()` 调用从 driver 到 worker 的路径，指出数据在哪里复制、哪些部分可重试、哪些状态不会自动恢复，解释 placement group 的 bundle 和策略，描述 Serve 副本收到请求前后的队列，以及指出 KubeRay 控制器与 Kubernetes Deployment 各自负责什么。

一句话心智模型是：**Ray 是一个把“函数调用、带身份的进程、对象引用和资源形状”统一到应用级调度器的运行时；Kubernetes 是它的外层进程与基础设施控制器。** 应用逻辑决定需要几个 actor、每个 actor 的资源和状态契约，Ray 决定把它们放到哪里并转发调用，Kubernetes 决定 Ray 进程和节点池是否存在、如何升级与隔离。

可把一次请求抽象成六元组：调用者 `D`（driver 或 Serve router）、执行单元 `E`（task worker 或 actor）、输入对象 `I`、输出对象 `O`、资源向量 `R`、故障策略 `F`。`R` 至少包含 CPU、内存、GPU、自定义标签和拓扑形状；`F` 至少包含重试次数、重启策略、超时和检查点。若任何一个元组成员没有明确定义，系统会采用默认值，默认值往往无法承受生产故障。

## 18.3 第一性原理：任务、Actor 与对象引用

### 18.3.1 Task 是可调度的调用记录

Ray task 可以看作函数 `f` 加上参数对象引用、资源声明和调用选项生成的一条调用记录。driver 调用 `f.remote(x)` 时不直接执行函数，而是向 Ray 提交任务，得到一个 `ObjectRef`。调度器在满足资源和依赖后选择 worker，worker 从对象存储获取参数，执行函数，把返回值序列化到对象存储，并把对象元数据写入控制面。driver 之后调用 `ray.get(ref)` 才同步等待结果。多个 task 可以立即提交形成并行度，`ray.get` 的位置决定了是否无意中串行化。

Task 的默认失败语义包括 worker 进程异常后的重试尝试，但具体次数、异常类型和是否允许重试由 Ray 版本与调用选项决定。重试只对“函数没有外部副作用”这一假设安全。若函数已经写入数据库、发送邮件或扣减配额，再次执行可能造成重复副作用。设计判断是把外部写操作放在幂等键保护的提交阶段，并将 task 结果和提交状态分离记录。

### 18.3.2 Actor 是带地址的有状态进程

定义 actor 类并调用 `.remote()` 会启动一个有身份的 worker 进程。方法调用进入 actor 的 mailbox；默认情况下方法按提交顺序串行执行，方法返回值是新的对象引用。actor 可以保存模型权重、连接池、缓存或计数器，避免每个 task 重新加载大对象。它也可以声明并发方法、异步方法或内部线程，但一旦引入并发，状态同步和顺序契约就由应用负责。

Actor 的“状态”通常分三层：进程内内存、Ray 对象引用中的不可变结果、外部持久化存储。进程重启会丢失第一层，除非 actor 初始化时从第二层或第三层恢复。把关键状态只存在 Python 属性里，再把 actor 数量设置为自动扩缩容，是典型的数据丢失设计。Actor 重启选项解决的是“重新创建进程”，不是“还原业务状态”；恢复函数和版本化检查点才是后者。

### 18.3.3 ObjectRef 是未来值而不是移动语义

`ObjectRef` 是对象存储中的未来值句柄，传递它通常只传元数据，实际字节在消费者需要时通过本地或远程对象管理器获取。小对象可能被内联在任务元数据中，大对象会占用对象存储和网络带宽；具体阈值和存储实现由版本、配置与对象类型决定。把一百兆模型权重作为普通 Python 参数交给每个 task，可能产生多次序列化和复制；先放入对象存储并复用引用，通常能减少重复传输，但仍要测量跨节点读取成本。

对象存储容量是调度和稳定性的一部分。对象放不下会触发 spill 到磁盘，磁盘慢会让看似 CPU 空闲的集群出现尾延迟；spill 路径写满会使新任务因依赖无法执行。`ray memory`、对象存储指标和节点磁盘指标要一起观察。仅看 worker CPU 利用率无法说明数据管道健康。

### 18.3.4 Driver、raylet 与全局控制状态

Driver 是提交 task、创建 actor 或运行 Serve 应用的进程。每个节点通常有 raylet，负责节点级资源、worker 启动、对象位置和本地调度；全局控制存储（GCS）保存集群级元数据，如任务、actor、节点和命名空间信息。控制路径和数据路径分开：任务元数据通过控制服务传播，用户对象通过对象管理器在节点间移动。版本会改变组件名称和实现细节，但“提交、放置、传递对象、回收元数据”的四步机制不变。

Driver 退出不一定会终止所有 task 或 actor。默认生命周期和命名空间决定哪些对象仍存在；长期运行的 detached actor 可在 driver 重连后继续提供服务，但这也意味着孤儿 actor 可能泄漏资源。生产系统应给每个应用设置 namespace、命名规则、TTL 或显式清理程序，并把 driver 的退出语义写进运行手册。

## 18.4 资源声明：把“需要什么”写成可调度事实

### 18.4.1 基本资源与自定义资源

Ray task 和 actor 可以通过 `num_cpus`、`num_gpus`、`memory` 等参数声明资源，也可使用节点上的自定义资源标签，例如 `{"accelerator_type:A10": 0.01}` 或 `{"zone:west": 1}`。资源声明是调度约束，不是 Linux cgroup 的完整隔离。声明 `num_cpus=2` 会让 Ray 为该执行单元预留两个逻辑 CPU，但若节点上的其他进程、后台线程或外部 Pod 未受同一控制，实际争用仍会发生。GPU 资源同理：Ray 的 `num_gpus=1` 需要底层节点提供可用设备和正确的可见性注入，不能替代 Kubernetes device plugin。

资源标签必须来自可信节点启动配置。让普通用户在提交 task 时声明任意自定义资源，只会缩小候选节点集合，不会凭空创建硬件；但如果节点标签由租户可修改，标签就失去安全意义。设计判断是把硬件标签写进节点镜像或 KubeRay workerGroupSpec，并通过准入控制禁止运行时伪造。

### 18.4.2 资源的软约束和硬约束

Ray 的资源请求通常是硬约束：没有足够资源就排队。应用可以通过 placement group、节点亲和和自定义资源实现组合约束。对低优先级任务，常见做法是声明较少 CPU、降低并发或把任务放入独立队列；不能把资源请求写成“如果有 GPU 就用，否则自动退回 CPU”而期待运行时替换。要实现降级，应用需要显式提交两条路径，并为结果标记执行硬件。

一个陷阱是把 `num_cpus=0` 当作“免费”。零 CPU task 仍需要 worker、网络、对象存储和 Python 线程，过多零资源任务可能压垮 raylet 和 GCS。另一个陷阱是把 `memory` 资源当作绝对内存限制；它主要参与调度，真正的 RSS、对象存储和 mmap 行为还受操作系统及容器限制。应把 Ray 资源、容器 cgroup 和节点监控三者的单位与语义记录在同一张表里。

### 18.4.3 资源形状与可行性

设节点集合为 `N`，节点 `n` 的资源向量为 `C_n`，任务的请求为 `R_i`。可调度的必要条件是存在某个节点使 `C_n - R_i` 每一维非负，并满足标签和拓扑条件。多 actor 应用还需要一组同时满足的向量，单个任务逐一可行并不意味着整体可行。这个“形状”观点解释了为什么集群总 CPU 足够，某个需要“4 CPU、1 GPU、同一互联域”的 actor 仍会排队。

对象存储、磁盘空间和网络带宽不一定被写进 Ray 资源向量，但会影响真实可行性。可为节点注册自定义资源 `object_store_budget` 或 `nvlink_domain` 作为保守近似，然后用实验校准。自定义资源过于细碎会导致碎片和低利用率，过于粗糙则会让调度成功但运行时抖动，取舍必须通过等待和尾延迟测量。

## 18.5 Placement Group：为一组执行单元预订形状

### 18.5.1 Bundle 和策略

Placement group（PG）由若干 bundle 组成，每个 bundle 是资源向量，策略决定 bundle 如何映射到节点。`PACK` 尽量把 bundle 放在少数节点，适合需要节点内共享内存或低延迟通信的流水线；`SPREAD` 尽量分散，适合副本故障域隔离；`STRICT_PACK` 要求全部 bundle 在同一节点，`STRICT_SPREAD` 要求每个 bundle 落到不同节点。不同版本可能新增策略或改变边缘行为，实验应打印实际分配并固定 Ray 版本。

PG 的语义是“资源预留与放置”，不是自动启动 worker，也不是网络拓扑的完美证明。创建成功表示调度器找到满足 bundle 的位置并预留资源；应用仍需用 `placement_group` 调度选项把 task 或 actor 放进 bundle。若忘记传递 PG，actor 可能跑在任意有空资源的节点，导致通信变慢或资源超卖的错觉。

### 18.5.2 原子准入与碎片

没有 PG 时，三个 actor 可能先启动两个，第三个因资源不足排队；已启动的 actor 却占住资源，整个应用无法进展。PG 让一组 bundle 要么整体被放置，要么整体等待，减少半启动。代价是可行解空间变小：一个 `STRICT_SPREAD` 的四 bundle 组无法使用只有三台节点的集群，即使每台节点还有大量 CPU；一个 `PACK` 组可能因为单节点显存不足而等待。

PG 是应用级 gang 语义的工具，但它不替代 Kubernetes 层的队列、公平和抢占。KubeRay worker Pod 可能尚未创建，Ray 认为 PG 无法满足；或者 Pod 已经存在但 Kubernetes 没有足够 GPU，Ray 仍会把 PG 留在等待状态。必须同时观察 Kubernetes pending 原因、Ray PG 状态和应用提交日志。

### 18.5.3 PG 与 autoscaling 的相互作用

Ray autoscaler 根据 pending resource request（包括 PG）决定是否启动节点。一个形状为四个两 GPU bundle 的 PG 可能触发四台 GPU 节点扩容；如果云提供商只能一次交付两台，PG 会持续 pending，扩容器会反复检查。建议给 PG 设置启动超时和可诊断的队列标签，区分“没有节点”与“节点已来但标签不匹配”。自动扩容只解决容量，不解决驱动错误、镜像拉取失败或拓扑不满足。

## 18.6 最小可运行实现：标准库 CPU actor 模拟器

本实验不安装 Ray，而是用 Python 标准库模拟 task、actor、资源和一次失败恢复。它的价值是暴露机制：提交与执行分离、actor 持有状态、资源不足形成队列、重试必须区分幂等。下面代码在 Python 3.11 或更新版本运行，使用 `queue`、`threading` 和 `dataclasses`。它不是 Ray 的兼容实现，也不代表真实调度公平性。

```python
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

if __name__ == "__main__":
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
```

### 18.6.1 如何运行和记录

在干净 Python 环境中执行 `python3 ch18_ray_actor_toy.py`。预期得到六个平方结果、一个 `actor-add` 结果和 `actor_value 15`；顺序可能因线程调度改变，等待时间也会随机器变化。若把 `workers` 改成 1，平方任务的等待时间会近似阶梯式增加；改成 4，等待降低但线程调度开销和上下文切换增加。若把 `actor.add` 改为“先修改值再抛异常”，重试将把值增加两次，输出变成 20，这正是非幂等 actor 方法在崩溃点不明确时的风险。

脚本把一个 actor 当作共享 Python 对象，而 Ray actor 实际运行在独立 worker 进程，状态不能依赖主进程内存。要更接近生产，可把 `Actor` 放入单独进程，用管道传输调用，并在进程被杀后从 JSON 检查点恢复；该扩展属于练习，不应被误称为 Ray 的网络协议。

### 18.6.2 测量表与解释

| 干预 | 应记录的指标 | 预期方向 | 机制解释 |
|---|---|---|---|
| workers 从 1 增至 2 | 平均等待、P95 等待 | 下降 | 并行执行减少队列头阻塞 |
| square 输入扩大十倍 | 执行时间、CPU 利用率 | 执行时间上升 | 计算时间与调度时间分离 |
| actor 方法增加 100 次调用 | actor 完成时间、最终值 | 时间近似线性 | 单一 actor 的串行状态边界 |
| 在提交前写入外部计数器 | 计数器值、重试次数 | 计数器可能重复 | 重试不等于事务回滚 |
| 队列加入 1000 个零资源调用 | 主线程和内存 | 元数据增长 | 调度器仍需保存调用记录 |

表中“预期方向”不是性能保证。真实 Ray 集群还会受到对象序列化、节点网络、GCS 压力和 autoscaler 延迟影响。实验报告至少记录 Python 版本、CPU 型号、线程数、输入分布、运行次数和中位数/P95，而不是只贴一次漂亮的总时间。

## 18.7 生产实现阅读：从 task 到 actor 的代码契约

### 18.7.1 Task 的最小模式

典型 task 包含纯计算、显式资源和有限重试。调用者应保存输入数据版本和 task 参数，结果对象应带 schema 版本。使用 `ray.wait` 分批消费结果可以限制 driver 内存，避免一次 `ray.get` 等待巨大列表。对失败 task，日志必须包括 task ID、尝试编号、节点、输入对象引用和异常类型；仅打印业务错误信息无法区分代码 bug、节点故障和对象丢失。

```python
import ray

ray.init(address="auto")

@ray.remote(num_cpus=2, max_retries=2, retry_exceptions=False)
def normalize(item_ref):
    item = ray.get(item_ref)
    return {"id": item["id"], "value": item["value"] / 100.0}

refs = [normalize.remote(ref) for ref in input_refs]
while refs:
    ready, refs = ray.wait(refs, num_returns=min(32, len(refs)))
    for result in ray.get(ready):
        write_idempotently(result)
```

这里 `retry_exceptions=False` 是设计选择：预期的业务异常被返回给调用者，节点级崩溃仍可能重试。实际可用参数随 Ray 版本变化，应在目标版本 API 文档中核对。`write_idempotently` 需要以输入 ID 和数据版本组成唯一键，不能假设调用者只会执行一次。

### 18.7.2 Actor 的初始化、命名和恢复

Actor 初始化是加载模型、建立连接和读取检查点的阶段，可能比单次方法长得多。应把初始化时间计入扩容冷启动预算，并用健康方法区分“进程存在”和“依赖已就绪”。命名 actor 可让其他 driver 通过 namespace 获取，但全局名字带来生命周期和权限问题：同一名称被旧应用占用时，新部署可能连接到错误版本。部署脚本应检查命名空间、应用版本和 actor metadata。

一个恢复友好的 actor 方法通常遵循三步：读取当前版本、计算新结果、以条件写入外部存储并推进版本号。检查点写入成功后才更新内存状态；如果写入失败，方法返回错误且可重试。若外部存储不支持条件写入，至少使用幂等键和 compare-and-set 模拟。初始化时加载最新完整检查点，再回放有限的追加日志，避免把全部历史保存在进程内存。

### 18.7.3 队列、背压与对象引用释放

提交速度超过执行速度时，任务队列和对象存储都会增长。背压可以在 driver 端限制未完成引用数，在 actor 端限制 mailbox 长度，在 Serve 端用请求队列和最大并发。释放无用 `ObjectRef`、分批 `ray.wait`、控制单项输入大小，通常比增加 worker 更有效。增加 worker 在对象带宽已经饱和时只会放大网络和 spill 压力。

监控至少包含：提交速率、完成速率、排队任务数、任务执行时长、依赖获取时长、对象存储使用率、spill 字节、节点可用资源、actor 重启次数和应用错误率。把“任务排队时间”和“函数执行时间”分开，否则一次扩容延迟会被误报为模型慢。

## 18.8 Ray Serve：从请求到模型副本

### 18.8.1 Serve 的组件和数据流

Ray Serve 在 Ray 集群上提供部署、路由和副本管理。HTTP 或 gRPC 入口通常由 Serve proxy 接收请求，按应用和部署路由到 replica actor；replica 调用模型并返回响应。部署描述副本数量、资源、并发、批处理和更新策略。入口、路由器和副本是不同故障域：入口存活不表示副本已经加载权重，副本可用也不表示入口网络策略允许用户访问。

请求路径可以拆成：监听端口、解析路由、生成请求上下文、选择副本、等待并发槽位、执行模型、序列化响应、记录延迟。每一步都可能排队。若模型有长 prefill 和短 decode，请求级队列会把两者混在一起，P99 可能因一个长请求阻塞而恶化；可用部署拆分、优先级或流式响应改善，但代价是跨部署传输。

### 18.8.2 副本资源与批处理

每个 Serve replica 通常是一个 actor，因此 `ray_actor_options` 中的 CPU、GPU、内存和自定义资源应与模型实际占用匹配。GPU 数量写成 1 只会影响调度，不会自动给模型选择正确精度、显存上限或并行度。副本并发设置过高会让单卡同时执行多个请求，显存峰值和尾延迟可能增加；设置过低又会让 GPU 空闲。应基于请求长度分布测量，而不是照抄示例值。

动态 batching 把时间窗口内的多个请求合并为一个模型调用。它能提高矩阵乘利用率，但增加首请求等待，并要求模型接口能接受可变批次、错误隔离和部分结果。批处理窗口、最大批大小和请求超时需要一起设定。一个批次内有一条坏输入时，若异常未被隔离，整个批次可能失败；生产实现应返回逐项状态或在批前验证。

### 18.8.3 Serve autoscaling 的信号与盲点

Serve autoscaling 通常根据部署队列长度、运行中的请求数、目标并发和观测窗口调整副本数。它不是 Kubernetes HPA 的简单替换：Serve 看应用请求，Kubernetes HPA 看 Pod 指标；Ray autoscaler 再根据资源缺口启动节点。三层扩缩容存在时间常数叠加：请求队列增长，Serve 决定增加副本，Ray 发现资源不足并请求节点，KubeRay 创建 Pod，镜像拉取和模型加载完成，最后副本接流量。若每层阈值都过于敏感，系统会产生振荡。

设计判断是先固定节点池容量测量 Serve 的扩缩容曲线，再开启 Ray 和 Kubernetes 扩容；设置最小副本、最大副本、冷却时间和启动超时；对模型加载时间长的部署保留预热副本。扩容信号应结合错误率和排队超时，不能只看平均并发。缩容前要等待正在生成的流式响应完成，否则用户会看到断流。

### 18.8.4 渐进发布和回滚

Serve 部署更新可能创建新版本副本、等待健康，再逐步把流量切换过去。版本标签必须进入指标和日志，便于比较错误率、延迟和模型质量。回滚不能只把流量权重改回旧版本：新版本留下的缓存、外部 schema 或迁移可能使旧版本无法读取。发布前应验证双向兼容，发布后保留旧副本直到观察窗口结束。

## 18.9 KubeRay：把 Ray 交给 Kubernetes 生命周期

### 18.9.1 CRD 角色与控制流

KubeRay 提供面向 Kubernetes 的自定义资源。`RayCluster` 描述 head Pod、worker group、资源、镜像、服务和生命周期；`RayJob` 把一个 Ray 作业提交到集群，可在完成后清理；`RayService` 将 Ray 集群与 Serve 应用绑定，支持服务级升级和流量切换。具体字段、版本和升级策略会随 KubeRay 发行版变化，生产集群应固定 CRD schema、operator 版本和兼容矩阵。

控制流通常是：用户提交 CR，Kubernetes API 持久化对象，KubeRay operator 观察对象并创建 head/worker Pod、Service 和配置，Ray head 启动 GCS 与调度器，worker 通过地址加入，Serve 或 RayJob 在 head 上注册应用。KubeRay 把 Pod 的期望状态交给 Kubernetes；Ray 再在 Pod 内调度 task、actor 和对象。两个控制器都可能报告“成功”，但成功含义不同：KubeRay 成功表示 RayCluster 对象的 Pod 达到就绪条件，Ray 成功表示应用资源和调用可运行。

### 18.9.2 Head、worker group 与资源准确性

Head Pod 承载控制服务，通常也能执行任务，但生产中应通过自定义资源或污点避免把大量计算压到 head。Worker group 可按 CPU、GPU、区域和生命周期拆分，每组定义副本数、最小/最大副本和 Pod 模板。Ray 节点启动时注册的资源必须与 Pod limits 一致：如果 Pod 有 8 CPU 但 Ray 启动参数只报告 4 CPU，资源会被浪费；反过来报告 8 CPU 可能让 Ray 超卖容器 cgroup。

GPU worker 的 Pod 需要正确的 device plugin、runtime、toleration、节点亲和和安全上下文。KubeRay 不能绕过 Kubernetes 的扩展资源分配；Ray `num_gpus` 只是消费 Pod 内已可见的 GPU。若驱动版本与镜像 CUDA 不兼容，Ray 可能成功启动 worker，任务在加载模型时才失败。因此 readiness probe 应包含最小 CUDA/模型依赖检查，而不仅是端口探活。

### 18.9.3 RayJob 与 RayService 的边界

RayJob 适合一次性或有限时长作业。提交者应记录作业 spec、代码包、输入版本和输出位置，并为失败设置重试和清理策略。若作业创建集群、运行几分钟就删除，镜像拉取和模型加载可能占据大部分时间；对高频短作业，常驻 RayCluster 或批处理队列更经济。

RayService 面向在线 Serve。它通常维护一个稳定服务集群和一个更新中的新集群或新版本，并在健康后切换流量。切换期间可能同时占用两倍 GPU，应在容量计划中预留。若新版本无法获得资源，旧版本应继续接流量而不是被先删掉；这要求检查 operator 的升级策略、PodDisruptionBudget 和超时设置，而不能只看 YAML 中的版本字段。

### 18.9.4 升级、滚动和灾难恢复

Kubernetes 可以重建 head Pod，但 Ray 的控制状态和对象数据是否可恢复取决于持久化配置与应用检查点。把 GCS、日志和检查点存储在临时盘上，节点重启后可能丢失集群元数据；把所有数据写到共享盘又会引入吞吐和故障域。建议区分可丢弃的任务缓存、可重建的模型副本和必须恢复的业务状态，并为每类数据选择对象存储、数据库或快照。

升级路径应按组件拆分：先验证 KubeRay operator 与 CRD，再升级 Ray 镜像，最后升级应用代码。任何一次升级都要记录 Ray 版本、Python 版本、序列化协议、CUDA/驱动和 Serve schema。滚动替换 worker 时，长 actor 可能被终止；如果没有重启策略和检查点，任务只会在控制面显示失败。可中断训练应使用 checkpoint 间隔与最大恢复时间预算决定滚动窗口。

## 18.10 状态、自动扩缩容与失败恢复

### 18.10.1 三种状态存储

把状态分成“瞬时执行状态、可重建派生状态、权威业务状态”。瞬时状态包括当前批次、网络连接和 GPU 显存，进程死亡即可丢弃。派生状态包括 embedding cache、对象 spill 和中间张量，可从输入版本重建。权威业务状态包括订单、计费、模型版本批准和训练检查点，必须写入具有一致性与备份的外部存储。Ray actor 适合承载前两类，第三类应以外部存储为真源。

状态恢复流程应明确恢复点 `C_k`、输入日志 `L_{k+1...n}` 和版本 `v`。重启时先读取 `C_k`，验证 schema 与代码兼容，再按幂等方式回放日志。若恢复时间超过 SLO，可保留热备 actor，但热备也需要同步和故障转移测试。仅增加副本数量不能使有状态 actor 自动高可用；两个副本同时写同一状态可能产生冲突。

### 18.10.2 Ray autoscaler、Serve autoscaler、Kubernetes HPA 的三层关系

Ray autoscaler 关注 Ray 节点资源需求，通常根据待调度资源和节点闲置时间增删节点。Serve autoscaler 关注某个部署的请求并发和队列。Kubernetes HPA 关注 Pod 指标，Cluster Autoscaler 关注 Pending Pod。四者串联后，任何一层看不到下一层的信号，都可能出现“Pod 足够但 Serve 没副本”“Serve 想扩容但 Ray 没 GPU”“Kubernetes 已扩容但镜像拉取未完成”的状态。

容量模型应写出各层时间：`T_total = T_queue + T_serve + T_ray + T_kube + T_image + T_model`。其中 `T_kube` 包含调度与 Pod 启动，`T_image` 包含拉取和解压，`T_model` 包含权重加载与 warmup。平均值掩盖尾延迟，至少测量 P50、P95、P99，并把扩容事件和请求 ID 关联。若请求超时小于最慢一层的启动时间，自动扩容对突发流量几乎没有帮助，应使用预热容量或排队降级。

### 18.10.3 重试、超时与取消

任务重试适合暂时性节点故障和幂等函数，不适合确定性输入错误。重试次数、指数退避和总截止时间必须同时设置；没有总截止时间的重试会把失败请求变成永不结束的资源泄漏。Actor 方法重试更危险，因为调用可能已在远端执行。若协议无法区分“尚未开始”和“已提交”，应使用请求 ID 和状态查询，而不是盲目重放。

Serve 请求取消需要沿调用链传播。客户端断开后，入口可以取消等待，但模型 kernel、Ray task 或外部数据库写入未必立刻停止。对长生成请求，应用应定期检查取消标记，在安全点释放 KV cache，并记录中止原因。取消不等于回滚；已经写入的外部状态仍需幂等补偿。

## 18.11 设计实验：测量 Ray 与直接 Kubernetes 服务

### 18.11.1 实验问题、级别与环境

实验级别为 L1/L2，CPU-only，预计 15 分钟，无云成本。若本机安装 Ray，可使用固定版本（例如 `ray==2.9.3`，具体版本需按目标平台验证）；若不能安装，使用上一节标准库模拟器完成机制实验。Kubernetes 对照可在 kind 或 minikube 上运行一个 Python HTTP Deployment；不要求 GPU。记录 Python、Ray、Kubernetes、容器运行时和内核版本，避免把版本差异误当成架构收益。

问题有三项：一，Ray task 的排队和对象传递开销相对于本地函数增加多少；二，Serve 两个副本的尾延迟在并发增加时如何变化；三，直接 Kubernetes Deployment 与 Ray Serve 在扩容、状态恢复和可观测性上的工作量差异。假设不是“Ray 一定更快”，而是“Ray 在应用级并发和状态编排上减少胶水代码，代价是额外控制面和序列化开销”。

### 18.11.2 命令与最小程序

创建虚拟环境后安装固定版本，运行 task 基准、Serve 应用和 HTTP 压测。不要在生产网络上执行压测；只对本地回环或隔离 kind 集群。每个场景预热至少 30 秒，重复五次，报告中位数和 P95。输入大小分别取 1 KiB、1 MiB、16 MiB，以观察对象传递拐点。

```bash
python3 -m venv .venv-ray
. .venv-ray/bin/activate
python -m pip install --upgrade pip
python -m pip install 'ray[serve]==2.9.3' requests
python ch18_ray_actor_toy.py
python serve_app.py --port 8000
python load.py --url http://127.0.0.1:8000 --concurrency 1,4,16 --duration 30
```

若安装了 Ray，可将 `ray_task_bench.py` 作为扩展脚本，比较本地调用、远程 task 和 actor 方法三列：提交延迟、执行延迟、总延迟和吞吐；无 Ray 时直接运行随章提供的 `ch18_ray_actor_toy.py`。`serve_app.py` 应暴露健康端点和业务端点，响应中包含部署版本与处理时间。`load.py` 应记录成功、超时、HTTP 错误、P50/P95/P99、请求字节和并发，而非只打印平均 QPS。

### 18.11.3 预期观察和解释

小 payload 下，远程 task 总延迟通常高于本地函数，因为有序列化、调度和进程间通信；payload 变大后，网络和对象存储带宽成为主导。actor 方法在多次调用共享模型时可摊薄初始化成本，但单 actor 的串行 mailbox 会限制并发。Serve 在并发上升时先表现为队列增长，再表现为副本扩展或请求超时；扩容速度取决于副本启动和模型加载，而非 Ray 调度器的瞬时决策。

直接 Kubernetes Deployment 的单请求路径可能更短：Ingress 到 Pod，应用进程直接处理。Ray Serve 的路径多一层 proxy 和 actor 调度，但可以在同一集群内用对象引用、批处理和多部署编排减少外部队列。实验只测到本地环境的差异，不能推断 GPU 集群、跨区域网络或生产流量下的成本。任何“更快”结论都应附上 payload、并发、版本和硬件。

### 18.11.4 对照实验：故障与恢复

依次杀掉一个 task worker、一个 Serve replica、一个 Ray worker Pod 和一个 Kubernetes Deployment Pod。记录从故障注入到恢复的时间、失败请求数、状态是否丢失、重试次数和日志线索。对有状态 actor，在杀进程前写入检查点，重启后比较快照版本；对无检查点 actor，明确记录状态丢失而不是把新进程存活当成恢复。

Kubernetes 对照应设置 `readinessProbe` 和 `livenessProbe`，观察 liveness 失败是否导致反复重启；Ray Serve 对照应观察副本健康和请求队列。若探针只检查 TCP 端口，两套系统都可能在模型未加载时报告健康，形成“看起来恢复、实际上拒绝请求”的失败案例。

## 18.12 与纯 Kubernetes 部署的系统比较

### 18.12.1 控制流和数据流

纯 Kubernetes 服务通常把每个模型副本作为一个 Pod，Service 负责发现，Ingress 或网关负责路由，队列和状态放在外部组件。控制流由 Deployment、HPA、Cluster Autoscaler 等对象组成，数据流由 HTTP/gRPC、消息队列和对象存储组成。Ray Serve 把副本路由、模型部署和部分批处理放进 Ray 控制面，数据流可通过对象引用和进程内调用连接多个部署。两者都需要外部持久化和观测；区别是“应用级调度逻辑放在哪一层”。

### 18.12.2 资源与扩缩容

Kubernetes 的 requests/limits、taint、affinity、quota 和 PodDisruptionBudget 是成熟基础能力，能够覆盖网络、存储和安全策略。Ray 的资源声明更接近函数和 actor，便于在一个应用内混合 CPU、GPU 和自定义资源，但它依赖底层 Pod 提供准确容量。纯 K8s HPA 根据 Pod 指标扩容，Ray Serve 根据请求和副本队列扩容；二者都需要 Cluster Autoscaler 或 KubeRay worker autoscaling 才能获得新节点。若把两个系统的 autoscaler 同时设成激进，容量和成本会失控。

### 18.12.3 状态、发布和运维

Deployment 替换 Pod 时，状态默认不保留，开发者需要 StatefulSet、卷或外部数据库。Ray actor 让状态留在进程中更方便，但状态丢失风险更隐蔽，必须主动设计检查点。Kubernetes 的发布工具和审计集成普遍成熟；Ray Serve 的多部署版本、批处理和流量切换更贴近模型服务，但需要学习 Ray Dashboard、Serve 指标和 KubeRay operator。

### 18.12.4 决策矩阵

| 需求 | 纯 Kubernetes 更合适的信号 | Ray/KubeRay 更合适的信号 |
|---|---|---|
| 单模型、无状态 HTTP | Pod 生命周期和网关已标准化 | 需要同进程多阶段调用 |
| 多步骤数据/模型流水线 | 外部队列和工作流已有团队能力 | 需要 ObjectRef、task DAG 和共享对象 |
| 长寿命有状态模型 | 可用 StatefulSet/数据库并愿意自建协议 | actor 状态、命名和方法顺序是主要抽象 |
| 细粒度 GPU 组合 | Pod 级资源和调度插件足够 | 运行时内需按 actor/task 动态分配 |
| 合规隔离 | 依赖成熟 Pod 安全与网络策略 | 需额外验证 Ray 端口、对象和命名空间 |
| 高频在线扩缩容 | 网关和 HPA 已有成熟 SLO | Serve 请求级路由和批处理明显有益 |
| 团队运维能力 | 只熟悉 Kubernetes | 愿意维护 Ray、KubeRay 和双层指标 |

矩阵只是起点。选择前应做同一输入、同一硬件、同一 SLO 的端到端实验，并计算迁移、培训和故障排查成本。不要因为 Ray API 看起来像 Python 函数就忽略其分布式失败语义。

## 18.13 失败诊所：八个“看起来健康”的故障

### 18.13.1 PG 一直 pending，但节点有空 CPU

症状是 `ray status` 显示资源充足，placement group 仍等待。原因可能是 bundle 需要 GPU、特定自定义标签或严格分散，而空闲资源分布不满足形状。排查先查看每个 bundle 的缺口，再对照 Kubernetes 节点标签和 Pod 资源；不要只看总 CPU。修复可以放宽策略、增加匹配节点或拆分应用，但必须说明通信和故障域变化。

### 18.13.2 Serve 副本 ALIVE，模型请求超时

副本进程启动且健康探针返回成功，但权重仍在后台加载，或模型初始化失败被吞掉。把 readiness 定义为“能完成一次带真实输入的最小推理”，并记录加载阶段、版本和显存。避免只用端口探针。若模型加载时间长，设置启动超时和预热副本，扩容时把冷启动纳入 SLO。

### 18.13.3 actor 重启后计数器归零

Ray 报告 actor 已重启，调用恢复但业务计数丢失。原因是状态只在进程内存，重启机制没有检查点。修复是把权威计数写入外部原子存储，或按间隔写版本化快照，初始化时验证并恢复。不要用“重启成功”替代“状态恢复成功”的指标。

### 18.13.4 task 重试导致重复写入

网络抖动触发 task 重试，数据库出现两条相同订单。函数有外部副作用且没有幂等键；Ray 无法回滚第一次写入。修复为以输入事件 ID 建唯一约束，写入和状态更新采用事务或 outbox，并把重试次数和提交结果分开记录。对不可幂等操作，关闭自动重试并改用显式确认协议。

### 18.13.5 对象 spill 把 CPU 打满

任务执行时间增加，CPU 利用率下降，但磁盘写入和对象存储接近上限。大对象超过内存触发 spill，读取时又发生网络和磁盘抖动。排查对象大小分布、spill 字节、磁盘延迟和 `ray memory`，按批消费并释放引用，调整对象存储与本地盘容量。增加 worker 只会产生更多对象，可能进一步恶化。

### 18.13.6 KubeRay worker Pod Running，Ray 仍看不到 GPU

Pod 状态为 Running，但 Ray 节点资源中没有 GPU。常见原因是 device plugin、runtime hook、CUDA 库或 Ray 启动参数不匹配。进入 Pod 检查设备文件和 `nvidia-smi`，查看 kubelet/device plugin 事件，再验证 Ray 节点资源表。修复要在 Kubernetes 层先让 GPU 可见，再在 Ray 层设置准确资源，不要在代码中硬编码“第 0 张卡”。

### 18.13.7 双层 autoscaler 产生振荡

请求高峰时 Serve 扩副本，Ray 扩 worker，Kubernetes Cluster Autoscaler 又扩节点；峰值过去后三层同时缩容，随后新请求再次触发冷启动。表现为副本数、节点数和 P99 周期性波动。修复是给每层设置不同的冷却与最小容量，保留少量热副本，优先根据队列和超时做 admission control。把扩缩容事件和请求时间线画在同一图上。

### 18.13.8 更新时旧服务被提前删除

RayService 或 operator 先删除旧集群，再等待新镜像和模型，造成长时间 503。原因是容量预算不足或升级策略未启用双版本。修复是预留一份更新容量，先创建并验证新版本，再切流，最后排空旧版本。若预算不允许双份，明确维护窗口并在网关层返回可重试的状态，而不是宣称零停机。

## 18.14 取舍与被拒绝的替代方案

第一种被拒绝方案是“所有东西都做成 task”。它简单，但模型权重、连接池和缓存每次重复初始化，网络与显存浪费；当状态需要顺序时，调用者会偷偷构造锁和队列，最终得到脆弱的自制 actor。第二种是“所有东西都做成 actor”。actor 数量过多会增加心跳、内存和命名管理，短任务吞吐反而下降；无状态计算更适合 task。

第三种是“只用 PG 就能完成分布式训练”。PG 提供资源形状，不提供梯度同步、检查点、故障恢复或公平队列；仍需训练框架、通信库和作业控制器。第四种是“把 Ray Serve 当成 API 网关”。Serve 适合模型副本和应用级路由，认证、WAF、限流、审计和跨地域流量治理往往仍应由专门网关承担。

第五种是“让 Kubernetes HPA 直接读取 Ray 队列并解决一切扩容”。跨控制器指标延迟、标签映射和权限配置可能比问题本身更复杂；优先选一层负责应用副本，另一层只负责节点容量。第六种是“把所有状态写入对象存储快照”。快照便宜但不一定支持原子更新、低延迟读取和并发冲突，权威事务仍应交给数据库或具备一致性语义的存储。

## 18.15 论文、官方文档与仓库综合

Ray 的论文把任务、actor、对象和资源调度放在同一运行时，解释了为什么应用级 API 可以减少分布式系统样板代码；论文中的规模和延迟不能直接当作今天版本的生产指标。Ray 官方文档描述 task、actor、placement group、对象存储、autoscaler 和 Serve API；这些页面是字段语义的首要来源，但版本切换会改变默认值。KubeRay 文档和 CRD schema 说明 operator 如何把 RayCluster、RayJob、RayService 映射到 Kubernetes 对象；部署前应阅读与当前 operator 版本对应的 schema，而不是复制旧博客。

阅读源码时可从三个路径交叉验证：Ray 核心的资源和调用状态机、Serve 的 proxy/router/replica 生命周期、KubeRay operator 的 reconcile 和状态条件。测试用例揭示边界，如节点失联、actor 重启、PG 取消和滚动更新；测试通过只证明该场景的契约。生产设计还要结合目标硬件、网络策略、镜像供应链和团队应急能力。

## 18.16 六个理解检查（含答案）

### 检查一：为何 `ray.get` 放在循环中会降低并行度？

**答案：** 每次提交一个 task 后立即等待结果，下一次提交被阻塞，driver 只保持一个未完成调用。把引用先收集，再用 `ray.wait` 分批收割，才能让调度器同时看到多个可运行任务。即使使用批量提交，外部写入仍需幂等，否则并行重试会产生重复副作用。

### 检查二：PG 创建成功是否保证 actor 使用了预留 bundle？

**答案：** 不保证。PG 只预留 bundle；创建 actor 或 task 时必须传递 placement group 和对应 bundle index。忘记传递时，执行单元可能落在普通节点。应从 actor metadata、节点资源和 PG 状态三处验证，而不是只看 PG 为 READY。

### 检查三：actor 自动重启后，哪些状态一定存在？

**答案：** 只有运行时仍能获得的元数据和应用显式写入的外部状态可以恢复。进程内 Python 属性、未落盘缓存和当前调用的局部变量不一定存在。重启选项创建新进程，检查点和幂等回放才恢复业务状态。

### 检查四：为什么 Serve 扩副本不等于立刻获得 GPU 吞吐？

**答案：** 新副本必须等待 Ray 资源、KubeRay worker、Kubernetes Pod 启动、镜像拉取、模型加载和 warmup；任一步骤慢，请求仍在队列。副本数量增加还可能受显存和对象带宽限制。应把冷启动时间和队列时间分别测量。

### 检查五：KubeRay 和 Kubernetes Deployment 各自负责什么？

**答案：** Kubernetes 负责 Pod、节点、网络、存储、权限和基础设施生命周期；KubeRay operator 将 RayCluster/Job/Service 期望状态映射为这些对象。Ray 在 Pod 内负责 task、actor、对象和应用级调度。KubeRay 不会替 Ray 处理对象引用，也不会替 Kubernetes 分配设备插件资源。

### 检查六：何时应选择纯 Kubernetes 服务？

**答案：** 当服务是少量无状态 HTTP 副本，团队已有成熟网关、HPA、发布和安全策略，且不需要进程内有状态 actor、对象引用 DAG 或请求级批处理时，纯 Kubernetes 往往更简单。若多阶段应用需要共享大对象、动态 actor 和 Serve 路由，Ray 的额外控制面可能换来更低的胶水复杂度。选择必须由同一 SLO 和故障实验验证。

## 18.17 练习

### 回忆练习

1. 写出 task、actor、ObjectRef、PG、RayService 五个术语各自的生命周期起点和终点。
2. 列出 Ray 资源、Pod requests/limits、GPU device plugin 三层中至少一个相同名称但不同语义的字段。
3. 解释为什么 `ALIVE`、`Running` 和“请求成功”是三个不同的健康层级。

### 推导练习

4. 一个节点有 8 CPU、1 GPU，四个 actor 各要 2 CPU 和 0.25 GPU。若 GPU 只能按整卡分配，最多能同时运行几个？若平台支持四路时间切片，哪些额外指标决定是否真的可用？
5. Serve 副本冷启动 45 秒，Ray 扩容 20 秒，Kubernetes Pod 调度与镜像 15 秒。请求超时为 10 秒，计算预热容量至少要覆盖什么，为什么？
6. 两个 2-GPU bundle 使用 `STRICT_SPREAD`，集群有两台各 4 GPU 节点；若一台节点维护，PG 会发生什么？比较 `PACK` 与 `SPREAD` 的通信和故障域变化。

### 实现练习

7. 扩展标准库模拟器：增加一个有限长度 mailbox，当队列超过阈值时拒绝新调用并返回重试时间。记录拒绝率与 P99 等待。
8. 为 actor 增加 JSON 检查点和版本号。模拟进程在提交前、提交后各崩溃一次，验证最终值不会重复提交。
9. 用 Ray task 写一个 `ray.wait` 流式处理器，每次最多保持 32 个未完成引用，比较与一次性 `ray.get` 的内存峰值。

### 诊断练习

10. 给出一个 RayService 503 时间线：旧副本被删、新 Pod Pending、GPU 配额不足。指出三个控制器各自的错误信号和最小修复。
11. 看到对象 spill 增长、CPU 低、磁盘延迟高时，列出至少四个要检查的指标，并解释为什么增加 worker 不是第一步。

### 设计练习

12. 为一个每小时批处理和一个全天在线生成服务设计共用 RayCluster。说明 namespace、PG、配额、优先级、检查点、Serve autoscaling 和抢占策略。
13. 设计 Ray 与纯 Kubernetes 的 A/B 迁移计划，要求同一输入、同一 GPU、同一 P95 SLO，列出停止条件和回滚路径。
14. 画出从客户端请求到模型响应的控制流、数据流和故障域，标记每一跳的超时、重试和幂等键。

## 18.18 安全边界与运行手册

Ray head、worker、Dashboard、Serve proxy、对象管理器和 GCS 之间会开放多个端口；不要把整个 Ray 集群端口直接暴露到公网。使用网络策略限制 namespace 和节点间流量，给 Dashboard 加认证和只读代理。Kubernetes ServiceAccount、镜像拉取凭据、对象存储密钥和数据库凭据应通过 Secret 管理，避免写入 task 参数、日志和环境回显。任务代码本身可执行任意 Python，不应把不可信租户的代码放进同一信任域；需要强隔离时使用独立集群、节点池或沙箱，而不是只依赖 Ray namespace。

对象引用可能包含敏感数据的序列化字节，spill 到本地盘后会留下取证痕迹。为对象存储和日志设置加密、保留周期和擦除策略，禁止把 PII 放进任务名、actor 名和指标标签。自定义资源标签不是安全边界，租户不能决定“安全节点”或“已清理节点”的标签。GPU 显存和驱动状态也可能跨进程残留，租户隔离应结合硬件分区、节点清理和漏洞修复。

运行手册至少包括：如何列出应用 namespace 和孤儿 actor；如何查看 PG、对象 spill、Serve 队列和 KubeRay conditions；如何暂停自动扩容；如何排空节点而不丢失检查点；如何回滚 Ray 镜像与 CRD；如何验证旧版本流量已归零；如何在事故后清理对象、日志和 Secret。每个操作都写明确认步骤和不可逆后果，避免把“删除 RayCluster”误当成“删除业务数据”。

## 18.19 总结与下一依赖

Ray 把分布式应用的最小语义提升到函数、actor、对象和资源形状；task 适合可重放的无状态计算，actor 适合有身份的长期状态，但状态恢复必须由检查点和外部真源保证。Placement group 让一组资源以形状被预留，却不自动提供拓扑、训练通信或公平队列。Ray Serve 把请求路由、副本、并发和批处理放到运行时，扩缩容要与 Ray 节点和 Kubernetes Pod 的时间常数协同。KubeRay 把 Ray 的进程生命周期交给 Kubernetes，不能替代设备插件、网络策略、存储一致性或应用幂等。

下一章进入推理优化与加速器栈。那里会把本章的 Serve replica、GPU 资源和对象传递继续向下追踪到编译、kernel fusion、FlashAttention、paged KV 和量化；本章实验得到的排队、冷启动和对象带宽基线将作为优化前对照。再往后的可靠性章节会重新检查这里的重试、检查点和双层 autoscaler，验证它们是否满足错误预算。

## 18.20 来源地图与可复现记录

下表列出本章主要依据。访问日期均为 2026-10-05；版本敏感的 API 以目标环境实际安装版本为准。

| 来源与 URL | 类型与版本/日期 | 支持的主张 |
|---|---|---|
| Ray Tasks，https://docs.ray.io/en/latest/ray-core/tasks.html | 官方文档，Ray 2.x 系列，访问 2026-10-05 | task 提交、ObjectRef、重试与资源声明语义（事实） |
| Ray Actors，https://docs.ray.io/en/latest/ray-core/actors.html | 官方文档，Ray 2.x 系列，访问 2026-10-05 | actor 生命周期、方法调用、重启与命名空间（事实） |
| Ray Placement Groups，https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html | 官方文档，Ray 2.x 系列，访问 2026-10-05 | bundle、PACK/SPREAD/STRICT 策略与资源预留（事实/机制） |
| Ray Memory Management，https://docs.ray.io/en/latest/ray-core/objects/object-spilling.html | 官方文档，Ray 2.x 系列，访问 2026-10-05 | 对象存储、引用、spill 和内存诊断（事实） |
| Ray Serve，https://docs.ray.io/en/latest/serve/ | 官方文档，Ray 2.x 系列，访问 2026-10-05 | proxy、deployment、replica、批处理和 autoscaling（事实/机制） |
| KubeRay 文档，https://docs.ray.io/en/latest/cluster/kubernetes/ | 官方文档，KubeRay 1.x 系列，访问 2026-10-05 | RayCluster、RayJob、RayService 与 operator 控制流（事实） |
| KubeRay GitHub 与 CRD schema，https://github.com/ray-project/kuberay | 开源实现，提交版本需按部署锁定，访问 2026-10-05 | reconcile、conditions、升级与字段兼容性（代码/机制） |
| Moritz 等，“Ray: A Distributed Framework for Emerging AI Applications”，https://www.usenix.org/conference/osdi18/presentation/moritz | 论文，OSDI 2018，访问 2026-10-05 | task/actor/object 抽象和应用级运行时动机（论文事实） |
| Kubernetes Device Plugins，https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/ | 官方文档，按目标 Kubernetes 版本核对，访问 2026-10-05 | Pod 资源、设备插件、节点调度和控制器边界（事实） |
| Kubernetes HPA，https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/；Cluster Autoscaler，https://github.com/kubernetes/autoscaler | 官方文档与代码，访问 2026-10-05 | Pod 指标扩缩容与节点扩容的语义边界（事实/机制） |

建议读者在实际实验中保存以下证据：`python --version`、`ray --version`、`kubectl version`、镜像 digest、Ray 集群节点资源表、PG 状态快照、Serve 配置、压测输入分布、P50/P95/P99、对象 spill 与磁盘指标、故障注入时间线和恢复日志。源码、YAML、实验命令与输出应随章节一起归档；若无法安装 Ray，使用 `ch18_ray_actor_toy.py` 完成机制实验，并明确标记为 CPU 模拟而非分布式运行时性能结论。

### 术语补充

- **task**：由函数、参数引用、资源和失败策略组成的一次可调度调用
- **actor**：拥有身份和进程内状态、通过方法 mailbox 接收调用的执行单元
- **ObjectRef**：指向分布式对象存储未来值的句柄
- **placement group**：由多个资源 bundle 及放置策略组成的原子资源形状
- **Serve replica**：承载一个 Serve deployment 版本的 actor 进程
- **KubeRay operator**：把 Ray 自定义资源的期望状态协调为 Kubernetes 对象的控制器
- **幂等**：同一请求重复执行不会改变最终业务结果的性质
- **冷启动**：从扩容决定到副本完成模型加载并可接流量的时间
