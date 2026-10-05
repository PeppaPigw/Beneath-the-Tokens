---
id: ch17-kubernetes-gpu-orchestration
title: Kubernetes 与 GPU 编排：调度、资源隔离与升级
slug: /chapters/17-kubernetes-gpu-orchestration
description: 从 OCI 镜像到 Kubernetes GPU 节点，建立设备插件、资源请求、拓扑感知、gang scheduling、配额与可回滚升级的可测量模型
sidebar_position: 17
level: systems
prerequisites:
  - ch06-pytorch-execution
  - ch08-distributed-collectives
  - ch10-training-ops
  - ch15-inference-execution
  - ch16-model-serving-system
learning_objectives:
  - 能说明 OCI、Docker 和 Kubernetes 在 GPU 工作负载中的边界与数据流
  - 能解释 device plugin、operator、runtime hook 如何把 GPU 变成可调度资源
  - 能正确设置 GPU requests/limits、节点池、污点、亲和性和拓扑约束
  - 能为多 GPU 训练设计 gang scheduling、配额和公平策略
  - 能用 CPU 模拟器和 YAML 实验验证资源分配、失败恢复与升级回滚
  - 能建立包含驱动、固件、镜像、插件和工作负载的升级与回滚账本
  - 能识别 GPU 隔离、跨租户数据泄漏、供应链和特权容器风险
estimated_hours: 24
hardware: CPU-only simulation; GPU cluster optional
risk_level: L4
last_verified: 2026-10-05
---

# 第17章　Kubernetes 与 GPU 编排：调度、资源隔离与升级

> 一块 GPU 不是一个可以随手塞进容器的数字。它有驱动、固件、显存、拓扑邻居、复位语义和不可见的共享状态。OCI 规定了镜像和运行时如何封装进程，Docker 提供了常用的构建与运行界面，Kubernetes 则负责把声明放到节点上并维持期望状态。真正的 GPU 平台还需要 device plugin 把设备容量报告给 kubelet，需要 operator 协调驱动、插件和监控，需要调度器理解 NUMA、PCIe、NVLink 或其他互联拓扑，还要在升级和故障时安全地排空节点、回滚版本。本章把这些层拆开，给出可在无 GPU 的 CPU 环境运行的模拟实验，并明确哪些结论只能在目标硬件上复测。

本章使用“事实、机制、测量、推断、设计判断”五种标签。Kubernetes API 的字段语义属于事实；控制器、kubelet、容器运行时和设备插件之间的调用顺序是从官方文档与源码重构出的机制；实验输出是测量；由测量推导的容量和排障优先级是推断；配额大小、升级窗口和安全边界是设计判断。一个成功调度的 Pod 不能证明 GPU 内核性能正常，一个通过健康检查的节点也不能证明多租户隔离成立。

## 17.1 为什么 GPU 编排比 CPU 编排更难

### 17.1.1 GPU 资源不是均匀的整数

CPU 通常可以切成 millicore，内存也能以字节为单位近似分割。GPU 的“1”可能表示整卡、时间切片、MIG 实例、虚拟函数，甚至只是一个由插件定义的扩展资源名称。两张都标记为 `nvidia.com/gpu: 1` 的卡，可能有不同显存、计算能力、互联带宽和固件版本。如果调度器只看数量，作业可能被放到数量足够但拓扑不合适的节点上，表现为通信带宽骤降或初始化失败。

GPU 还拥有设备外的资源：主机到设备的 PCIe 通道、NUMA 内存、Pinned memory、网卡队列、共享内存、CPU 线程和电源预算。训练作业在申请四张 GPU 时，实际承诺的是一组互相可达的设备和足够的主机资源。把 GPU 数字从 YAML 复制到另一个集群，并不等于复制了同样的性能或隔离。

### 17.1.2 控制面与数据面是不同故障域

Kubernetes 控制面保存对象和调度决策，kubelet 在节点上执行 Pod，容器运行时创建容器和 Linux namespace，设备插件通过 gRPC 向 kubelet 注册，operator 负责安装或升级驱动、插件、监控和自定义资源。训练进程和推理进程属于数据面。控制面显示 Pod 为 `Running`，只说明容器进程被认为存活；它不保证 CUDA 初始化、内核编译、显存余量或集体通信已经成功。

故障域的划分决定排障顺序。若 API Server 不可用，无法创建新 Pod，但正在运行的容器可能继续工作；若 kubelet 失联，节点状态会陈旧，调度器可能暂时不再放置新 Pod；若 device plugin 崩溃，已分配设备的容器未必立刻退出，但新 Pod 不能获得扩展资源；若驱动挂死，节点上的多个 Pod 可能同时出现设备错误。应分别记录控制面事件、节点事件、运行时日志和应用日志。

### 17.1.3 “容器化”不等于“隔离了 GPU”

容器默认隔离进程、文件系统和网络命名空间，GPU 设备通常通过 `/dev` 节点和运行时注入的环境变量暴露。若容器拥有过宽的 Linux capability、主机 PID 命名空间或特权模式，进程可能访问不应看到的设备和内核接口。即便设备文件只映射了一张卡，驱动错误、共享内存、DMA 和固件状态也可能使隔离边界比 CPU cgroup 更复杂。MIG 等硬件分区可以改善显存和计算隔离，但不应被宣传为对侧信道、功耗争用或驱动漏洞的绝对防护。

## 17.2 从 OCI 到 Kubernetes：一条请求如何到达 GPU

### 17.2.1 OCI 镜像和运行时的边界

OCI Image Specification 定义层、配置和清单的格式；OCI Runtime Specification 描述如何创建容器进程、根文件系统、namespace、cgroup 和环境。镜像本身不包含“这是一张 GPU”的含义，GPU 相关的库和用户态工具只是文件层。真正打开 `/dev/nvidia0`、加载驱动库或设置可见设备，要由节点上的运行时、hook 或设备插件完成。

Docker 是构建镜像和调用运行时的常用 CLI 与守护进程组合。`docker build` 产生 OCI 兼容镜像，`docker run` 通过运行时参数将设备映射进容器。Kubernetes 不直接解析 Dockerfile，也不把 Docker daemon 当作必需组件；现代集群常使用 containerd 或 CRI-O，通过 Container Runtime Interface（CRI）与 kubelet 通信。一个镜像在 Docker 本地能运行，不代表在 Kubernetes 节点上具备匹配的驱动、运行时 hook 或安全策略。

### 17.2.2 Kubernetes API 对 GPU 的声明

Pod 通过容器的 `resources.limits` 请求扩展资源，例如 `nvidia.com/gpu`。对 GPU 这类不可压缩扩展资源，官方约束是只写 `limits`（Kubernetes 将其当作 request），或同时写相等的 requests 与 limits；不能只写 GPU request，也不能写不同数值。调度器把扩展资源视为节点可分配数量，kubelet 在节点上进行最终分配和设备注入。

一个最小示例需要同时描述镜像、资源、节点选择和容错策略。下面的 YAML 是设计实验，不代表所有发行版都使用相同扩展资源名；在生产中应以 `kubectl describe node` 和设备插件文档为准。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-smoke
spec:
  restartPolicy: Never
  nodeSelector:
    accelerator.example.com/class: a100
  tolerations:
  - key: accelerator
    operator: Equal
    value: nvidia
    effect: NoSchedule
  containers:
  - name: test
    image: registry.example.com/ml/cuda-smoke:12.4
    command: ["python", "-c", "print('probe')"]
    resources:
      requests:
        cpu: "500m"
        memory: "2Gi"
        nvidia.com/gpu: 1
      limits:
        cpu: "1"
        memory: "4Gi"
        nvidia.com/gpu: 1
```

### 17.2.3 Device plugin 的注册与分配

Kubernetes device plugin API 让节点代理发现和分配专用硬件。插件通常在每个节点运行，启动后向 kubelet 注册资源名称和版本，通过 `ListAndWatch` 报告设备健康状态，通过 `Allocate` 为被选中的 Pod 返回设备节点、环境变量、挂载和容器运行时配置。kubelet 将这些信息交给 CRI，运行时据此创建带有相应设备的容器。

这里有三个容易混淆的时间点。第一，调度器只看插件已经上报到 Node Status 的可分配数量，不会在调度时调用插件。第二，`Allocate` 发生在 Pod 已被绑定到节点后，若此时设备变为不健康，Pod 可能进入创建失败而不是重新调度。第三，插件报告健康并不等于应用级健康；驱动可以响应查询，但 NCCL、CUDA 或模型加载仍可能失败。应把插件健康、驱动健康和应用探针分开观测。

插件返回的设备列表可能包含多个设备，顺序和稳定性应有明确契约。若作业请求两张 GPU，插件必须保证返回的是可用且满足其策略的一对设备，而不是任意两张。若要表达 NUMA 或互联关系，通常需要拓扑管理器和插件提供的 topology hints；没有 hints 时，调度器可能把设备放在不同 PCIe 根复杂体下。

### 17.2.4 Operator 是协调器而不是魔法

GPU operator 往往由自定义资源、控制器、DaemonSet、节点标签和升级逻辑组成。它可以安装内核模块、容器工具包、device plugin、DCGM exporter、节点特性探测器，并按照依赖顺序等待组件就绪。Operator 的期望状态是“节点具备一套兼容的软件栈”，但它不能替应用证明模型正确，也不能修复硬件物理故障。

安装驱动涉及内核、签名、Secure Boot 和重启，风险高于普通 Deployment 滚动更新。升级控制器若在驱动尚未就绪时先重启工作负载，可能导致大规模训练中断。因此 operator 应支持暂停、分批节点选择、前置检查和回滚；平台团队要知道自定义资源的版本、控制器版本和底层驱动版本是否兼容。

### 17.2.5 Runtime hook 与容器可见性

NVIDIA Container Toolkit 等组件会根据运行时请求把驱动库和设备节点注入容器。常见机制包括 OCI prestart hook、CDI（Container Device Interface）描述文件或 CRI 配置。注入应保持最小权限：只映射被分配设备、必要的库和只读信息，避免把整个 `/dev` 或主机文件系统暴露进去。`CUDA_VISIBLE_DEVICES` 只是用户态提示，不能替代内核层设备访问控制；应用若能打开未授权的设备节点，环境变量并不能提供隔离。

## 17.3 调度模型：从数量到拓扑和共同启动

### 17.3.1 节点池和污点

节点池按硬件、驱动、区域和生命周期分组，例如推理池、训练池、抢占式池和高带宽互联池。每个池应有可验证的标签，如 GPU 型号、显存、驱动分支、拓扑能力、机架和区域。标签是调度输入，不是可信证明；节点注册和自动扩缩容流程必须防止普通租户自行伪造“高端 GPU”标签。

GPU 节点常用污点 `accelerator=nvidia:NoSchedule`，迫使 Pod 明确添加 toleration。这样 CPU 工作负载不会意外占据昂贵节点。Toleration 只表示“允许被调度到带污点的节点”，不表示“必须使用 GPU”；还需要资源请求、节点亲和性和镜像兼容性。节点池缩容时，要先排空可中断作业，再处理有本地缓存或长时间初始化的工作负载。

### 17.3.2 资源 requests、limits 与配额

对 CPU 和内存，requests 主要用于调度，limits 还参与 cgroup 限制。对整卡 GPU，通常没有像 CPU 那样的可压缩时间片；`nvidia.com/gpu: 1` 表示分配一个扩展资源实例。某些平台提供共享 GPU、时间分片或 MIG 资源名，但它们的语义必须由插件文档和测量确认。不要把 `nvidia.com/gpu: 0.5` 当作通用做法。

Namespace 的 `ResourceQuota` 可以限制 GPU 总量、Pod 数、CPU、内存和对象数量。配额应按租户和优先级层次设计：训练租户有保证份额和突发上限，交互租户有并发但较小的单作业上限，系统租户保留恢复空间。配额只控制声明的上限，不会阻止单个容器在 GPU 上创建过多线程、占满主机网络或耗尽共享存储；还需要 LimitRange、网络策略和存储配额。

配额计算要考虑碎片。四卡节点可能剩两张卡，无法满足一个请求四卡的作业；全局剩余量看起来充足，实际可行分配却为零。建议同时记录按节点池、区域和拓扑域的可行容量，以及因形状不匹配造成的碎片。扩缩容器只按总 GPU 数量决策，会在多卡作业高峰时频繁抖动。

### 17.3.3 亲和性、反亲和性和拓扑分布

Node affinity 可以要求 GPU 型号、区域或驱动标签；pod anti-affinity 可以避免副本落在同一故障域；topology spread constraints 能把服务分散到区域或机架。多 GPU 训练还需要“聚合”约束：同一作业的 rank Pod 应尽量靠近，以降低 AllReduce 延迟。Kubernetes 默认调度器的分数插件并不了解所有互联细节，必要时应使用调度框架插件、拓扑管理器或厂商调度器，并通过 NCCL 拓扑报告验证结果。

Topology Manager 在 kubelet 层协调 CPU、设备插件和其他资源的 topology hints。它可以采用 `best-effort`、`restricted`、`single-numa-node` 等策略。策略越严格，局部性越好，但可调度容量越小。NUMA 对齐失败时，Pod 可能仍然启动，性能却因跨节点内存访问下降。平台应把对齐状态作为可观测事件，而不是只看 Pod 是否为 `Running`。

### 17.3.4 Gang scheduling：要么一起开始，要么不开始

数据并行或张量并行训练通常需要一组 rank 同时参与。若调度器先启动部分 Pod，剩余 Pod 长时间排队，已启动的 rank 会占用 GPU 却无法有效训练，形成“半成品占用”。Gang scheduling（队列组调度）要求满足最小可运行数量后再绑定整组 Pod，或在等待期间不占用设备。原生 Gang Scheduling 的可用性取决于 Kubernetes 版本和 feature gate；较旧集群通常需要 Volcano、Kueue 或调度框架插件，不能假设默认 scheduler 已提供整组原子准入。

Gang 语义需要回答几个细节：最小启动数是全部 rank 还是允许弹性训练的下限；等待多久后释放已分配资源；一个 rank 崩溃时是否整体重启；抢占时按整组还是按单 Pod；不同作业间如何公平。实现可通过调度框架插件、批处理队列控制器或 JobSet 等组件完成。无论采用哪个实现，都要测试控制器重启、节点故障和 API 延迟，避免组状态卡在“部分已绑定”。

## 17.4 设计实验：CPU 模拟器和 YAML 计划

### 17.4.1 实验问题与假设

我们先不依赖 GPU，模拟三个节点池、两种 GPU 形状和四个训练作业。问题是：在总容量相同的情况下，普通 FIFO、带配额的公平队列和 gang scheduling 是否会产生不同的等待时间、碎片和失败率？假设如下：

1. 只按总 GPU 数量调度会让四卡作业在碎片存在时长时间等待。
2. gang scheduling 能减少半启动作业，但可能提高短作业的首等待时间。
3. 令牌桶配额能防止单租户占满池子，却可能留下暂时无法借用的保留容量。
4. 拓扑约束使可行容量降低，但多卡通信成本更稳定。

### 17.4.2 纯 Python CPU 模拟器

下面脚本仅使用标准库，模型是离散事件近似。每台节点有 GPU 数、拓扑域和租户配额；作业包含所需 GPU 数、持续步数、是否 gang、租户和优先级。模拟器不计算神经网络，也不代表真实 GPU 吞吐。

```python
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
```

改变 `gang`、`quota`、节点的 `domain` 和提交顺序，记录每个作业的等待时间、运行时间、节点碎片、拒绝次数和公平指标。可以定义 Jain 公平指数：设每个租户获得的有效 GPU 时间为 `x_i`，租户数为 `n`，则 `J=(Σx_i)^2/(nΣx_i^2)`。该指数只描述分配均匀程度，不说明作业是否满足 deadline，也不说明 GPU 利用率。

### 17.4.3 YAML 设计实验

在测试集群中准备两个节点池：`gpu-small` 每节点四张卡，`gpu-large` 每节点八张卡。先创建 Namespace、ResourceQuota、PriorityClass、节点标签和污点，再提交一个四 rank 的 gang 作业、两个单卡推理 Pod 和一个故意超出配额的作业。实验步骤如下：

1. `kubectl get nodes --show-labels`，记录 GPU 型号、驱动、拓扑标签和可分配扩展资源。
2. 创建只允许 `team-a` 使用四卡配额的 Namespace，观察超额 Pod 的准入错误。
3. 提交 gang 作业，先只让三个 GPU 节点可用，确认 Pod 是否保持等待而不是半启动。
4. 增加第四个可用 GPU，记录从满足最小组到全部 rank `Running` 的时间。
5. 删除一个节点或注入 device plugin 不健康状态，观察作业重试、队列释放和告警。
6. 在不改变镜像的情况下切换节点池标签，比较 NCCL 拓扑和 step time；若没有 GPU，至少比较模拟器中的跨域惩罚。

YAML 设计应明确 owner、版本、优先级、最大重试次数、服务账户、网络策略和终止宽限期。不要让实验使用默认 ServiceAccount 或集群管理员权限。所有对象加上 `experiment=true` 标签，结束后按标签清理，避免残留配额和优先级影响其他工作负载。

## 17.5 升级与回滚：把 GPU 平台当作有依赖的发布列车

### 17.5.1 版本矩阵

GPU 工作负载至少有六个版本维度：节点内核、GPU 驱动、容器工具包、CUDA 用户态库、device plugin/operator、框架和模型镜像。它们不是任意组合都兼容。驱动通常向后兼容一组用户态库，但具体上限、内核模块和固件要求必须查厂商矩阵；“容器里带了 CUDA”并不表示节点驱动足够新。

维护一张机器可读的兼容矩阵，字段包括节点池、驱动分支、固件版本、运行时、插件版本、operator 版本、支持的镜像 digest、已知限制和验证日期。发布前做静态检查：镜像声明的最低驱动、插件 API 版本、内核签名、MIG 配置、监控 exporter 和安全策略都必须满足。矩阵是事实记录，不是测试替代品。

### 17.5.2 升级顺序和分批策略

常见顺序是先在隔离节点池验证新驱动和运行时，再升级 operator 控制器，随后滚动 device plugin，最后切换工作负载镜像。若驱动升级需要重启，必须先 `cordon` 节点，再根据工作负载可中断性选择 `drain` 或排空队列。对长训练作业，应保存 checkpoint、确认恢复时间和数据一致性，再允许迁移。

分批比例要和故障域对应。先升级一个节点或一个机架，运行 GPU 烟雾测试、框架初始化、显存分配、集体通信和代表性推理；观察一段完整的训练 step 和尾延迟后再扩大。自动扩缩容器应暂时固定，避免升级期间新节点混入而掩盖版本问题。每批都记录升级前后指标差异，而不是只看 Pod 成功率。

### 17.5.3 烟雾测试与可观测性

最低烟雾测试包括：列出设备、读取驱动和 CUDA 版本、分配少量显存、运行矩阵乘、启动多进程通信、检查设备重置行为、验证监控指标和日志脱敏。真实模型测试还应包括 checkpoint 加载、混合精度、长上下文 KV、异常取消和节点重启恢复。测试容器的镜像 digest 和命令固定，避免每次拉到不同内容。

关键指标按层分组：调度层记录队列等待、配额拒绝、碎片、gang 等待和抢占；节点层记录 GPU 利用率、显存、温度、ECC/XID 错误、PCIe 重传和功耗；应用层记录 step time、吞吐、首 token、p99、loss、NaN、checkpoint 成功率和重试。GPU 利用率高不代表作业有效，可能只是忙于通信或内存复制；loss 不变也不代表训练正确，可能是数据管道卡住。

### 17.5.4 回滚触发器和不可逆变化

提前写出回滚触发器，例如烟雾测试失败、XID 错误增加、p99 超过基线二倍、loss 异常、checkpoint 无法读取、跨租户设备可见性错误或节点重启循环。回滚动作可能包括停止新作业、恢复旧节点池、重新标记节点、回退 operator、恢复镜像 digest 和重放 checkpoint。驱动或固件降级有时不可逆或需要冷重启，必须在维护窗口前验证。

不要把“回滚 Deployment”误当作“回滚驱动”。Kubernetes 对象可以恢复旧镜像，但旧容器仍可能加载新驱动；operator 可能马上把旧配置改回新状态；MIG 切分和持久化卷格式变更也可能无法自动恢复。升级计划要为每种层定义独立回滚和数据恢复路径，并保存最后一个已知良好的节点模板。

### 17.5.5 节点排空和训练恢复

排空是有状态工作负载的协议，不是强制删除。先停止新准入，等待作业到达安全检查点，标记节点不可调度，再迁移或终止。PodDisruptionBudget 只能限制自愿中断数量，不能阻止驱动故障、节点断电或管理员强删。训练控制器要区分“可重试的节点失败”和“代码或数据不可重试的失败”，并对 checkpoint 使用原子写入、校验和与代际目录。

恢复时验证三件事：模型参数与优化器状态匹配同一全局 step，数据分片和随机种子不会重复或跳过，安全策略和镜像版本仍然允许继续。若只能恢复模型权重而不能恢复优化器状态，应把它记录为降级恢复并重新评估收敛，而不是悄悄当作无缝续训。

## 17.6 失败诊所：五个看起来健康却不可靠的案例

### 案例一：Pod `Running`，但 CUDA 初始化失败

现象是 Deployment 全部绿色，应用日志显示设备数为零或 `driver/library version mismatch`。原因可能是节点驱动与镜像用户态库不兼容、runtime hook 未配置、设备插件分配为空或容器只读路径覆盖了注入库。排查顺序应从 Pod 的 `resources`、事件、容器运行时配置、插件日志到宿主机驱动版本，不能只重启 Pod。修复后用固定 digest 的烟雾测试验证，并在探针中执行真实的 CUDA 初始化而非只返回 HTTP 200。

### 案例二：四卡训练长期排队，集群仍显示有空闲 GPU

四卡作业需要同一拓扑域，但剩余卡分散在多个节点或不同 NUMA 域。普通调度器按总数计算，无法找到满足形状的节点。解决方法是报告按形状和拓扑域的可用容量，使用 gang 与拓扑约束，或调整节点池规格。把作业改成四个单卡 Pod 只能掩盖问题，可能导致通信性能和收敛行为改变。

### 案例三：配额公平，租户仍然相互影响

两个 Namespace 各有两卡配额，但共享同一节点的 PCIe、CPU、内存带宽和电源限制。GPU 利用率下降、尾延迟抖动，配额指标却正常。需要增加 CPU/memory requests、NUMA 对齐、节点池隔离、功耗和热限制监控；若硬件支持，使用 MIG 或其他分区。配额控制“能分配多少”，不自动控制“分配后性能是否稳定”。

### 案例四：升级完成，训练 loss 悄悄改变

所有 Pod 都成功启动，但 CUDA、通信库或精度默认值变化，导致数值误差积累。若只做启动探针，不会发现。应在升级前保存短训练基线：固定数据、种子、batch、混合精度和 checkpoint，比较 loss 曲线、梯度范数、吞吐与通信时间。允许合理的浮点差异，但要设定统计阈值和人工复核点。无法解释的变化应停止扩批并回滚镜像或驱动。

### 案例五：device plugin 重启后设备“丢失”

插件 DaemonSet 滚动更新，kubelet 暂时看到可分配 GPU 为零，新的 Pod 全部 Pending；旧 Pod 仍在运行但监控断流。若控制器同时重启所有节点，会造成全局停顿。应采用分批、就绪门、节点池上限和恢复演练；把插件注册延迟、健康设备数和 kubelet 分配错误纳入发布门禁。


### 案例六：节点 drain 卡住，PDB 不是理由充分的强删许可

维护窗口开始后执行 `kubectl drain`，命令因 PodDisruptionBudget、长时间 checkpoint 或 Job finalizer 一直等待。先确认 Pod 所属控制器、最近 checkpoint、终止宽限期和 PDB 当前 allowed disruptions；对训练作业，应等待安全边界或降低新准入，而不是直接 `--force --delete-emptydir-data`。若必须中止，要由负责人确认数据损失范围，保存事件和 checkpoint 校验，并在维护后验证没有重复训练或写坏对象存储。

### 案例七：自动扩缩容器增加了节点，Pod 仍然 Pending

Cluster Autoscaler 已把节点数量从四台扩到六台，但新节点没有目标 GPU 标签、带有未预期污点，或驱动和 device plugin 尚未 Ready。检查顺序是 capacity、Node Ready、allocatable、标签/污点、插件注册和镜像拉取；不要只看云厂商实例数量。扩容脚本应在节点通过 GPU 烟雾测试后再添加可调度标签，避免调度器把作业放到“有卡但不可用”的半成品节点。

### 案例八：Operator 升级后 CRD 成功，旧 ResourceClaim 却无法恢复

Helm 或 operator 报告升级成功，但旧 CRD schema 的字段被新版本舍弃，ResourceClaim 卡在 finalizer，设备插件无法读取句柄。升级前应导出 CRD、对象和 claim 状态，确认 schema 的存储版本和转换 webhook，先在 canary 集群做旧对象往返测试。若出现不兼容，暂停删除和新作业，恢复旧 operator 或使用官方迁移工具；不要手工删除 finalizer 以“清理”生产设备占用。

## 17.7 安全边界与最小权限

GPU 集群的安全目标至少包括镜像供应链、设备访问、租户隔离、数据驻留、升级权限和审计。镜像使用签名和 digest 固定，构建阶段扫描 CUDA、驱动用户态库和系统包漏洞；禁止工作负载从不受信任的注册表拉取高权限镜像。准入策略检查特权模式、hostPID、hostNetwork、任意 hostPath、Linux capabilities、服务账户和可写设备节点。

设备插件和 operator 往往需要 DaemonSet 级别权限，必须与普通租户工作负载分离。operator 的服务账户只授予管理其自定义资源和目标 Namespace 的权限；驱动安装组件若需访问宿主机，应放在受控节点池并缩短生命周期。不要让租户自行修改节点标签、污点、RuntimeClass、PriorityClass 或设备插件配置。

GPU 显存和缓存可能残留前一租户的数据。硬件复位、MIG 重建、驱动清零策略和应用级显式清理要结合验证；不能仅依赖容器退出。共享前缀、模型权重和日志中的 prompt 也要按租户权限隔离。监控标签避免包含原始文本、用户标识或机密文档标题，调试转储需经过脱敏和审批。

网络与存储同样是边界。多节点训练通常需要高带宽网络，NetworkPolicy 不能因为“训练需要通信”就放开整个集群；应限定 rank 之间的端口和 Namespace。Checkpoint 存储使用最小读写权限、服务端加密和版本保留，防止恶意作业覆盖其他租户的恢复点。升级操作和强制删除 Pod 属于高风险动作，审计记录操作者、对象、原因、前后版本和结果。

安全策略可能降低可调度容量或性能，例如禁止特权驱动安装、启用签名验证、关闭跨租户缓存。应把这些成本写入容量模型，而不是为了通过基准临时关闭策略。任何绕过 admission、使用 hostPath 访问设备或授予集群管理员权限的实验都必须在隔离环境中进行，并在结束后撤销。

## 17.8 权衡与被拒绝的替代方案

**只用 Docker Compose 管理多 GPU**：适合单机实验，无法提供跨节点调度、配额、节点排空和期望状态。我们保留 Compose 作为本地复现工具，不把它当成集群控制面。

**只按 GPU 数量扩容**：实现简单，但忽略多卡形状、拓扑、显存和碎片。我们接受总量指标用于粗粒度自动扩缩容，同时增加按形状的可行容量和队列等待作为门禁。

**所有作业共享一个高优先级队列**：短期吞吐可能高，长期会让交互和小租户饥饿。我们采用租户配额、带老化的公平队列和有限的优先级抢占，牺牲少量峰值吞吐换取可预测性。

**升级时一次滚动整个集群**：操作步骤少，但故障半径大，无法区分驱动、插件和镜像问题。我们选择按节点池和故障域分批，并保留旧池作为快速回退路径。

**把 `CUDA_VISIBLE_DEVICES` 当作安全边界**：这是用户态选择，不是内核访问控制。我们要求设备插件、运行时、Pod Security 和宿主机权限共同限制设备，并通过跨租户测试验证。

## 17.9 论文、文档与代码综合阅读

阅读 Kubernetes 文档时，先确认扩展资源、device plugin、Topology Manager、ResourceQuota 和调度框架的版本；字段语义可能随版本变化。阅读 OCI 规范时，关注镜像与运行时的责任边界，避免把 Docker 行为误推到 CRI。阅读 GPU operator 和 device plugin 源码时，沿着注册、ListAndWatch、Allocate、节点标签和 DaemonSet 更新路径画时序图。阅读调度器插件时，区分过滤（能否放置）、打分（更偏好哪里）、绑定（最终写入）和抢占（释放谁）。

将文档与实验对照：如果文档声称某拓扑策略能保证单 NUMA，实验应检查 kubelet 事件、设备分配和实际 CPU/内存位置；如果 operator 声称支持无中断升级，实验应包括正在运行的训练和 device plugin 短暂不可用；如果配额声称公平，实验应在突发、取消和重试场景下测量。源码和文档只能说明机制，不能替代目标硬件上的吞吐和故障测试。

## 17.10 六个理解检查（含答案）

### 检查一：为什么镜像里有 CUDA 仍可能无法使用 GPU？

**答案**：镜像只提供用户态文件和进程环境，GPU 设备节点、内核驱动、运行时 hook/CDI 和 device plugin 由节点与集群配置提供。驱动与用户态库版本、设备注入和权限任一不匹配，都可能导致初始化失败。`CUDA_VISIBLE_DEVICES` 不能替代设备访问控制。

### 检查二：调度器和 device plugin 各自在什么时候做决定？

**答案**：调度器依据 Node Status 中的可分配扩展资源、亲和性、配额和其他约束，把 Pod 绑定到节点；绑定后 kubelet 调用 device plugin 的 `Allocate` 获取具体设备、挂载和环境配置，再交给运行时创建容器。调度器不会在过滤阶段实时调用插件来挑选具体设备。

### 检查三：为什么四卡作业在全局还有四张空闲卡时仍可能 Pending？

**答案**：四张卡可能分散在不同节点、机架或 NUMA/互联域，无法满足 gang 和拓扑约束；也可能被不同配额保留、形状不匹配或节点被污点隔离。应查看按节点池和拓扑域的可行容量，而不是只看全局总数。

### 检查四：requests 与 limits 对 GPU 该如何设置？

**答案**：整卡等不可压缩扩展资源通常要求 requests 与 limits 相等，并使用插件定义的整数资源名。共享 GPU、时间切片或 MIG 的写法取决于具体实现，不能普遍使用小数。还要为 CPU、内存、临时存储和网络配置合理请求，避免 GPU 被主机瓶颈拖住。

### 检查五：gang scheduling 解决了什么，又带来什么代价？

**答案**：它保证训练作业满足最小可运行组后再占用或启动，减少半启动作业长期占卡和死锁。代价是等待时间可能上升、调度器和控制器状态更复杂，抢占和故障恢复必须按整组处理。对支持弹性训练的作业，最小组和扩缩容语义需要显式定义。

### 检查六：为什么回滚旧镜像不一定回滚 GPU 平台？

**答案**：旧镜像仍运行在新驱动、固件、runtime、device plugin 或 MIG 配置上，operator 还可能持续把节点改回新状态；某些驱动和固件降级需要重启或不可逆。完整回滚必须有版本矩阵、旧节点池或模板、checkpoint 恢复和独立的驱动/插件回退步骤。

## 17.11 练习

1. **回忆**：列出 OCI、Docker、CRI、kubelet、device plugin、operator 六个名词各自负责的最小职责，并为每个职责写一个可观测日志来源。
2. **推导**：给定三节点，每节点四卡，其中一节点的卡跨两个 PCIe 根复杂体；设计一个四卡拓扑约束，计算在一张卡故障后的可行容量，并说明碎片如何变化。
3. **实现**：扩展 CPU 模拟器，加入 Node 级污点、Namespace 配额、PriorityClass 和 `retry_after`。输出每个租户的 Jain 公平指数、p95 等待时间和 gang 半启动次数。
4. **诊断**：构造一个 Pod `Running` 但矩阵乘失败的故障注入实验，分别让驱动版本不匹配、device plugin 返回空设备、容器缺少库路径。写出不会泄露 prompt 的日志字段。
5. **设计**：为一个 64 卡训练集群设计节点池、队列、配额、PDB、checkpoint 和升级分批计划。给出明确的回滚触发器和停止条件。
6. **安全**：审查一份带 `privileged: true`、`hostPID: true`、`hostPath: /` 和集群管理员服务账户的 GPU Pod，列出每项风险与最小替代权限。说明哪些测试必须在隔离集群完成。
7. **测量**：在可用 GPU 集群上比较单 NUMA 与跨 NUMA 的 step time、NCCL 带宽、p99 和功耗。记录硬件、驱动、框架、镜像 digest、数据集和温度，避免把一次运行误当成普遍结论。
8. **升级演练**：建立旧/新驱动与 operator 的双节点池，先升级一个节点，运行烟雾和短训练，再故意注入通信失败，验证停止扩批、保存 checkpoint、恢复旧池和审计记录。

## 17.12 小结与下一依赖

GPU 编排的核心不是把 `nvidia.com/gpu: 1` 写进 YAML，而是维护一条从镜像、运行时、驱动、设备插件到调度器和应用的可验证链路。OCI 和 Docker 解释封装边界，Kubernetes 提供声明式控制面，device plugin 把设备报告和分配接入 kubelet，operator 协调节点软件栈，Topology Manager 和调度插件处理局部性，gang scheduling、配额和队列处理共同启动与公平性。升级则要求把驱动、固件、插件、镜像、模型和 checkpoint 当成有依赖的发布列车，并且每一层都能独立停止、观测和回滚。

下一章依赖本章的资源和生命周期契约，转向 Ray 等分布式应用运行时：Kubernetes 负责节点、设备和 Pod 的边界，Ray 的 actor、task、placement group 和 Serve 负责应用级状态与调度。进入第18章前，读者应能运行 CPU 模拟器、解释每个假设、读懂一份真实集群的资源和拓扑报告，并写出一份包含回滚触发器的升级计划。


## 17.13 设备分区、DRA 与共享语义

传统 device plugin 用一个扩展资源名描述可分配设备，适合“整卡给一个 Pod”的简单场景。当平台需要按显存、计算能力、互联位置或安全域选择设备时，Dynamic Resource Allocation（DRA）提供了更丰富的声明方式。工作负载通过 ResourceClaim 或 ResourceClaimTemplate 请求 DeviceClass，DRA 驱动依据属性筛选并返回设备句柄，调度器在绑定前把声明与节点可用性关联起来。DRA 仍然需要节点驱动和运行时注入；它解决的是资源声明与分配表达力，不会自动解决应用级通信或显存超量。

采用 DRA 前要确认集群版本、feature gate、驱动和控制器的兼容矩阵。ResourceClaim 是独立对象，可能有生命周期和 finalizer；删除工作负载而忘记清理 claim，会让设备看起来被占用。升级时先验证旧 claim 能否被新驱动读取，确认 schema 往返兼容，再扩大范围。当前实现对 DRA 资源的抢占能力可能有限，高优先级作业不一定能驱逐低优先级 claim，因此队列需要把等待和释放策略写入 admission 控制器，而不是假设默认 scheduler 会替你完成。

MIG（Multi-Instance GPU）把支持的 GPU 划分为具有独立计算和显存配额的实例。MIG profile 不是任意显存切片，配置改变通常需要停止使用设备的 Pod，某些平台还要重启或重新初始化 GPU。切换 profile 前应 cordon 节点、确认无用户作业、保存 checkpoint，并记录旧 profile 到新 profile 的映射。实例 UUID、节点标签和资源名可能随着重配置改变，应用不要把这些值写死在代码里。MIG 能缩小故障和显存争用范围，但驱动、PCIe、功耗和监控仍可能共享。

Time-slicing 通过多个副本让 Pod 交错使用整张 GPU，调度器看到的“副本数”是可超卖的逻辑容量。它不提供 MIG 的显存或故障隔离，一个进程的非法访问或驱动错误仍可能影响同卡租户；请求两个副本也不等于得到两倍算力。若平台允许 time-slicing，必须把每个副本的最大并发、显存行为、抢占延迟和故障域写入服务等级，必要时拒绝大于一的请求，避免用户误解。监控应同时显示物理卡利用率和每个逻辑租户的有效样本数，否则一个贪婪作业可能让整卡利用率达到 100%，其他作业却几乎没有进展。

MPS、用户态多进程和框架级批处理又是另一组共享机制。它们可以提高利用率，也把内存、错误传播和计费复杂度推到应用层。无论使用哪种机制，都要定义“设备分配单位”“显存上限”“故障影响范围”“可观察的账单单位”和“清理动作”。如果这五项无法回答，宁可采用整卡或 MIG 的保守模式。

## 17.14 状态机、证据链与容量账本

排查 GPU Pod 时，建议把状态拆成五个相互独立的轴。第一轴是对象状态：API Server 是否接受对象、generation 是否更新、owner 是否存在。第二轴是调度状态：Pod 是否进入队列、过滤失败原因、是否通过 permit、是否已 bind。第三轴是节点状态：kubelet Lease、Node Ready、allocatable、驱动和 device plugin 注册。第四轴是容器状态：镜像拉取、sandbox、设备注入、进程退出码、探针。第五轴是应用状态：模型加载、首个训练 step、通信 rendezvous、吞吐和 checkpoint。

每个轴都要有“能证明”和“不能证明”的证据。例如 `kubectl get pod` 的 `Running` 能证明容器进程被 kubelet 观察到，却不能证明 readiness、GPU 初始化或应用接受请求；`nvidia-smi` 能证明宿主机工具能查询设备，不能证明容器获得了正确的设备集合；DCGM 的利用率能证明某些 GPU 周期忙碌，不能证明有效样本、收敛或租户公平。把证据写成表格后，排障就能避免从一个绿色状态跳到另一个猜测。

容量账本应把静态、动态和保留三类量分开。静态容量包括节点型号、GPU 数、MIG profile、CPU、内存、NIC 和本地存储；动态容量包括当前分配、健康设备、队列中的 claim、驱动升级占用和节点排空；保留容量包括系统 DaemonSet、监控、故障余量、SLA 预留和可抢占池。自动扩缩容只看静态总数会在升级或故障时过度承诺，只看动态空闲又可能把系统保留吃掉。

建议每次调度决策都保留摘要：请求形状、过滤失败的第一原因、候选节点数量、最终节点、配额和优先级、是否跨拓扑域、设备插件返回的句柄类型。摘要中不要记录完整 prompt、Secret 或原始文档。长期统计可按租户输出等待 p50/p95、可行容量、拒绝率、有效 GPU 小时、抢占次数、恢复时间和碎片率。碎片率应说明分母，例如“无法满足队列头部形状的空闲 GPU 比例”，否则不同团队会用同一个词描述不同现象。

## 17.15 端到端升级演练脚本

为了把升级从口号变成可重复过程，可把一次 canary 演练分成十个门。门一是备份：导出 API 对象、operator 自定义资源、etcd 快照（由管理员完成）和最近 checkpoint。门二是冻结：暂停自动扩缩容和新作业准入，保留系统恢复作业。门三是选择：只挑一个没有高优先级作业的节点，并记录其故障域和旧版本摘要。门四是排空：cordon、等待可中断作业检查点、遵守 PDB，超时则停止而不是强删。

门五是升级节点镜像、驱动或运行时，记录内核、固件和容器工具包版本。门六是等待 operator、device plugin 和监控组件达到 Ready，确认 socket、allocatable 和健康设备数恢复。门七是运行三层烟雾：宿主机查询、容器内 CUDA 分配、代表性框架和通信。门八是运行短训练或推理回放，比较吞吐、p95、显存高水位、数值和日志脱敏。门九是执行恢复演练：人为终止 worker，在 checkpoint 边界恢复，验证数据和随机状态。门十才是解除 cordon、恢复准入并记录 go/no-go 决定。

每个门都有停止条件。若驱动模块无法加载、设备插件资源数量与预期不符、容器可见未授权设备、NCCL 带宽低于基线、数值差异超阈值或 checkpoint 校验失败，立即停在当前门，保留证据并回滚。回滚过程按相反顺序执行，但不假设所有变化可逆：固件刷写、MIG profile、CRD schema 和数据格式可能需要专门恢复。若旧节点池仍在线，优先把流量和作业迁回旧池，再处理新池；若只能原地回滚，应先评估重启和数据风险。

升级报告至少包含：变更范围与负责人、对象和镜像摘要、每门开始结束时间、观测指标及对照、异常事件、是否触发回滚、剩余风险和下一步。报告面向下一次升级，而不是为了证明这次“成功”。没有报告的手工修复会把临时命令变成未记录的系统状态，下一次故障将无法判断哪些组件真正改变过。


## 17.16 实验期望输出与解释指南

CPU 模拟器在默认输入下应打印四个作业，每行包含作业名、等待时间和运行时间。由于 `train-a` 先占用四张卡，`serve-b` 可以在另一节点获得一张卡；`train-c` 要求四卡 gang，必须等 `train-a` 释放整组资源，等待时间应明显高于 `serve-b`。如果把 `gang` 改为 `False`，`train-c` 可能分散启动，等待时间下降但会出现跨域通信惩罚；这正是“吞吐看起来更好、有效训练更差”的反例。若把 `team-a` 配额降到三，`train-a` 和 `train-c` 都应因配额检查失败而不能放置，日志应包含租户和请求形状，而不是只打印通用的 `no resources`。

YAML 设计实验中，资源和节点都满足时，事件顺序应接近 `Scheduled`、镜像拉取、容器创建、设备分配、`Started`、就绪探针成功。若节点只有污点而 Pod 没有 toleration，`FailedScheduling` 应指出 taint；若 GPU 扩展资源不足，应指出 insufficient 扩展资源；若 device plugin 被停止，Node 的 allocatable 会下降，新的 Pod 应保持 Pending 或在创建阶段失败。恢复插件后，记录注册延迟、allocatable 恢复时间和既有 Pod 是否受到影响；不能因为一个 Pod 最终 Running 就忽略中间的设备不可用窗口。

gang 演练的成功标准不是“所有 Pod 最终运行”，而是等待期间没有半启动 rank 长期占用 GPU，满足最小组后整组在一个受控窗口内绑定，任一 rank 失败时控制器释放或重试整组。拓扑实验应同时报告节点选择、GPU 互联图、NCCL 或模拟通信时间；只报告 GPU 利用率无法解释跨 NUMA 造成的尾延迟。升级演练应产出旧/新版本矩阵、每个 canary 门的时间戳、烟雾和短训练对照、checkpoint 校验结果以及 go/no-go 决定。

| 场景 | 预期输出 | 解释与下一步 |
| --- | --- | --- |
| 节点和资源均满足 | 事件依次出现 `Scheduled`、设备分配、容器启动、探针成功 | 只能证明控制契约成立；还要运行真实 CUDA/框架 smoke |
| 缺少 GPU toleration | `FailedScheduling` 指向 taint | 给目标 Pod 增加最小 toleration，或确认它不应进入 GPU 池 |
| 扩展资源不足 | `Pending` 且事件写明 insufficient 扩展资源 | 检查按节点池和拓扑域的可行容量，不要盲目重试 |
| device plugin 停止 | allocatable 降低，新 Pod Pending 或创建失败 | 记录注册恢复时间，既有 Pod 需单独验证设备是否仍可用 |
| gang 未满足 | 整组等待、没有长期半启动 rank | 若长期等待，检查队列、配额、节点形状和抢占策略 |
| 升级 canary 失败 | 触发停止门，旧池仍可接管 | 保存事件、版本和 checkpoint，再按既定顺序回滚 |


将实验输出分成三层解读。第一层是控制契约：对象是否被接受、调度是否符合约束、插件是否上报设备。第二层是资源行为：等待、碎片、抢占、显存和通信是否按模型变化。第三层是应用结果：首 step、吞吐、数值、checkpoint 和用户可见错误。低层通过不能推导高层通过；例如 `Scheduled` 不能证明 CUDA 初始化，CUDA 初始化也不能证明训练收敛。若结果与假设矛盾，先保存原始事件和版本信息，再缩小实验规模，最后更新假设，不要直接修改阈值直到“通过”。

## 17.17 来源地图与可复现记录

下表区分来源支持的事实与本章设计判断。访问日期统一为 2026-10-05；版本随集群升级应重新核对。

| 来源 | 版本/日期 | 支持的主张 | 类型 |
| --- | --- | --- | --- |
| OCI Image Specification，opencontainers.org/image-spec | v1.1.x（访问 2026-10-05） | 镜像清单、层和配置的格式边界 | 事实 |
| OCI Runtime Specification，opencontainers.org/runtime-spec | v1.3.0（2025；访问 2026-10-05） | 容器进程、namespace、cgroup 和运行时配置 | 事实 |
| Kubernetes Device Plugins 文档，kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/ | 在线文档（访问 2026-10-05；以目标集群版本为准） | 注册、ListAndWatch、Allocate 与扩展资源 | 事实/机制 |
| Kubernetes Resource Management 文档，kubernetes.io/docs/concepts/configuration/manage-resources-containers/ | 在线文档（访问 2026-10-05；以目标集群版本为准） | requests、limits 和不可压缩资源语义 | 事实 |
| Kubernetes Topology Manager，kubernetes.io/docs/tasks/administer-cluster/topology-manager/ | 在线文档（访问 2026-10-05；以目标集群版本为准） | NUMA hint、策略和 kubelet 协调 | 事实 |
| Kubernetes Scheduling Framework，kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/ | 在线文档（访问 2026-10-05；以目标集群版本为准） | 过滤、打分、绑定和扩展点 | 事实/机制 |
| Kubernetes ResourceQuota，kubernetes.io/docs/concepts/policy/resource-quotas/ | 在线文档（访问 2026-10-05；以目标集群版本为准） | Namespace 配额与准入行为 | 事实 |
| Kubernetes Pod Security Standards，kubernetes.io/docs/concepts/security/pod-security-standards/ | 在线文档（访问 2026-10-05；以目标集群版本为准） | 特权、host namespace、capability 等安全边界 | 事实 |
| Kubernetes JobSet/批处理工作负载文档，kubernetes.io | 2024–2025 | 作业组与批处理控制器的设计参考 | 机制 |
| NVIDIA Kubernetes device plugin，github.com/NVIDIA/k8s-device-plugin | 在线仓库（访问 2026-10-05；版本随集群核对） | GPU 扩展资源、设备分配和插件配置示例 | 代码/机制 |
| NVIDIA GPU Operator 文档，docs.nvidia.com/datacenter/cloud-native/gpu-operator | 在线文档（访问 2026-10-05；版本随集群核对） | 驱动、toolkit、插件、监控和升级组件关系 | 事实/机制 |
| NVIDIA Container Toolkit 文档，docs.nvidia.com/datacenter/cloud-native/container-toolkit | 在线文档（访问 2026-10-05；版本随集群核对） | OCI hook/CDI、设备和库注入 | 事实/机制 |
| Kubernetes SIG Scheduling 与 Volcano 项目文档 | 2024 | gang、队列、公平和抢占策略的实现参考 | 机制 |
| NVIDIA NCCL 文档与拓扑工具 | 2.20+，2024 | 集体通信、拓扑报告和带宽测量方法 | 事实/测量 |
| 本章 `chapter17-draft.md` CPU 模拟器 | Python 3.11+，标准库 | FIFO、配额、gang 与碎片的可重复离散事件实验 | 测量/设计 |

### 可复现记录

- 纯 CPU 实验：Python 3.11 或更新版本；仓库附带 `ch17_gpu_sim.py`，运行 `python ch17_gpu_sim.py` 可直接复现默认输出；随机性为零，输入顺序固定。
- Kubernetes YAML 实验：Kubernetes 1.28–1.30、containerd 或 CRI-O、已安装并验证的 GPU device plugin；先在隔离 Namespace 运行，记录 `kubectl version`、节点标签、插件镜像 digest 和 operator 版本。
- GPU 测量：记录 GPU 型号、显存、固件、驱动、CUDA、通信库、内核、CPU/NUMA、网络、温度、功耗、镜像 digest、数据和种子；至少重复三次并报告均值、标准差和 p95。
- 任何升级实验都先保存 checkpoint 与对象清单，设置明确停止条件；若设备错误、跨租户可见性异常、数据校验失败或 p99 超过基线二倍，立即停止扩批并执行回滚计划。

### 安全边界声明

本章的 CPU 模拟器不访问真实设备，不验证 GPU 隔离，也不证明 operator 或驱动升级安全。YAML 中的镜像、资源名、标签和权限均为示例，不能直接复制到生产。真实集群实验必须由有权限的管理员在隔离节点池执行，使用最小服务账户、签名镜像和可恢复 checkpoint；不得为了通过实验关闭 Pod Security、审计、网络策略或租户隔离。任何涉及特权驱动安装、固件刷写、强制删除有状态训练、跨租户数据访问或接受新法律条款的动作，都需要事先批准、维护窗口和人工复核。
