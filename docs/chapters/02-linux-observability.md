---
id: ch02-linux-process-files-observability
title: Linux、进程、文件系统与可观测性
description: 从系统调用到容器边界，建立 AI worker 故障诊断的操作系统底座
slug: /chapters/02-linux-process-files-observability
sidebar_position: 2
level: foundation
prerequisites:
  - ch01-ai-infrastructure
learning_objectives:
  - 能解释进程、线程、系统调用、虚拟内存和 page cache 的因果关系
  - 能用文件描述符、信号、cgroup 与 namespace 诊断容器化 worker
  - 能设计 CPU 实验并使用 /proc、perf、eBPF 交叉验证假设
  - 能区分事实、机制、测量、推断和设计判断
estimated_hours: 18
hardware: Linux CPU baseline; container runtime optional
risk_level: L1
last_verified: 2026-10-05
---

# 第2章　Linux、进程、文件系统与可观测性

> AI worker 并不是“一个 Python 脚本加一块 GPU”。它是由进程、线程、系统调用、虚拟地址、页缓存、文件描述符、调度器、cgroup 和容器边界共同组成的运行实体。本章从 Linux 内核可观察的事实出发，建立一条从源码级机制到故障诊断的路径。所有结论都明确标注为事实、机制、测量、推断或设计判断。

## 2.1 为什么 AI 基础设施必须懂 Linux

一次推理请求进入 worker 后，至少要经过套接字读取、用户态解析、内核态拷贝或映射、线程调度、文件或 page cache 访问、模型运行时调用以及响应写回。即使模型算子在加速器上执行，CPU 仍负责排队、内存管理、驱动 ioctl、日志和进程生命周期。

[事实] Linux 内核把资源抽象为任务（task）、地址空间、文件描述符、命名空间和控制组。用户态程序只能通过受支持的系统调用改变这些资源；普通 Python、C++ 或 Rust API 最终都会落到一组 syscall 上。

[机制] 一个“慢请求”不是单一事件。它可能在用户态等待锁，等待运行队列中的时间片，触发缺页并从磁盘填充页，阻塞于 socket，或被 cgroup 的 CPU/内存限制延迟。相同的端到端延迟可以由完全不同的机制产生，所以只看应用日志无法定位根因。

[设计判断] 读本章时，先画出 worker 的状态与资源图：哪些线程属于哪个进程，哪些 fd 指向同一 socket，哪些文件通过 mmap 映射，哪个 cgroup 施加 CPU 和内存上限，哪个 namespace 改变了它看到的 PID、挂载点和网络。图越接近内核事实，排障越少靠猜。

### 2.1.1 五类证据

- **事实**：内核、man page 或代码明确规定的语义，例如 `fork()` 创建子进程、`read()` 返回字节数。
- **机制**：由事实推导的因果过程，例如缺页处理会先查页表，再决定从匿名页、page cache 或交换区取页。
- **测量**：使用命令、计数器、tracepoint 或实验得到的观察值，例如某进程 `minor-faults` 每秒 2 万次。
- **推断**：基于测量提出的可证伪解释，例如缺页突增可能由首次 mmap 访问引起。
- **设计判断**：在目标和约束下的选择，例如将长上下文请求放入独立 cgroup。

[设计判断] 文中每个结论都应该能回答“证据在哪里、如何反驳、适用边界是什么”。如果一句话无法归类，通常说明它混淆了事实与经验。

## 2.2 进程、线程与任务：调度器真正看到什么

### 2.2.1 进程是资源视图，线程是执行单元

[事实] 在 Linux 中，进程通常对应一个地址空间、文件描述符表、信号处理设置和若干线程；线程共享进程的大部分资源，但拥有独立的寄存器上下文、栈和调度实体。内核内部把进程和线程都表示为 `task_struct`，线程组由 `tgid` 标识，单个线程有自己的 `tid`。

[机制] 调度器选择的是可运行任务，而不是抽象的“进程”。一个多线程 worker 如果创建 32 个 CPU 线程，最多会同时占用 32 个逻辑 CPU（还受亲和性和 cgroup 限制）。线程在用户态共享堆和全局对象，因此锁竞争、缓存一致性和错误传播也共享。

[测量] `/proc/<pid>/status` 可查看 `Threads`、`VmRSS`、`voluntary_ctxt_switches` 等字段；`/proc/<pid>/task/` 下的每个目录对应一个线程。`ps -L -p <pid>` 或 `top -H` 可观察线程级 CPU 时间，但采样窗口和显示精度要写进记录。

[推断] 若 worker 的进程 CPU 利用率只有 200%，而机器有 32 个核，不代表“CPU 空闲且没有瓶颈”。它可能只有两个线程活跃、被锁串行化、受 cgroup 配额限制，或大部分时间在 IO 睡眠。需要同时看每线程状态、运行队列、iowait 和 cgroup throttling。

### 2.2.2 生命周期与状态

[事实] 常见任务状态包括运行（`R`）、可中断睡眠（`S`）、不可中断睡眠（`D`）、停止（`T`）和僵尸（`Z`）。僵尸已经结束但父进程尚未 `wait()` 回收其退出状态；它几乎不占用运行内存，却会占用 PID 表项。

[机制] `fork()` 复制进程的资源视图，现代 Linux 通常使用写时复制（COW）避免立即复制每个物理页；`execve()` 用新程序映像替换当前地址空间；`waitpid()` 让父进程读取子进程退出状态并释放僵尸记录。容器内 PID 1 若不正确转发或回收子进程，长时间运行的 worker 可能积累僵尸。

[测量] `/proc/<pid>/stat` 的第三字段是状态，`ps -eo pid,ppid,stat,wchan:32,cmd` 可以把不可中断任务等待的内核函数名显示出来（内核和权限可能限制符号）。`pstree -ap` 适合检查父子关系，但命名空间会让宿主机和容器看到不同 PID。

[设计判断] 服务进程应明确谁负责子进程回收、优雅停止和超时。不要依赖“容器被删掉就自然清理”的愿望；发布脚本、信号处理器和 watchdog 都应有可测的退出路径。

### 2.2.3 线程模型与 Python worker

[事实] CPython 的 GIL（全局解释器锁）限制同一解释器内的 Python 字节码并行，但 C 扩展可以在等待 IO 或执行释放 GIL 的代码时并行。多进程、原生线程池、异步事件循环和外部运行时各自有不同的内存、调度和故障边界。

[机制] 线程池过大不仅增加并发，还增加上下文切换、锁竞争、L1/L2 缓存失效和 page cache 争用。若 AI 运行时内部再开线程，应用层的 8 个线程可能实际生成 8×16 个原生线程，迅速超过 CPU 配额。

[测量] 记录线程总数、每个线程的 CPU 时间、上下文切换、运行队列长度，以及运行时环境变量（例如 BLAS/OpenMP 线程数）。测量前后应固定输入形状和线程亲和性，否则“线程数优化”会被请求分布混淆。

## 2.3 系统调用：用户态和内核态的窄门

### 2.3.1 一次 `read()` 发生了什么

[事实] 用户程序不能直接执行特权内核代码。它通过 syscall 指令（在 x86-64 上常见 `syscall` 指令）切换到内核入口，传递系统调用号和寄存器参数。内核验证地址和权限，查找文件对象，执行操作，再以返回值或负错误码返回用户态。

[机制] 以 `read(fd, buf, n)` 为例：内核先从当前任务的 fd 表取得 `struct file`，再调用其 `f_op->read_iter` 或相关实现；若数据在 socket 接收队列或 page cache 中，可能立即复制到用户缓冲区；若没有数据且 fd 未设置 `O_NONBLOCK`，任务会被加入等待队列并睡眠。数据到达后由唤醒路径使任务重新可运行。

[源码级线索] Linux VFS 把路径名解析与具体文件系统分开，`struct file` 保存打开实例，`struct inode` 描述文件元数据，`struct address_space` 连接文件与 page cache。不同文件系统和 socket 实现会提供不同的操作函数表，因此同一个 `read()` 入口可能走到不同的内核代码。

[测量] `strace -f -ttT -e trace=%file,%network -p <pid>` 可以记录 syscall 起止时间；`-T` 显示单次调用耗时，`-f` 跟踪子线程/子进程。`strace` 会改变时序并增加开销，不能把跟踪状态下的延迟当作生产基线。

[推断] 如果 `strace` 显示大量 `futex()` 且每次耗时很短，可能是线程锁或条件变量竞争；如果 `read()` 长时间处于 `D` 状态，可能等待块设备或网络文件系统；如果 `mmap()` 很快但后续首次访问产生大量缺页，瓶颈在延迟加载，而不是映射调用本身。

### 2.3.2 syscall 成本与批量化

[机制] syscall 有固定入口、寄存器保存、权限检查和返回开销。对小消息频繁调用 `write()`，固定成本可能大于复制成本；使用 `writev()`、`sendmsg()`、`splice()` 或用户态批量队列可以减少调用次数，但会增加缓冲和调试复杂度。

[测量] 用 `perf stat -e cycles,instructions,syscalls:sys_enter_read,syscalls:sys_enter_write`（事件可用性因内核配置不同）比较每请求 syscall 数与 CPU 周期。报告应注明是否包含网络栈、数据拷贝和等待时间。

[设计判断] 对 AI worker，优先减少“每 token、每日志行、每小张量一次 syscall”的模式。批量化要与尾延迟预算一起评估，不能只以 syscall 数下降为成功标准。

## 2.4 虚拟内存：地址、页表、TLB 与缺页

### 2.4.1 虚拟地址不是物理内存

[事实] 每个进程看到独立的虚拟地址空间。页表把虚拟页号映射到物理页框，并携带可读、可写、可执行、用户/内核等权限位。CPU 的 MMU 使用页表完成地址转换；TLB 缓存近期转换以避免每次访问都遍历页表。

[机制] 在常见 4 KiB 页大小下，虚拟地址可拆为页号和页内偏移。若页大小为 (P)，地址 (v) 对应页号 ⌊(v/P)⌋ 和偏移 (v \bmod P)。大页可减少 TLB 项和页表层级，但会增加内部碎片和分配/回收约束。

[事实] 内存访问可能触发三类不同事件：TLB 命中直接得到物理地址；TLB 未命中但页表映射存在，需要页表遍历；页表项无效或权限不符，触发 page fault 异常。缺页异常不等于磁盘 IO，有些是匿名页首次分配或 COW 写入，可在内存中完成。

[机制] 缺页处理大致经过：保存异常现场、定位 `vm_area_struct`、检查访问权限、分配或查找物理页、必要时从文件或交换区读入、更新页表、刷新相关 TLB 项，再重新执行触发指令。若物理内存不足，回收和写回会把一次访问放大为毫秒级甚至更长延迟。

[测量] `/proc/<pid>/stat` 中的 `minflt` 和 `majflt` 提供进程级 minor/major fault 累计值；`perf stat -e page-faults,minor-faults,major-faults` 可按进程或命令统计。major fault 通常需要存储层参与，但具体含义取决于内核和映射类型，不能直接等同于“磁盘坏了”。

[推断] 如果模型加载后首次请求很慢、随后变快，候选机制包括文件页首次进入 page cache、mmap 页首次 fault、JIT 编译或运行时缓存建立。应通过重复冷/热实验、page fault 计数和 IO 计数器区分，而不是只看总延迟。

### 2.4.2 RSS、虚拟大小与共享页

[事实] `VmSize` 或 VIRT 代表虚拟地址空间大小，不等于实际占用物理内存；RSS 是当前驻留在物理内存中的页，PSS 会按共享页比例分摊，USS 近似进程独占页。共享库、COW 和 mmap 文件会使这些数值差异很大。

[测量] `/proc/<pid>/smaps_rollup` 可提供 RSS、PSS、Private_Dirty 等汇总（权限和内核版本可能不同）。容器内看到的 RSS 还应和 cgroup 的 `memory.current`、`memory.stat` 对照，因为后者包括不同层级的记账对象。

[设计判断] 容量规划不要用单一 RSS 阈值推断“还能启动多少 worker”。应按模型权重、激活、线程栈、page cache、共享页、碎片和突发请求建立上界，并留出内核和 sidecar 余量。

### 2.4.3 OOM 与过度承诺

[事实] 当 cgroup 或节点内存压力达到限制，内核可能触发 OOM 选择器，向某个进程发送 SIGKILL；被杀进程无法捕获或清理。容器运行时通常把退出码 137 等现象报告给上层，但退出码只是症状。

[机制] cgroup v2 的 `memory.max` 限制组内可用内存，`memory.high` 可触发回收和节流而不立即杀死；内核选择 victim 时会考虑进程的 oom_score_adj、内存使用和层级。page cache 也可能计入 cgroup 内存，因此“模型权重没变”仍可能因日志或数据缓存导致 OOM。

[测量] 采集 `memory.events` 中的 `high`、`max`、`oom`、`oom_kill`，并关联进程退出时间、工作集、page cache 和请求长度。若只有应用日志里的“Killed”，证据不足以区分 cgroup OOM、宿主机 OOM、人工 kill 或节点重启。

## 2.5 page cache、文件 IO 与 mmap

### 2.5.1 page cache 是内核的文件数据缓存

[事实] 普通文件读写通常经过 page cache。读取时，内核可从缓存页复制数据；缓存未命中则从块设备填充页面。写入通常先修改缓存页并标记 dirty，再由 writeback 线程异步写回；`fsync()` 请求把相关数据和元数据推进到持久介质，但具体持久性受文件系统和设备语义影响。

[机制] page cache 让多个进程映射同一模型文件时共享物理页，降低重复读盘和内存占用；但缓存争用会驱逐热点代码、词表或其他租户数据。容器并不会自动拥有一份独立 page cache，隔离取决于宿主内核和 cgroup 记账。

[测量] `/proc/meminfo` 中的 Cached、Active(file)、Dirty、Writeback 展示全局情况；`/proc/<pid>/io` 提供进程累计读写字节和 syscall 次数。`iostat -xz`、`pidstat -d` 或 eBPF 工具可观察设备队列和等待，但需标明采样窗口。

[推断] 若多个副本启动时读同一模型文件，后续副本变快可能是 page cache 热了，而不是每个副本真正拥有独立的高速存储。节点重启、内存压力或不同挂载点可能使结果失效，应做冷缓存和热缓存两套测试。

### 2.5.2 `mmap` 与零拷贝的边界

[事实] `mmap()` 把文件区域映射到进程虚拟地址，访问映射地址时按页触发 fault；这避免了应用显式 `read()` 到用户缓冲区的一次拷贝，但不保证没有内核到设备、页表或 cache miss 成本。

[机制] 模型运行时常用只读 mmap 加载权重，以利用按需分页和进程间共享。随机访问会造成许多缺页和低局部性；顺序预读、`madvise()` 或显式加载可以降低首次延迟，却可能增加启动内存和 IO 峰值。

[测量] 用 `mincore()`（需权限和实现支持）或 page fault 计数估计哪些页驻留；比较 `MAP_PRIVATE`、`MAP_SHARED`、预读与不预读配置时，记录启动时间、首请求延迟、RSS/PSS、设备读带宽和质量结果。

[设计判断] “零拷贝”应被当作有条件的设计目标，而不是营销标签。只要跨越内核、设备或不同地址空间，仍可能需要拷贝、同步或缓存失效；是否值得做应由端到端测量决定。

### 2.5.3 写放大与日志陷阱

[机制] 同步日志、频繁 `fsync()`、小块随机写和容器 overlay 文件系统的 copy-up 都可能放大 IO。AI worker 在每 token 写一行日志，会把本应批量的顺序写变成高频 syscall 和 writeback 压力。

[测量] 统计日志字节/请求、`write()` 次数、fsync 延迟、设备 util、dirty 页和应用 p99。降低日志级别只是实验变量，不能直接当成长期方案，因为可观测性损失也应计入设计判断。

## 2.6 文件描述符：把“资源”统一成整数

### 2.6.1 fd 表与 open file

[事实] 文件描述符是进程 fd 表中的非负整数索引，指向内核 `struct file`；多个 fd 可以通过 `dup()` 或 `fork()` 指向同一个 open file description，共享文件偏移和状态标志。路径名不是 fd，路径解析在 `open()` 时完成，之后 fd 可继续使用，即使文件被重命名或删除。

[机制] socket、pipe、eventfd、timerfd、epoll 实例、设备节点和普通文件都可表现为 fd，使事件循环能够统一等待。关闭一个 fd 只减少一次引用；底层对象在所有引用释放前仍存在。忘记关闭会导致 fd 泄漏，最终触发 `EMFILE`（进程上限）或 `ENFILE`（系统级表耗尽）。

[测量] `/proc/<pid>/fd/` 的符号链接显示当前 fd 指向；`lsof -p <pid>` 汇总文件、socket 和设备；`cat /proc/<pid>/limits` 显示 `Max open files`。对服务应记录 fd 数、监听 socket、连接池、日志文件和临时文件数量。

[推断] “连接池耗尽”可能表现为请求排队、超时或 `EMFILE`，但不一定是网络故障。若 fd 数随请求单调增长且 GC 后不下降，优先检查异常路径是否漏关 socket、临时文件或 epoll 注册对象。

### 2.6.2 阻塞、非阻塞与 epoll

[事实] 阻塞 fd 在无数据时会让任务睡眠；`O_NONBLOCK` 让调用在暂时不可用时返回 `EAGAIN/EWOULDBLOCK`。`select/poll/epoll` 让一个线程等待多个 fd 的就绪事件；epoll 通过内核事件集合减少每次扫描大量 fd 的成本，但不会替应用读取完数据或处理惊群。

[机制] 边沿触发（EPOLLET）要求应用在收到事件后循环读取直到 `EAGAIN`，否则可能错过后续通知；水平触发更易编程但可能反复报告未处理事件。线程池和事件循环混用时，fd 所属线程、取消和关闭顺序必须明确。

[测量] `strace` 可观察 `epoll_wait` 返回数和耗时；eBPF 可统计每个 fd 的等待时间和错误码。压测时按连接数、消息大小、事件批大小和线程数分桶，不要只比较 QPS。

### 2.6.3 pipe、socket 与背压

[机制] pipe 和 socket 都有内核缓冲区。写端填满后，阻塞写会睡眠，非阻塞写返回 `EAGAIN`；这就是内核层面的背压。若上游忽略错误、无限重试或创建无界用户队列，最终会把内核可控的背压变成进程 OOM。

[设计判断] AI worker 的日志、指标和任务队列都应有有界缓冲与丢弃/降级策略。可观测性系统本身不能拖垮数据面；应区分必须保留的错误事件与可采样的调试事件。

## 2.7 信号：异步控制与不可捕获的终止

### 2.7.1 信号语义

[事实] 信号是内核向线程组或特定线程传递的异步事件。SIGTERM 通常用于请求优雅终止，SIGINT 对应交互中断，SIGHUP 常用于重新加载，SIGKILL 不能被捕获、阻塞或忽略，SIGSTOP 也不能被捕获。

[机制] 信号可能在任意用户态指令边界被递送；处理器应只做异步信号安全操作，例如设置原子标志或写入 self-pipe，不能在 handler 中调用可能加锁、分配内存或使用 stdio 的函数。主循环看到标志后再执行清理和退出。

[测量] `/proc/<pid>/status` 的 `SigQ`、`SigBlk`、`SigIgn`、`SigCgt` 显示队列与处理设置；`kill -0 <pid>` 只检查存在性/权限，不会发送可见信号。`strace -f -e signal=all` 能帮助关联信号与退出，但会改变时序。

[推断] worker 收到 SIGTERM 后仍在运行，不一定“忽略了信号”：可能主线程阻塞在不可中断 IO、子线程持锁、退出超时后被运行时升级为 SIGKILL，或信号发给了错误的 PID namespace。应记录信号发送时间、处理器日志、线程栈和容器 stop 超时。

### 2.7.2 优雅关闭的时序

[设计判断] 推荐顺序是：停止接收新请求；标记实例不再就绪；让正在处理的请求在 deadline 内完成或取消；关闭监听 fd；回收子进程和线程；刷新必要日志；最后退出。每一步应有时间预算和强制终止路径。

[事实] 若进程是容器 PID 1，默认信号转发和僵尸回收行为可能与普通进程不同，取决于运行时和 init 方案。不要假设 shell 包装脚本会自动把 SIGTERM 转给真正的 worker。

## 2.8 调度、CPU 亲和性与 cgroup

### 2.8.1 调度器的可运行队列

[事实] Linux 调度器在逻辑 CPU 上选择可运行任务，CFS 等策略根据权重和虚拟运行时间分配份额；实时策略有不同语义。任务睡眠时不占用运行队列，但唤醒、迁移、锁竞争和 NUMA 远端访问仍会产生成本。

[测量] `pidstat -w` 查看上下文切换，`vmstat 1` 查看运行队列、阻塞与上下文切换，`perf sched timehist` 可追踪调度延迟。`taskset -cp` 和 cpuset cgroup 能限制亲和性；实验必须记录是否启用 CPU 隔离、频率调节和超线程。

[机制] cgroup v2 的 `cpu.max` 以“配额/周期”限制 CPU，例如 `20000 100000` 约允许每 100 ms 使用 20 ms 的总 CPU 时间；超额后会发生 throttling。`cpu.weight` 是相对权重，不是硬上限。多线程进程共享组配额，线程越多不会突破总配额。

[测量] `cpu.stat` 中的 `nr_periods`、`nr_throttled`、`throttled_usec` 可验证是否被节流。若应用 CPU 使用率看似低而延迟高，检查 throttling 比盯着宿主机总体 idle 更有价值。

[设计判断] 给 AI worker 配额时，用请求形状和尾延迟 SLO 反推 CPU 预算。过度超卖会让 p99 随邻居负载抖动；过度独占则增加成本。亲和性、NUMA 和线程数应通过测量而非信仰决定。

### 2.8.2 cgroup 内存和 IO

[事实] cgroup 可对 CPU、内存、块 IO、进程数等资源进行层级记账和控制。cgroup v2 统一层级通常挂载在 `/sys/fs/cgroup`，具体控制文件由内核配置和权限决定。

[机制] `pids.max` 限制组内任务数，线程也计数；达到限制时 `clone/fork` 失败。`io.max`、`io.weight` 可影响块设备调度，但不一定控制已在 page cache 中的数据。资源限制与 namespace 是正交概念：cgroup 控制“能用多少”，namespace 控制“能看到什么”。

[推断] 容器中 `fork: Resource temporarily unavailable` 可能是 pids cgroup 达到上限，而非宿主机 PID 不足；`malloc` 失败可能来自 memory.max 或地址空间限制。读取对应 cgroup 文件并关联 errno 才能确认。

## 2.9 namespace 与容器：隔离视图，不是魔法虚拟机

### 2.9.1 七类常见 namespace

[事实] Linux 提供 PID、mount、network、UTS、IPC、user、cgroup 等 namespace。PID namespace 让进程拥有不同的 PID 视图；mount namespace 隔离挂载树；network namespace 隔离接口、路由和端口；user namespace 映射 UID/GID 与能力。

[机制] 容器运行时通常组合 namespace、cgroup、rootfs 和安全策略（如 seccomp、能力集）。容器内的 PID 1 只是该 namespace 的第一个进程，宿主机仍可看到对应的不同 PID。容器不是独立内核，系统调用仍进入宿主内核，内核漏洞或共享资源争用不能靠 rootfs 解决。

[测量] `lsns -p <pid>` 显示进程所属 namespace；`readlink /proc/<pid>/ns/*` 可比较两个进程是否共享同一 namespace。`nsenter -t <host-pid> -m -n -p`（需权限）可从宿主视角进入目标 namespace，执行前要确认不会改变生产状态。

[设计判断] 把容器边界当作“可观察和可控制的边界”，而不是“绝对隔离”。敏感工作负载需要最小能力、只读根文件系统、设备白名单、seccomp 和专用节点；这些控制的开销与可调试性应在上线前测量。

### 2.9.2 overlayfs 与模型文件

[机制] overlayfs 将 lower（只读层）和 upper（可写层）合并成统一路径。对 lower 层文件的首次写入可能触发 copy-up，把整个文件复制到 upper；大型模型或缓存目录因此产生巨大 IO 和空间峰值。

[测量] 比较容器内写入与宿主 bind mount、临时卷的 `stat -f`、IOPS、启动时间和磁盘使用。不要只看镜像大小，因为运行时 upper 层和 page cache 才决定节点压力。

### 2.9.3 容器中的时钟、PID 和网络陷阱

[事实] 时间命名空间、PID 视图和网络命名空间会影响日志、超时和端口诊断。容器内看到的 `localhost` 指向该 network namespace，不一定是宿主机或另一容器。

[推断] “服务监听 127.0.0.1 但健康检查失败”可能是探针位于不同 namespace；“日志时间倒退”可能是时钟同步、不同时间源或跨进程时钟混用。应记录单调时钟用于时延，墙上时钟仅用于关联外部事件。

## 2.10 perf 与 eBPF：可观测性强项和边界

### 2.10.1 perf 适合回答什么

[事实] `perf stat` 可读取硬件/软件性能计数器，`perf record`/`perf report` 可进行采样剖析，`perf sched` 可研究调度。计数器包括 cycles、instructions、cache-misses、context-switches、page-faults 等，但事件名称和权限依赖 CPU、内核和安全配置。

[机制] 采样剖析以固定频率或事件溢出中断记录调用栈，得到统计意义上的热点，而非每条指令的完整日志。采样频率越高，开销和扰动越大；用户态符号、JIT、内联和容器路径可能需要额外符号信息才能正确解码。

[测量] 运行 `perf stat -r 5 -d -- <command>` 至少重复多次，报告均值、离散范围、CPU 亲和性和热身方式。`perf stat` 的“缓存未命中率”只是计数器比值，不能直接说明哪一行代码导致缺失。

### 2.10.2 eBPF 适合回答什么

[事实] eBPF 程序可挂载在内核 tracepoint、kprobe、uprobes、perf event、cgroup 和网络钩子等位置，由验证器检查后在内核中执行。工具如 bpftrace、BCC、libbpf 可用于 syscall、调度、IO、网络和内存事件观测。

[机制] eBPF 可以按 PID、cgroup、容器身份和请求 ID 聚合延迟，避免把所有事件复制到用户态；ring buffer、map 和 helper 提供数据交换。它看到的是挂点暴露的事件，不是任意内核内部状态；编译内核、BTF、权限和安全策略会影响可用性。

[边界] eBPF 不能自动证明因果。观察到 `read()` 延迟与 p99 同步上升，只能说明相关；要证明它导致排队，需要控制变量或实验。高频挂钩可能显著增加 CPU 开销，生产环境必须设置采样率和最大 map 大小。

[测量] 使用 bpftrace 前先记录内核版本、BTF 是否可用、`kernel.unprivileged_bpf_disabled`、容器能力和工具版本。用自监控统计丢事件、map 满、ring buffer backlog 和程序运行时间，否则“没有事件”可能只是权限或丢样本。

### 2.10.3 观测边界表

| 工具 | 擅长 | 盲区/风险 |
|---|---|---|
| 应用日志 | 业务阶段、请求 ID、错误上下文 | 记录点之外的等待；同步写日志会扰动时序 |
| `/proc`、cgroup 文件 | 资源快照、累计计数 | 采样间隔内的瞬态；字段语义随内核版本变化 |
| strace | syscall 序列和单次耗时 | 开销大，难覆盖高吞吐，可能改变调度 |
| perf | CPU、缓存、调度统计 | 计数器权限、符号和采样偏差 |
| eBPF | 内核/用户事件聚合、低侵入采样 | 挂点、权限、版本限制；高频程序有开销 |
| core dump/调试器 | 崩溃现场、栈和变量 | 只代表某一时刻；敏感数据泄漏和暂停成本 |

[设计判断] 至少用两类独立证据交叉验证关键结论。例如“CPU 配额导致 p99”需要应用延迟、`cpu.stat` throttled_usec 和调度/运行队列证据，而不是单一 dashboard。

## 2.11 AI worker 的系统结构与故障诊断

### 2.11.1 一个典型 worker

[机制] 典型在线 worker 包含：监听 socket 的主线程；接收和解析线程；请求队列；分词/预处理线程；模型运行时线程池；设备驱动提交线程；后处理与响应线程；日志/指标导出线程。每层都可能创建 fd、分配内存、访问 page cache 或发起 syscall。

[设计判断] 为每个请求携带 request_id、模型版本、输入/输出 token、队列进入/离开时间、计算开始/结束时间和错误阶段。不要把设备事件时间、CPU wall-clock 和客户端时间混为一个字段；时钟源和同步方法必须写清。

### 2.11.2 故障树

1. **请求无响应**：检查监听 fd、accept 队列、事件循环、线程死锁、cgroup throttling、网络 namespace 和上游超时。
2. **首请求特别慢**：检查模型文件 page cache、mmap 缺页、JIT/编译缓存、设备上下文初始化和冷连接。
3. **运行一段时间后 OOM**：检查 RSS/PSS、page cache、碎片、fd/线程泄漏、请求队列和 cgroup `memory.events`。
4. **p99 周期性尖峰**：检查 CPU 频率、配额周期、writeback、GC、批处理等待、长请求和邻居争用。
5. **滚动发布卡住**：检查 SIGTERM 处理、PID 1、子进程回收、监听 fd 继承、readiness 状态和终止宽限期。
6. **吞吐下降但 CPU 很低**：检查 IO 睡眠、锁/futex、设备同步、page fault、网络等待和错误重试。

[测量] 每个分支都应有“最小证据包”：应用分段延迟、线程栈、`/proc` 与 cgroup 快照、syscall/调度摘要、请求形状、版本和时间窗口。证据包不足时，只能写候选机制，不能把故障归因到 GPU 或 Linux 某个子系统。

### 2.11.3 三个失败案例

#### 案例 A：CPU 利用率不高但请求排队

[事实] 某容器 `cpu.max=20000 100000`，四线程 worker 在高峰期 p99 从 300 ms 升到 3 s；宿主机总体 CPU 仍有空闲。

[测量] `cpu.stat` 显示 `nr_throttled` 和 `throttled_usec` 与 p99 同步增长；线程栈大多处于可运行状态而非 IO 睡眠。

[机制] cgroup 配额按组总量限制，四线程同时可运行只会更快消耗 20 ms 配额，随后整个组被节流到下一个周期。宿主机空闲不等于该组可用。

[推断] 主要放大因素是配额过低与批处理等待叠加；需要用提高配额、减少线程、调整批窗口的 A/B 实验分离各因素。

[设计判断] 以 SLO 反推配额并保留突发余量，同时监控 throttling，而不是只设置一个平均 CPU 百分比。

#### 案例 B：模型热身后仍偶发 major fault

[事实] worker 启动时 mmap 12 GB 权重，首轮请求后平均延迟下降，但每隔几分钟仍出现数秒尖峰。

[测量] 进程 major fault 在尖峰时增加，节点 `Active(file)` 接近上限，另一批处理任务同时读取大文件，设备队列深度升高。

[机制] page cache 受全局内存压力驱逐；mmap 页再次访问需要从存储层填充。热身只预取了常用路径，不能保证所有层和所有请求形状常驻。

[推断] 邻居批处理 IO 是候选放大因素；需在隔离 IO cgroup、固定请求集和关闭批任务的对照中验证。

[设计判断] 对关键权重采用本地高速盘/预加载和 IO 隔离，并把冷缓存恢复时间写入容量与发布预算。

#### 案例 C：发布后 worker 不退出

[事实] 编排器发送 SIGTERM，容器 30 秒后被 SIGKILL；日志显示 shell 脚本退出但模型子进程仍在。

[测量] 容器 PID 1 是 `/bin/sh -c ...`，模型进程位于同一 PID namespace 但没有收到 SIGTERM；监听 socket 由子进程继承，readiness 一直未撤销。

[机制] shell 未执行 `exec`，信号和子进程回收责任不符合预期；旧进程继续占用 fd，优雅期限耗尽后被强杀。

[推断] 连接泄漏和突然 SIGKILL 可能造成重试风暴；需检查客户端重试、socket reset、未完成请求和错误预算。

[设计判断] 使用明确的 init/exec 方案，先撤 readiness 再发终止，记录每阶段 deadline，并在发布演练中验证真正的子进程退出。

## 2.12 CPU 实验：从 syscall 到 cgroup 的可复现实验

本实验不依赖 GPU，目标是观察同一程序在不同 IO、内存和 CPU 限额下的变化。所有数值由读者运行后填写，本节不预先声称结果。

### 2.12.1 环境记录

[测量] 记录发行版和内核 (`uname -a`)、CPU 型号与核数 (`lscpu`)、内存 (`free -h`)、挂载和文件系统 (`findmnt`)、perf 版本、容器运行时（若使用）以及 governor/频率策略。关闭或记录后台下载、编译和其他高 IO 任务。

### 2.12.2 实验程序

```c
// io_mem.c：刻意制造文件读取、匿名内存和线程工作
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static volatile uint64_t sink;
void *spin(void *arg) {
    uint64_t n = (uint64_t)(uintptr_t)arg;
    for (uint64_t i = 0; i < n; i++) sink = sink * 1664525u + 1013904223u;
    return NULL;
}
int main(int argc, char **argv) {
    size_t mb = argc > 1 ? strtoull(argv[1], 0, 10) : 512;
    int fd = open("model.bin", O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    char *buf = aligned_alloc(4096, 4096);
    for (size_t i = 0; i < mb * 256; i++) {
        if (read(fd, buf, 4096) != 4096) break;
    }
    free(buf); close(fd);
    pthread_t t[4];
    for (int i = 0; i < 4; i++) pthread_create(&t[i], 0, spin, (void *)(uintptr_t)500000000);
    for (int i = 0; i < 4; i++) pthread_join(t[i], 0);
    printf("sink=%llu\n", (unsigned long long)sink);
}
```

[事实] 该程序只是实验装置，不是安全的生产 worker；它没有输入校验、优雅关闭、错误恢复或并发控制。为生成测试文件可使用 `fallocate -l 2G model.bin`，应确认磁盘有足够空间。

### 2.12.3 基线与观测命令

```bash
cc -O2 -pthread io_mem.c -o io_mem
/usr/bin/time -v ./io_mem 512
perf stat -r 5 -d ./io_mem 512
strace -f -c ./io_mem 64
pidstat -druwt -p $(pgrep -n io_mem) 1
```

[测量] 分别记录冷 page cache、热 page cache、线程数 1/4、文件大小 64/512/2048 MB。每种配置至少重复五次，报告中位数和范围；冷缓存实验需要管理员权限且会影响同机任务，若无法安全执行，改用不同文件或重启后的受控环境并注明限制。

### 2.12.4 cgroup 变量

在 cgroup v2 环境中，可创建实验组（需权限）：

```bash
mkdir -p /sys/fs/cgroup/ai-lab
printf '20000 100000' > /sys/fs/cgroup/ai-lab/cpu.max
printf '2G' > /sys/fs/cgroup/ai-lab/memory.max
printf '4' > /sys/fs/cgroup/ai-lab/pids.max
echo $$ > /sys/fs/cgroup/ai-lab/cgroup.procs
./io_mem 512
cat /sys/fs/cgroup/ai-lab/cpu.stat
cat /sys/fs/cgroup/ai-lab/memory.events
```

[事实] 写入 cgroup 控制文件需要相应权限；不要在生产 cgroup 上执行。某些系统使用 hybrid v1/v2 或只读挂载，命令会失败。

[机制] 比较无配额与 20% CPU 配额时，预期会看到运行时间、调度等待或 throttled_usec 改变；比较 2 GB 内存上限与更大上限时，预期会看到回收、major fault、分配失败或 OOM 风险变化。实际结果取决于文件系统、内存容量和后台负载。

### 2.12.5 结果解释模板

- **事实**：命令输出的计数值、内核版本和实验条件。
- **机制**：指出哪些计数器与 page cache、fault、调度或 cgroup 语义相符。
- **测量限制**：预热是否充分、perf 事件是否可用、系统是否有邻居负载、采样是否丢失。
- **推断**：列出最可能机制和至少一个替代解释。
- **设计判断**：若这是 worker，选择线程数、CPU 配额、预加载策略或日志级别的理由和代价。

[设计判断] 实验的成功标准不是“得到漂亮曲线”，而是当结果反常时能够指出下一条证据路径。例如 CPU 配额降低却运行时间未变，可能是程序大部分时间在 page cache 命中或实验输入太小；应测量 `cpu.stat` 和 IO，而不是删掉反常样本。

## 2.13 六道理解检查（含答案）

### 检查 1：进程 RSS 只有 4 GB，为什么 cgroup memory.current 可能达到 6 GB？

答案要点：[事实] RSS 是某个进程的驻留页，而 cgroup 记账可能包含多个进程、page cache、内核记账对象和共享页；[机制] 模型文件 page cache、日志缓存、线程栈和 sidecar 会计入组内存；[测量] 对照 `smaps_rollup`、`memory.stat`、`memory.current` 和组内所有 PID；[推断] 只有在字段时间对齐后，才能判断是哪类对象增长。

### 检查 2：`mmap()` 返回很快，为什么首次推理仍可能很慢？

答案要点：[事实] mmap 建立的是虚拟映射，不保证所有页面已驻留；[机制] 首次访问触发缺页，可能从 page cache 或存储层填充并更新页表；[测量] 比较前后 major/minor fault、设备读带宽和冷热缓存；[设计判断] 是否预读取决于首请求 SLO、启动内存和 IO 峰值。

### 检查 3：容器内 PID 1 和宿主机看到的 PID 为什么不同？

答案要点：[事实] PID namespace 提供不同的 PID 视图；[机制] 同一 task 在嵌套 namespace 有多个 PID，容器内 PID 1 是该视图的 init；[测量] 比较 `/proc/<pid>/status`、`lsns` 和宿主/容器 `ps`；[推断] 发信号前必须确认目标是哪个 namespace 的 PID。

### 检查 4：eBPF 观察到 `read()` 延迟高，能否直接宣布磁盘是根因？

答案要点：不能。[事实] eBPF 只说明挂点处事件和耗时；[机制] read 延迟也可能来自 page fault、文件系统锁、网络文件系统或 cgroup IO 限制；[测量] 需要设备队列、major fault、IO cgroup、调用栈和对照实验；[设计判断] 生产挂钩要控制采样率并监控丢事件。

### 检查 5：为什么 worker 收到 SIGTERM 后仍可能被 SIGKILL？

答案要点：[机制] 优雅退出可能卡在不可中断 IO、锁、子进程或未完成请求；编排器到达宽限期后发送 SIGKILL；[测量] 记录信号时间、阶段 deadline、线程栈、PID 1 和子进程；[设计判断] 先撤 readiness、限制请求 deadline、确保 exec/信号转发和回收路径。

### 检查 6：线程数增加但吞吐下降，列出至少三个机制。

答案要点：[机制] 锁/futex 竞争、cgroup CPU 配额被更快耗尽、上下文切换和缓存失效、NUMA 远端访问、运行时内部过度订阅、page cache/IO 争用都可能发生；[测量] 看每线程 CPU、context-switches、cpu.stat、perf cache-misses、调度延迟和请求分桶；[推断] 必须用逐步增加线程的实验找拐点，而不能依据线程数本身下结论。

## 2.14 练习：从内核现象到 AI 决策

1. **进程图**：启动一个主进程、预处理线程池和模型子进程，画出 PID/TID、fd 和 cgroup 关系，标记谁负责回收谁。
2. **syscall 预算**：用 strace 统计每请求 `read/write/futex/epoll_wait` 次数，设计一次批量化改动，并预测它对 p99 和可观测性的影响。
3. **虚拟内存实验**：比较匿名数组首次写入、只读 mmap 顺序访问和随机访问的 minor/major fault，解释页表和局部性。
4. **page cache 对照**：冷、热缓存分别加载同一模型文件，记录首请求延迟、PSS、设备读带宽，写出至少两个外推边界。
5. **fd 泄漏故障**：编写一个异常路径不关闭 socket 的小程序，观察 `/proc/<pid>/fd`、`EMFILE` 和恢复方法；说明为什么重启不是根治。
6. **信号演练**：用 shell 包装脚本和 `exec` 两种方式运行 worker，比较 SIGTERM 转发、子进程退出和宽限期表现。
7. **cgroup 容量**：给 worker 设置不同 `cpu.max`、`memory.high` 和 `pids.max`，建立请求 p95、throttled_usec、memory.events 与线程数的关系图。
8. **namespace 诊断**：在 network namespace 中监听端口，分别从宿主和容器执行 `ss -lntp`，解释为何 `localhost` 结论不同。
9. **perf/eBPF 边界**：用 perf 采样一次 CPU 热点，再用 eBPF 统计 syscall 延迟；列出两者都无法回答的问题，并设计第三个证据。
10. **AI worker 事故报告**：任选本章三个失败案例，按事实、机制、测量缺口、推断、短期缓解、长期设计写一页复盘。禁止使用“系统抖了”作为根因。

## 2.15 来源地图：主张如何追溯

### 事实来源

- `man 2 fork`, `execve`, `waitpid`, `read`, `mmap`, `epoll_wait`, `signal`, `clone`：https://man7.org/linux/man-pages/ 。核对返回值、错误码、阻塞和信号语义，并记录 man-pages 版本。
- Linux 内核文档的进程、内存管理和 cgroup v2 章节：https://docs.kernel.org/ 。重点核对 `memory.max/high/events`、`cpu.max`、页回收和 namespace 语义。
- `proc(5)`、`proc_pid_status(5)`、`proc_pid_smaps(5)`：https://man7.org/linux/man-pages/man5/proc.5.html 。核对 RSS、fault、线程和信号字段定义。

### 机制来源

- Linux VFS、page cache、writeback、mmap、COW 的内核源码与文档：从 https://git.kernel.org/ 选择与目标内核匹配的 tag；不要把某版本实现细节推广到所有版本。
- cgroup v2 admin guide：https://docs.kernel.org/admin-guide/cgroup-v2.html 。核对层级、控制文件、记账与限制的交互。
- OCI runtime specification：https://github.com/opencontainers/runtime-spec 。核对容器配置字段和 namespace/cgroup 组合；运行时实现仍可能有差异。

### 测量来源

- perf 文档与 `perf help`：https://perf.wiki.kernel.org/ 。核对事件名、采样和权限；不同 CPU 的硬件计数器不可直接横比。
- eBPF 文档与 bpftrace reference：https://ebpf.io/ 、https://bpftrace.org/ 。核对 verifier、挂点、BTF、map 和工具版本。
- Brendan Gregg 的性能方法资料（用于方法参考，不替代内核文档）：https://www.brendangregg.com/linuxperf.html 。将命令转化为本机可复现实验，并记录扰动。

### 推断与设计判断

- 任何“某工具证明了根因”的句子都应改写为候选机制，附上可证伪测量。
- 任何线程数、缓存预热、cgroup 配额或容器隔离建议，都要写出目标（延迟、吞吐、成本、恢复）和代价（内存、复杂度、可观测性、风险）。
- 来源若只覆盖某内核、文件系统、CPU 或运行时，必须在报告中标注版本和外推边界。事实可以复用，机制需要版本核对，测量只对其环境负责。

建议维护“主张—证据”表：主张、类型、来源/实验编号、内核与运行时版本、采样窗口、反例、负责人和最后复核日期。它能防止事故复盘把一次测量升级成永久真理。

## 2.16 本章小结

Linux 为 AI worker 提供执行、内存、文件、网络和隔离的共同语义。进程是资源视图，线程是调度单元；系统调用是跨越用户态与内核态的窄门；虚拟内存、TLB 和 page fault 决定地址如何变成可用数据；page cache 与 mmap 让权重共享和按需加载成为可能，也引入冷启动和回收风险；fd 把文件、socket、pipe 和事件统一成可等待资源；信号决定停止和故障传播；cgroup 限制“能用多少”，namespace 改变“能看到什么”。

`/proc`、strace、perf 和 eBPF 能把黑盒 worker 变成可观察系统，但每个工具都有盲区和扰动。最可靠的诊断是沿因果链组合独立证据：请求分段、线程状态、syscall、fault、page cache、cgroup、调度和容器视图。对于 AI worker，首要问题不是“哪条命令最酷”，而是“哪个观测能区分候选机制”。

最后，把实验结果分层书写：事实是什么，机制如何发生，测量是否可靠，推断怎样被证伪，设计判断牺牲了什么。掌握这套语言后，面对 p99 飙升、OOM、SIGTERM 卡住、fd 泄漏或容器内外视图不一致，你就能从症状回到内核机制，再回到可执行的容量、隔离和发布决策。
