# VABPnn compare-1.0.0

独立正式对照版，适用于 Minecraft Java 26.1 / Fabric Loader 0.19.3。从 compare-1.0.0-beta 继承完整 7,680 节点、双头全连接图，在每个有向节点对的关联矩阵上新增动态记忆。原正式 1.4.1 与 beta 源码和历史包保留。

## 矩阵记忆

每头仍有完整的 7,680×7,680 FP32 图偏置 B，所有有向节点对（包含自身）参与注意力。另有同尺寸动态记忆 M；它是推理状态，不计为可学习参数。Q/K 每头 8 维；以下逐节点对计算：

```text
x = Q_i · K_j / √8 + B_hij
φ = [Q_i(8), K_j(8), M_previous, log(1+Δt_seconds), 1]
ρ = softplus(W_forget · φ)
f = exp(-ρ · Δt_seconds)
u = sigmoid(W_write · φ)
r = sigmoid(W_read · φ)
M_new = f · M_previous + (1-f) · u · tanh(x)
S = x + gain_h · r · M_new
Attention = softmax_row(S)
```

遗忘、写入、读取三个门均可学习；门系数按头共享，逐节点对的偏置与记忆依然完整、独立。Δt 来自实际采集时间戳，单位为秒，初始遗忘速率 0.1/s。初始零记忆与首帧 Δt=0 保持 beta 的首帧注意力。记忆在 [-1,1] 内，不随帧数无限累积。

新增参数为 2×3×19+2 = **116**；原块级 Transformer 118,092,354，现为 **118,092,470**（约 1.181 亿）。其中逐节点对偏置仍为 117,964,800；保留的 PNN、JEPA、动作头、分钟支路等另计。

人工示范训练使用真实 CUDA 反向传播与 AdamW，梯度覆盖新门、读增益、逐节点对偏置，以及门对 Q/K 的直接依赖。采用**单步截断递归梯度**：缓存的上一帧记忆作为 stop-gradient 边界，当前记忆更新可导；没有跨整个游玩历史保存计算图或实施无限 BPTT。新门有独立 AdamW 步数，继承 beta 的旧优化器步数不会错误校正新参数。

## 原流程与冻结

1920×1024 像素 PNN、16×16 块（120×64）、动态高贡献点与 CSR、模因、秒/分钟轴、30 Hz XYZ、JEPA 与贝叶斯鼠标、15 项操作及 Ctrl/Shift 沿用 beta。

- `/vabpnn start` 开始人工示范训练。
- F8 连续模型操作：冻结所有神经权重、门系数、AdamW 与 EMA；关联矩阵记忆、高贡献点、连接与模因继续按预测更新。
- `/vabpnn start predict` 或 `/vabpnn predict on` 仅预测，权重冻结且不注入键鼠，关联记忆仍更新。
- F6/F7 监控，F9 暂停，F10 或 `/vabpnn save` 保存。
- `/vabpnn status` 增加关联记忆帧数、门训练步数、|M| 均值与峰值。JNI 只传少量标量，不回读完整矩阵或游戏帧。

记忆是临时状态，沿用现有 episode、模式切换、场景与时间回退的重置边界；切换后新帧继续积累。普通预测不会因冻结而清空记忆；同一帧重复提交不推进两次。显存队列仍有界并丢旧帧。

## 存档与版本

本版只写 `vabpnn/models/compare-1.0.0.bin`，v10 / VABPNNF / 23-part，完整大小 **1,939,219,348 字节**。保存所有神经权重、AdamW 矩与两个训练计数；临时记忆 M 不写入存档，重启后从零开始。

没有本版存档时，可从同目录的 `compare-1.0.0-beta.bin` **只读迁移** v9：先检查整文件布局、大小与 SHA256，再继承所有旧参数、AdamW 矩、旧计数及持久状态，116 个新参数按默认值初始化，新矩与新计数为零。原 beta 文件不覆盖、不重命名。正式 1.4.1 与历史正式模型不作为迁移源。

新包 `Mod/build/libs/vabpnn-compare-1.0.0.jar`；元数据 version=1.0.0，Mod ID=vabpnn。只启用一个 VABPnn 包；安装目录沿用 `D:/Minecraft/JavaEdition/BlockHelm/.minecraft/versions/26.1-fabric-0.19.3/mods`。切换后重启游戏。

```powershell
.\switch-mod.ps1 -Variant compare
.\switch-mod.ps1 -Variant beta
.\switch-mod.ps1 -Variant formal
```

## 构建与验证

用 IDEA 打开此工程的 Mod，VS 算子仍在本工程 CUDAkernel/SelfCudaKernel/SelfCudaKernel；核心为 DenseBlockTransformer.h/.cu。依赖仍为现有 CUDA/JNI，没有新增 cuBLAS 或 Python 运行时。

```powershell
cd Mod
.\gradlew.bat --offline nativeTest nativeBenchmark
.\gradlew.bat --offline scopeUiSmokeTest memeScopeUiSmokeTest glCaptureSmokeTest runClientGameTest build
# 可通过 -PcompatibilityCheckpoint=<beta.bin的绝对路径> 补充实际 beta 迁移测试
```

不设 10 ms 的测试门槛，也不以降节点、稀疏化、低秩化、停训练来达成帧预算。现有 over10ms 仅为计时统计。

已验证的 CUDA 数值：原完整注意力输出最大误差 3.57×10⁻⁷；4,886 项含新门的首帧梯度最大误差 3.61×10⁻⁶；660 项暖记忆/门/Q/K/逐节点对偏置有限差分均有非零梯度，最大误差 5.15×10⁻⁹。真实贝叶斯/阈值/learned-k 反向以及秒、分钟频率验证通过。

全尺寸两个头各 58,982,400 项偏置获得非零梯度，AdamW 与冻结下记忆推进验证通过。块 Transformer GPU 工作空间 4,749,114,320 字节；完整管线约 5.4 GB，另需图像池与游戏显存。本机 RTX 5070 Laptop 独立前向 16 次平均 21.3120 ms、最高 21.4530 ms；反向加 AdamW 50.5623 ms，仅为本机观测。

实际集成验证全部通过：普通与 4x MSAA 的 GL/CUDA 互操作、真实 beta 模型只读迁移与 v10 保存恢复、15 个神经/优化器持久部分冻结、两个普通 Windows 监控窗口，以及 Minecraft 26.1 的全部 15 项操作、背包示范、Ctrl/Shift、F8 连续控制且动态记忆、仅预测不注入、30 Hz XYZ、启动/停止/恢复。完整模型显存为 5,377,260,539 字节；Minecraft 回归观测的算子峰值 93.321 ms，不作为帧预算失败。

集成验证、源码/JAR 核验与安装记录分别保存在 Mod/build/memory-integration-summary.json、memory-artifact-proof.json 与 memory-deployment.json。
