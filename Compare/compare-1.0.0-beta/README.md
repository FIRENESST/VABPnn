# VABPnn compare-1.0.0-beta

适用于 Minecraft Java 26.1、Fabric Loader 0.19.3；这是从正式 1.4.1 复制出的独立对照工程。正式源码、JAR 和训练权重保持原样。

## 对照内容

1920×1024 的 PNN 仍逐像素计算，16×16 像素组成一个块，共 120×64 = 7,680 个块。主图 Transformer 改为这 7,680 个块的全连接图；宽度 16、两个注意力头，每头 8 维，前馈宽度 32。每个头为每个有向节点对（包含自身）保存独立的 FP32 图偏置。

注意力严格计算 `softmax(QKᵀ / √8 + B)`，每行包含全部 7,680 个节点；不使用稀疏邻接、低秩偏置、参数共享或 top-k 注意力。图后的 15 个可学习读出查询将块表示汇聚成现有动作头的输入，再接原 CTMC / 贝叶斯、阈值、网络 k、互斥规则和 JEPA / 贝叶斯鼠标。

节点输入为本块最近 8 次原始有符号投票（15 项 / 次）、秒 / 分钟时间编码、12 项坐标特征和网格位置，共 142 维。历史先放进节点的输入投影，再对当前 7,680 个节点做完整注意力；原版对 15 个动作及历史帧逐项做注意力的秒级实现被替换。原分钟摘要分支继续独立参与动作预测。秒和分钟频率也是可学习参数；没有观察到的历史槽填零。

PNN、动态高贡献点、局部 / 长程 CSR、块控制器、模因、JEPA、贝叶斯鼠标和最终游戏操作沿用 1.4.1。主图所有新参数、Q/K/V、FFN、归一化和读出层均有真实反向传播及 AdamW；训练使用上一帧对应的人工示范。k 的后验 / 软门依赖也纳入传给块图的梯度。

## 参数口径

| 部分 | 可学习参数 |
| :--- | ---: |
| 两头逐有向节点对图偏置：2×7,680² | 117,964,800 |
| 节点编码：7,680×16 | 122,880 |
| 输入投影：142×16 | 2,272 |
| Q/K/V 与注意力输出投影 | 1,024 |
| 前馈网络与偏置 | 1,072 |
| 两层归一化 gain / bias | 64 |
| 15 项动作读出查询 | 240 |
| 秒 / 分钟频率 | 2 |
| **块级 Transformer 合计** | **118,092,354（约 1.181 亿）** |

上述为比较版块 Transformer 自身的参数；保留的 PNN 三值参数、JEPA、贝叶斯 / 动作头、分钟分支和其他模块另计，不能把整个混合系统误报成仅 1.18 亿。图偏置梯度、AdamW 矩和注意力概率都是显存中的真实完整数组。

## 使用与存档

新包为 `Mod/build/libs/vabpnn-compare-1.0.0-beta.jar`。Fabric 元数据使用合法语义版本 `1.0.0-beta`，显示名为 VABPnn Compare，工程和发布标识为 compare-1.0.0-beta。Mod ID 保持 vabpnn，命令、Mixin 和热键沿用原流程；正式版与比较版只启用一个 JAR。

进入世界后 `/vabpnn start` 开始人工示范训练；F8 连续模型控制，冻结包括完整图偏置在内的全部神经权重 / AdamW / EMA，动态高贡献点、连接和模因仍推进；`/vabpnn start predict` 或 `predict on` 只预测、冻结神经学习且不注入动作。F6 / F7 为监控窗口，F9 暂停，F10 保存。

比较版只读写 `vabpnn/models/compare-1.0.0-beta.bin`。这是独立 v9 / VABPNNE / 23-part 存档，完整大小 1,939,217,948 字节；在原布局之后另存小型 Transformer / AdamW、完整图偏置、第一矩和第二矩。不会自动读取正式版或历史存档，因此首次启动须重新示范训练；原权重仍保留。不记录游戏帧、图片或视频文件，运行时画面与中间张量都在显存。

对照版不设置 10ms 的通过 / 失败条件，也不因超过 10ms 缩图、减少节点、冻结梯度或关闭训练。队列仍有界并丢弃旧帧，维持原输入释放和时效规则。日志里的 over10ms 是原来的计时统计，比较版测试不会把它视为失败。

部署目录仍为 `D:/Minecraft/JavaEdition/BlockHelm/.minecraft/versions/26.1-fabric-0.19.3/mods`。安装时将正式 1.4.1 改为 `.jar.disabled` 留存，启用比较包。切换后重启游戏；可从本工程根目录运行：

```powershell
.\switch-mod.ps1 -Variant formal
.\switch-mod.ps1 -Variant compare
```

## 构建和验证

VS 源码仍在本对照工程的 `CUDAkernel/SelfCudaKernel/SelfCudaKernel/`；核心新文件为 DenseBlockTransformer.h / .cu。用 IDEA 打开本工程的 Mod，不把它挂进正式项目。

```powershell
cd Mod
.\gradlew.bat --offline nativeTest
.\gradlew.bat --offline nativeBenchmark
.\gradlew.bat --offline scopeUiSmokeTest memeScopeUiSmokeTest glCaptureSmokeTest runClientGameTest build
```

nativeTest 跑完整块图的 CPU 数值 / 反向传播、动作损失与时间频率检查，及保持不变的像素 / 图像布局 / JEPA 验证。复制保留的旧动作级注意力专用测试不作为新结构的数值参考。nativeBenchmark 在真实 7,680 节点上做全量前向、反向和 AdamW，报告耗时但不做帧预算限制。GL 存档测试采用顺序创建工作器，避免在 8 GiB GPU 上同时驻留三个完整模型。

已验证：

- 小图采用完全相同的生产 CUDA 核函数，与独立 CPU 完整注意力相比，输出最大误差 3.57×10⁻⁷；4,770 项梯度最大误差 3.61×10⁻⁶。实际贝叶斯 / 阈值 / learned-k 损失反传最大误差 4.24×10⁻⁶；秒、分钟频率均通过独立有限差分。
- 全尺寸两个头各 58,982,400 项偏置都获得非零梯度，实际 AdamW 更新成功。Transformer 工作空间 2,380,723,288 字节；完整管线模型约 2,869 MiB，包含训练状态和必要缓存。
- 全尺寸独立前向 16 次平均 10.4135ms、最高 11.0655ms；一次完整反向加 AdamW 为 29.7728ms。这是本机 RTX 5070 Laptop 的记录，未作为通过条件。
- 普通 Windows 监控窗口、无 MSAA / 4x MSAA、真实 GL/JNI、显存连续处理、尺寸变化、示范训练、整模型 v9 保存 / 恢复和 15 个神经 / 优化器持久部分逐字节冻结检查通过。
- 实际 Minecraft 26.1 的 15 项操作、Ctrl / Shift、背包左右键示范、F8 连续模型控制、只预测不注入、动态 CSR、30Hz XYZ、F9 与 start / stop / restart 回归通过；算子最高 53.094ms，按对照要求不判超预算失败。

最终包、DLL、源码与正式版未改动核验在 `Mod/build/compare-artifact-proof.json`，安装记录在 `Mod/build/compare-deployment.json`。仅用原生 CUDA / 现有 JNI，不新增 cuBLAS 或 Python 运行时依赖。
