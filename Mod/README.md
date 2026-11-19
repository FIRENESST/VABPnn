# VABPnn Fabric 2.0.0-alpha

适用于 Minecraft Java **26.1**、Fabric Loader **0.19.3** 和 Fabric API **0.155.2+26.1.2**。运行环境为 Windows x64 和 NVIDIA GPU。Minecraft 的 OpenGL 与 CUDA 需要使用同一块 NVIDIA 显卡。

## 构建与安装

用 IDEA 打开 `Mod/`，选择 JDK 26 作为 Gradle JVM。工程的 Java 编译目标为 25。在 `Mod/` 目录运行构建命令。

```powershell
.\gradlew.bat build
```

产物为 `build/libs/vabpnn-2.0.0-alpha.jar`，已包含 CUDA DLL。将这个 JAR 放入目标 Fabric 实例的 `mods/`，同一个实例只加载一个 VABPnn 版本。`-sources.jar` 用于阅读源码。

## 游戏内操作

进入世界后，输入 `/vabpnn start` 初始化 GPU 并开始人工示范学习。模组反馈显示在聊天栏。

| 命令 | 行为 |
| --- | --- |
| `/vabpnn` | 查看可用命令、快捷键和自训练状态 |
| `/vabpnn start` | 从待命进入人工示范。已启动时保留当前模式 |
| `/vabpnn stop` | 回到待命，释放模型控制的输入 |
| `/vabpnn selftrain on` / `off` | 开始 Scorer 奖励自训练或切换回冻结的仅预测。无参数可查状态 |
| `/vabpnn explore on` / `off` | 开关阈值外动作尝试。无参数可查状态。开启时启动奖励自训练 |

- **F7** 显示或隐藏网络和模因传播监控窗口。
- **F8** 切换冻结权重的模型控制。奖励自训练期间按 F8 会先退出奖励学习。
- **F9** 暂停或恢复已启动的流程。

模型控制、监控窗口、Scorer 自训练和阈值外探索默认关闭。奖励训练需要运行 [Scorer](../Scorer/) 并在单人游戏实例加载 `vabpnn-telemetry-0.5.0.jar`。模型和记忆会自动保存。

## Alpha 模型与存档

1920×1024 帧继续由 GPU 上的 PNN 全分辨率处理。每 4×4 个神经元块组成一个 JEPA patch。PNN 的有符号投票和膜电位构成 480 个 patch 的 48 维特征。JEPA 对相邻帧做时序自监督，保留 EMA 目标编码器。当前观测潜变量与上一帧预测合成 15 个空间 token。

动作 Transformer 用 15 个动作查询对最多 8 帧的空间 token 做双头交叉注意力。模型宽度 16，前馈层宽度 32。贝叶斯先验、秒／分钟时间轴、位置与记忆融合、置信度阈值、冲突约束和模型选择的 k 接在解码器输出之后。鼠标与滚轮仍按已有 15 项原子动作协议交给游戏。

人工示范更新解码器和输出头。Scorer 以当前生命实际执行的回执更新策略，通过最近观测的潜变量梯度给 PNN 局部直通估计信用。预测分支及更早的历史在奖励反传中截断，JEPA 编码器与预测器继续用真实相邻帧的自监督目标学习。F8／仅预测冻结权重和长期记忆，立即重生清空旧帧配对和决策历史。

Alpha 模型使用独立的 release-2.0.0-alpha.bin（v16）和 experience-2.0.0-alpha.vdb。首次运行可从 1.4.2／1.4.1／1.4.0 的模型只读继承 PNN 三值权重、连续主权重及块内出边。JEPA、动作头及优化器重新开始学习。

## 独立后台训练

在 `Mod/` 运行对应脚本。只有启动脚本会开启训练。

```powershell
.\start-background-training.ps1
.\background-training-status.ps1
.\stop-background-training.ps1
```

启动脚本使用 `run/background-training/` 内的独立世界和模型副本，以隐藏的 Minecraft 客户端运行，默认 1920×1024、目标 30 FPS。首次运行若需要指定世界，可传入 `-SourceInstance` 和 `-WorldName`。状态、日志、模型和记忆文件都在该运行目录下。

## 目录索引

| 路径 | 用途 |
| --- | --- |
| `src/main/`、`src/client/` | Fabric 主入口、客户端采集、输入执行与界面 |
| `src/backgroundTraining/` | 可选的后台训练入口 |
| `src/glSmokeTest/`、`src/gametest/` | OpenGL/JNI 与 Minecraft 客户端测试 |
| `native/`、`native/build/` | JNI 桥接、原生构建脚本、CUDA 测试与基准 |
| `../CUDAkernel/SelfCudaKernel/` | CUDA 模型实现和 Visual Studio 工程 |
| `gradle/` | Gradle Wrapper 和构建配置 |
| `build/libs/` | 各版本 JAR。近期验证日志和报告位于 `build/` |
| `logs/validation/` | 按主题归档的历史验证日志 |
| `run/` | 开发实例与独立后台训练数据 |

- [1.x 版本记录与历史验证](docs/RELEASE_HISTORY.md)
- [FBO 与显存采集链路](docs/FBO_PIPELINE.md)
- [1.4.2 长期记忆设计](docs/1.4.2-long-term-memory.md)
