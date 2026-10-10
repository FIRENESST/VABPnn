# VABPnn 工程检查与优化（0.9.5）

日期：2026-10-10。基线为 0.9.4，优化版为 0.9.5。本次检查 Java 输入/面板、JNI 显存池与 worker、CUDA 投票/SNN/时间决策/JEPA/在线学习和构建脚本。

## 实施的优化

- [OnlineLearning.cu](C:/Users/43828/Desktop/MyProject/VABPnn/CUDAkernel/SelfCudaKernel/SelfCudaKernel/OnlineLearning.cu:77)：输出头的 1087 个参数分配给一个 256 线程块，每个参数由一个线程更新。梯度和旧滚轮预测先计算，再经过块同步进行写入；AdamW 校正项每次训练仅计算一次。保持梯度裁剪、衰减、学习率、损失累加顺序和参数布局。
- [SpikeConnections.cu](C:/Users/43828/Desktop/MyProject/VABPnn/CUDAkernel/SelfCudaKernel/SelfCudaKernel/SpikeConnections.cu:61)：块内 CSR 整数行偏移改用 [CUB BlockScan](https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1BlockScan.html)，减少原先手写扫描的块同步。CSR 入边集合和属性仍由原出边转置产生；列内顺序仍由原有原子散射决定。
- [capture_bridge.cu](C:/Users/43828/Desktop/MyProject/VABPnn/Mod/native/capture_bridge.cu:244)：未到自动保存时间且无手动请求时，跳过重复路径复制和元数据锁。首次恢复、手动/自动保存、保存确认和关闭保存流程保留。
- [NetworkMonitor.java](C:/Users/43828/Desktop/MyProject/VABPnn/Mod/src/client/java/cn/vabpnn/mod/client/scope/NetworkMonitor.java:80)：相同面板状态复用已有不可变对象，减少分配和原子发布。绘制定时器和状态变化的反馈保持原样。
- [CaptureController.java](C:/Users/43828/Desktop/MyProject/VABPnn/Mod/src/client/java/cn/vabpnn/mod/client/capture/CaptureController.java:299)：删除 tick 内紧邻生产交付方法的一次重复场景校验；交付方法仍先校验场景并处理输入释放。
- [build-native.ps1](C:/Users/43828/Desktop/MyProject/VABPnn/Mod/native/build-native.ps1)：CUDA 翻译单元独立缓存，生产架构与 sm_120 测试对象分开。数值测试和基准复用同一批测试对象。缓存记录源码、项目头文件、JNI/CUDA 版本头、nvcc/cl 二进制、工具链版本、构建参数与对象 SHA-256。使用 .NET 哈希兼容 Gradle 启动的系统 PowerShell。
- [build.gradle](C:/Users/43828/Desktop/MyProject/VABPnn/Mod/build.gradle:91)：单独改测试 .cu 不触发生产 DLL 构建。生产桥接、头文件、算子和脚本仍参与 Gradle 输入检查。

原实现逐文件保留在 Mod/build/optimization-backup；没有批量删除代码或模型。

## 性能对比

RTX 5070 Laptop GPU；Release/O3；1920×1024；每档 64 帧预热、512 帧测量；优化前后各运行两轮，表中平均值为两轮均值。强制 256 个高贡献节点，长程边超过 80,000；每六帧包含一次完整可视化取样。

| 档位 | 优化前算子平均 ms | 优化后算子平均 ms | 平均耗时下降 | 优化后 GPU 复制/算子/回传最慢 ms |
| --- | ---: | ---: | ---: | ---: |
| 推理，JEPA 训练关闭 | 1.921 | 1.918 | 基本持平 | 3.471 |
| 模型控制 + JEPA | 2.108 | 2.095 | 基本持平 | 3.331 |
| 人工示范 + JEPA（默认） | 3.377 | 3.164 | 6.3% | 5.797 |
| 压力档：JEPA 每帧 + 像素学习 + 重连 | 3.859 | 3.646 | 5.5% | 6.365 |

优化后共 4096 个测量帧，GPU 复制/算子/控制回传超过 10 ms 的数量为 0。压力档算子最慢 6.299 ms，包含复制和回传最慢 6.365 ms。推理两档差异在本机波动范围内，不宣称显著提升。

模型与训练缓存依然为 535,989,595 字节（包含 GPU 可视化摘要），没有增加全局显存缓冲。上述计时不包含 Minecraft 渲染、GL blit、GL/CUDA map/unmap、初始化、文件保存和系统调度，不能作为全游戏每帧 10 ms 的保证。

强制运行 Gradle 原生任务的无改动验收：2.96 秒，compiled=0/reused=7，跳过 DLL 链接，DLL SHA 与修改时间均不变。改动一个 .cu 的增量验收为 compiled=1/reused=6。首次完整构建/数值验证的时间含测试运行，未与无改动构建作速度倍数比较。

## 行为与兼容验证

- 七组 CUDA 测试通过：全尺寸/奇数尺寸/保护区、76,677,120 个三值参数与原始票数、时间图/马尔科夫/贝叶斯/阈值/k、JEPA 梯度与 AdamW/EMA、在线示范、CSR/LIF/重连/长程边、GPU 面板摘要。
- 在线训练加入独立 CPU 首次 AdamW 参考，逐项核对全部策略参数和冻结主干；非零滚轮监督核对原 MSE。模型控制仍不制造人工监督标签。
- 真实 GL/JNI 测试覆盖 MSAA=0/4、背压、池退休/尺寸变化、GL 状态恢复、显存复制和训练，无游戏帧读回。
- 面板测试覆盖不可变摘要、过期状态、交互、隐藏/重开、关闭和窗口不抢焦点。
- Minecraft 26.1 临时世界验证通过：默认待命、手动 start、人工示范默认模式、13 路操作、背包左右键按住/松开、F8/F9、状态/HUD、两次 F10 保存、stop/重启与断开连接。背包示范弹出提示仍已删除。
- VS x64 Debug 编译通过，保留原 CUDA 工程与调试配置。
- 只读恢复现有 online-v2.bin 通过：14 个 SHA-256 检查点载荷，DecisionParameters=15,544 字节，已有有效示范和决策步数均为 67,612。验证不保存或重置真实模型。

像素网格与 16×16 块、连接规则、模型结构、参数量、训练频率、概率与动作规则、默认示范模式、热键和命令、显存链路及检查点布局均保持兼容。数学与流程测试通过不代表网络已经学会自主游戏。

## 验收产物

正式模组：[vabpnn-0.9.5.jar](C:/Users/43828/Desktop/MyProject/VABPnn/Mod/build/libs/vabpnn-0.9.5.jar)；SHA-256：C2EBC69A15FD76B3162DD119DB6E68E7FBA5D66AAEFBED323F059A1261694A22。

原生 DLL SHA-256：DAA9644CF76678CB223BF83D2B109049A5DD7FE5622584453D7904B693C3573F。

日志位于 Mod/build/native：

- optimization-before-1.log / optimization-before-2.log
- optimization-after-1.log / optimization-after-2.log
- optimization-final-native-validation.log / optimization-online-reference.log
- optimization-java-validation.log / optimization-minecraft-validation.log
- optimization-cache-validation.log / optimization-vs-debug-validation.log
- optimization-checkpoint-compatibility.log

复现：在 Mod 中执行 gradlew.bat build glCaptureSmokeTest scopeUiSmokeTest runClientGameTest --offline；原生数值验证与基准执行 powershell.exe -NoProfile -ExecutionPolicy Bypass -File native/build-native.ps1 -TestOperator -Benchmark。

部署已完成：正式实例 mods 中启用 vabpnn-0.9.5.jar，0.9.4 保留为 .jar.disabled。部署后的 SHA-256 复核确认已有 online-v1.bin、online-v2.bin 和其他模组全部保持原内容。DLL 的 21 个导出符号也与 0.9.4 完全相同；接口核对日志为 optimization-exports-validation.log。
