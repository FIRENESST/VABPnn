# VABPnn compare-1.0.3

Minecraft Java 26.1 / Fabric Loader 0.19.3 的独立对照版本。保留 compare-1.0.0 的 7,680 节点、双头完整关联矩阵及动态记忆门控，沿用 compare-1.0.1 恢复的置信度阈值。本版在 compare-1.0.2 的 Scorer 奖励接口上新增可调的随机探索，用于模型起步；不改变默认人工示范流程和网络架构。

## 运行

先启动 VABPnn/Scorer/启动评分器.cmd，并重启装有本版及 vabpnn-telemetry-0.5.0 的 Minecraft。默认评分器地址是本机 http://127.0.0.1:9876，初版奖励接入支持单人游戏的集成服务器。

- /vabpnn start：人工示范，边玩边学。
- F8：连续模型控制，冻结所有神经权重、优化器及 JEPA EMA；仍应用学习到的 confidence > threshold，再执行 top-k 和互斥规则。
- /vabpnn start predict 或 /vabpnn predict on：仅预测，权重冻结，不注入键鼠。
- /vabpnn selftrain on：显式进入模型控制及奖励训练。等待遥测和评分器绑定到当前世界、玩家及会话之后才执行动作。
- /vabpnn selftrain off：关闭奖励训练，释放模型持有的键鼠，转为冻结的仅预测模式。
- /vabpnn selftrain：查看连接原因、奖励游标、GPU 更新次数、待配对动作数和丢弃次数。
- /vabpnn explore on：显式进入 Scorer 自训练，并以 20% 概率探索。探索分支允许尝试未过阈值的动作，每次最多选择两个；其余预测仍执行原阈值策略。
- /vabpnn explore on 0.1：将探索比例改为 10%；范围 0..1。
- /vabpnn explore strict 0.2：只在已过阈值的候选中增加随机性。
- /vabpnn explore off：关闭额外探索，继续原来的 Scorer 自训练；F8 或 predict on 同时退出探索和自训练。
- /vabpnn explore：查看当前强度、是否允许未过阈值尝试及累计探索提案。
- F8、仅预测命令或人工动作键/鼠标操作会退出奖励训练；F9 暂停捕获并清除旧回执，恢复后重新绑定。
- F6/F7 监控窗口，F10 或 /vabpnn save 保存，/vabpnn status 查看完整状态。

正常 F8 控制、仅预测和未开启探索的奖励训练都保留置信度阈值。只有显式开启的探索分支可以暂时扩展候选集；不会修改学习到的阈值。没有候选对象或 k=0 时仍发布新的合法休息指令，不沿用上一帧的移动量。关联记忆、高贡献点和连接仍按原流程动态更新；冻结的是可学习权重。

## 奖励学习范围

当前奖励接口只更新已有的 1,022 个动作策略参数：15 项动作的视觉证据权重/偏置/原始票权重，以及 k 头的权重和偏置。没有增加网络架构或额外可学习参数。PNN、完整图 Transformer、记忆门参数、阈值参数、位置/分钟支路与 JEPA 在奖励控制时冻结；回到人工示范模式仍使用原来的训练流程。

奖励训练采用 on-policy REINFORCE。阈值决定候选集；k 按现有输出分布在可用候选数内重新归一化采样，动作按贝叶斯后验进行不放回采样并排除互斥项。GPU 缓存实际采样顺序的联合 log 概率与精确 score 梯度，包括 k 的概率/软门特征对证据头的链式梯度。

探索以固定 ε 选择分支。探索 k 分布为 0.5×原 softmax + 0.5×均匀分布，动作不放回采样权重为 0.5×后验概率 + 0.5，避免几乎零概率的动作永远没有机会；仍检查既定互斥关系。起步探索临时使用已有 k=0、1、2 头，保留可学习支持元数据。缓存梯度对应真正采样的分支/k/顺序混合分布，不能以普通 top-k 的梯度代替随机动作的梯度。鼠标和滚轮位移继续由原算子提供，未训练的位移头不会被随机大位移替代。设置改变会释放模型持有输入、清除旧梯度/执行缓存并重新绑定当前 Scorer 会话；没有增加新的网络参数。奖励是对应已执行窗口内 score 的平均值乘以归一化奖励；不把奖励伪装成人工标签。

奖励除以 Scorer 的 reward_clip 并限制到 [-1,1]，每个成功匹配的窗口聚合更新一次。使用独立 Adam 状态，学习率 0.0001，β₁=0.9、β₂=0.999，逐参数梯度限制在 [-1,1]，不施加权重衰减。零奖励不更新权重、矩、步数或策略版本。

这只证明奖励链路和梯度实现正确，不保证模型已具备高质量自主行为。奖励范围之外的模块不会被悄悄标记为“已奖励训练”；vabpnn.selftrain.modules 只接受 policy_heads。

## 执行证据与边界

只有 ActionExecutor 确实成功应用的新 GPU 指令才生成执行回执。回执校验帧号、采集时间、实际掩码、控制 epoch、策略版本及指令租期，采用遥测 SDK 发布的集成服务器 tick，拒绝伪造客户端时间替代服务器时间。

奖励匹配同一 world_id/player_id/session_id/dimension 和 Scorer stream/config_version，窗口为 (start_tick,tick]。没有执行回执、重复/过期/越界奖励、缓存溢出、断流、配置变化或旧策略帧不能更新权重。每次策略更新清除旧策略执行记录，不用后来图像替代原来的采样状态。

真实死亡停止新动作，最多保留三秒用于结算已执行的最后死亡窗口；只处理死亡 terminal，不在此阶段用普通窗口推进策略。收到 GPU 完成回执后退出奖励训练并切到冻结的仅预测模式。暂停、人工接管、失焦、场景边界立即清除训练回执；无法精确配对的死亡窗口会跳过。快速复活若已错过死亡结算状态，冻结并清理，避免挂在旧 episode。

CPU 仅接收少量动作与训练元数据。图像、逐像素神经状态、完整矩阵和策略 score 都保留在 GPU。缓存固定 512 帧，奖励请求队列最多 8 条；新增显存约 2.2 MB。HTTP 为本机异步连接，限制响应 2 MB，完整请求/响应截止时间不超过 1.5 秒，不在渲染线程等待网络。

## 保留的视觉流程

游戏呈现画面 → 独立 FBO 纹理 → OpenGL/CUDA 互操作 → 有界显存队列 → 1920×1024 PNN → 16×16 块（120×64，即 7,680 节点） → 完整双头图 Transformer/动态矩阵记忆 → 秒/分钟时间轴与位置 → 贝叶斯概率/阈值/k → JEPA 贝叶斯鼠标位移 → Minecraft 输入。

图 Transformer 仍为 118,092,470 个可学习参数，含两个完整 7,680×7,680 有向关联偏置矩阵及 116 个共享记忆门参数。人工示范保持原来的真实 CUDA 反向传播与 AdamW，记忆梯度使用单步截断递归。动态记忆不是权重，推理时继续递进，episode/模式/时间回退沿用原重置规则。

本对照版按原要求不设每帧 10 ms 验收门槛；没有缩减全连接图以规避计算量。

## 检查点和历史

只写 vabpnn/models/compare-1.0.3.bin，v11 / VABPNNG / 24 部分，完整大小 1,939,227,548 字节。前 23 部分维持 v10 布局，新增 8,200 字节保存独立奖励 Adam 状态、策略版本和计数。临时矩阵记忆、采样缓存和 HTTP 回执不写入模型文件，重启不会自动启用奖励训练。

没有本版检查点时先只读继承 compare-1.0.2.bin（v11，保留奖励优化器），再继承 compare-1.0.1.bin、compare-1.0.0.bin（v10），最后兼容 compare-1.0.0-beta.bin（v9）。先校验整文件大小、布局和 SHA256，再加载；原模型不覆盖，v10/v9 的奖励优化器从零开始。探索强度和临时采样状态不写进模型，重新进游戏默认关闭。正式 1.4.1 不作为迁移源。历史源码、JAR 和用户模型保留。

工程在 Compare/compare-1.0.3/Mod 与同级 CUDAkernel。仅启用一个 Mod ID=vabpnn 的模型包，遥测的 ID 独立，可共存。安装目标：
D:/Minecraft/JavaEdition/BlockHelm/.minecraft/versions/26.1-fabric-0.19.3/mods/vabpnn-compare-1.0.3.jar

## 验证

JDK 26、CUDA 13.4、Windows x64、NVIDIA CUDA/OpenGL GPU；没有增加 Python 或深度学习运行时依赖。IDEA 打开 Mod，VS 使用同级 SelfCudaKernel 工程。

    .\gradlew.bat --offline nativeTest
    .\gradlew.bat --offline glCaptureSmokeTest
    .\gradlew.bat --offline runClientGameTest
    .\gradlew.bat --offline build

nativeTest 检查原视觉/JEPA/全图/门控，以及普通/探索混合分布全部 1,022 个策略 score 的有限差分、冷启动动作多样性、20% 分支频率、最多两项/互斥/阈值保护、正负奖励方向、零/重复/冻结保护、连续非阻塞流收据生命周期和阈值约束。

glCaptureSmokeTest 检查异步 HTTP、纯奖励 ledger、真实 GL/JNI/CUDA 动作与执行回执、奖励优化器保存恢复；通过完整检查点 SHA 验证非策略神经部分精确不变。真实 Minecraft 测试使用独立临时世界、独立 Scorer 数据和模型，验证正/负/死亡奖励及 F8/仅预测/人工接管/暂停保护。测试夹具只用于联调，不安装到用户模型目录。

build/exploration-game-validation.json 指向组合验证日志，build/exploration-artifacts.json 记录包内容核对，build/exploration-deployment.json 记录安装哈希及历史保留核对。
