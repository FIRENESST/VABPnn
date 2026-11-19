# VABPnn compare-1.0.1

适用于 Minecraft Java 26.1 / Fabric Loader 0.19.3。这是 compare-1.0.0 的阈值恢复补丁，保留 7,680 节点双头全连接矩阵、116 个记忆门参数和原游戏流程。Transformer 参数仍为 118,092,470。

## 连续预测与执行

所有模式都先判断每项动作的 `confidence > learned threshold`，只把通过的动作交给 top-k，再执行互斥规则。F8 仍连续预测并消费新帧，神经权重与优化器冻结；仅预测模式使用相同筛选而不注入操作。动态关联记忆、高贡献点和连接继续更新。
当没有对象通过阈值时，即使 k 大于零也返回有效空动作包；模型持有的普通键会按现有执行器规则释放，鼠标位移和滚轮不会越过筛选。`/vabpnn status` 会显示候选动作、k，以及“没有对象超过学习阈值”的状态。

## 使用与存档

重启 Minecraft 后 `/vabpnn start` 开始示范，F8 冻结模型控制；`/vabpnn predict on` 仅预测，F9 暂停，F10 保存。
本版只写 `vabpnn/models/compare-1.0.1.bin`，v10 格式保持不变。首次启动优先只读继承同目录 `compare-1.0.0.bin`；没有时再只读迁移 beta。旧模型和旧包保留。此补丁本身不启动 Scorer 自训练。

## 验证

`nativeTest glCaptureSmokeTest runClientGameTest build` 已通过：完整 CUDA 数值/记忆门回归；正 k、部分/空候选、鼠标和滚轮筛选、严格阈值与冻结验证；真实 beta 迁移、v10 存档恢复；Minecraft 连续控制、仅预测、15 项操作、Ctrl/Shift、背包示范和启动/停止。
JAR/DLL、保留源码和安装核验见 `Mod/build/threshold-deployment.json`。对照版沿用无 10 ms 验收门槛。

完整矩阵记忆机制见 [compare-1.0.0 说明](../compare-1.0.0/README.md)。
