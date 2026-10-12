# VABPnn

基于 CUDA 与 Minecraft Fabric 的视觉、脉冲神经网络和在线示范学习项目，包含 CUDA 算子、JNI 桥接及游戏客户端模组。

## 项目结构

- [CUDAkernel/SelfCudaKernel](CUDAkernel/SelfCudaKernel/)：CUDA 算子、模型和 Visual Studio 工程。
- [Mod](Mod/)：Fabric 模组、JNI 桥接、Gradle 构建脚本及验证程序。
- [模组使用与技术说明](Mod/README.md)：启动命令、热键、模型结构和训练流程。
- [图像采集管线](Mod/docs/FBO_PIPELINE.md)：OpenGL / CUDA 显存交互说明。
- [当前工程结构](Mod/docs/1.4.1-project-structure.md)：模块位置、构建入口和模型兼容关系。
- [历史工程检查记录](ENGINEERING_OPTIMIZATION.md)：0.9.5 的优化与验证记录。

## 构建

当前构建配置面向 Windows x64，需要 NVIDIA GPU、CUDA Toolkit、Visual Studio C++ 工具链和 JDK 26。Minecraft 与 Fabric 依赖版本见 [Mod/gradle.properties](Mod/gradle.properties)。

先按本机安装位置调整 `Mod/gradle.properties` 中的 JDK 路径，以及 `Mod/native/build-native.ps1` 中的 CUDA、JDK 和 Visual Studio 路径。保留 `CUDAkernel` 与 `Mod` 的同级目录关系，然后运行：

```powershell
cd Mod
.\gradlew.bat build
```

生成的模组 JAR 位于 `Mod/build/libs/`。更多构建和验证方法见 [Mod/README.md](Mod/README.md)。

## 使用

将构建出的模组 JAR 放入匹配版本的 Minecraft Fabric 实例的 `mods` 目录。进入世界后执行 `/vabpnn start`，默认进入人工示范训练；其他命令和快捷键见模组说明。

本仓库包含源码与构建配置。编译缓存、开发游戏实例、存档和本地训练模型由 `.gitignore` 排除。
