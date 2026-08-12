# Exp00.0 服务器环境审计

状态：**PASS（审计完成），训练环境 Gate：STOP**。

实测环境与任务预期不一致：容器和 PyTorch 均只看到 **1 张 NVIDIA GeForce RTX 2080 Ti**，显存由 `nvidia-smi` 报告为 **22528 MiB**，PyTorch 报告约 **22001 MiB**。当前不是“两张各约 11 GB”的可见拓扑，不能执行双卡并行策略。

系统为 Ubuntu 22.04.5 LTS，驱动 595.71.05，驱动声明 CUDA 13.2；系统 Miniconda Python 3.12.3，PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、NumPy 2.3.2、Pillow 11.3.0、matplotlib 3.10.5。

当前缺少 Ultralytics、OpenCV、scikit-learn、pandas、shapely；审计没有为此创建虚拟环境，也未启动训练。正式训练前必须确认单卡 22GB 配置是否就是实际租用规格，并固定所需依赖。完整原始命令输出见 `environment_report.md`，完整包状态见 `pip_freeze.txt`。

