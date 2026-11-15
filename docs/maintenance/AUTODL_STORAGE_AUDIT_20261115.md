# AutoDL 数据盘清理前审计

**`STORAGE_AUDIT_COMPLETE_PENDING_CLEANUP_APPROVAL`**。Exp15 seed142／242 四臂保持 HOLD。首选方案只涉及两个已有核验离机字节的 SDXL 文件。预计释放 **8.33 GB**，使数据盘可用空间从 **6.79 GB** 增至 **15.12 GB**。逐文件绝对路径、SHA256 和恢复位置见私有审批清单。

## 挂载与运行状态

2026-11-15 23:10:53（北京时间），AutoDL 数据盘为 XFS，容量 50 GiB，已用 46.90 GB，可用 6.79 GB，占用 88%。`df -i` 显示 inode 占用 5%。根文件系统和共享内存为独立挂载。初次 `du -x` 实际分配量约 46.90 GB，与当时 `df` 已用字节一致。表观大小约 46.71 GB。扫描在同一文件系统内完成，未沿链接穿越其他文件树。

在可见的 AutoDL 进程与 GPU 上未发现训练、模型推理、下载或解压任务，也未发现审查范围内的模型文件被打开或映射。Exp15 数据盘只见 seed42 J-C 的正式账本，J-E0 的 seed142／242 文件为已结束的工程预检和单步 Smoke。审计基线为 `phase/exp15-shared-region-supcon` 的 `826634659a036dcccd8cf4e168beeb75639b6d31`。服务器旧工作区在历史分支，包含仍需保留的独立结果和未跟踪文件。

## 主要占用和处置

| 对象 | 实际分配 | 分类与依据 |
| --- | ---: | --- |
| DreamShaper 历史模型树 | 8.224 GB | **ARCHIVE_REQUIRED**。已查离机模型范围未找到同字节权重。历史可变 upstream revision 缺少可验证的不可变身份。 |
| xichen 历史模型获取缓存 | 7.372 GB | **UNKNOWN_HOLD**。inpainting 版本缺 VAE，恢复与备份核验未完成。 |
| SDXL FP16 传输包 | 6.941 GB | **DELETE_CANDIDATE**。已核验离机 TAR 同字节。 |
| SDXL FP16 完整运行快照 | 6.941 GB | 首选范围是第二文本编码器单个权重文件，实际分配 1.389 GB，**DELETE_CANDIDATE**。该文件回填前，历史 SDXL 离线加载暂停。 |
| 服务器旧工作区 | 6.306 GB | **KEEP**。主要是旧实验结果，同时含训练环境、Git 和用户文件。 |
| Exp15、Exp14 的 seed42 正式训练目录 | 1.979 GB、1.871 GB | **KEEP**。正式终点、诊断检查点、账本和评价证据。 |
| 早期不完整 SDXL 获取目录 | 0.307 GB | **UNKNOWN_HOLD**。包含一份同字节编码器及截断权重，保留获取证据。 |

已核验的首选 SDXL 普通文件均为单链接，FIEMAP 无共享 extent。现有离机 TAR 与服务器 TAR 的 SHA256 一致。服务器 staged、TAR 成员和离机展开快照均逐文件通过冻结清单的 19 项大小与 SHA256。官方固定 revision 的元数据与完整文件清单一致，四个固定权重端点从离机环境作 HEAD 检查均可达。恢复优先使用已核验离机字节。保留模型来源、配置、tokenizer、清单和部署、加载记录。 [SDXL 固定 revision](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/tree/115134f363124c53c7d878647567d04daf26e41e) · [历史 SDXL 部署](../exp13_phase1b_r3_handoff.md)

DreamShaper 的历史 loader 读取默认 safetensors 并转为 FP16。另一精度和 safety checker 权重仍属于 19 项冻结校验，任一冻结成员缺失都会中断原样复查。在任何清理审批之前，需要用已有指纹清单对账离机权重。模型加载环境及生成样本、审核、配准、Mask 和 Pair 证据保持原位。 [历史获取](../exp13_1_r1_model_acquisition_recovery.md) · [M1 执行](../exp13_2_m1_execution_review.md)

J-A 两份第 70 步工程权重用于 J-B 历史校准与 seed42 正式入口。J-E0 未来入口读取绑定的预检，独立预检复查读取四份单步权重。J-C 的正式终点和中期梯度检查点也有实际读取方。保留有关模型、协议、账本以及所有小型证据。旧首步和 AMP 故障复查的独有状态没有已核验离机权重，以 **ARCHIVE_REQUIRED** 处理。另有约 0.217 GB 的两件 Exp14 传输对象已核验离机同字节，作为 **DELETE_CANDIDATE** 备案。A10 发布和传输证据均保持 **KEEP**。 [J-C 交接](../exp15/EXP15_JC_HANDOFF.md) · [J-E0 交接](../exp15/EXP15_JE0_HANDOFF.md)

## 待审批范围

按十进制 15 GB 目标，单独处理服务器 SDXL TAR 后预计有 13.73 GB 可用，加上第二文本编码器文件后预计 **15.12 GB**，对应 **14.08 GiB**。若需达到 15 GiB 或留出更宽裕的空间，可另行审批用同版 SDXL UNet 文件替换第二文本编码器，预计可用约 18.87 GB。两套方案择一审批，保留全部模型配置和其余权重。任何实际操作前要重新核对可用量、文件身份、占用和离机 SHA256。达到目标后停止，训练恢复仍需单独授权。

本轮没有删除、移动、重命名、下载模型、传输大文件、卸载环境或运行模型任务。冻结数据与实验文件保持原样。
