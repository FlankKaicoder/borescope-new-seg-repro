# EXP18 完整 CPU 证据包

`package_delivery.py` 只处理 EXP18 输出 `full_a10` 和 `resume_20261128_01`，不跟随 symlink，不加载图像、模型或 `.pt`。服务器仅保存清单，通过 SSH 把归档二进制流直接写到本地；原服务器产物、旧资料包和失败 partial 均不删除、不覆盖。

执行前必须已有完整 200 Epoch / 4200 Batch、全轨迹最大匹配重算、独立总审查和1899输入/446源码末端校验，且恢复进程 exit0。工具会重新检查这些 Gate 的 JSON、SHA 绑定及完成数量。

先把本工具复制到服务器记录目录；所有审查和日志文件均关闭后，生成排他清单：

```bash
PYTHON=/root/autodl-tmp/borescope-new-seg-repro/.venv/bin/python
ROOT=/root/autodl-tmp/exp18_cpu/20261125T131451Z
$PYTHON -B $ROOT/resume_20261128_01/package_delivery.py inventory --root $ROOT
```

输出只含清单路径、SHA 和统计。将 `resume_20261128_01/artifact_inventory.json` 原字节复制到本地工作目录后，使用输出的SHA：

```powershell
python -B diagnostics/exp18_cpu/package_delivery.py capture `
  --inventory <本地清单JSON> --inventory-sha256 <清单SHA256> `
  --server-script /root/autodl-tmp/exp18_cpu/20261125T131451Z/resume_20261128_01/package_delivery.py
```

目标 `E:/autodlwenjian/exp18/cpu-20261128` 必须尚不存在，由 capture 排他创建。工具下载到 `core.tar.gz.partial`，逐成员核验类型、大小、SHA、mode/uid/gid/纳秒mtime及symlink目标；另用gzip分块读完整解压流直到EOF，强制校验CRC32/ISIZE trailer；再做四个JSON/CSV/log文件的样本恢复，全部通过才发布 `core.tar.gz`。成员索引为 `files.csv.gz`，最终回执 `archive_verification.json` 明确记录 `gzip_full_stream_verified=true` 和完整解压字节数。失败保留 `.partial` 和 `capture_failure.json`，不可直接覆盖重试，需换用新批准目录或人工恢复。

清单包含自身之外所有 regular、symlink 和目录；归档再包含清单自身。IPC/socket等特殊项列为排除，不打包。硬链接的device/inode/link-count保留在清单中，归档把每个regular成员独立保存完整字节。symlink恢复只保留链接定义，不跟随链接目标。不可变输入及既有a10源码包沿用 `cpu-20261125` 资料，不重复收入本包。

本工具初次交付仅完成静态检查，尚未执行实际打包或验收；它的存在不代表 CPU_READY。
