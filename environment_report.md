# Exp00.0 Environment Audit

Generated UTC: 2026-08-12T10:07:34+00:00

## nvidia-smi

```text
Wed Aug 12 18:07:34 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.05              Driver Version: 595.71.05      CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2080 Ti     On  |   00000000:DA:00.0 Off |                  N/A |
| 18%   27C    P8             23W /  250W |       1MiB /  22528MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

## nvidia-smi-L

```text
GPU 0: NVIDIA GeForce RTX 2080 Ti (UUID: GPU-31992cd1-feb5-9d9c-ab76-93451ae5cb4a)
```

## nvcc

```text
MISSING: nvcc is not on PATH
```

## uname

```text
Linux autodl-container-22ec429c79-6876b0d3 5.15.0-94-generic #104-Ubuntu SMP Tue Jan 9 15:25:40 UTC 2024 x86_64 x86_64 x86_64 GNU/Linux
```

## os-release

```text
PRETTY_NAME="Ubuntu 22.04.5 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.5 LTS (Jammy Jellyfish)"
VERSION_CODENAME=jammy
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=jammy
```

## python

```text
Python 3.12.3
```

## gcc

```text
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

```

## df

```text
Filesystem      Size  Used Avail Use% Mounted on
overlay          30G   53M   30G   1% /
tmpfs            64M     0   64M   0% /dev
/dev/md0        7.0T  6.1T  983G  87% /autodl-pub
AutoFS:fs1      4.0T  1.4T  2.6T  35% /autodl-pub/data
shm              24G     0   24G   0% /dev/shm
/dev/nvme2n1p2  1.8T   31G  1.6T   2% /usr/bin/nvidia-smi
tmpfs           4.0K  4.0K     0 100% /run/nvidia-ctk-hook
tmpfs           252G     0  252G   0% /proc/asound
tmpfs           252G     0  252G   0% /proc/acpi
tmpfs           252G     0  252G   0% /proc/scsi
tmpfs           252G     0  252G   0% /sys/firmware
```

## free

```text
               total        used        free      shared  buff/cache   available
Mem:           503Gi        48Gi        16Gi        66Mi       437Gi       451Gi
Swap:             0B          0B          0B
```

## Python packages

| Import | Version/status |
|---|---|
| `torch` | 2.8.0+cu128 |
| `torchvision` | 0.23.0+cu128 |
| `ultralytics` | MISSING (ModuleNotFoundError: No module named 'ultralytics') |
| `cv2` | MISSING (ModuleNotFoundError: No module named 'cv2') |
| `numpy` | 2.3.2 |
| `sklearn` | MISSING (ModuleNotFoundError: No module named 'sklearn') |
| `pandas` | MISSING (ModuleNotFoundError: No module named 'pandas') |
| `matplotlib` | 3.10.5 |
| `PIL` | 11.3.0 |
| `shapely` | MISSING (ModuleNotFoundError: No module named 'shapely') |

## CUDA visibility check

```text
torch.version.cuda= 12.8
torch.cuda.is_available= True
torch.cuda.device_count= 1
0 NVIDIA GeForce RTX 2080 Ti 23069917184 22001.1875 MiB
```
