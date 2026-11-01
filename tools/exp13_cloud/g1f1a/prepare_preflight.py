"""Additive local preparation/final reporting. Never invokes live or networking."""
import argparse
import ast
import json
from datetime import datetime, timezone
from pathlib import Path

from protocol import (ROOT, PREVIOUS, PROMPT, EXPECTED_ROI, EXPECTED_MASK, require, file_sha,
                      validate_inputs, request_metadata, write_json, CONFIRM)


def history_snapshot():
    return {str(path.relative_to(PREVIOUS.parent)): file_sha(path)
            for path in sorted(PREVIOUS.parent.rglob('*')) if path.is_file()}


def prepare():
    # Reject mismatched frozen inputs before copying anything; never regenerate them.
    require(file_sha(PREVIOUS / 'roi_b_4x3_original.png') == EXPECTED_ROI, 'Frozen ROI mismatch: STOP')
    require(file_sha(PREVIOUS / 'roi_b_4x3_api_rgba_mask.png') == EXPECTED_MASK, 'Frozen mask mismatch: STOP')
    write_json(ROOT / 'history_before_sha256.json', history_snapshot())
    (ROOT / 'inputs').mkdir(exist_ok=False)
    for source, name in [('roi_b_4x3_original.png', 'roi.png'), ('roi_b_4x3_api_rgba_mask.png', 'mask_rgba.png')]:
        with (ROOT / 'inputs' / name).open('xb') as f:
            f.write((PREVIOUS / source).read_bytes())
    with (ROOT / 'prompt_v1.txt').open('xb') as f:
        f.write(PROMPT.encode('utf-8'))
    write_json(ROOT / 'frozen_input_checks.json', validate_inputs())
    write_json(ROOT / 'request_manifest.json', request_metadata())
    write_json(ROOT / 'gate_approval.TEMPLATE.json', {
        'status': 'HOLD_NOT_APPROVED', 'independent_gate_pass': False, 'user_approved': False,
        'source': 'corrosion2', 'model': 'gpt-image-2.5-sunburst', 'max_posts': 1,
        'bundle_lock_sha256': 'POPULATE_ONLY_AFTER_INDEPENDENT_GATE_AND_EXPLICIT_USER_APPROVAL',
        'approval_reference': '',
        'note': 'Template only. Do not treat it as approval or rename it during preflight.'})
    print('FROZEN_INPUTS_PROMPT_REQUEST_PREPARED; POST=0')


def freeze_bundle():
    names = ['protocol.py', 'single_repair.py', 'audit_saved_output.py', 'Start-Corrosion2Repair.ps1',
             'prompt_v1.txt', 'inputs/roi.png', 'inputs/mask_rgba.png', 'request_manifest.json']
    write_json(ROOT / 'bundle_lock.json', {'experiment': 'Exp13-G1F-1',
               'files': {name: file_sha(ROOT / name) for name in names},
               'max_posts': 1, 'source': 'corrosion2', 'model': 'gpt-image-2.5-sunburst',
               'live_approval': 'NOT_GRANTED_BY_THIS_FILE'})
    print('BUNDLE_FROZEN; APPROVAL_RECEIPT_ABSENT')


def finalize():
    require(not (ROOT / 'STARTED.json').exists() and not (ROOT / 'live_result').exists(), 'Unexpected live state')
    require(not (ROOT / 'gate_approval.json').exists(), 'Unexpected approval during preflight')
    before = json.loads((ROOT / 'history_before_sha256.json').read_text(encoding='utf-8'))
    require(before == history_snapshot(), 'Historical files changed')
    checks = validate_inputs()
    tests = json.loads((ROOT / 'offline_tests/test_report.json').read_text(encoding='utf-8'))
    require(tests['status'] == 'PASS', 'Offline tests failed')
    loaders = sorted(ROOT.glob('loader_result_*.json'))
    require(len(loaders) == 1, 'Expected exactly one preflight credential check')
    loader = json.loads(loaders[0].read_text(encoding='utf-8-sig'))
    require(loader['status'] == 'DPAPI_LOADER_DRY_RUN_PASS' and not loader['live_mode']
            and loader['api_post_count'] == 0, 'DPAPI dry run failed')
    dry = json.loads(Path(loader['child_report_path']).read_text(encoding='utf-8'))
    require(dry['status'] == 'DRY_RUN_PASS' and dry['api_post_count'] == 0, 'Dry run failed')
    # Inspect actual AST call sites, not substring counts or runtime transport invocation.
    tree = ast.parse((ROOT / 'single_repair.py').read_text(encoding='utf-8'))
    post_sites = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == 'request'
                  and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 'POST']
    require(len(post_sites) == 1, 'Unexpected POST call-site count')
    static = {
        'status': 'STATIC_PREFLIGHT_PASS_NOT_LIVE_VALIDATION', 'generation_post_call_sites': post_sites,
        'transport': 'Python stdlib http.client.HTTPSConnection; no automatic redirect/retry',
        'generation_post_count_this_stage': 0, 'live_function_executed': False,
        'maximum_future_generation_posts': 1, 'production_started_marker_created': False,
        'approval_receipt_exists': False, 'default_mode': 'DRY_RUN',
        'gates': ['independent gate receipt', 'explicit user-approved confirmation',
                  'bundle SHA lock', 'frozen input validation', 'exclusive durable STARTED marker'],
        'lock_tests': tests['marker_tests'], 'multipart_tests': tests['multipart_tests'],
        'key_management': 'G1F-0B DPAPI block adapted into new launcher; historical loader unchanged.',
        'raw_preservation': 'raw_output.bin saved before decode; named image is a byte-identical copy',
        'url_output': 'At most one output GET; no redirect/retry or Authorization forwarding; signed URL not logged.',
        'remaining_limits': ['No local test proves relay forwarding, upstream model identity, or mask adherence.',
                             'No live integration/paid transport validation performed.']}
    write_json(ROOT / 'live_executor_static_audit.json', static)
    write_json(ROOT / 'history_integrity_check.json', {'status': 'PASS', 'files_checked': len(before),
               'all_bytes_unchanged': True})
    command = f"& '{ROOT / 'Start-Corrosion2Repair.ps1'}' -Live -ConfirmSinglePaidRequest {CONFIRM}"
    report = f'''# Exp13-G1F-1A corrosion2 单次修复 Preflight

最终状态：`G1F_1A_PREFLIGHT_PASS / LIVE_NOT_STARTED / STOP`

## 冻结输入检查

TRAIN corrosion2 / singleton_2。ROI `[0,0,320,240)`，320×240。

- ROI SHA256：`{checks['roi_sha256']}`，PASS。
- RGBA Mask SHA256：`{checks['mask_sha256']}`，PASS。
- 透明区 2394 像素，覆盖率 3.1171875%，完整对应原 JSON 精确 polygon；内部 alpha=0、外部 alpha=255。
- 原图/JSON 身份、尺寸、TRAIN 归属与直接裁剪像素核对通过。未生成替代输入；G1F-0/G1F-0B 共 {len(before)} 个文件逐项哈希未变。

## 固定 Prompt

{PROMPT}

- 文件：`prompt_v1.txt`。
- 文件 SHA256：`{checks['prompt_file_sha256']}`。
- 实际请求字符串 SHA256：`{checks['prompt_request_string_sha256']}`。
- UTF-8、无 BOM、无末尾换行。不调用 strip/rstrip，不归一化或改写；multipart part 的 CRLF 是协议分隔符，不属于 prompt 值。离线解析 multipart 后已核对请求值精确相等。

## 请求参数和凭据

- Base URL：`https://direct-api.ccccapi.cc`。
- Method/endpoint：`POST /v1/images/edits`，model=`gpt-image-2.5-sunburst`，n=1。
- 独立 multipart 文件字段：`image[]` 与 `mask`；文本字段 `model`、`prompt`、`n`。`image[]` 沿用 G1E 成功协议，不将 mask 合并进图片或 prompt。
- size、quality、response_format 未额外指定；服务端输出尺寸不能由 ROI 输入尺寸推定。
- 自动重试=0，fallback=0，Prompt 搜索=0，并发=0，HTTP redirect 禁用；超时 600 秒后也不重发。
- DPAPI 使用已验证 G1F-0B 的 CurrentUser 解密/子进程环境传递方式，复用其逻辑并另存新启动器。User 配置读取与 Python 子进程密钥存在性检查 PASS；DPAPI 文件未变、父进程/永久环境及 Codex 认证未修改。
- 不记录密钥、长度或指纹；密钥不在命令行，临时 BSTR 清零、SecureString 释放。托管字符串内存无法保证立即物理清零。

## 永久单次额度锁与静态审计

生产锁固定在本目录 `STARTED.json`。仅在独立 Gate 记录、显式确认和全部 SHA 校验通过后，发送前使用排他创建、flush 和 fsync 持久化。一旦存在即拒绝再次发送；无删除/清锁/重置功能。失败、超时、解析失败或送达/计费不确定均占用额度。独立 `live_result` 目录也禁止复用。

本阶段仅测试了 `offline_tests` 中的测试 marker；生产 marker 未创建，Live CLI/函数未调用。当前没有 `gate_approval.json`，只有明确 HOLD 的模板。未来批准记录必须绑定 `bundle_lock.json` 的 SHA256、限定 source/model/max_posts，并注明批准来源。它是本地防误操作机制，不能防止有权限的人改脚本或删除文件。

静态审计发现且仅发现 1 个生成 POST 调用点；使用不自动重试、不跟随跳转的标准库 HTTP 连接。离线测试覆盖 multipart 完整字段和 prompt 字节、永久锁排他拒绝、部分写入 marker 仍拒绝重发、默认 Dry Run、Gate 拒绝和诊断缩放。真实 relay 兼容性尚未验证。

## 输出保全与后处理设计

未来收到图像后先保存未修改字节到 `raw_output.bin`，再保存同字节命名图像，记录格式、尺寸、通道、SHA256、响应元数据、usage 和耗时。JSON/base64/URL 均有处理；URL 最多单次 GET，不传 API Key，不跟随跳转，不记录含令牌 URL。任何错误立即停止，无第二次 POST。

原始结果不覆盖。非 320×240 输出生成等比例诊断观察图；比例不一致则记录 letterbox、有效区域、缩放比例与采样信息损失，Mask 内外统计排除诊断补边。不把缩放当作几何配准证明。

预备九视图：原始 ROI、精确 polygon、API raw output 缩略图、ROI 尺寸观察图、绝对差异热图、Mask 内修复区、机械边缘、黑色空腔和 Mask 外 UI。局部窗口是观察窗口，不是额外权威分割。未实现 hard/feather 候选，未授予训练资格。

独立保留三类结论：原始结果视觉合理性、原始配对几何一致性、后处理 Pair 可用性，全部等待人工审核。A CORROSION_REMOVAL、B MECHANICAL_GEOMETRY、C OUTSIDE_MASK_FIDELITY、D PAIR_USABILITY、E PROVENANCE 均由人工 Gate 决定；不自动给出 VISUAL_PASS 或 TRAIN_PAIR_ELIGIBLE。单次成功不能授权批量化。

## Dry Run 与下一步命令

Dry Run PASS；API POST 总数 **0**；网络请求 0；GPU/训练/TEST 0。DPAPI 子进程仅构造并检查请求元数据，安装网络拒绝钩子。所有真实请求代码只做静态审计。

以下命令仅供 **ChatGPT 独立 Gate PASS、用户明确批准、并由后续授权步骤创建绑定当前 bundle 的 `gate_approval.json` 后**，用户主动执行。当前运行会因没有批准记录而拒绝：

```powershell
{command}
```

## 尚未解决的风险

顶部 UI 仍被裁断；mask 紧邻真实金属边缘与空腔，可能误填空腔或改变机械间隙。relay 是否将 mask 正确转发、上游实际身份及严格 mask 服从均未知；模型可能改变画风、结构、Mask 外 UI 或输出尺寸。冻结 Prompt 改变了修复措辞，其效果未经本次付费验证。若请求失败，计费状态可能未知，必须保留占用额度并停止。生成后仍需几何/语义/Pair 分项审核。
'''
    with (ROOT / 'G1F_1A_PREFLIGHT_REPORT.md').open('x', encoding='utf-8', newline='\n') as f:
        f.write(report)
    with (ROOT / 'FUTURE_LIVE_COMMAND.txt').open('x', encoding='utf-8') as f:
        f.write('ONLY AFTER INDEPENDENT GATE + EXPLICIT USER APPROVAL + MATCHING RECEIPT. NOT AUTHORIZED NOW.\n' + command + '\n')
    write_json(ROOT / 'preflight_summary.json', {
        'status': 'G1F_1A_PREFLIGHT_PASS / LIVE_NOT_STARTED / STOP',
        'api_post_count': 0, 'network_requests': 0, 'live_executed': False,
        'production_started_marker_exists': False, 'gate_receipt_exists': False,
        'dpapi_loader_report': str(loaders[0]), 'dry_run_report': loader['child_report_path'],
        'bundle_lock_sha256': file_sha(ROOT / 'bundle_lock.json'),
        'prompt_file_sha256': checks['prompt_file_sha256'],
        'prompt_request_string_sha256': checks['prompt_request_string_sha256'],
        'source': 'corrosion2', 'further_action': 'WAIT_INDEPENDENT_CHATGPT_GATE',
        'completed_utc': datetime.now(timezone.utc).isoformat()})
    print('G1F_1A_PREFLIGHT_PASS / LIVE_NOT_STARTED / STOP; API_POST=0')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'freeze', 'finalize'])
    args = parser.parse_args()
    {'prepare': prepare, 'freeze': freeze_bundle, 'finalize': finalize}[args.action]()
