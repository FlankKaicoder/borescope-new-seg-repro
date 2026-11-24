"""R4-C first-order statistic bridge and independent, single-use budget."""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path('/root/autodl-tmp/exp17_j2_r4c_detach_mechanism_v3')
REPO = Path('/root/autodl-tmp/borescope-exp17-j2')
R4B = Path('/root/autodl-tmp/exp17_j2_r4b_multibatch_20261124_v1')
FORMAL = Path('/root/autodl-tmp/exp17_j2_seed42_formal_v1')
CHECKPOINT = FORMAL / 'recovery_r1_v1/recovery_slot_0.pt'
ARGS = FORMAL / 'native_initializations/init_0001/runtime/args.yaml'
CONFIG = Path('/root/autodl-tmp/exp17_j2_seed42_formal_gate_20261123/runtime_config.json')
PACKET = Path('/root/autodl-tmp/exp17_j2_engineering_20261122/coverage_r3/fixed_cpu_batches/epoch000_batch001_native_cpu.pt')
BOUND = {str(CHECKPOINT): '79b237241517e9112687f9ecd155d21c096ce54243e9a8267b21a963d58c675e',
         str(ARGS): 'd1dfef12c7d5808e8530443f84577eee40d09f42d01cffc47214efa2a42c495a',
         str(CONFIG): '02f0f6548ee1cfd57af449fd1c9118417252b3a528273a1c80d2eee28c96d7b8',
         str(PACKET): '802114bde743ce7c58e997be279987148f33d91898d43b511ad5040939edd929',
         str(REPO / 'results/exp17g_formal_protocol/formal_training_protocol_v1.json'): '613e4e154a872e0fd88791013f8e63716100470bb4a2f73084b702bb5cd37e44'}
CODE_SHA = '3ccdfe26b414c9c5c6c45a1be93a9fa38210cae8c36bc7e1a8a8c227a7b30f42'
LABELS = ['A_full', 'A_N_prefix', 'B_full', 'B_N_prefix', 'A_yolo', 'B_yolo', 'A_supcon8', 'B_supcon8', 'A_pair4', 'B_pair4']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''): h.update(b)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def write(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())

class Budget:
    def __init__(self, root, authorization):
        self.start = time.monotonic(); self.root = root
        self.file = (root / 'calls.jsonl').open('x', encoding='utf8')
        self.used = []; self.active = None; self.previous = '0' * 64
        self.RSS_peak = 0; self.cgroup_peak = 0
        self.event('BEGIN', authorization_sha256=digest(authorization))

    def event(self, event, **fields):
        row = dict(event=event, previous=self.previous, elapsed=time.monotonic()-self.start, **fields)
        row['sha256'] = digest(row)
        self.file.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        self.file.flush(); os.fsync(self.file.fileno()); self.previous = row['sha256']

    def check(self):
        if time.monotonic()-self.start >= 1200: raise RuntimeError('1200-second limit')
        status = Path('/proc/self/status').read_text()
        rss = int(next(x.split()[1] for x in status.splitlines() if x.startswith('VmRSS:'))) * 1024
        current = int(Path('/sys/fs/cgroup/memory.current').read_text())
        limit = int(Path('/sys/fs/cgroup/memory.max').read_text())
        self.RSS_peak=max(self.RSS_peak,rss);self.cgroup_peak=max(self.cgroup_peak,current)
        if rss > 20*1024**3 or limit-current < 2*1024**3: raise RuntimeError('Host memory ceiling/reserve')
        return {'RSS_bytes': rss, 'cgroup_current_bytes': current}

    def begin(self, label):
        self.check()
        if self.active is not None or len(self.used) >= len(LABELS) or LABELS[len(self.used)] != label:
            raise RuntimeError('Duplicate, pending, out-of-order or excess call: '+label)
        self.active = label; self.used.append(label); self.event('ATTEMPT_BEGIN', label=label)

    def commit(self, **fields):
        self.check()
        if self.active is None: raise RuntimeError('No pending call')
        self.event('COMMIT', label=self.active, **fields); self.active = None

def bridge_type(torch):
    class StatisticBridge(torch.autograd.Function):
        @staticmethod
        def forward(ctx, base, mean, variance, x_detached, gamma_detached, epsilon, observer):
            assert not x_detached.requires_grad and not gamma_detached.requires_grad
            ctx.save_for_backward(mean, variance, x_detached, gamma_detached)
            ctx.epsilon, ctx.observer = epsilon, observer
            return base.clone()

        @staticmethod
        def backward(ctx, h):
            mean, variance, x, gamma = ctx.saved_tensors
            dims = (0,) + tuple(range(2, x.ndim)); shape = (1, -1) + (1,)*(x.ndim-2)
            r = (variance+ctx.epsilon).sqrt()
            gm = -gamma/r * h.sum(dims)
            gv = -gamma/(2*r.pow(3)) * (h*(x-mean.reshape(shape))).sum(dims)
            if ctx.observer is not None: ctx.observer(gm, gv)
            return h, gm, gv, None, None, None, None
    return StatisticBridge

class Offload:
    """Lossless save_on_cpu semantics with host checks; no recomputation."""
    def __init__(self, torch, budget):
        self.torch, self.budget = torch, budget
        self.packed_bytes = 0; self.pack_calls = 0; self.unpack_calls = 0

    def pack(self, tensor):
        self.budget.check()
        size = tensor.numel()*tensor.element_size()
        free = int(Path('/sys/fs/cgroup/memory.max').read_text())-int(Path('/sys/fs/cgroup/memory.current').read_text())
        if size+2*1024**3 > free: raise RuntimeError('Offload allocation would breach host reserve')
        self.packed_bytes += size; self.pack_calls += 1
        return tensor.device, tensor.detach().cpu()

    def unpack(self, saved):
        self.budget.check(); self.unpack_calls += 1
        device, tensor = saved
        return tensor.to(device)
