import sys

import numpy as np
import torch
import marlin

import time

def benchmark(f, warmup=1, iter=10):
    for i in range(warmup + iter):
        f()
        # We do not synchronize here in order to hide the kernel launch overhead during benchmarkining as this will also
        # happen during realistic model inference as many launches are submitted to the kernel queue.
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    time.sleep(1.)
    return res

def get_problem(m, n, k, groupsize=-1):
    if groupsize == -1:
        groupsize = k
    dev = torch.device('cuda:0')
    A = torch.randint(-128, 127, (m, k), dtype=torch.int8, device=dev)
    A_ref = torch.randn((m, k), dtype=torch.half, device=dev)
    B = torch.randint(low=-2**31, high=2**31, size=(k * n // 8,), device=dev)
    B_ref = torch.randn((k, n), dtype=torch.half, device=dev)
    max_par = 16
    C = torch.zeros((16 * 4 * max_par, n), dtype=torch.int32, device=dev)
    D = torch.zeros((m, n), dtype=torch.half, device=dev)
    s1 = torch.ones((m, 1), dtype=torch.float, device=dev)
    s2 = torch.ones((1, n), dtype=torch.float, device=dev)
    if groupsize == k:
        s3 = torch.tensor([], dtype=torch.half, device=dev)
    else:
        s3 = torch.ones((k // groupsize, n), dtype=torch.half, device=dev)
    torch.cuda.synchronize()
    return A, B, C, D, A_ref, B_ref, s1, s2, s3

def benchmark_dense(A, B, D):
    res = benchmark(lambda: torch.matmul(A, B, out=D))
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * D.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 2 * B.numel() + 2 * D.numel()) / res / 10 ** 9
    }

def benchmark_quant(A, B, C, D, s1, s2, s3, thread_k, thread_n, sms):
    workspace = torch.zeros(D.shape[1] // 128 * 16, device=torch.device('cuda:0'))
    res = benchmark(lambda: marlin.w4a8_mul(A, B, C, D, s1, s2, s3, workspace, thread_k, thread_n, sms))
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * D.shape[1] / res / 10 ** 12,
        'GB/s': (A.numel() + 4 * B.numel() + 2 * D.numel() + 4 * C.numel() + 4 * s1.numel() + 4 * s2.numel() + 2 * s3.numel()) / res / 10 ** 9
    }

# Pass the SM count for known GPUs to avoid the kernel having to query this information (this is very minor)
gpu = torch.cuda.get_device_name(0)
if 'A100' in gpu:
    SMS = 108
elif 'A10' in gpu:
    SMS = 72
elif '3090' in gpu:
    SMS = 82
elif 'A6000' in gpu:
    SMS = 84
else:
    SMS = -1

MODELS = {
    'ideal': [
        (4 * 256 * SMS, 256 * SMS)
    ],
    'Llama7B': [
        (4096, 3 * 4096),
        (4096, 4096),
        (4096, 2 * 10752),
        (10752, 4096)
    ],
    'Llama13B': [
        (5120, 3 * 5120),
        (5120, 5120),
        (5120, 2 * 13568),
        (13568, 5120)
    ],
    'Llama33B': [
        (6656, 3 * 6656),
        (6656, 6656),
        (6656, 2 * 17664),
        (17664, 6656)
    ],
    'Llama65B': [
        (8192, 3 * 8192),
        (8192, 8192),
        (8192, 2 * 21760),
        (21760, 8192)
    ],
    'Falcon180B': [
        # Note that parallel attention and FC allows layer fusions
        (14848, 14848 * 5 + 1024),
        (14848 * 5, 14848)
    ]
}

# Set to true in order to run a more complete benchmark sweep; the default is reproduce README experiments
ALL = False

for groupsize in [-1, 128] if ALL else [128]:
    print('groupsize=%d' % groupsize)
    print()
    for model, layers in MODELS.items():
        print(model)
        if ALL:
            batchsizes =  [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288]
        else:
            batchsizes = [1, 2, 4, 8, 16, 32, 64, 128]
        for batch in batchsizes:
            if not ALL and model != 'ideal' and batch != 16:
                continue
            tot_q = {'s': 0, 'TFLOP/s': 0, 'GB/s': 0, 'speedup': 0} 
            for layer in layers:
                A, B, C, D, A_ref, B_ref, s1, s2, s3 = get_problem(batch, layer[1], layer[0], groupsize)
                res_d = benchmark_dense(A_ref, B_ref, D)
                if model == 'ideal' and batch == 16:
                    # This is a special case constructed to be optimal for a thread-shape different than the default one
                    res_q = benchmark_quant(A, B, C, D, s1, s2, s3, 64, 256, SMS)
                else:
                    res_q = benchmark_quant(A, B, C, D, s1, s2, s3, -1, -1, SMS)
                res_q['speedup'] = res_d['s'] / res_q['s']
                tot_q['s'] += res_q['s']
                for k in tot_q:
                    if k != 's':
                        tot_q[k] += res_q[k] * res_q['s']
            for k in tot_q:
                if k != 's':
                    tot_q[k] /= tot_q['s']
            print('batch=%04d: s=%.5f, TFLOP/s=%07.3f, GB/s=%08.3f, speedup=%.2f' % (
                batch,
                tot_q['s'],
                tot_q['TFLOP/s'],
                tot_q['GB/s'],
                tot_q['speedup']
            ))
        print()
