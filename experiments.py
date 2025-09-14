


# plots for T4, A100, H100, H200
# block size
# tiled vs naive
# loop unroll
# o0 vs o2 optim



import pandas as pd
import re
import matplotlib.pyplot as plt

def parse_gpu_benchmarks(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    data = []
    current_config = {}
    current_gpu = None
    
    for line in content.strip().split('\n'):
        line = line.strip()
        
        # Parse config header
        if 'unroll=' in line and 'opt=' in line:
            unroll = int(re.search(r'unroll=(\d+)', line).group(1))
            opt = int(re.search(r'opt=(\d+)', line).group(1))
            current_config = {'unroll': unroll, 'opt': opt}
        
        # Parse GPU info
        elif line.startswith('GPU 0:'):
            gpu_match = re.search(r'GPU 0: ([^,]+)', line)
            current_gpu = gpu_match.group(1) if gpu_match else None
        
        # Parse benchmark results
        elif 'Running with block size' in line:
            block_size = int(re.search(r'block size (\d+)', line).group(1))
            kernel_type = re.search(r'using (\w+) kernel', line).group(1)
            gpu_time = float(re.search(r'GPU Time: ([\d.]+) ms', line).group(1))
            gflops = float(re.search(r'GFLOPS: ([\d.]+)', line).group(1))
            
            data.append({
                **current_config,
                'gpu': current_gpu,
                'block_size': block_size,
                'kernel_type': kernel_type,
                'gpu_time_ms': gpu_time,
                'gflops': gflops
            })
    
    return pd.DataFrame(data)

def plot_benchmarks(df):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: GFLOPS - Naive Kernel
    naive_data = df[df['kernel_type'] == 'naive']
    for gpu in naive_data['gpu'].unique():
        gpu_data = naive_data[naive_data['gpu'] == gpu].sort_values('block_size')
        ax1.plot(gpu_data['block_size'], gpu_data['gflops'], 
                marker='o', label=gpu)
    
    ax1.set_xlabel('Block Size')
    ax1.set_ylabel('GFLOPS')
    ax1.set_title('Naive Kernel: GFLOPS vs Block Size')
    ax1.set_xticks([8, 16, 32])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: GFLOPS - Tiled Kernel
    tiled_data = df[df['kernel_type'] == 'tiled']
    for gpu in tiled_data['gpu'].unique():
        gpu_data = tiled_data[tiled_data['gpu'] == gpu].sort_values('block_size')
        ax2.plot(gpu_data['block_size'], gpu_data['gflops'], 
                marker='o', label=gpu)
    
    ax2.set_xlabel('Block Size')
    ax2.set_ylabel('GFLOPS')
    ax2.set_title('Tiled Kernel: GFLOPS vs Block Size')
    ax2.set_xticks([8, 16, 32])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: GPU Time - Naive Kernel
    for gpu in naive_data['gpu'].unique():
        gpu_data = naive_data[naive_data['gpu'] == gpu].sort_values('block_size')
        ax3.plot(gpu_data['block_size'], gpu_data['gpu_time_ms'], 
                marker='o', label=gpu)
    
    ax3.set_xlabel('Block Size')
    ax3.set_ylabel('GPU Time (ms)')
    ax3.set_title('Naive Kernel: GPU Time vs Block Size')
    ax3.set_xticks([8, 16, 32])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: GPU Time - Tiled Kernel
    for gpu in tiled_data['gpu'].unique():
        gpu_data = tiled_data[tiled_data['gpu'] == gpu].sort_values('block_size')
        ax4.plot(gpu_data['block_size'], gpu_data['gpu_time_ms'], 
                marker='o', label=gpu)
    
    ax4.set_xlabel('Block Size')
    ax4.set_ylabel('GPU Time (ms)')
    ax4.set_title('Tiled Kernel: GPU Time vs Block Size')
    ax4.set_xticks([8, 16, 32])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_h100_comparison(df):
    # Filter for H100 data (adjust GPU name as needed)
    h100_data = df[df['gpu'].str.contains('H100', case=False, na=False)]
    
    if h100_data.empty:
        print("No H100 data found. Available GPUs:", df['gpu'].unique())
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Tiled vs Untiled (Naive) comparison
    naive_data = h100_data[h100_data['kernel_type'] == 'naive'].sort_values('block_size')
    tiled_data = h100_data[h100_data['kernel_type'] == 'tiled'].sort_values('block_size')
    
    ax1.plot(naive_data['block_size'], naive_data['gflops'], 
            marker='o', linestyle='--', label='Untiled (Naive)')
    ax1.plot(tiled_data['block_size'], tiled_data['gflops'], 
            marker='s', linestyle='-', label='Tiled')
    
    ax1.set_xlabel('Block Size')
    ax1.set_ylabel('GFLOPS')
    ax1.set_title('H100: Tiled vs Untiled')
    ax1.set_xticks([8, 16, 32])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Tiled with unroll vs without unroll
    tiled_only = h100_data[h100_data['kernel_type'] == 'tiled']
    
    tiled_unroll_0 = tiled_only[tiled_only['unroll'] == 0].sort_values('block_size')
    tiled_unroll_1 = tiled_only[tiled_only['unroll'] == 1].sort_values('block_size')
    
    if not tiled_unroll_0.empty:
        ax2.plot(tiled_unroll_0['block_size'], tiled_unroll_0['gflops'], 
                marker='o', linestyle='--', label='Tiled (no unroll)')
    if not tiled_unroll_1.empty:
        ax2.plot(tiled_unroll_1['block_size'], tiled_unroll_1['gflops'], 
                marker='s', linestyle='-', label='Tiled (with unroll)')
    
    ax2.set_xlabel('Block Size')
    ax2.set_ylabel('GFLOPS')
    ax2.set_title('H100: Tiled with/without Unroll')
    ax2.set_xticks([8, 16, 32])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_speedup_comparison(df):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    naive_data = df[df['kernel_type'] == 'naive']
    tiled_data = df[df['kernel_type'] == 'tiled']
    
    # Plot speedup factor for each GPU
    for gpu in df['gpu'].unique():
        gpu_naive = naive_data[naive_data['gpu'] == gpu].sort_values('block_size')
        gpu_tiled = tiled_data[tiled_data['gpu'] == gpu].sort_values('block_size')
        
        # Merge on block_size to ensure matching pairs
        merged = gpu_naive.merge(gpu_tiled, on=['gpu', 'block_size'], suffixes=('_naive', '_tiled'))
        if not merged.empty:
            speedup = merged['gflops_tiled'] / merged['gflops_naive']
            ax.plot(merged['block_size'], speedup, marker='o', label=gpu)
    
    ax.set_xlabel('Block Size')
    ax.set_ylabel('Speedup Factor (Tiled/Naive)')
    ax.set_title('Tiled vs Naive: Speedup Factor')
    ax.set_xticks([8, 16, 32])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Usage
df = parse_gpu_benchmarks('figures/log.txt')
plot_speedup_comparison(df)
print(df.head())
plot_benchmarks(df)
plot_h100_comparison(df)