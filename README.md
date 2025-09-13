# What did I do?

Implement the tiled matrix multiplication with NVIDIAs shared memory

## How to run
#### Preferred (tested)

Sign up to Modal Labs and run with `uv modal run run_on_modal.py`.

Is that it? Yes, one command to develop and debug on arbitrary NVIDIA hardware (A100, H100, H200, B200) for a few dollars per month.

#### Locally
If you have access to a nvidia gpu:
Compile and run locally
Set `GPU_SM_ARCH` based on your GPU, e.g., for a H200 is use `GPU_SM_ARCH=90`
Compile in debug mode without optimizations:
```bash
mkdir bin && nvcc \
    -arch=compute_${GPU_SM_ARCH} \
    -code=sm_${GPU_SM_ARCH},compute_${GPU_SM_ARCH} \
    -g \
    -O0 \
    -Xcompiler -Og \
    -lineinfo \
    -v \
    -o bin/matmul_binary \
    cuda/matmul_host.cu \
    -lcuda \
    && ./bin/matmul_binary
```