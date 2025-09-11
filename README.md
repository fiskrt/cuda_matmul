If you have access to a nvidia gpu:
just compile and run locally



If no access sign up to Modal Labs and use `uv modal run run_on_modal.py`

Set your GPU_SM_ARCH based on your GPU, e.g., for a H200 is use `GPU_SM_ARCH=90`
Compile in debug mode without optimizations:
```bash
nvcc \
    -arch=compute_${GPU_SM_ARCH} \
    -code=sm_${GPU_SM_ARCH},compute_${GPU_SM_ARCH} \
    -g \
    -O0 \
    -Xcompiler -Og \
    -lineinfo \
    -v \
    -o matmul_binary \
    matmul.cu \
    matmul_host.cu \
    -lcuda
```