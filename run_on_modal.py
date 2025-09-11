from pathlib import Path
import modal

app = modal.App("cuda-matmul")

GPU_CONFIG = "T4"
COMPILE_CONFIG = GPU_CONFIG

match COMPILE_CONFIG:
    case "T4":
        GPU_SM_ARCH = "75"      # Turing 12nm
    case "A100":
        GPU_SM_ARCH = "80"      # Ampere 7nm
    case "A10G":
        GPU_SM_ARCH = "86"      # Ampere 8nm
    case "L4":
        GPU_SM_ARCH = "89"      # Lovelace 5nm
    case "H100" | "H200":
        GPU_SM_ARCH = "90"      # Hopper 5nm
    case "B200":
        GPU_SM_ARCH = "120"     # Blackwell, not supported by 12.4 compiler
    case _:
        raise ValueError(
            f"Not sure how to compile architecture-specific code for {COMPILE_CONFIG}"
        )

@app.local_entrypoint()
def main():
    print(f"🔥 showing nvidia-smi output for {COMPILE_CONFIG}")
    nvidia_smi.remote()

    print(f"🔥 compiling our CUDA program for {COMPILE_CONFIG}")
    prog = nvcc.remote()
    output_path = Path("bin/matmul_binary")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(prog)

    print(f"🔥 running our CUDA program on the base Modal image on {GPU_CONFIG}")
    raw_cuda.remote(Path("matmul_binary").read_bytes())


base_image = modal.Image.debian_slim(python_version="3.11")


@app.function(gpu=COMPILE_CONFIG, image=base_image)
def nvidia_smi():
    import subprocess
    assert not subprocess.run(["nvidia-smi"]).returncode


@app.function(gpu=GPU_CONFIG, image=base_image)
def raw_cuda(prog: bytes):
    import subprocess
    filepath = Path("./prog")
    filepath.write_bytes(prog)
    filepath.chmod(0o755)
    subprocess.run(["./prog"])


arch = "x86_64"
distro = "debian11"  # the distribution and version number of our OS (GNU/Linux)
filename = "cuda-keyring_1.1-1_all.deb"  # NVIDIA signing key file
cuda_keyring_url = f"https://developer.download.nvidia.com/compute/cuda/repos/{distro}/{arch}/{filename}"
major, minor = 12, 4
max_cuda_version = f"{major}-{minor}"

cudatoolkit_image = (
    base_image.apt_install("wget")
    .run_commands(
        [
            f"wget {cuda_keyring_url}",
            f"dpkg -i {filename}",
        ]
    )
    .apt_install(f"cuda-compiler-{max_cuda_version}")  # nvcc and dependencies
    .env({"PATH": "/usr/local/cuda/bin:$PATH"})
)


cudatoolkit_image = cudatoolkit_image.add_local_file("matmul.cu", "/root/matmul.cu")
cudatoolkit_image = cudatoolkit_image.add_local_file("matmul_host.cu", "/root/matmul_host.cu")
@app.function(image=cudatoolkit_image)
def nvcc():
    import subprocess
    from pathlib import Path

    assert not subprocess.run(
        [
            "nvcc",  # run nvidia cuda compiler
            f"-arch=compute_{GPU_SM_ARCH}",  # generate PTX machine code compatible with the GPU architecture and future architectures
            f"-code=sm_{GPU_SM_ARCH},compute_{GPU_SM_ARCH}",  # and a SASS binary optimized for the GPU architecture
            "-g",  # include debug symbols
            "-O0",  # no optimization from nvcc, keep PTX assembly simple
            "-Xcompiler",  # and for the gcc toolchain
            "-Og",  # also limit the optimization
            "-lineinfo",  # add line numbers in machine code
            "-v",  # show verbose log output
            "-o",  # and send binary output to
            "matmul_binary",  # a binary called invsqrt_demo
            "matmul.cu",  # compiling the kernel
            "matmul_host.cu",  # and the host code
            "-lcuda",  # and linking in some symbols from the CUDA driver API
            # note that cudart is linked by default
        ],
        check=True,
    ).returncode

    return Path("./matmul_binary").read_bytes()
