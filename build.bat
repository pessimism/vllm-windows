@echo off
set DISTUTILS_USE_SDK=1
set VLLM_TARGET_DEVICE=cuda
REM replace YOUR_GPU_ARCH with the GPU architectures you want to build against, for example, for RTX 30XX, RTX 40XX and RTX 50XX: 8.6;8.9;12.0
set TORCH_CUDA_ARCH_LIST=12.0a
REM (replace 12 with your desired cpu threads to use in parallel to speed up compilation)
set MAX_JOBS=12

REM Setup Visual Studio x64 Native Tools environment if not already loaded
where cl >nul 2>nul
if %ERRORLEVEL% neq 0 (
    if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" (
        echo Initializing MSVC x64 build environment...
        call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
    ) else (
        echo WARNING: Could not find vcvars64.bat. Please run from VS x64 Native Tools Command Prompt.
    )
)

REM Optional variables:

REM To include cuDSS (only if you have cuDSS installed)
set USE_CUDSS=1
set "CUDSS_LIBRARY_PATH=C:\Program Files\NVIDIA cuDSS\v1.0\lib\13"
set "CUDSS_INCLUDE_PATH=C:\Program Files\NVIDIA cuDSS\v1.0\include"

REM To include cuSPARSELt (only if you have cuSPARSELt installed)
set USE_CUSPARSELT=1
set "CUSPARSELT_INCLUDE_PATH=C:\Program Files\NVIDIA cuSPARSELt\v0.10\include"
set "CUSPARSELT_LIBRARY_PATH=C:\Program Files\NVIDIA cuSPARSELt\v0.10\lib\13"

REM To include cuDNN:
set USE_CUDNN=1
set "CUDNN_LIBRARY_PATH=C:\Program Files\NVIDIA\CUDNN\v9.28\lib\13.5\x64"
set "CUDNN_INCLUDE_PATH=C:\Program Files\NVIDIA\CUDNN\v9.28\include\13.5"

REM To build with Rust vllm-rs support:
REM Download and install rust: https://static.rust-lang.org/rustup/dist/aarch64-pc-windows-msvc/rustup-init.exe
REM Add rustc.exe and cargo.exe folder to Windows PATH (default path is C:\Users\YOUR_WINDOWS_USER\.cargo\bin), then close and open again the VC console.

REM Flash Attention v3 build has been disabled inside WSL2 and Windows due to compiler being killed on WSL2, and extremely long compiling times on Windows. Hopper is not available on Windows, so FA3 has no sense anyway. 
REM Build can be forcefully enabled using the following environment var:
REM set VLLM_FORCE_FA3_WINDOWS_BUILD=1

if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat

REM Install torch 2.11 CUDA 13 (change cu130 with your installed CUDA version)
uv pip install torch==2.11+cu130 torchaudio==2.11+cu130 torchvision==0.26.0+cu130 --index-url https://download.pytorch.org/whl/cu130

uv pip install -r requirements/build/cuda.txt
uv pip install -r requirements/cuda.txt
uv pip install -r requirements/windows.txt

REM Optional: Humming kernels for quantization GEMM (clones large git repo if enabled)
REM set INSTALL_HUMMING=1
if "%INSTALL_HUMMING%"=="1" (
    uv pip install -r requirements/humming.txt
)

uv pip install -e . --no-build-isolation --no-deps -vvv
