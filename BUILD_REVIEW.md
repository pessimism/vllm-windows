# vLLM Windows Build Review & Progress

## 1. Executive Summary

This document tracks the review, investigation, warning-as-errors configuration, and build fixes for `vllm-windows` on Windows (x86_64 / RTX 50-series Blackwell architecture).

---

## 2. Environment & Prerequisites

- **OS**: Windows 11 / Windows Server x64
- **Host GPU**: NVIDIA GeForce RTX 5060 Ti (sm_120 / Blackwell architecture)
- **CUDA Toolkit**: CUDA 13.5 (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5`)
- **Visual Studio**: VS 2026 Community (`C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat`)
- **Python**: 3.12 (managed via `uv` in `./venv`)
- **Build Parallelism**: 4 worker jobs (`MAX_JOBS=4`) to avoid system memory exhaustion.
- **Accelerators/Libraries**:
  - cuDSS v1.0 (`C:\Program Files\NVIDIA cuDSS\v1.0`)
  - cuSPARSELt v0.10 (`C:\Program Files\NVIDIA cuSPARSELt\v0.10`)
  - cuDNN v9.28 (`C:\Program Files\NVIDIA\CUDNN\v9.28`)

---

## 3. Findings & Issues Identified

### 3.1 Package Restore Bug: `flashinfer-jit-cache` Platform Marker
- **Location**: `requirements/cuda.txt:30`
- **Issue**:
  ```text
  flashinfer-jit-cache @ https://github.com/SystemPanic/flashinfer-windows/releases/download/v0.6.11.post3+spark/flashinfer_jit_cache-0.6.11.post3-cp39-abi3-win_arm64.whl; sys_platform == "win32" and platform_machine != 'ARM64'
  ```
  The condition mistakenly specified `platform_machine != 'ARM64'` for an ARM64-specific wheel (`win_arm64.whl`), causing `uv pip install` to fail on standard `win_amd64` machines with:
  `A URL dependency is incompatible with the current platform: wheel is compatible with win_arm64, but you're on win_amd64`.
- **Fix**: Changed marker to `sys_platform == "win32" and platform_machine == 'ARM64'`.

### 3.2 Investigation: Decoupling `humming-windows` Git Dependency
- **Location**: `requirements/cuda.txt:50`
- **Context**:
  ```text
  humming-kernels[cu13] @ git+https://github.com/SystemPanic/humming-windows.git@v0.1.17 ; sys_platform == "win32"
  ```
- **Analysis**:
  1. **Direct Git Dependency**: `uv pip` must clone `SystemPanic/humming-windows` during package restore rather than fetching a prebuilt binary wheel.
  2. **Bloated `[cu13]` Extra**:
     `humming-kernels` defines `[cu13]` to pull:
     - `nvidia-cuda-runtime`
     - `nvidia-cuda-cccl`
     - `nvidia-cuda-nvcc` (over 3 GB download from PyPI)
     - `nvidia-cuda-nvrtc`
     along with `triton-windows`.
     On a Windows build machine with local CUDA 13 Toolkit already installed, downloading duplicate multi-gigabyte PyPI compiler wheels causes massive download delays, disk bloat, and cache thrashing.
  3. **Build Independence**: vLLM C++/CUDA extension compilation (`setup.py` / `CMakeLists.txt`) does **not** link against or compile `humming`.
  4. **Runtime Protection**: Every call in vLLM is guarded dynamically via `has_humming()` from `vllm.utils.import_utils`. If `humming` is not installed, vLLM gracefully skips Humming quantization paths and uses standard Cutlass / Marlin / FP8 kernels.
- **Decoupling Strategy**:
  - Separate `humming-kernels` out of core `requirements/cuda.txt` into an optional requirement file (`requirements/quantization/humming.txt` or an extra `[humming]`).
  - For Windows users requiring Humming quantization kernels, build and publish precompiled wheels under GitHub releases (similar to `flashinfer-windows`), or avoid the `[cu13]` extra so PyPI does not re-download the 3GB `nvidia-cuda-nvcc` compiler distribution when local CUDA is present.

### 3.3 Visual Studio Native Tools Environment
- **Issue**: Standard command shells do not have MSVC tools (`cl.exe`, `link.exe`, `lib.exe`, etc.) on `PATH` or required environment variables (`INCLUDE`, `LIB`, `WindowsSdkDir`).
- **Fix**: Ensure `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat` is invoked automatically in `build.bat` if `cl.exe` is not in `PATH`.

### 3.4 Elimination of Brittle PowerShell Hack in `build.bat`
- **Issue**: `build.bat` was executing a fragile PowerShell command that text-replaced `/Zc:__cplusplus` out of `CMakeLists.txt` on every build run.
- **Fix**: Removed `/Zc:__cplusplus` directly from `CMakeLists.txt:523` and `534`, and removed the PowerShell patching block from `build.bat`.

### 3.5 Warning Noise Suppression (`D9025`)
- **Issue**: Every single translation unit emitted multiple warnings:
  `cl : Command line warning D9025 : overriding '/Zi' with '/Z7'`
  Flooding the build logs with thousands of redundant warning lines.
- **Fix**: Added `/wd9025` and replaced `/Zi` across all `CMAKE_*_FLAGS*` variants.

### 3.6 Critical Windows 64-bit Portability Bug: Marlin Scalar Type Truncation
- **Location**: `csrc/quantization/marlin/marlin_dtypes.cuh:16`
- **Issue**:
  ```cpp
  template <long scalar_type_id>
  class MarlinScalarType {};
  ```
  On Windows (LLP64), `long` is 32-bit (`int32_t`), whereas `ScalarTypeId` is 64-bit (`int64_t`). Passing a 64-bit scalar type ID (e.g. `1125899906910725i64`) caused `error #69-D: integer conversion resulted in truncation` and caused all 24 Marlin and Marlin MoE kernel templates to fail instantiation.
- **Fix**: Changed template parameter to `template <vllm::ScalarTypeId scalar_type_id>`.

### 3.7 Macro Redefinition in `csrc/core/registration.h`
- **Location**: `csrc/core/registration.h:5-18`
- **Issue**: Standard and MSVC implementations reserve macros starting with an underscore followed by a capital letter (`_CONCAT`). When `registration.h` is included before MSVC's `<xatomic.h>` (such as in `csrc/moe/torch_bindings.cpp`), MSVC's un-guarded `#define _CONCAT(A, B) A##B` triggers `warning C4005: '_CONCAT': macro redefinition` (fatal under `/WX`).
- **Fix**: Renamed macros to `VLLM_CONCAT` and `VLLM_STRINGIFY`, keeping standard `CONCAT` and `STRINGIFY` aliases and avoiding collision with MSVC internal headers.

### 3.8 Nonstandard Function Template Redeclaration
- **Location**: `csrc/topk_histogram_4096_overflow.cuh:74`
- **Issue**: `exact_topk_rescan` definition specified default template and function arguments after a forward declaration in `csrc/topk_histogram_4096.cuh` that lacked defaults, triggering `error #802-D`.
- **Fix**: Moved default arguments to the forward declaration in `csrc/topk_histogram_4096.cuh` and removed them from the definition in `csrc/topk_histogram_4096_overflow.cuh`.

### 3.9 Deprecated `std::is_pod_v` in C++20
- **Location**: `csrc/layernorm_kernels.cu:116`, `csrc/layernorm_quant_kernels.cu:98`
- **Issue**: `static_assert(std::is_pod_v<...>)` triggered `error #1444-D` under C++20.
- **Fix**: Replaced with `std::is_standard_layout_v<...> && std::is_trivially_copyable_v<...>`.

### 3.10 Unused Variable in Selective Scan
- **Location**: `csrc/mamba/selective_scan_fwd.cu:421`
- **Issue**: `static_assert bool kHasZ = true;` was declared but shadowed by `VLLM_STABLE_DISPATCH_BOOL(..., kHasZ, ...)`.
- **Fix**: Removed the redundant unused variable declaration.

### 3.11 Implicit 64-bit to 8-bit Truncation in Custom All-Reduce
- **Location**: `csrc/custom_all_reduce.cu:151`
- **Issue**: `bytes.emplace_back(handles[i].begin(), handles[i].end())` where `handles[i]` holds `int64_t` caused `warning C4244: '=': conversion from 'const __int64' to 'char', possible loss of data` (fatal under `/WX`).
- **Fix**: Rewrote the loop to construct a `std::string` with explicit `static_cast<char>`.

### 3.12 NVCC Warnings-as-Errors Configuration for External Headers
- **Location**: `CMakeLists.txt:516`
- **Issue**: Upstream CUTLASS headers contain `#20012-D` (ignored device annotation) and `#2908-D` (implicit this capture), and CUDA CCCL headers contain `#177-D` (unused variable).
- **Fix**: Configured `-Werror all-warnings` with `--diag-suppress=20012,2908,177,68,1444` for external headers while keeping `/WX` on MSVC.

### 3.13 sm_120 (Blackwell) `__launch_bounds__` Exceeding SM Capacity
- **Location**: `csrc/persistent_topk.cuh:612`
- **Issue**:
  ```cpp
  __global__ void __launch_bounds__(kThreadsPerBlock, 2) persistent_topk_kernel(...)
  ```
  `kThreadsPerBlock` is 1024. Asking for a minimum of 2 blocks per SM requires 2048 threads per SM (`1024 * 2 = 2048`).
  On sm_120 (RTX 50-series Blackwell), the hardware limit is 1536 threads per SM. Ptxas aborted compilation with:
  `ptxas error: Value of threads per SM for entry ... is out of range. .minnctapersm will be ignored`
  `ptxas fatal: Ptx assembly aborted due to errors`
- **Fix**: Changed `__launch_bounds__(kThreadsPerBlock, 2)` to `__launch_bounds__(kThreadsPerBlock, VLLM_BLOCKS_PER_SM(kThreadsPerBlock))` using `launch_bounds_utils.h` (which dynamically evaluates to 1 on sm_86/89/120 and 2 on sm_80/90/100).

### 3.14 Signed Integer Shift into Sign Bit in Hadamard Kernel
- **Location**: `csrc/quantization/hadamard/hadacore/hadamard_transform_cuda.cu:104-107`
- **Issue**: `val_1n[i] << 16` shifted a signed 16-bit value into the 32-bit sign bit, triggering `error #68-D: integer conversion resulted in a change of sign`.
- **Fix**: Wrapped operands in `static_cast<uint32_t>` and `0x0000FFFFu` to perform unsigned bit arithmetic.

### 3.15 Compiler Flag Cleanups
- Replaced `-Zi` as well as `/Zi` with `-Z7` / `/Z7` across NVCC and MSVC flags in `CMakeLists.txt` to eliminate command-line override noise.
- Removed invalid `/wd9025` driver warning suppressions that caused `D9014` invalid option warnings.

### 3.16 Integer Conversion Loss of Data in `csrc/core/scalar_type.hpp`
- **Location**: `csrc/core/scalar_type.hpp:90-164`
- **Issue**:
  `member_id_field_width` returned `size_t` (`uint64_t` on Windows x64).
  In `ScalarType::id()` and `ScalarType::from_id()`:
  - `bit_offset + bits` produced `uint64_t`, which was then used to initialize `std::pair<Id, uint32_t>` and `std::pair<..., int>`, triggering `warning C4244: 'initializing': conversion from '_Ty' to '_Ty2', possible loss of data` (fatal under `/WX`).
  - This header is included across all Marlin quantization kernels (`sm80_kernel_*.cu`), causing compilation of all Marlin targets to halt.
- **Fix**:
  - Changed `member_id_field_width()` return type to `uint32_t`.
  - Added UB-safe mask `(bits >= 64) ? ~uint64_t(0) : ((uint64_t(1) << bits) - 1)`.
  - Used matching `uint32_t` state tracking throughout `id()` and `from_id()`.

### 3.17 MSVC 2026 Warning C5285 in CUTLASS `cute/container/array_subbyte.hpp`
- **Location**: `.deps/cutlass-src/include/cute/container/array_subbyte.hpp:598`
- **Issue**:
  CuTe in CUTLASS 3.x/4.x specializes `template <class T> struct is_reference<cute::subbyte_reference<T>> : true_type {};`.
  In MSVC 18.12 (Visual Studio 2026), the compiler strictly enforces standard N5014 forbidding specialization of `std::is_reference`, generating `warning C5285: cannot declare a specialization for 'std::is_reference': Specializing this standard library template is forbidden by N5014 [meta.rqmts]/4` (treated as error under `/WX`).
- **Fix**: Added `/wd5285` to `CMAKE_CXX_FLAGS` and `-Xcompiler=/wd5285` to `VLLM_GPU_FLAGS` in `CMakeLists.txt`.

### 3.18 NVCC C++20 Module Keyword Diagnostic `#3189-D` in PyTorch Headers
- **Location**: `torch/csrc/api/include/torch/nn/modules/container/any_module_holder.h` and `torch/python.h`
- **Issue**:
  When compiling under C++20 (`-std=c++20`), nvcc EDG frontend treats `module` as a potential C++20 keyword and emits remark/warning `#3189-D: "module" is parsed as an identifier rather than a keyword because the tokens that follow it do not match those of a preprocessor directive`. Under `-Werror all-warnings`, this remark in PyTorch's public headers caused `_qutlass_C` (`qutlass/csrc/gemm.cu`) to fail.
- **Fix**: Added `3189` to `--diag-suppress` in `VLLM_GPU_FLAGS` in `CMakeLists.txt`.

### 3.19 MSVC Warning C4267 in PyTorch Public Headers
- **Location**: `ATen/core/function_schema.h:439` and `torch/csrc/dynamo/compiled_autograd.h:667`
- **Issue**:
  When external extension modules like `_qutlass_C` (`bindings.cpp` and `gemm.cu`) include PyTorch headers, PyTorch instantiates standard library templates (e.g. `std::optional<int>` and `std::vector<std::pair<int, int>>`) passing `size_t` (64-bit on x64) to `int` (32-bit). Because STL templates are instantiated in the translation unit, MSVC flags `warning C4267: 'initializing': conversion from 'size_t' to 'int', possible loss of data` (treated as error under `/WX`).
- **Fix**: Added `/wd4267` to `CMAKE_CXX_FLAGS` and `-Xcompiler=/wd4267` to `VLLM_GPU_FLAGS` in `CMakeLists.txt`.

### 3.20 Macro Redefinition in Vendored Dependencies (`vllm-flash-attn` and `deepgemm`)
- **Location**: `.deps/vllm-flash-attn-src/csrc/common/registration.h:5` and `.deps/deepgemm-src/csrc/utils/registration.h:5`
- **Issue**:
  Like in finding 3.7, vendored copies of `registration.h` in `vllm-flash-attn` and `deepgemm` defined `#define _CONCAT(A, B) A##B`. When compiling `_vllm_fa2_C` (`flash_api_torch_lib.cpp`), MSVC's `<xatomic.h>` was included after `registration.h`, triggering `warning C4005: '_CONCAT': macro redefinition` (fatal under `/WX`).
- **Fix**: Replaced with `VLLM_CONCAT` and `VLLM_STRINGIFY` implementations matching `csrc/core/registration.h`.

### 3.21 FlashAttention-v3 (FA3) Incompatibility on Windows
- **Location**: `build.bat:43`, `.deps/vllm-flash-attn-src/hopper/flash_api.cpp:302-350`
- **Issue**:
  - `build.bat` had active `set VLLM_FORCE_FA3_WINDOWS_BUILD=1`.
  - FlashAttention-v3 (`_vllm_fa3_C`) is written strictly for Hopper (SM90) architecture, which has no Windows drivers or consumer hardware.
  - In `hopper/flash_api.cpp`, `#ifdef` and `#ifndef` directives were placed directly inside macro arguments (`PACKGQA_SWITCH`, `ARCH_SWITCH`), which is undefined behavior under C++ ISO standards and fails on MSVC (`error C2121: '#': invalid character: possibly the result of a macro expansion`, `warning C5101`).
- **Fix**:
  - Commented out `set VLLM_FORCE_FA3_WINDOWS_BUILD=1` in `build.bat`.
  - Configured `set(FA3_ENABLED OFF CACHE BOOL "Enable FA3" FORCE)` on Windows in `cmake/external_projects/vllm_flash_attn.cmake`.
  - Updated `cmake/patches/windows_flash_attn.patch` so clean checkouts disable `FA3_ENABLED` automatically on Windows.

### 3.22 Recursive Submodule Bloat in `vllm-flash-attn` (`composable_kernel`)
- **Location**: `cmake/external_projects/vllm_flash_attn.cmake:39-47`
- **Issue**:
  - `FetchContent_Declare(vllm-flash-attn ...)` without submodule restrictions defaults to recursive git submodule initialization.
  - `vllm-flash-attn` references `third_party/aiter` (AMD ROCm), which in turn submodules `ROCm/composable_kernel.git` (a multi-gigabyte repository).
  - During CMake configuration, git clones gigabytes of unused ROCm code on a Windows CUDA build.
- **Fix**:
  - Added `GIT_SUBMODULES "csrc/cutlass"` and `GIT_SUBMODULES_RECURSE FALSE` to `FetchContent_Declare(vllm-flash-attn)` so git only initializes the essential Cutlass headers required by FA2.

---

## 4. Iteration Log

| Iteration | Description | Result |
| :--- | :--- | :--- |
| **01** | Investigated build prerequisites, environment, and `build.bat`. | Identified shortcut target for VS 2026 x64 Native Tools (`vcvars64.bat`). |
| **02** | Investigated `humming-windows` dependency. | Confirmed it pulls direct Git repo + 3GB `nvidia-cuda-nvcc` PyPI package via `[cu13]`. Verified vLLM uses runtime `has_humming()` guard. |
| **03** | Fixed `requirements/cuda.txt` ARM64 platform marker on `flashinfer-jit-cache`. | Resolved dry-run package resolution error on `win_amd64`. |
| **04** | Installing core package dependencies. | Completed successfully in `venv`. |
| **05** | Configured warnings-as-errors; cleaned `/Zc:__cplusplus`; added `vcvars64.bat` in `build.bat`. | Initial build failed; exposed `D9025` warning flood, `_CONCAT` collision, Marlin `long` truncation, `exact_topk_rescan` nonstandard redeclaration, and `std::is_pod_v`. |
| **06** | Reduced MAX_JOBS to 4; fixed Marlin `ScalarTypeId` bug; fixed `_CONCAT`; fixed `exact_topk_rescan`; fixed `std::is_pod_v`; fixed `custom_all_reduce.cu`; suppressed external CUTLASS/CCCL header warnings and D9025. | Verified `_C_stable_libtorch` targets (`torch_bindings`, `selective_scan_fwd`, `custom_all_reduce`, `layernorm_kernels`) compiled cleanly. Revealed `topk.cu` ptxas `__launch_bounds__` error for sm_120 and hadamard sign shift. |
| **07** | Fixed sm_120 launch bounds in `persistent_topk.cuh`; fixed hadamard `uint32_t` shift; cleaned `-Zi` flag replacements. Resumed Ninja build with `MAX_JOBS=4`. | Target 1 through 44 succeeded (including `topk.cu`!). Target 45 (`sm80_kernel_*.cu`) revealed MSVC `C4244` in `scalar_type.hpp`. |
| **08** | Fixed `scalar_type.hpp` integer conversion bug; retesting compilation with `MAX_JOBS=4`. | All 24 Marlin kernels compiled cleanly! Target 30/353 (`scaled_mm_sm120_fp8.cu`) revealed MSVC 2026 warning `C5285` in CUTLASS `array_subbyte.hpp`. |
| **09** | Added `/wd5285` to suppress C5285 in CUTLASS; resumed build with `MAX_JOBS=4`. | Targets progressed past 82/398 (CUTLASS and Hadamard compiled cleanly!). Target 83/398 (`_moe_C_stable_libtorch`'s `torch_bindings.cpp`) triggered `C4005: '_CONCAT': macro redefinition` in MSVC `xatomic.h`. |
| **10** | Renamed `_CONCAT`/`_STRINGIFY` to `VLLM_CONCAT`/`VLLM_STRINGIFY` in `registration.h`; resumed build with `MAX_JOBS=4`. | Passed both `_moe_C_stable_libtorch` and `_C_stable_libtorch` `torch_bindings.cpp`. Target 38/316 (`_qutlass_C/gemm.cu`) failed on nvcc `#3189-D` (`"module" is parsed as an identifier`) in PyTorch's headers. |
| **11** | Added `3189` to `--diag-suppress` in `CMakeLists.txt`; resumed build with `MAX_JOBS=4`. | `#3189-D` suppressed; targets 113/393 & 114/393 (`_qutlass_C`'s `bindings.cpp` & `gemm.cu`) encountered MSVC warning `C4267` (`size_t` to `int` loss of data) in PyTorch's `function_schema.h` and `compiled_autograd.h`. |
| **12** | Added `/wd4267` to MSVC compiler flags; resumed build with `MAX_JOBS=4`. | Progressed through target 125/398. Target 126/398 (`_vllm_fa2_C`'s `flash_api_torch_lib.cpp`) failed on vendored `registration.h` in `.deps/vllm-flash-attn-src`. |
| **13** | Patched vendored `registration.h` in `vllm-flash-attn` and `deepgemm`; resumed build with `MAX_JOBS=4`. | Target `_vllm_fa2_C` (`flash_api_torch_lib.cpp`) compiled cleanly! Target `_vllm_fa3_C` (`hopper/flash_api.cpp`) failed on directives inside macro args. |
| **14** | Disabled FA3 on Windows (non-applicable for consumer GPUs and sm_120; unblocked FA2); resumed build with `MAX_JOBS=4`. | Build completed 409/409 compilation units cleanly! Linking of all `.pyd` extensions succeeded. |
| **15** | Restricted `vllm-flash-attn` FetchContent submodules to `csrc/cutlass` (preventing multi-gigabyte ROCm `composable_kernel` clone) and ran clean rebuild with `MAX_JOBS=4`. | In progress (343 total compilation targets). |
