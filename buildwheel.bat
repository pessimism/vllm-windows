call venv\Scripts\activate.bat
set DISTUTILS_USE_SDK=1
set VLLM_TARGET_DEVICE=cuda
set TORCH_CUDA_ARCH_LIST=12.0a
set MAX_JOBS=12
python setup.py bdist_wheel --dist-dir dist