#pragma once

#include <Python.h>

#ifndef VLLM_CONCAT
#define VLLM_CONCAT_IMPL(A, B) A##B
#define VLLM_CONCAT(A, B) VLLM_CONCAT_IMPL(A, B)
#endif
#ifndef CONCAT
#define CONCAT(A, B) VLLM_CONCAT(A, B)
#endif

#ifndef VLLM_STRINGIFY
#define VLLM_STRINGIFY_IMPL(A) #A
#define VLLM_STRINGIFY(A) VLLM_STRINGIFY_IMPL(A)
#endif
#ifndef STRINGIFY
#define STRINGIFY(A) VLLM_STRINGIFY(A)
#endif

// A version of the TORCH_LIBRARY macro that expands the NAME, i.e. so NAME
// could be a macro instead of a literal token.
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

// REGISTER_EXTENSION allows the shared library to be loaded and initialized
// via python's import statement.
#define REGISTER_EXTENSION(NAME)                                               \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                     \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                           \
  }
