cmake_minimum_required(VERSION 3.20)
project(NanoInfer VERSION 0.1.0 LANGUAGES CXX CUDA)

# Default to Release if not specified (e.g., on first configure).
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- CUDA standard: auto-detect based on nvcc version ----
# CUDA 12.x+ fully supports C++20; CUDA 11.x only partially.
# Fall back to C++17 for 11.x to avoid compilation errors.
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(CMAKE_CUDA_COMPILER_VERSION AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.0")
    message(STATUS "CUDA ${CMAKE_CUDA_COMPILER_VERSION} detected — using C++17 for CUDA (11.x has incomplete C++20 support)")
    set(CMAKE_CUDA_STANDARD 17)
endif()

# ---- GPU architecture ----
# Auto-detect from the GPU if possible, else default to sm_75 (RTX 2060).
# Override with: -DCMAKE_CUDA_ARCHITECTURES="89" (RTX 4090) or "80" (A100)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75")
endif()

# ---- Platform-specific flags ----
if(MSVC)
    # MSVC: force UTF-8 for both CXX and CUDA (nvcc passes -Xcompiler to cl.exe)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/utf-8")
else()
    # GCC/Clang: enable warnings, C++20
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
endif()

# Find packages
find_package(CUDAToolkit REQUIRED)

# ---- xgrammar (structured output / constrained decoding) ----
# Vendored at third_party/xgrammar/ (v0.2.5 + Python-bindings-off patch)
set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(XGRAMMAR_BUILD_CXX_TESTS OFF CACHE BOOL "" FORCE)
set(XGRAMMAR_ENABLE_CPPTRACE OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/xgrammar
                 ${CMAKE_BINARY_DIR}/xgrammar)

# --- Library: nanoinfer ---
set(NANOINFER_SOURCES
    src/tensor.cpp
    src/allocator.cpp
    src/ops/elementwise.cu
    src/ops/norm.cu
    src/ops/rope.cu
    src/ops/gemm.cu
    src/ops/attention.cu
    src/ops/activation.cu
    src/ops/unary.cu
    src/ops/softmax.cu
    src/ops/reduce.cu
    src/ops/argmax.cu
    src/ops/embedding.cu
    src/ops/sampling.cpp
    src/ops/lm_head.cu
    src/model/model_loader.cpp
    # src/model/transformer_block.cpp  # TODO: fix namespace + copy issues
    src/kv_cache/block_allocator.cpp
    src/kv_cache/hash_prefix_cache.cpp
    src/kv_cache/radix_prefix_cache.cpp
    src/kv_cache/paged_attention.cu
    src/server/http_server.cpp
    src/server/server_engine.cpp
    src/engine/scheduler.cpp
    src/engine/engine_server.cpp
    src/engine/draft_engine.cpp
    src/engine/batch_loop.cpp
    src/backend/gemm_backend.cpp
    src/backend/attention_backend.cpp
    src/tokenizer/tokenizer.cpp
    src/model/model_config.cpp
    src/model/model_arch.cpp
)

set(NANOINFER_HEADERS
    include/nanoinfer/tensor.h
    include/nanoinfer/allocator.h
    include/nanoinfer/ops/dispatch.h
    include/nanoinfer/ops/elementwise.h
    include/nanoinfer/ops/norm.h
    include/nanoinfer/ops/rope.h
    include/nanoinfer/ops/gemm.h
    include/nanoinfer/ops/attention.h
    include/nanoinfer/ops/activation.h
    include/nanoinfer/ops/unary.h
    include/nanoinfer/ops/softmax.h
    include/nanoinfer/ops/reduce.h
    include/nanoinfer/model/model_loader.h
    include/nanoinfer/model/model_config.h
    include/nanoinfer/model/arch_spec.h
    include/nanoinfer/model/model_arch.h
    include/nanoinfer/model/transformer_block.h
    include/nanoinfer/ops/embedding.h
    include/nanoinfer/ops/sampling.h
    include/nanoinfer/ops/lm_head.h
    include/nanoinfer/kv_cache/prefix_cache.h
    include/nanoinfer/kv_cache/block_allocator.h
    include/nanoinfer/kv_cache/paged_attention.h
    include/nanoinfer/server/http_server.h
    include/nanoinfer/server/server_engine.h
    include/nanoinfer/engine/scheduler.h
    include/nanoinfer/tokenizer/tokenizer.h
    include/nanoinfer/engine/engine.h
    include/nanoinfer/engine/draft_engine.h
    include/nanoinfer/engine/batch_loop.h
    include/nanoinfer/backend/registry.h
    include/nanoinfer/backend/gemm_backend.h
    include/nanoinfer/backend/attention_backend.h
)

# engine_server.cpp + draft_engine.cpp contain inline CUDA kernels — compile with nvcc
set_source_files_properties(src/engine/engine_server.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/engine/draft_engine.cpp PROPERTIES LANGUAGE CUDA)
# gemm_backend.cpp carries the F32->F16 cast kernel — compile with nvcc
set_source_files_properties(src/backend/gemm_backend.cpp PROPERTIES LANGUAGE CUDA)

add_library(nanoinfer STATIC ${NANOINFER_SOURCES} ${NANOINFER_HEADERS})
target_include_directories(nanoinfer PUBLIC include)
target_link_libraries(nanoinfer PUBLIC CUDA::cudart CUDA::cublas xgrammar)

# Windows: link winsock for HTTP server
if(WIN32)
    target_link_libraries(nanoinfer PUBLIC ws2_32)
endif()

# --- Tests ---
function(add_nanoinfer_test test_name test_source)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} PRIVATE nanoinfer CUDA::cudart CUDA::cublas)
    target_include_directories(${test_name} PRIVATE include)
    set_target_properties(${test_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_nanoinfer_test(test_tensor tests/test_tensor.cpp)
add_nanoinfer_test(test_fp16 tests/test_fp16.cpp)
add_nanoinfer_test(test_ops tests/test_ops.cu)
add_nanoinfer_test(test_quality tests/test_quality.cu)
add_nanoinfer_test(test_model tests/test_model.cpp)
add_nanoinfer_test(test_e2e tests/test_e2e.cpp)
add_nanoinfer_test(test_full_model tests/test_full_model.cpp)
add_nanoinfer_test(test_paged_attention tests/test_paged_attention.cu)
add_nanoinfer_test(test_attention_fp16 tests/test_attention_fp16.cpp)
add_nanoinfer_test(test_paged_attention_v2 tests/test_paged_attention_v2.cu)
add_nanoinfer_test(test_paged_prefill tests/test_paged_prefill.cu)
add_nanoinfer_test(test_paged_model tests/test_paged_model.cpp)
add_nanoinfer_test(test_generate tests/test_generate.cu)
add_nanoinfer_test(test_sampling tests/test_sampling.cpp)
add_nanoinfer_test(test_structured_output tests/test_structured_output.cpp)
add_nanoinfer_test(test_structured_e2e tests/test_structured_e2e.cpp)
add_nanoinfer_test(test_lm_head tests/test_lm_head.cu)

# --- Install ---
install(TARGETS nanoinfer ARCHIVE DESTINATION lib)
install(DIRECTORY include/nanoinfer DESTINATION include)

add_nanoinfer_test(test_concurrent tests/test_concurrent.cu)
add_nanoinfer_test(test_scheduler tests/test_scheduler.cu)
add_nanoinfer_test(test_tokenizer tests/test_tokenizer.cpp)
add_nanoinfer_test(test_engine_paged tests/test_engine_paged.cpp)
add_nanoinfer_test(test_server_engine tests/test_server_engine.cpp)
add_nanoinfer_test(test_batch_loop tests/test_batch_loop.cpp)
add_nanoinfer_test(test_prefix_cache tests/test_prefix_cache.cpp)
add_nanoinfer_test(test_radix_engine tests/test_radix_engine.cpp)
add_nanoinfer_test(test_mixed_users tests/test_mixed_users.cpp)
add_nanoinfer_test(test_spec_decode tests/test_spec_decode.cpp)
add_nanoinfer_test(test_verify_incremental tests/test_verify_incremental.cpp)
add_nanoinfer_test(test_attention_divergence tests/test_attention_divergence.cpp)

# --- Tools ---
add_executable(bench_server tools/bench_server.cpp)
target_link_libraries(bench_server PRIVATE nanoinfer CUDA::cudart CUDA::cublas)
target_include_directories(bench_server PRIVATE include)

add_executable(bench_simulation tools/bench_simulation.cpp)
target_link_libraries(bench_simulation PRIVATE nanoinfer CUDA::cudart CUDA::cublas)
target_include_directories(bench_simulation PRIVATE include)
set_target_properties(bench_simulation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(bench_throughput tools/bench_throughput.cpp)
target_link_libraries(bench_throughput PRIVATE nanoinfer CUDA::cudart CUDA::cublas)
target_include_directories(bench_throughput PRIVATE include)
set_target_properties(bench_throughput PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(serve tools/serve.cpp)
target_link_libraries(serve PRIVATE nanoinfer CUDA::cudart CUDA::cublas)
target_include_directories(serve PRIVATE include)

add_executable(bench_fp16 tools/bench_fp16.cpp)
target_link_libraries(bench_fp16 PRIVATE nanoinfer CUDA::cudart CUDA::cublas)
target_include_directories(bench_fp16 PRIVATE include)

add_executable(bench_structured tools/bench_structured.cpp)
target_link_libraries(bench_structured PRIVATE nanoinfer CUDA::cudart CUDA::cublas)
target_include_directories(bench_structured PRIVATE include)

add_executable(quick_generate tools/quick_generate.cpp)
target_link_libraries(quick_generate PRIVATE nanoinfer CUDA::cudart CUDA::cublas)
target_include_directories(quick_generate PRIVATE include)

add_executable(bench_eval tools/bench_eval.cpp)
target_link_libraries(bench_eval PRIVATE nanoinfer CUDA::cudart CUDA::cublas)
target_include_directories(bench_eval PRIVATE include)

add_executable(test_fp8 tools/test_fp8.cu)
target_link_libraries(test_fp8 PRIVATE nanoinfer CUDA::cudart CUDA::cublas)
target_include_directories(test_fp8 PRIVATE include)

add_executable(bench_batch_quality tools/bench_batch_quality.cpp)
target_link_libraries(bench_batch_quality PRIVATE nanoinfer CUDA::cudart CUDA::cublas)
target_include_directories(bench_batch_quality PRIVATE include)

add_executable(bench_fp8_gemm tools/bench_fp8_gemm.cpp)
target_link_libraries(bench_fp8_gemm PRIVATE nanoinfer CUDA::cudart CUDA::cublas)
target_include_directories(bench_fp8_gemm PRIVATE include)

add_executable(dump_weight tools/dump_weight.cpp)
target_link_libraries(dump_weight PRIVATE nanoinfer CUDA::cudart CUDA::cublas)
target_include_directories(dump_weight PRIVATE include)

add_executable(bench_concurrency tools/bench_concurrency.cpp)
target_link_libraries(bench_concurrency PRIVATE nanoinfer CUDA::cudart CUDA::cublas)
target_include_directories(bench_concurrency PRIVATE include)

add_executable(bench_prompt_lengths tools/bench_prompt_lengths.cpp)
target_link_libraries(bench_prompt_lengths PRIVATE nanoinfer CUDA::cudart CUDA::cublas)
target_include_directories(bench_prompt_lengths PRIVATE include)

add_nanoinfer_test(bench_mixed_lengths tools/bench_mixed_lengths.cpp)
add_nanoinfer_test(bench_max_users tools/bench_max_users.cpp)

# Speculative decoding benchmark (inline CUDA kernels -> compile with nvcc)
add_executable(bench_spec_decode tools/bench_spec_decode.cpp)
target_link_libraries(bench_spec_decode PRIVATE nanoinfer CUDA::cudart CUDA::cublas)
target_include_directories(bench_spec_decode PRIVATE include)
set_source_files_properties(tools/bench_spec_decode.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(bench_spec_decode PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

