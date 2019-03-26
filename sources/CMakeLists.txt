add_executable(cuda_benches main.cu basic_arithmetic.cu constants.cu kernels.cu mont_mul.cu mul_128_to_256.cu
    mul_256_to_512.cu square_256_to_512.cu ell_point.cu ecc_exp.cu CUDA_init.cu finitefield.cu multiexp.cu FFT.cu experiemental.cu)
target_compile_features(cuda_benches PUBLIC cxx_std_14)

#target_compile_definitions(cuda_benches PRIVATE PRINT_BENCHES_INPUT)
#target_compile_definitions(cuda_benches PRIVATE PRINT_BENCHES_OUTPUT)
target_compile_definitions(cuda_benches PRIVATE ZERO_BANK_CONFLICTS)
 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties(cuda_benches PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#target_include_directories(cuda_benches PRIVATE ${PROJECT_SOURCE_DIR}/depends/CUB)

#Checking all of the routines for correctness
option(WITH_CORRECTNESS_CHECK "check correctness of all routines" OFF)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

add_library(CudaPTX OBJECT shuffle_ptx.cu)
set_property(TARGET CudaPTX PROPERTY CUDA_PTX_COMPILATION ON)

install(TARGETS CudaPTX
   OBJECTS DESTINATION bin/ptx 
)

if (WITH_CORRECTNESS_CHECK)
    # dependencies
    # add_subdirectory(lib/Catch2)
    # target_link_libraries(tests Catch2::Catch2)

    # executable
    # add_executable(correctness_checks test.cpp)
    # target_compile_definitions(correctness_checks PRIVATE Py_LIMITED_API)
    # target_include_directories(correctness_checks PUBLIC ${PYTHON_INCLUDE_DIRS})
    
    # target_link_libraries(correctness_checks PRIVATE ${PYTHON_LIBRARIES})
endif()

