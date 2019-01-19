add_executable(cuda_benches main.cu basic_arithmetic.cu constants.cu host_funcs.cpp kernels.cu mont_mul.cu mul_128_to_256.cu
    mul_256_to_512.cu square_256_to_512.cu ell_point.cu ecc_exp.cu CUDA_init.cu)
target_compile_features(cuda_benches PUBLIC cxx_std_14)
target_compile_definitions(cuda_benches PRIVATE PRINT_BENCHES)
 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties(cuda_benches PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

#Checking all of the routines for correctness
option(WITH_CORRECTNESS_CHECK "check correctness of all routines" OFF)

if (WITH_CORRECTNESS_CHECK)
    # dependencies
    find_package(PythonLibs 2.6 REQUIRED)

    # executable
    add_executable(correctness_checks test.cpp)
    target_compile_definitions(correctness_checks PRIVATE Py_LIMITED_API)
    target_include_directories(correctness_checks PUBLIC ${PYTHON_INCLUDE_DIRS})
    
    target_link_libraries(correctness_checks PRIVATE ${PYTHON_LIBRARIES})
endif()

