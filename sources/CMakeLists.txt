add_executable(test_cuda main.cu)
target_compile_features(test_cuda PUBLIC cxx_std_14)
 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties( test_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#target_link_libraries(test_cuda PUBLIC ff gmp)

#add_library(test_cuda_lib SHARED lib_exports.cu)
#target_compile_features(test_cuda_lib PUBLIC cxx_std_14)