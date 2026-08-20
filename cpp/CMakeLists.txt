cmake_minimum_required(VERSION 3.16)
project(resolutive_inference_cpp LANGUAGES CXX)

add_executable(test_integer_lag8_kernel tests/test_integer_lag8_kernel.cpp)
target_include_directories(test_integer_lag8_kernel PRIVATE include)
target_compile_features(test_integer_lag8_kernel PRIVATE cxx_std_17)

if(MSVC)
  target_compile_options(test_integer_lag8_kernel PRIVATE /W4 /WX)
else()
  target_compile_options(test_integer_lag8_kernel PRIVATE -Wall -Wextra -Werror)
endif()

enable_testing()
add_test(NAME integer_lag8_kernel_parity COMMAND test_integer_lag8_kernel)
