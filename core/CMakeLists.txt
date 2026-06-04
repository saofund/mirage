cmake_minimum_required(VERSION 3.15)
project(mirage_core LANGUAGES CXX)

# mirage_core — the native C++20 engine core (topology kernel; operators, op-log
# evaluator, selection engine and renderer to follow). Built with MSVC/CMake on
# Windows; cross-platform by design. Validated against the Python kernel spec.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  add_compile_options(/W4 /permissive- /utf-8)  # /utf-8: source+exec charset (i18n-safe)
else()
  add_compile_options(-Wall -Wextra)
endif()

add_library(mirage_core STATIC
  src/mesh.cpp
)
target_include_directories(mirage_core PUBLIC include)

add_executable(mirage_stats src/main.cpp)
target_link_libraries(mirage_stats PRIVATE mirage_core)
