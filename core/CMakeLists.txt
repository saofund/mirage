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

# nlohmann/json — the op-log IS a JSON program (the same dialect the Python
# meshlang reads/writes), so the core carries a JSON model natively.
include(FetchContent)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
FetchContent_MakeAvailable(nlohmann_json)

add_library(mirage_core STATIC
  src/mesh.cpp
  src/select.cpp
  src/program.cpp
  src/lint.cpp
  src/raytrace.cpp
)
target_include_directories(mirage_core PUBLIC include)
target_link_libraries(mirage_core PUBLIC nlohmann_json::nlohmann_json)
find_package(Threads REQUIRED)
target_link_libraries(mirage_core PUBLIC Threads::Threads)

add_executable(mirage_stats src/main.cpp)
target_link_libraries(mirage_stats PRIVATE mirage_core)

# Offline path-traced renderer (Cycles-class) — the ground-truth render pillar.
add_executable(mirage_render src/render_main.cpp)
target_link_libraries(mirage_render PRIVATE mirage_core)

# Optional pybind11 module — built when configured with -Dpybind11_DIR=... and
# -DPython_EXECUTABLE=... (see tests/test_cpp_core.py, the differential harness).
find_package(Python COMPONENTS Interpreter Development.Module QUIET)
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
  pybind11_add_module(_mirage_core src/bindings.cpp)
  target_link_libraries(_mirage_core PRIVATE mirage_core)
  message(STATUS "mirage: building pybind11 module _mirage_core")
else()
  message(STATUS "mirage: pybind11 not found — skipping _mirage_core (native lib/exe only)")
endif()

# Native GL viewport (.exe). Off by default (pulls GLFW + glad via FetchContent);
# enable with -DMIRAGE_BUILD_VIEWER=ON.
option(MIRAGE_BUILD_VIEWER "Build the native GL viewport (mirage_viewer.exe)" OFF)
if(MIRAGE_BUILD_VIEWER)
  include(FetchContent)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(glfw GIT_REPOSITORY https://github.com/glfw/glfw.git GIT_TAG 3.4)
  FetchContent_MakeAvailable(glfw)
  FetchContent_Declare(glad2 GIT_REPOSITORY https://github.com/Dav1dde/glad.git GIT_TAG v2.0.8
                       SOURCE_SUBDIR cmake)
  FetchContent_MakeAvailable(glad2)
  glad_add_library(glad_gl STATIC API gl:core=3.3)
  FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG v1.91.5)
  FetchContent_MakeAvailable(imgui)
  add_executable(mirage_viewer
    viewer/viewer.cpp
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
  target_include_directories(mirage_viewer PRIVATE ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
  target_link_libraries(mirage_viewer PRIVATE mirage_core glfw glad_gl)
  message(STATUS "mirage: building native viewer (mirage_viewer) with Dear ImGui")
endif()
