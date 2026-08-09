cmake_minimum_required(VERSION 3.22)
project(fuzeframe LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# android/app/src/main/cpp -> repo root
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)

# The algorithm core is platform independent by design: flat row-major float
# buffers, no Apple types. Everything Apple specific lives in the .mm files and
# HHSRKernels.metal, none of which are listed here -- on Android every stage
# takes the CPU path that already exists as the fallback beside each GPU one.
set(CORE_SRC
    ${REPO_ROOT}/core/align.cpp
    ${REPO_ROOT}/core/robustness.cpp
    ${REPO_ROOT}/core/kernels.cpp
    ${REPO_ROOT}/core/merge.cpp
    ${REPO_ROOT}/core/pipeline.cpp
    ${REPO_ROOT}/core/pipeline_paths.cpp
    ${REPO_ROOT}/core/grey_pyramid.cpp
    ${REPO_ROOT}/core/raw_io.cpp
    ${REPO_ROOT}/core/dng_writer.cpp
    ${REPO_ROOT}/core/snr_tuning.cpp
    ${REPO_ROOT}/core/preset_lut.cpp
    ${REPO_ROOT}/core/render_isp.cpp
)

# Same set the Xcode target builds: every .cpp under LibRaw/src except the
# *_ph.cpp aggregators, which re-include the split implementations and produce
# duplicate symbols at link time.
file(GLOB_RECURSE LIBRAW_SRC ${REPO_ROOT}/vendor/LibRaw/src/*.cpp)
list(FILTER LIBRAW_SRC EXCLUDE REGEX ".*_ph\\.cpp$")

# prof.mm and debug_utils.mm are Apple-only, but the core calls into both from
# portable code without a guard. portable_platform.cpp supplies those symbols.
add_library(fuzeframe SHARED
    hhsr_jni.cpp
    portable_platform.cpp
    ${CORE_SRC}
    ${LIBRAW_SRC})

target_include_directories(fuzeframe PRIVATE
    ${REPO_ROOT}
    ${REPO_ROOT}/vendor/LibRaw
    ${REPO_ROOT}/vendor/LibRaw/libraw
)

target_compile_definitions(fuzeframe PRIVATE HAVE_LIBRAW=1 LIBRAW_NODLL=1)

# -w only, matching OTHER_CFLAGS in project.yml. Deliberately NOT -ffast-math:
# it would reassociate the merge accumulation and let this build drift from the
# iOS one, and matching output is the point of sharing the core.
target_compile_options(fuzeframe PRIVATE -w)

# z: the DNG writer includes zlib.h. log: __android_log_print in the bridge.
find_library(log_lib log)
find_library(z_lib z)
target_link_libraries(fuzeframe ${log_lib} ${z_lib})
