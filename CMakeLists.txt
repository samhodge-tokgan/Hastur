cmake_minimum_required(VERSION 3.20)
project(Hastur VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(HASTUR_UNIVERSAL "Build a universal arm64+x86_64 binary (macOS)" ON)
option(HASTUR_WITH_ONNX "Link ONNX Runtime (CoreML EP) and enable inference" ON)
set(HASTUR_ORT_VERSION "1.27.1" CACHE STRING "ONNX Runtime version to fetch")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# GCC < 9 (e.g. Rocky Linux 8's default 8.5) keeps std::filesystem in a separate
# library that must be linked explicitly. Empty (a no-op link) on newer toolchains.
set(HASTUR_FS_LIB "")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
  set(HASTUR_FS_LIB stdc++fs)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# The prebuilt ONNX Runtime with the CoreML EP is arm64-only, so an ONNX-enabled
# build is arm64-only (which is also what the Apple Neural Engine path requires).
if(APPLE AND HASTUR_WITH_ONNX)
  set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "arm64 (ONNX Runtime CoreML)" FORCE)
elseif(APPLE AND HASTUR_UNIVERSAL)
  set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "OFX host arch must match" FORCE)
endif()

# ---------------------------------------------------------------------------
# Vendor the OpenFX SDK (populate only; we compile the Support library ourselves
# rather than adding OpenFX's own CMake targets). Pointing SOURCE_SUBDIR at a
# directory with no CMakeLists.txt makes MakeAvailable download without building.
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(openfx
  GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openfx.git
  GIT_TAG OFX_Release_1.5.1
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR include            # no CMakeLists here -> populate only
)
FetchContent_MakeAvailable(openfx)

set(OFX_ROOT "${openfx_SOURCE_DIR}")
set(OFX_SUPPORT "${OFX_ROOT}/Support")

# ---------------------------------------------------------------------------
# ONNX Runtime (prebuilt binary release: headers + shared library).
#   macOS arm64 -> CoreML EP ; Linux x64 -> CUDA EP (GPU) + CPU.
# Populate-only (not a CMake project).
# ---------------------------------------------------------------------------
set(ORT_BASE "https://github.com/microsoft/onnxruntime/releases/download/v${HASTUR_ORT_VERSION}")
if(HASTUR_WITH_ONNX)
  set(ORT_HASH_ARG "")
  if(APPLE)
    set(ORT_URL "${ORT_BASE}/onnxruntime-osx-arm64-${HASTUR_ORT_VERSION}.tgz")
    set(ORT_HASH_ARG URL_HASH SHA256=e42b77a7281cc6e55141bf44fcfbac2c782b823a491bbb6ac33c781dd991f8a6)
    set(ORT_SHLIB "lib/libonnxruntime.${HASTUR_ORT_VERSION}.dylib")
    set(HASTUR_ACCEL "CoreML")
  elseif(WIN32)
    # Windows uses a STATIC-CRT ONNX Runtime built from source (NOT Microsoft's prebuilt),
    # so the bundled runtime + providers carry no external MSVCP140/VCRUNTIME140 dependency
    # and are immune to whatever VC++ runtime the host forces process-wide. (Nuke 16.1 ships
    # MSVCP140 14.36 in its app dir; prebuilt ORT 1.27 needs >=14.40 and fails to init with
    # LoadLibrary error 1114 — see docs/WINDOWS.md.) Built with --enable_msvc_static_runtime;
    # CUDA archs sm_60/61/70/72/75/80/86/89/90 (-real) + 90-virtual PTX (Pascal->Hopper native,
    # Blackwell+ via JIT). Hosted as a pinned repo release asset; rebuild recipe in the memory
    # note / docs. TensorRT provider dropped (unused; would reintroduce a dynamic-CRT dep).
    # NOTE(hastur): this is the shared static-CRT ORT artifact originally built for humbaba and
    # reused here verbatim (same ORT version + hash). Re-host under the Hastur repo's releases
    # once a Hastur ORT build tag exists; the binary itself is identical.
    set(ORT_URL "https://github.com/samhodge-tokgan/humbaba/releases/download/ort-static-win-${HASTUR_ORT_VERSION}/onnxruntime-win-x64-static-cuda9arch-${HASTUR_ORT_VERSION}.zip")
    set(ORT_HASH_ARG URL_HASH SHA256=76a9f67f3704486cc354824212492fcdc152bbe6801667140335aa9aa14026d4)
    set(ORT_SHLIB "lib/onnxruntime.dll")          # runtime (IMPORTED_LOCATION)
    set(ORT_IMPLIB "lib/onnxruntime.lib")         # import lib (IMPORTED_IMPLIB)
    set(HASTUR_ACCEL "CUDA")
  elseif(UNIX)  # Linux
    set(ORT_URL "${ORT_BASE}/onnxruntime-linux-x64-gpu_cuda12-${HASTUR_ORT_VERSION}.tgz")
    # (hash added once pinned; FetchContent fetches without it meanwhile)
    set(ORT_SHLIB "lib/libonnxruntime.so.${HASTUR_ORT_VERSION}")
    set(HASTUR_ACCEL "CUDA")
  else()
    message(FATAL_ERROR "Unsupported platform for HASTUR_WITH_ONNX")
  endif()

  FetchContent_Declare(onnxruntime URL "${ORT_URL}" ${ORT_HASH_ARG})
  FetchContent_MakeAvailable(onnxruntime)
  set(ORT_DIR "${onnxruntime_SOURCE_DIR}")
  set(ORT_DYLIB "${ORT_DIR}/${ORT_SHLIB}")

  add_library(onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_DYLIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_DIR}/include"
  )
  if(WIN32)
    # On Windows link against the import library; IMPORTED_LOCATION is the DLL.
    set_target_properties(onnxruntime PROPERTIES
      IMPORTED_IMPLIB "${ORT_DIR}/${ORT_IMPLIB}")
  endif()
  message(STATUS "ONNX Runtime accelerator EP: ${HASTUR_ACCEL}")
endif()

# ---------------------------------------------------------------------------
# Eigen (header-only, pinned) — linear algebra for the MHR mesh model (Track C).
# NOTE(track-c): this block is owned by the MHR mesh track; it only adds the
# header-only Eigen dependency + the MhrModel library/validation target and does
# not touch the OpenFX / ONNX-Runtime machinery above. Populate-only (Eigen's own
# CMake is heavy and unneeded for header-only use).
# ---------------------------------------------------------------------------
FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR none-existent-so-populate-only
)
FetchContent_MakeAvailable(eigen)

# MHR parametric body model: mhr_assets.bin loader + Eigen LBS. Dependency-light
# (no OFX / ORT), so it builds and unit-tests standalone.
add_library(hastur_mhr STATIC src/MeshAssets.cpp src/MhrModel.cpp)
target_include_directories(hastur_mhr PUBLIC src "${eigen_SOURCE_DIR}")
set_target_properties(hastur_mhr PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Oracle-vs-C++ mesh validation (per-vertex RMSE in mm). Run against the
# generated test-assets: mhr_validate <mhr_assets.bin> <fixtures.bin>.
add_executable(mhr_validate tests/mhr_validate.cpp)
target_link_libraries(mhr_validate PRIVATE hastur_mhr)

# ---------------------------------------------------------------------------
# OpenFX C++ Support library
# ---------------------------------------------------------------------------
add_library(ofxsupport STATIC
  ${OFX_SUPPORT}/Library/ofxsCore.cpp
  ${OFX_SUPPORT}/Library/ofxsImageEffect.cpp
  ${OFX_SUPPORT}/Library/ofxsInteract.cpp
  ${OFX_SUPPORT}/Library/ofxsLog.cpp
  ${OFX_SUPPORT}/Library/ofxsMultiThread.cpp
  ${OFX_SUPPORT}/Library/ofxsParams.cpp
  ${OFX_SUPPORT}/Library/ofxsProperty.cpp
  ${OFX_SUPPORT}/Library/ofxsPropertyValidation.cpp
)
target_include_directories(ofxsupport PUBLIC
  ${OFX_ROOT}/include
  ${OFX_SUPPORT}/include
  ${OFX_SUPPORT}/Plugins/include
)
# The Support library expects the plugin to define getPluginIDs; kOfxThreadsuite etc.
target_compile_definitions(ofxsupport PUBLIC OFX_SUPPORTS_MULTITHREAD)
set_target_properties(ofxsupport PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Per-platform .ofx.bundle layout (OFX packaging spec).
# ---------------------------------------------------------------------------
# Isolation (B): the bundled ONNX Runtime is given a PRIVATE name so the dynamic
# loader can never deduplicate it against a host's or sibling plugin's copy of the
# same library (which would bind us to a different, ABI-incompatible ORT and crash).
# OFX_LIB_ORIG is the name the .ofx links against; OFX_LIB_BUNDLED is the private
# name we rename it to (and repoint the .ofx + providers at) post-build.
if(APPLE)
  set(OFX_ARCH_DIR "MacOS")        # Contents/MacOS/*.ofx
  set(OFX_LIB_DIR "Frameworks")    # bundled shared libs
  set(OFX_RPATH "@loader_path/../Frameworks")
  set(OFX_LIB_ORIG "libonnxruntime.1.dylib")
  set(OFX_LIB_BUNDLED "libonnxruntime_hastur.1.dylib")
elseif(WIN32)
  set(OFX_ARCH_DIR "Win64")        # Contents/Win64/*.ofx (OFX-spec arch dir)
  set(OFX_LIB_DIR "Win64")         # DLLs alongside the .ofx
  set(OFX_RPATH "")                # Windows has no rpath; DLLs found via loader dir
  set(OFX_LIB_ORIG "onnxruntime.dll")
  # The .ofx delay-imports "onnxruntime.dll"; the WinLoader delay hook redirects that
  # import to this PRIVATE name. Without it, Windows ships its own onnxruntime.dll in
  # System32 (Windows ML, a DIFFERENT ORT version) — hosts like Nuke make it resident,
  # and the loader's base-name dedup then binds us to the OS copy (ORT 1.17) instead of
  # ours, so our API-27 calls are rejected / crash. The provider DLLs do NOT import
  # onnxruntime.dll (shared-provider bridge via onnxruntime_providers_shared.dll), so no
  # PE patching is needed; ORT locates them relative to its own (renamed) module dir.
  set(OFX_LIB_BUNDLED "onnxruntime_hastur.dll")
else()                             # Linux
  # OFX packaging spec: the 64-bit Linux arch dir is "Linux-x86-64" (hyphens).
  # Using an underscore makes OFX hosts (Natron) silently skip the binary.
  set(OFX_ARCH_DIR "Linux-x86-64") # Contents/Linux-x86-64/*.ofx
  set(OFX_LIB_DIR "Linux-x86-64")  # .so alongside the .ofx
  set(OFX_RPATH "$ORIGIN")
  set(OFX_LIB_ORIG "libonnxruntime.so.1")
  set(OFX_LIB_BUNDLED "libonnxruntime_hastur.so.1")
endif()

# ---------------------------------------------------------------------------
# The plugin: a loadable module named Sam3dBody.ofx
# ---------------------------------------------------------------------------
add_library(Sam3dBody MODULE src/HasturPlugin.cpp)
target_link_libraries(Sam3dBody PRIVATE ofxsupport ${HASTUR_FS_LIB})
set_target_properties(Sam3dBody PROPERTIES
  PREFIX ""
  OUTPUT_NAME "Sam3dBody"
  SUFFIX ".ofx"
)
if(APPLE)
  # OFX plugins are Mach-O bundles.
  target_link_options(Sam3dBody PRIVATE -bundle)
endif()

# Isolation (D0): export ONLY the OFX C entry points, so none of our C++ / static-lib
# symbols leak into the host process and interpose on the host or a sibling OFX plugin.
# (Windows already exports only the __declspec(dllexport) OFX entry points.)
if(NOT WIN32)
  target_compile_options(Sam3dBody PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
endif()
if(APPLE)
  target_link_options(Sam3dBody PRIVATE
    "-Wl,-exported_symbols_list,${CMAKE_SOURCE_DIR}/packaging/ofx_exports.macos.sym")
elseif(UNIX)
  target_link_options(Sam3dBody PRIVATE
    "-Wl,--version-script=${CMAKE_SOURCE_DIR}/packaging/ofx_exports.map")
endif()

if(HASTUR_WITH_ONNX)
  # M4: the full body-only pipeline. Every stage engine authored in isolation is
  # compiled into the plugin module and orchestrated by Sam3dBodyPipeline.
  target_sources(Sam3dBody PRIVATE
    src/DetectorEngine.cpp
    src/Sam3dBodyEngine.cpp
    src/HandRefinerEngine.cpp
    src/MhrModel.cpp
    src/MeshAssets.cpp
    src/CropAffine.cpp
    src/CameraSolver.cpp
    src/OrtSessionManager.cpp
    src/SoftwareRasterizer.cpp
    src/Sam3dBodyPipeline.cpp
  )
  # Engine headers live in src/; MhrModel needs the header-only Eigen.
  target_include_directories(Sam3dBody PRIVATE src "${eigen_SOURCE_DIR}")
  if(WIN32)
    # onnxruntime.dll and its dependents live next to the .ofx (Contents/Win64),
    # which the loader does NOT search by default for a LoadLibrary'd plugin.
    # WinLoader.cpp (DllMain) adds that directory to the DLL search path, and
    # delay-loading onnxruntime.dll defers its resolution until after DllMain runs.
    target_sources(Sam3dBody PRIVATE src/WinLoader.cpp)
    target_link_libraries(Sam3dBody PRIVATE delayimp)
    target_link_options(Sam3dBody PRIVATE /DELAYLOAD:onnxruntime.dll)
    # MSVC's <cmath> only defines M_PI when _USE_MATH_DEFINES is set beforehand.
    target_compile_definitions(Sam3dBody PRIVATE _USE_MATH_DEFINES NOMINMAX)
  endif()
  target_link_libraries(Sam3dBody PRIVATE onnxruntime)
  target_compile_definitions(Sam3dBody PRIVATE HASTUR_WITH_ONNX=1)
  # Find the bundled shared library relative to the .ofx binary at runtime.
  if(NOT WIN32)
    set_target_properties(Sam3dBody PROPERTIES
      BUILD_WITH_INSTALL_RPATH ON
      INSTALL_RPATH "${OFX_RPATH}"
    )
  endif()
endif()

# ---------------------------------------------------------------------------
# Assemble the .ofx.bundle (macOS layout: Contents/{Info.plist,MacOS/*.ofx})
# ---------------------------------------------------------------------------
set(OFX_BUNDLE_NAME "Sam3dBody")
set(OFX_BUNDLE_BINARY "Sam3dBody.ofx")
set(OFX_BUNDLE_IDENTIFIER "com.tokgan.openfx.Sam3dBody")
set(BUNDLE_DIR "${CMAKE_BINARY_DIR}/${OFX_BUNDLE_NAME}.ofx.bundle")

add_custom_command(TARGET Sam3dBody POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_DIR}/Contents/${OFX_ARCH_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:Sam3dBody>" "${BUNDLE_DIR}/Contents/${OFX_ARCH_DIR}/${OFX_BUNDLE_BINARY}"
  COMMENT "Assembling ${OFX_BUNDLE_NAME}.ofx.bundle (${OFX_ARCH_DIR})"
  VERBATIM
)

# Info.plist is a macOS bundle requirement (Linux bundles don't use it).
if(APPLE)
  configure_file("${CMAKE_SOURCE_DIR}/packaging/Info.plist.in" "${CMAKE_BINARY_DIR}/Info.plist" @ONLY)
  add_custom_command(TARGET Sam3dBody POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/Info.plist" "${BUNDLE_DIR}/Contents/Info.plist"
    VERBATIM)
endif()

# Bundle the ONNX Runtime shared library inside the .ofx.bundle so the plugin is
# self-contained. The plugin links against the SONAME (@rpath/libonnxruntime.1.dylib
# on macOS, libonnxruntime.so.1 on Linux), so the bundled file must carry that name
# (the tarball's real file is fully versioned).
if(HASTUR_WITH_ONNX)
  add_custom_command(TARGET Sam3dBody POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_DIR}/Contents/${OFX_LIB_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy "${ORT_DYLIB}"
            "${BUNDLE_DIR}/Contents/${OFX_LIB_DIR}/${OFX_LIB_BUNDLED}"
    COMMENT "Bundling ONNX Runtime shared library"
    VERBATIM
  )
  # The CUDA EP lives in separate provider shared libraries (cuda/tensorrt/shared);
  # bundle them alongside the .ofx so the EP loads. (Linux: .so, Windows: .dll.)
  if(WIN32)
    file(GLOB ORT_PROVIDER_LIBS "${ORT_DIR}/lib/onnxruntime_providers_*.dll")
  elseif(NOT APPLE)
    file(GLOB ORT_PROVIDER_LIBS "${ORT_DIR}/lib/libonnxruntime_providers_*.so")
  endif()
  if(NOT APPLE)
    foreach(pl ${ORT_PROVIDER_LIBS})
      add_custom_command(TARGET Sam3dBody POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${pl}" "${BUNDLE_DIR}/Contents/${OFX_LIB_DIR}/"
        VERBATIM)
    endforeach()
  endif()

  # Isolation (B): rename the bundled ONNX Runtime to a private name and repoint the
  # .ofx (and, on Linux, the provider libs that NEED it) at that name, so the loader
  # can never deduplicate it against a host's / sibling plugin's copy.
  if(NOT "${OFX_LIB_ORIG}" STREQUAL "${OFX_LIB_BUNDLED}")
    set(_bundled "${BUNDLE_DIR}/Contents/${OFX_LIB_DIR}/${OFX_LIB_BUNDLED}")
    set(_ofxbin "${BUNDLE_DIR}/Contents/${OFX_ARCH_DIR}/${OFX_BUNDLE_BINARY}")
    if(APPLE)
      add_custom_command(TARGET Sam3dBody POST_BUILD
        COMMAND install_name_tool -id "@rpath/${OFX_LIB_BUNDLED}" "${_bundled}"
        COMMAND install_name_tool -change "@rpath/${OFX_LIB_ORIG}" "@rpath/${OFX_LIB_BUNDLED}" "${_ofxbin}"
        COMMAND codesign --force --sign - "${_bundled}"
        COMMAND codesign --force --sign - "${_ofxbin}"
        COMMENT "Isolating ONNX Runtime (private install name ${OFX_LIB_BUNDLED})"
        VERBATIM)
    elseif(UNIX)
      find_program(PATCHELF patchelf)
      if(NOT PATCHELF)
        message(FATAL_ERROR "patchelf not found (needed to privately rename the bundled "
          "ONNX Runtime). Install it: dnf install patchelf  /  apt-get install patchelf")
      endif()
      add_custom_command(TARGET Sam3dBody POST_BUILD
        COMMAND ${PATCHELF} --set-soname "${OFX_LIB_BUNDLED}" "${_bundled}"
        COMMAND ${PATCHELF} --replace-needed "${OFX_LIB_ORIG}" "${OFX_LIB_BUNDLED}" "${_ofxbin}"
        COMMENT "Isolating ONNX Runtime (private soname ${OFX_LIB_BUNDLED})"
        VERBATIM)
      foreach(pl ${ORT_PROVIDER_LIBS})
        get_filename_component(_pln "${pl}" NAME)
        add_custom_command(TARGET Sam3dBody POST_BUILD
          COMMAND ${PATCHELF} --replace-needed "${OFX_LIB_ORIG}" "${OFX_LIB_BUNDLED}"
                  "${BUNDLE_DIR}/Contents/${OFX_LIB_DIR}/${_pln}"
          VERBATIM)
      endforeach()
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# Model set (INTENTIONALLY not bundled here). SAM 3D Body is a MODEL-LESS plugin:
# it ships NO Meta weights. The ONNX models + mesh assets are DERIVED from Meta's
# gated SAM-3D-Body checkpoints (SAM License, no-redistribution) and are therefore
# USER-GENERATED locally — never committed and never staged into the build.
#
# The one wire-up script that produces the whole set from a user's gated
# checkpoints is tools/convert_models.{sh,ps1} (it runs the tools/export_* +
# extract_mhr_assets tools in order). packaging/make_pkg.sh injects convert_models
# into the installed bundle's Contents/; after install the user runs it to populate
# Contents/Resources. tools/fetch_models.{sh,ps1} is a pointer to convert_models
# (+ an optional HASTUR_MODELS_BASE_URL mirror hook for a user-hosted set).
#
# The BODY-ONLY (M4) pipeline resolves these four REQUIRED files from
# Contents/Resources (or $HASTUR_MODEL_DIR / the model-dir param):
#   - person_detector.onnx   (person / bbox detector; tools/export_detector.py)
#   - sam3dbody_body.onnx     (SAM 3D body ViT regressor; tools/export_sam3dbody.py)
#   - mhr_assets.bin          (MHR static buffers, C++ FK+LBS; tools/extract_mhr_assets.py)
#   - pose_corrective.onnx    (pose-corrective vertex-offset MLP; tools/export_pose_corrective.py)
# The M7 hand refiner adds two OPTIONAL files (absence -> graceful body-only):
#   - sam3dbody_hand.onnx     (hand-decoder 2nd pass; tools/export_hand.py)
#   - mhr_wrist.bin           (wrist-twist joint_rotation[77,41], wrist-angle gate; export_hand.py)
#
# DELIBERATELY no per-model install()/copy custom-command here: adding one would
# require the weights to exist at build/pack time, which would defeat the model-less
# posture. A self-contained build simply drops the convert_models output into
# ${BUNDLE_DIR}/Contents/Resources before packaging.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Provider-check tool: prints the ONNX Runtime execution providers (verifies
# CoreML is available). Built only with ONNX enabled.
# ---------------------------------------------------------------------------
if(HASTUR_WITH_ONNX)
  add_executable(ort_check tests/ort_check.cpp)
  target_link_libraries(ort_check PRIVATE onnxruntime ${HASTUR_FS_LIB})
  if(WIN32)
    # No rpath on Windows: copy the ORT DLLs next to ort_check.exe so it can run.
    file(GLOB ORT_ALL_DLLS "${ORT_DIR}/lib/*.dll")
    foreach(dll ${ORT_ALL_DLLS})
      add_custom_command(TARGET ort_check POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:ort_check>"
        VERBATIM)
    endforeach()
  else()
    set_target_properties(ort_check PROPERTIES
      BUILD_RPATH "${ORT_DIR}/lib"
      INSTALL_RPATH "${ORT_DIR}/lib"
    )
  endif()

  # ---------------------------------------------------------------------------
  # pipeline_smoke: offline end-to-end exercise of Sam3dBodyPipeline (no OFX
  # host). Compiles the same engine sources as the plugin + a stb-based image
  # I/O harness. Run: pipeline_smoke <in.jpg> <out.png> [model_search_path].
  # ---------------------------------------------------------------------------
  add_executable(pipeline_smoke
    tools/pipeline_smoke.cpp
    src/DetectorEngine.cpp
    src/Sam3dBodyEngine.cpp
    src/HandRefinerEngine.cpp
    src/MhrModel.cpp
    src/MeshAssets.cpp
    src/CropAffine.cpp
    src/CameraSolver.cpp
    src/OrtSessionManager.cpp
    src/SoftwareRasterizer.cpp
    src/Sam3dBodyPipeline.cpp
  )
  target_include_directories(pipeline_smoke PRIVATE src tools "${eigen_SOURCE_DIR}")
  target_link_libraries(pipeline_smoke PRIVATE onnxruntime ${HASTUR_FS_LIB})
  if(WIN32)
    file(GLOB ORT_ALL_DLLS2 "${ORT_DIR}/lib/*.dll")
    foreach(dll ${ORT_ALL_DLLS2})
      add_custom_command(TARGET pipeline_smoke POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:pipeline_smoke>"
        VERBATIM)
    endforeach()
  else()
    set_target_properties(pipeline_smoke PROPERTIES
      BUILD_RPATH "${ORT_DIR}/lib"
      INSTALL_RPATH "${ORT_DIR}/lib")
  endif()
endif()

# ---------------------------------------------------------------------------
# Install: drop the bundle into the user's OFX plugin dir (override with
# -DOFX_INSTALL_DIR=...). Use `cmake --install build` or the `install-local` target.
# ---------------------------------------------------------------------------
if(NOT DEFINED OFX_INSTALL_DIR)
  if(APPLE)
    set(OFX_INSTALL_DIR "$ENV{HOME}/Library/OFX/Plugins")
  elseif(WIN32)
    # Standard OFX plugin location on Windows.
    set(OFX_INSTALL_DIR "$ENV{CommonProgramFiles}/OFX/Plugins")
  else()  # Linux
    set(OFX_INSTALL_DIR "$ENV{HOME}/OFX/Plugins")
  endif()
endif()
install(DIRECTORY "${BUNDLE_DIR}" DESTINATION "${OFX_INSTALL_DIR}" USE_SOURCE_PERMISSIONS)

add_custom_target(install-local
  COMMAND ${CMAKE_COMMAND} -E make_directory "${OFX_INSTALL_DIR}"
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${OFX_INSTALL_DIR}/${OFX_BUNDLE_NAME}.ofx.bundle"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${BUNDLE_DIR}" "${OFX_INSTALL_DIR}/${OFX_BUNDLE_NAME}.ofx.bundle"
  DEPENDS Sam3dBody
  COMMENT "Installing bundle to ${OFX_INSTALL_DIR}"
)
