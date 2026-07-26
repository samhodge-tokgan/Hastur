// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// TinyExrImpl.cpp — the single translation unit that instantiates the vendored
// tinyexr (BSD-3-Clause) implementation. Every other TU includes vendor/tinyexr.h
// for declarations only; the heavy implementation body lives here exactly once.
//
// tinyexr uses miniz (public domain, vendor/miniz.{h,c}) for ZIP(S) DEFLATE, which
// is compiled as a sibling C source and linked in. We keep tinyexr's default
// TINYEXR_USE_MINIZ=1 (its `#include <miniz.h>` resolves against vendor/ on the
// include path). No thread pool, no OpenMP: EXR writes here are per-frame and small.

// Silence third-party warnings (tinyexr is vendored verbatim).
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"
