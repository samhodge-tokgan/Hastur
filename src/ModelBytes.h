// SPDX-License-Identifier: LicenseRef-Tokgan-Proprietary
// Copyright (c) Tokgan. Proprietary — licensed under the Tokgan EULA; see NOTICE.
//
// Model files as bytes, decrypting when the tree they live in is sealed (#43).
//
// Every load site here took a PATH and handed it to ORT. A sealed tree has no
// such path: filenames are derived from the content key, so there is nothing on
// disk called "sam3dbody_body.onnx". These helpers keep the call sites talking
// in real filenames and do the translation in one place.
//
// A plaintext tree needs no licence and behaves exactly as before, which is
// what lets the sealed rollout land one loader at a time.
#pragma once

#include <string>

namespace hastur {

// The bytes of `model_path`. Throws std::runtime_error with an operator-facing
// message on a missing file, a missing licence, or a failed decrypt.
//
// The SealedTree behind this is cached per directory: the body models live in
// one tree and several engines load from it, and re-reading the licence for
// each would be wasted work.
std::string LoadModelBytes(const std::string& model_path);

// Does this model exist? Replaces std::filesystem::exists() at the sites that
// probe for an optional variant, which cannot work against derived names.
bool ModelExists(const std::string& model_path);

// NOTE: models with external data (.onnx.data) cannot use LoadModelBytes —
// CreateSessionFromArray cannot resolve the sidecar. Those need a materialised
// directory, which lands with the tracker in phase 4. Nothing here loads one.

}  // namespace hastur
