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
#include <vector>

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

// Resolve a model directory to one ORT can open BY PATH, decrypting if sealed.
//
// Needed for models with external data (.onnx.data): ORT resolves the sidecar
// by name at session-create time, and CreateSessionFromArray cannot do that at
// all — so the tracker cannot use LoadModelBytes.
//
// `logicals` names every file the group needs, because a sealed tree has no
// readable listing: the names on disk are HMACs. On a plaintext tree this
// returns the directory unchanged.
//
// The returned directory is RAM-backed (/dev/shm) and is shredded when the
// process exits. It is NOT created on a platform without /dev/shm — writing
// plaintext weights to a disk would defeat the point — so sealed trees with
// external data are a Linux feature today. Plaintext trees work everywhere.
std::string MaterialiseModelDir(const std::string& model_dir,
                                const std::vector<std::string>& logicals);

// Is there a real file at `model_path` that ORT can open directly?
//
// True for a plaintext tree and for a materialised directory; false for a
// sealed single-file model, whose bytes only exist once decrypted.
//
// This decides HOW a session is created, and it is not cosmetic: ORT resolves
// an .onnx.data sidecar RELATIVE TO THE MODEL FILE, and refuses outright for a
// model loaded from a buffer ("External data path for model loaded from bytes
// escapes working directory"). So anything with external data must go through
// a real path, which is what MaterialiseModelDir provides.
bool ModelOnDisk(const std::string& model_path);

}  // namespace hastur
