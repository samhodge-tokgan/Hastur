// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// Registration glue for the single .ofx binary. The bundle registers one plugin
// (the SAM 3D Body effect) from OFX::Plugin::getPluginIDs. As additional
// stages/effects land they can each expose an append<Name>() here and be pushed
// into the host's plugin list alongside it (mirroring humbaba's multi-plugin bundle).
//
// (M4: the M0 ORT-liveness anchor that once forced a link dependency on the
// bundled ONNX Runtime is gone — the real detector/body/pose-corrective sessions
// in Sam3dBodyPipeline now make libonnxruntime_hastur a genuine dependency.)
#pragma once

#include "ofxsImageEffect.h"

namespace hasturreg {

// TODO(stages): declare append<Name>(OFX::PluginFactoryArray&) here for any extra
// effects that should ship in the same bundle (e.g. a mesh/overlay companion plugin).

// Best-effort user messages. DaVinci Resolve's render context does not support the
// OFX persistent-message suite (MessageSuiteV2); calling set/clearPersistentMessage
// there THROWS, which unwinds out of the render action and the host reports the whole
// render as kOfxStatErrUnsupported. Wrap the calls so a missing/unsupported message
// suite can never fail a render (the message is simply dropped on such hosts).
inline void SafeSetMessage(OFX::ImageEffect& e, OFX::Message::MessageTypeEnum type,
                           const std::string& id, const std::string& msg) {
  try { e.setPersistentMessage(type, id, msg); } catch (...) {}
}
inline void SafeClearMessage(OFX::ImageEffect& e) {
  try { e.clearPersistentMessage(); } catch (...) {}
}
}  // namespace hasturreg
