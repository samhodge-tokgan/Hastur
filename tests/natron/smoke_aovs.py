# Headless Natron smoke for the SAM 3D Body OFX plugin + multi-plane AOVs.
#
# License-free in-host verification: confirms the plugin loads, declares its AOV
# planes via the Foundry/Natron multi-plane extension, and (optionally) renders
# the full pipeline end-to-end. Run with an arm64 Natron matching the plugin arch:
#
#   OFX_PLUGIN_PATH=/path/to/build \
#   HASTUR_MODEL_DIR=/path/to/models \
#   HASTUR_INPUT=/path/to/person.png \
#   NatronRenderer --clear-openfx-cache -t tests/natron/smoke_aovs.py
#
# Optional: HASTUR_RENDER=1 also renders the beauty pass to $HASTUR_OUTPUT.
# Exit 0 = the expected AOV planes were all declared; non-zero otherwise.
import os, sys
def log(*a): print("[AOV-SMOKE]", *a); sys.stdout.flush()

MODELS = os.environ.get("HASTUR_MODEL_DIR", "")
INPUT = os.environ.get("HASTUR_INPUT", "")
OUTPUT = os.environ.get("HASTUR_OUTPUT", "/tmp/hastur_beauty.exr")
PLUGIN_ID = "com.tokgan.Sam3dBody"
EXPECTED = {"Depth", "Position", "Normal", "Pref", "ST",
            "CryptoObject00", "CryptoObject01"}

def layer_name(L):
    for m in ("getLayerName", "getName"):
        f = getattr(L, m, None)
        if f:
            try: return f()
            except Exception: pass
    return str(L)

try:
    import NatronEngine
    natron = NatronEngine.natron
    if PLUGIN_ID not in natron.getPluginIDs():
        log("FAIL: plugin", PLUGIN_ID, "not registered"); os._exit(3)
    try: app = app1  # noqa: F821  (bound by NatronRenderer)
    except NameError: app = natron.getInstance(0)

    node = app.createNode(PLUGIN_ID)
    if MODELS:
        try: node.getParam("modelDir").setValue(MODELS)
        except Exception as e: log("modelDir warn:", e)
    if INPUT:
        node.connectInput(0, app.createReader(INPUT))

    names = {layer_name(L) for L in node.getAvailableLayers(-1)}
    log("declared layers:", sorted(names))
    missing = EXPECTED - names
    if missing:
        log("FAIL: missing AOV planes:", sorted(missing)); os._exit(4)
    log("PASS: all AOV planes declared")

    if os.environ.get("HASTUR_RENDER") == "1" and INPUT:
        w = app.createWriter(OUTPUT); w.connectInput(0, node)
        app.render(w, 1, 1)
        ok = os.path.exists(OUTPUT) and os.path.getsize(OUTPUT) > 0
        log("beauty render:", "OK" if ok else "MISSING", OUTPUT)
        if not ok: os._exit(5)
    os._exit(0)
except Exception as e:
    import traceback; log("FAIL:", e); traceback.print_exc(); os._exit(2)
