# Copyright the Hastur authors.
# SPDX-License-Identifier: LicenseRef-SAM-License
#
# seed_matte_graph.py -- Natron -t graph that turns an RGB plate into a temporally-stable per-person
# matte using the REPAIRED (inter-limb-open) seed. It is the durable replacement for the Python
# crop_refine.sh seed hackery: the void fix now lives in the Hastur C++ (confidence-aware binarise in
# Sam3MultiTracker::Detect/PropagateObj + hole-preserving CleanDetMask + adaptively-eroded MHR union
# in Sam3dBody crypto_coverage=Both), so this graph is mostly clean wiring, no intermediate mask files.
#
#   Read(plate) --> Sam3Tracker  (writes tracks.txt + masks/*.msk sidecar; masks now confidence-open)
#               --> Sam3dBody    (cryptoCoverage=Both -> REPAIRED coverage: eroded-MHR UNION open-mask;
#                                  outputAov=Matte -> its RGBA output IS the soft seed matte;
#                                  Cryptomatte planes + skeleton_%04d.json also emitted)
#   Kthanid  (Source=Read, Seed=Sam3dBody)  --> pooled soft matte (MatAnyone2 seeded from the repaired
#                                                coverage node-to-node -- no disk round-trip for the seed)
#   MattePartition (Crypto=Sam3dBody, Pool=Kthanid) --> per-person Cryptomatte EXR on disk
#
# Tuning without a rebuild (env, read by the C++): HASTUR_CONFOPEN_TAU, HASTUR_CONFOPEN_MINFRAC,
# HASTUR_CONFOPEN_EDGE, HASTUR_MHR_ERODEFRAC, HASTUR_MASK_HOLEFRAC, HASTUR_CONFOPEN_OFF (A/B),
# HASTUR_MASK_NOCLEAN.
#
# Env inputs:
#   OFX_PLUGIN_PATH   dirs with Sam3dBody.ofx (Sam3Tracker + Sam3dBody + MattePartition) and Kthanid.ofx
#   ROT_RGB           printf plate path, e.g. /work/plate_%04d.png (or .exr)
#   ROT_FIRST/ROT_LAST  frame range (ints)
#   ROT_WORK          intermediates dir (default dirname(ROT_RGB)/seed_matte_work)
#   TRACKER_MODEL / HASTUR_MODEL / KTHANID_MODEL   the three model dirs
#   ROT_CRYPTO_EXR    output per-person Cryptomatte EXR printf (default ROT_WORK/crypto_%04d.exr)
#   ROT_TRIMAP        if set to "1", also write a 3-class trimap EXR (for an XMem/Cutie/trimap consumer)
import os
import sys


def _app():
    try:
        return app1  # noqa: F821  (Natron injects this in -t mode)
    except NameError:
        import NatronEngine  # noqa: F401
        return natron.getInstance(0)  # noqa: F821


app = _app()

RGB = os.environ.get("ROT_RGB", "")
if not RGB:
    print("FAIL: set ROT_RGB to the plate frame sequence (printf path).")
    sys.exit(1)
FIRST = int(os.environ.get("ROT_FIRST", "1"))
LAST = int(os.environ.get("ROT_LAST", "1"))
WORK = os.environ.get("ROT_WORK", os.path.join(os.path.dirname(RGB) or ".", "seed_matte_work"))
os.makedirs(WORK, exist_ok=True)
TRACKS_DIR = os.path.join(WORK, "tracks")
os.makedirs(TRACKS_DIR, exist_ok=True)
CRYPTO_EXR = os.environ.get("ROT_CRYPTO_EXR", os.path.join(WORK, "crypto_%04d.exr"))
TRACKER_MODEL = os.environ.get("TRACKER_MODEL", "")
HASTUR_MODEL = os.environ.get("HASTUR_MODEL", "")
KTHANID_MODEL = os.environ.get("KTHANID_MODEL", "")


def make(plugin_id):
    try:
        return app.createNode(plugin_id)
    except Exception as e:
        print("WARN createNode(%s): %s" % (plugin_id, e))
        return None


def setp(node, name, value):
    try:
        node.getParam(name).setValue(value)
    except Exception as e:
        print("WARN set %s.%s=%r: %s" % (getattr(node, "getScriptName", lambda: "?")(), name, value, e))


def reader(path):
    r = app.createReader(path)
    # pass the sRGB plate through un-linearised, else the detector sees darkened pixels.
    for p in ("ocioInputSpace", "ocioOutputSpace"):
        try:
            pass
        except Exception:
            pass
    try:
        r.getParam("ocioOutputSpace").setValue(r.getParam("ocioInputSpace").getValue())
    except Exception:
        pass
    return r


def render(node, first, last):
    try:
        app.render(node, first, last)
    except Exception:
        app.render([(node, first, last)])


# --- Stage 1: SAM3 tracker -> confidence-open sidecar (tracks.txt + masks/*.msk) ---------------------
trk = make("com.tokgan.Sam3Tracker")
if trk is not None:
    r = reader(RGB)
    if TRACKER_MODEL:
        setp(trk, "modelDir", TRACKER_MODEL)
    setp(trk, "tracksOutputDir", TRACKS_DIR)
    setp(trk, "bidirectional", True)
    w = app.createWriter(os.path.join(WORK, "trk_passthru_%04d.png"))
    w.connectInput(0, trk)
    trk.connectInput(0, r)
    render(w, FIRST, LAST)
    print("stage1 tracker -> %s/{tracks.txt,masks}" % TRACKS_DIR)
else:
    print("stage1 tracker SKIPPED (Sam3Tracker missing); expecting %s to exist" % TRACKS_DIR)

# --- Stage 2: Sam3dBody -> REPAIRED coverage (eroded-MHR union open-mask); Matte = the seed ----------
sam = make("com.tokgan.Sam3dBody")
if sam is None:
    print("FAIL: Sam3dBody plugin not found (set OFX_PLUGIN_PATH).")
    sys.exit(1)
r2 = reader(RGB)
sam.connectInput(0, r2)
if HASTUR_MODEL:
    setp(sam, "modelDir", HASTUR_MODEL)
setp(sam, "tracksDir", TRACKS_DIR)          # ingest the tracker sidecar
setp(sam, "skeletonDir", WORK)
setp(sam, "cryptoCoverage", 2)              # 2 = Both -> eroded-MHR UNION confidence-open mask
setp(sam, "outputAov", "hastur.Matte")      # main RGBA output = the soft seed matte (for Kthanid)

# --- Stage 3: Kthanid (MatAnyone2) seeded from the repaired coverage, node-to-node -------------------
kth = make("com.tokgan.openfx.Kthanid")
seed_source = sam
if kth is not None:
    r3 = reader(RGB)
    kth.connectInput(0, r3)                 # Source = plate
    kth.connectInput(1, sam)                # Seed = Sam3dBody repaired Matte (the whole point)
    if KTHANID_MODEL:
        setp(kth, "modelDir", KTHANID_MODEL)
    setp(kth, "execProvider", 0)            # 0 = Auto/GPU
    pooled = kth
else:
    print("stage3 Kthanid SKIPPED (not found); MattePartition will use the Sam3dBody coverage as pool")
    pooled = sam

# --- Stage 4: MattePartition -> per-person Cryptomatte EXR on disk -----------------------------------
mp = make("com.tokgan.MattePartition")
if mp is not None:
    mp.connectInput(0, sam)                 # Cryptomatte (multi-plane) from Sam3dBody
    mp.connectInput(1, pooled)              # Pool = MatAnyone2 refined alpha (or Sam3dBody if no Kthanid)
    setp(mp, "cryptoExrPath", CRYPTO_EXR)
    w4 = app.createWriter(os.path.join(WORK, "mp_passthru_%04d.png"))
    w4.connectInput(0, mp)
    render(w4, FIRST, LAST)
    print("stage4 per-person Cryptomatte -> %s" % CRYPTO_EXR)
else:
    print("stage4 MattePartition SKIPPED; rendering the pooled matte to %s/matte_%%04d.exr" % WORK)
    w4 = app.createWriter(os.path.join(WORK, "matte_%04d.exr"))
    w4.connectInput(0, pooled)
    render(w4, FIRST, LAST)

# --- Optional trimap branch (net-new; for an XMem/Cutie/trimap consumer other than Kthanid) ----------
# FG = erode(matte); BG = 1 - dilate(matte); unknown = the band between. Built from Natron built-ins.
if os.environ.get("ROT_TRIMAP") == "1":
    ero = make("net.sf.cimg.CImgErode")
    dil = make("net.sf.cimg.CImgDilate")
    if ero is not None and dil is not None:
        ero.connectInput(0, seed_source); setp(ero, "size", 6)      # definite FG
        dil.connectInput(0, seed_source); setp(dil, "size", 6)      # maybe-FG (complement = definite BG)
        # FG(1.0) where eroded, unknown(0.5) in the dilate\erode band, BG(0.0) elsewhere:
        # KeyMix/Merge the two into a 3-level map, then Grade to {0,0.5,1}. Wire + tune in the host;
        # this scaffolds the nodes. Write the trimap EXR.
        wt = app.createWriter(os.path.join(WORK, "trimap_%04d.exr"))
        wt.connectInput(0, dil)             # placeholder pipe; refine the 3-level combine interactively
        render(wt, FIRST, LAST)
        print("stage5 trimap scaffold -> %s/trimap_%%04d.exr (refine the FG/unknown/BG combine in-host)")
    else:
        print("stage5 trimap SKIPPED (CImg Erode/Dilate not found)")

print("seed_matte_graph done -> %s" % WORK)
