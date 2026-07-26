# External person tracks

Hastur can be driven by **external per-frame person tracks** — bounding boxes with
stable ids (and optional instance masks) produced by an upstream tracker such as a
SAM 3 video predictor — instead of running its own detector and geometric (cam_t)
association.

## Why

The internal detector + `AssignTrackIds()` gives stable `person_NN` ids only when a
single plugin instance sees frames **in order** and holds its track table between
renders. That breaks in clone-per-frame hosts and swaps ids on fast or crossing
subjects. With external tracks the ids arrive **pre-assigned per frame**, so:

- `person_NN` == the tracker's id — **stable by construction**, no swaps;
- every frame is **self-contained** — no in-order or stateful/sequential render is
  required (works in any host, including render-clone hosts);
- the per-person **MHR body pose** is consistent, because each stable box drives the
  pose under a fixed id;
- coverage can be the tracker's **matte-quality instance mask** (set
  `cryptoCoverage = SAM 3 mask`), falling back to the mesh silhouette when a track
  has no mask.

The internal detector path is unchanged and remains the default; external tracks
engage automatically when a sidecar is found.

## Enabling

Set the **Person tracks** param (`tracksDir`) to a directory or file holding the
sidecar, or export `HASTUR_TRACKS` (path-list; `:`-separated on POSIX, `;` on
Windows). When a sidecar resolves for the frame, the detector and cam_t association
are bypassed. The param value is used verbatim — unlike *Model dir*, there is no
bundle-`Resources` fallback (tracks are shot-specific, not shipped with the plugin).

## Sidecar format

Plain text, dependency-free. One whitespace-separated record per line; `#` begins a
comment. Coordinates are **full-frame pixels, xyxy, top-down** (y = 0 at the top —
the orientation Hastur's image buffer already uses; OFX is natively bottom-up, so a
writer must flip):

```
# hastur-tracks v1
# ids 0 1 2 5 7                              <- optional: every id in the clip
# frame track_id x0 y0 x1 y1 [mask_relpath]
1 0 320.5 88 512 940 masks/0001_00.msk
1 7 700 120 880 900
2 0 322 90 514 942 masks/0002_00.msk
```

- **frame** — integer, matched to `lround(pipeline time)` == the plate frame number.
- **track_id** — the tracker's stable id; becomes `person_NN` directly (may be
  sparse/large, e.g. `7` → `person_07`).
- **x0 y0 x1 y1** — box in full-frame top-down pixels.
- **mask_relpath** — optional, relative to the sidecar's directory (see below).

Malformed rows are skipped. A frame with no rows renders as a pass-through (empty)
frame, exactly like "no persons detected".

### `# ids` header

The optional `# ids ...` line declares every track id that appears anywhere in the
clip. It is used to bake a **complete Cryptomatte manifest** (via *Bake camera
data*) so a decoder on disk resolves every `person_NN`, even on frames where an id
is absent. Without it, the manifest falls back to the per-frame ordinal range.

### Instance masks (`.msk`)

A compact little-endian binary, so the plugin needs no image decoder:

| field | type | notes |
|---|---|---|
| magic | 4 bytes | ASCII `MSK1` |
| w, h  | int32 ×2 | little-endian |
| data  | `w*h` uint8 | 0..255 coverage, row-major |

The mask is a **full-frame stretched** coverage (any resolution covering the whole
frame). It is consumed as a `DetMask` — the same path a detector instance mask
takes (`CleanDetMask` floor + enclosed-hole fill + edge gain, then resampled to
frame resolution) — feeding the `SAM 3 mask` / `Both` Cryptomatte coverage modes.

The SAM 3 Tracker node writes the mask at the tracker's **high-resolution**
`pred_masks_high_res` output (1008², the encoder input size) as soft `sigmoid`
coverage — not the 288² low-res logits used internally for association. This is a
~3.5× linear-resolution matte with only ~1.9× upscaling to a 1080p plate.

## On-disk layouts

Either works; the first that resolves wins per entry:

1. **Single file** `tracks.txt` holding all frames (primary — one tracker pass
   writes it once). A directory param resolves `tracks.txt` inside it.
2. **Per-frame files** via a printf pattern `tracks.%04d.txt` (an entry containing
   `%`, or found inside a directory). Good for very long clips; each per-frame file
   may repeat the `# ids` header.

## Offline check

```
pipeline_smoke <frame.png> <out.png> <model_search_path> <tracks_path>
# HASTUR_SMOKE_FRAME=<n> selects the frame row (default 0); SAM 3 mask coverage,
# no downsize (external boxes are full-res).
```

Render the same clip **out of order** and confirm `person_NN` is identical per human
— that proves the ids need no in-order render. `external_tracks_validate` covers the
parser, `.msk` decode, frame-keying, coordinate order, and the `%04d` layout.
