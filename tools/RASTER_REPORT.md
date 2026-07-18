# SoftwareRasterizer validation (Track D / M6)

C++ CPU rasterizer that renders a posed MHR human mesh to neutral-grey RGBA with a
coverage alpha at input-frame resolution — the C++ equivalent of the reference
Python pyrender path (`visualization/renderer.py`,
`Renderer.__call__(..., return_rgba=True)`).

## Results

`tests/raster_validate.cpp` vs the reference fixtures
(`test-assets/raster_fixtures.bin`, 3 posed meshes, 36 874 tris each):

| fix | res      | ss1 alpha IoU | ss2 alpha IoU | shaded-L1 (both-covered) | both px |
|-----|----------|---------------|---------------|--------------------------|---------|
| 0   | 512×640  | 1.00000       | 0.98990       | 0.000000                 | 29 902  |
| 1   | 640×480  | 1.00000       | 0.98623       | 0.000000                 | 14 237  |
| 2   | 480×720  | 1.00000       | 0.99073       | 0.000000                 | 36 012  |

- **WORST ss1 alpha IoU = 1.00000**, **WORST shaded-L1 = 0.000000** → the C++ port
  reproduces the reference rasterizer's silhouette and shading to float precision.
- **premultiplied-alpha invariant OK**: `premult.rgb == straight.rgb * alpha` to
  < 1e-5.
- **ss2 alpha IoU ≈ 0.986–0.991** is *not* error: at `ssaa=2` the silhouette is
  anti-aliased (fractional edge alpha), and thresholding that soft edge at 0.5
  against the reference's crisp 1-px edge accounts for the whole difference. The
  ss=2 render is the intended production output (clean silhouettes); ss=1 is the
  bit-exact apples-to-apples check against the reference.

Reproduce:

```
tools/dump_test_meshes.py                       # regenerate fixtures (numpy ref)
clang++ -std=c++17 -O2 -I src tests/raster_validate.cpp \
        src/SoftwareRasterizer.cpp src/MeshAssets.cpp -o build/raster_validate
./build/raster_validate test-assets/raster_fixtures.bin
```

## How the reference was generated

**pyrender was NOT usable.** pyrender needs a headless OpenGL context, and on this
Apple-silicon Mac every backend fails to load:

- `PYOPENGL_PLATFORM=osmesa` → `Unable to load OpenGL library 'OSMesa'` (no OSMesa
  installed).
- `egl` → `Unable to load EGL library` (macOS has no EGL).
- `pyglet` → `Unsupported PyOpenGL platform: pyglet` for `OffscreenRenderer`.

Per the M6 brief this falls back to **geometric + independent-oracle validation**:
`tools/dump_test_meshes.py` contains a **pure-NumPy software rasterizer** that
implements the *same spec* the C++ targets (CV pinhole projection, z-buffer,
perspective-correct barycentric interpolation, per-vertex-normal Lambert
neutral-grey shading, coverage alpha). The C++ port is validated against it. The
geometrically meaningful, implementation-independent check is the **coverage / IoU**
(any correct rasterizer must reproduce the projected silhouette); shaded-L1
additionally confirms the shading math is ported correctly. The reference render was
also eyeballed — an upright, correctly-oriented clay human (confirming the 180°-X /
`cam_t[0]` reconciliation below) — and the C++ ss=2 output is visually identical.

Mesh source: the M2 mesh-oracle fixtures (`test-assets/fixtures/fix*.npz`, already in
metres in the MHR camera frame); faces from `test-assets/mhr_assets.bin`. The
cameras are **synthesised** to frame each mesh (these zero/random poses carry no real
camera solve) — the rasterizer contract only consumes `(focal, center, cam_t)`.

## Coordinate / sign reconciliation (the important caveat)

`renderer.py` uses pyrender (OpenGL): it places the mesh at `verts`, applies a **180°
rotation about X** (`diag(1,-1,-1)`), sets `camera_translation = cam_t.copy();
camera_translation[0] *= -1`, and views from the origin with an `IntrinsicsCamera`.
Those three things exist purely to turn the OpenGL camera math back into the
**standard CV pinhole projection** the pipeline's `perspective_projection` uses.
Working it through, an OpenGL-camera point of a vertex is
`(vx+cam_t_x, -vy-cam_t_y, -vz-cam_t_z)`, and the pyrender projection of that is
identical to the plain CV projection the C++ implements directly:

```
v_cam = verts + cam_t          # camera at origin, +z forward, y down
u = focal * v_cam.x / v_cam.z + center.x
v = focal * v_cam.y / v_cam.z + center.y
depth = v_cam.z                 # metres, >0 in front
```

So the C++ rasterizer does **not** re-apply the 180°-X flip or negate `cam_t[0]`; it
consumes `Camera{focal, center, cam_t}` and the frame-space `Mesh.verts` from the
contract as-is. This was confirmed visually (the rendered human is upright and
right-way-round, not mirrored or upside-down).

## Other caveats

- **Color space: LINEAR.** RGB is `grey * lightscalar` in linear light with no gamma
  / sRGB encoding. The base grey is a linear value (default 0.6 — a clay mid-grey;
  the reference demo default is ivory `(1,1,0.9)`, parameterised via
  `RasterOptions::grey`). **The OFX plugin owns color management / the display
  transform** — treat this output as linear scene-referred grey.

- **Alpha convention.** Output alpha is **mesh coverage** (1 inside, 0 outside,
  fractional on anti-aliased edges), returned as grey-on-transparent. Both
  **straight** and **premultiplied** are supported via `RasterOptions::premultiply`
  (OFX hosts default to premultiplied). Straight color is the coverage-weighted
  average over covered subsamples; premultiplied color is additionally scaled by
  alpha (invariant verified in the test).

- **Lighting is an approximation, by design.** Shading is a view-space Lambert with
  a 3-point rig derived from `create_raymond_lights` (θ=π/6; φ=0,2π/3,4π/3) plus
  ambient 0.3, rather than pyrender's exact PBR + 3 directional lights. The brief
  explicitly favours "a correct, well-tested rasterizer over matching pyrender's
  exact lighting — the neutral-grey clay look + correct silhouette/alpha at input
  resolution is what matters." A **view-stable matcap** is available as an
  alternative (`RasterOptions::matcap`). The reference and C++ use the *same* shading
  spec, so shaded-L1 = 0; the lighting is not claimed to be pyrender-identical.

- **Double-sided shading.** Face winding in the MHR topology is not relied upon;
  per-pixel normals are oriented toward the camera, so back-facing triangles at a
  silhouette still shade correctly (no black holes). Backface culling is therefore
  off by default.

- **Anti-aliasing.** Silhouette AA is edge SSAA (`ssaa=2` → 4× subsamples,
  box-downsampled). No AA inside the mesh is needed since it is a single solid
  color-per-normal surface; SSAA only cleans the coverage boundary.

- **Clipping.** A simple near-plane reject drops any triangle with a vertex at
  `depth <= near_z` (default 1e-3 m). Full frustum clipping is not implemented —
  fine for framed human meshes several metres in front of the camera; revisit if a
  mesh can straddle the near plane.

## Files

- `src/SoftwareRasterizer.h` / `src/SoftwareRasterizer.cpp` — the rasterizer
  (`RgbaImage Render(const Mesh&, const Camera&, int W, int H, const RasterOptions&)`),
  std-only (no Eigen / ORT / OFX).
- `tools/dump_test_meshes.py` — fixture + reference generator (NumPy oracle).
- `tests/raster_validate.cpp` — standalone validator (built with clang++ above).
- `test-assets/raster_fixtures.bin` — packed (verts, faces, cam, reference RGBA).
- `test-assets/fixtures/raster0{0,1,2}.npz` — inspectable per-fixture dumps.

> Not wired into CMake (engine/rasterizer CMake is deferred to M4); build the test
> standalone as shown. Fixtures are not committed by this task.
