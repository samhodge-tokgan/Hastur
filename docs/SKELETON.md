# MHR skeleton data

Hastur can surface the per-person MHR skeleton — 3D joints, their 2D screen
projections, and the **per-joint rig transforms** — as a structured, serializable
record keyed by the stable `person_NN` track id. Everything is already computed by
the pipeline; this is serialization, not new geometry.

`person_NN` is the single spine of the pipeline: the same id binds the refined
matte (Cryptomatte / Matte Partition node) and the skeleton here, so downstream
gets **{matte + identity + pose}** per person.

## What a joint carries

Per person, per frame (`SkeletonPerson` in `src/Skeleton.h`), all in the **MHR
camera frame** (metres, the `[1,2]` flip applied — see `MeshTypes.h`):

| field | shape | notes |
|---|---|---|
| `track_id` | int | `person_NN` identity (−1 = unassigned) |
| `joints3d` | `kNumJoints`×3 | posed joint positions (m) |
| `joints2d` | `kNumJoints`×2 | screen px (top-down), projected per person |
| `joint_xforms` | `kNumJoints`×8 | **the rig**: `[tx,ty,tz, qx,qy,qz,qw, s]` global posed transform per joint |
| `keypoints3d` | `kNumKeypoints`×3 | regressed mhr70 keypoints (m) |
| `pose` | `kParamDim` (519) | the MHR-head parameter block `[global6d\|body_cont\|shape\|scale\|hands\|expr]` |
| `cam` | — | `focal`, `(cx,cy)`, `cam_t` used for the 2D projection |

Plus a **static** `hierarchy.joint_parents` (`kNumJoints`, root = −1) emitted once —
with the transforms this is a full **skeleton rig**, not just a point cloud.

`joint_xforms` translation always equals `joints3d`; the quaternion is unit and in
the same flipped camera frame as the positions (it is post-multiplied by the
180°-about-X flip so orientation matches the flipped joint positions).

## 2D projection

A joint projects to screen pixels with the person's own camera placement:

```
P = joint3d + cam_t
u = focal * P.x / P.z + cx
v = focal * P.y / P.z + cy
```

Joints behind the camera (`P.z <= 0`) are emitted as `(0,0)`. `ProjectJoints()` in
`src/Skeleton.cpp` does exactly this.

## Serialization

- **In-memory** — `Mesh.joint_xforms` on the pipeline result + the `SkeletonFrame`
  struct.
- **JSON** — `SkeletonToJson()` emits one compact object per frame (hand-rolled; no
  JSON lib in-repo, mirroring the Cryptomatte manifest). Flat float arrays with the
  strides above.
- (Planned) a compact binary sidecar (mirror `mhr_binfmt`) and EXR metadata
  attributes baked like the Cryptomatte manifest.

## Try it

```
HASTUR_SMOKE_SKELETON=out.skel.json HASTUR_SMOKE_FRAME=181 \
  pipeline_smoke frame.png out.png <model_search_path> [tracks_path]
```

writes `out.skel.json` for the frame. `skeleton_validate` unit-tests the projection,
extraction, xform consistency, and JSON well-formedness (no models needed).
