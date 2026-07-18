# M5 -- C++ Sam3dBodyPipeline vs Python SAM-3D-Body reference (numeric parity)


---

## dancing  (2250x1500)

C++ detector box: [762.0, 134.8, 1516.8, 1441.0] (score 0.983)

### (a) FIXED-BBOX -- pipeline parity (same box to both)
- pred[519]: max_abs = 1.7142e-02, Pearson = 0.999999
- verts per-vertex RMSE = 1.9901 mm (mean 1.6208 mm, max 6.9077 mm)
- cam_t rel err = 8.9115e-04  (cpp [0.1375, 0.8718, 3.8479] vs ref [0.1378, 0.8719, 3.8444])
- focal rel err = 0.0000e+00  (cpp 2704.16 vs ref 2704.16)
- keypoint 2D reproj err = 1.039 px mean (median 1.092, max 3.236)
- silhouette IoU (ref proj vs C++ render) = 0.5755; same-rasterizer geom IoU = 0.6204

(b) END-TO-END: ViTDet detector unavailable; skipped.
