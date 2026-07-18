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
- silhouette IoU (ref proj vs C++ render) = 0.9745; same-rasterizer geom IoU = 0.9808

ViTDet-H box: [729.3, 109.7, 1529.2, 1453.1]  (C++ ssdlite box above)

### (b) END-TO-END -- each with its own detector
- pred[519]: max_abs = 5.5889e-02, Pearson = 0.999993
- verts per-vertex RMSE = 7.8120 mm (mean 7.2830 mm, max 18.3029 mm)
- cam_t rel err = 9.0096e-04  (cpp [0.1375, 0.8718, 3.8479] vs ref [0.141, 0.8719, 3.8471])
- focal rel err = 0.0000e+00  (cpp 2704.16 vs ref 2704.16)
- keypoint 2D reproj err = 2.859 px mean (median 2.826, max 7.203)
- silhouette IoU (ref proj vs C++ render) = 0.9615; same-rasterizer geom IoU = 0.9655

---

## sample3_yoga  (1132x750)

C++ detector box: [427.7, 155.7, 847.8, 706.6] (score 0.997)

### (a) FIXED-BBOX -- pipeline parity (same box to both)
- pred[519]: max_abs = 1.8550e-02, Pearson = 1.000000
- verts per-vertex RMSE = 3.3074 mm (mean 2.9902 mm, max 6.4687 mm)
- cam_t rel err = 8.2786e-04  (cpp [0.0873, 1.2691, 2.8929] vs ref [0.087, 1.2693, 2.8903])
- focal rel err = 0.0000e+00  (cpp 1357.91 vs ref 1357.91)
- keypoint 2D reproj err = 1.223 px mean (median 1.255, max 2.049)
- silhouette IoU (ref proj vs C++ render) = 0.9732; same-rasterizer geom IoU = 0.9851

ViTDet-H box: [438.3, 169.9, 847.7, 703.6]  (C++ ssdlite box above)

### (b) END-TO-END -- each with its own detector
- pred[519]: max_abs = 1.0534e-01, Pearson = 0.999986
- verts per-vertex RMSE = 32.2676 mm (mean 20.6274 mm, max 83.9602 mm)
- cam_t rel err = 9.7028e-04  (cpp [0.0873, 1.2691, 2.8929] vs ref [0.086, 1.2717, 2.8937])
- focal rel err = 0.0000e+00  (cpp 1357.91 vs ref 1357.91)
- keypoint 2D reproj err = 13.709 px mean (median 15.189, max 28.127)
- silhouette IoU (ref proj vs C++ render) = 0.9624; same-rasterizer geom IoU = 0.9677

---

## sample1  (750x764)

C++ detector box: [22.5, 14.9, 655.7, 591.7] (score 0.922)

### (a) FIXED-BBOX -- pipeline parity (same box to both)
- pred[519]: max_abs = 1.6386e-02, Pearson = 1.000000
- verts per-vertex RMSE = 1.4449 mm (mean 1.1073 mm, max 4.7201 mm)
- cam_t rel err = 1.8087e-03  (cpp [0.2367, 1.0447, 1.9824] vs ref [0.2365, 1.0452, 1.9784])
- focal rel err = 0.0000e+00  (cpp 1070.61 vs ref 1070.61)
- keypoint 2D reproj err = 0.774 px mean (median 0.656, max 1.612)
- silhouette IoU (ref proj vs C++ render) = 0.9818; same-rasterizer geom IoU = 0.9880

ViTDet-H box: [14.8, 16.0, 681.8, 611.2]  (C++ ssdlite box above)

### (b) END-TO-END -- each with its own detector
- pred[519]: max_abs = 1.9585e-01, Pearson = 0.999963
- verts per-vertex RMSE = 27.7225 mm (mean 19.4231 mm, max 73.1413 mm)
- cam_t rel err = 3.7010e-03  (cpp [0.2367, 1.0447, 1.9824] vs ref [0.232, 1.0499, 1.978])
- focal rel err = 0.0000e+00  (cpp 1070.61 vs ref 1070.61)
- keypoint 2D reproj err = 9.853 px mean (median 6.933, max 24.291)
- silhouette IoU (ref proj vs C++ render) = 0.9589; same-rasterizer geom IoU = 0.9638
