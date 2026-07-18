# Track F geometry validation report

Status: PASS (tolerance 0.0001, 6 cases)

C++ (`src/CameraSolver.{h,cpp}`, `src/CropAffine.{h,cpp}`) vs the
sam-3d-body reference (@ c259bfc). Reference runs on-device in fp16; we
validate the fp32/fp64 mathematical result the ONNX pipeline feeds.

| quantity | reference | max abs diff |
|---|---|---|
| GetBBoxCenterScale center | bbox_xyxy2cs | 0.000e+00 |
| GetBBoxCenterScale scale (padded) | bbox_xyxy2cs | 0.000e+00 |
| TopdownAffine square side | fix_aspect_ratio x2 | 8.133e-05 |
| TopdownAffine affine (forward) | get_warp_matrix | 3.740e-06 |
| TopdownAffine affine (inverse) | get_warp_matrix inv | 2.344e-08 |
| condition_info (CLIFF, intrin center) | _get_decoder_condition | 9.302e-08 |
| ray_cond [2,32,32] | get_ray_condition + CameraEncoder 1/16 | 4.867e-08 |
| perspective_projection (focal, cam_t) | camera_head | 5.790e-07 |
| crop warp (ImageNet CHW, 16x16x3 samples) | inverse affine + bilinear | 5.779e-07 |

