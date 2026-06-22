# Reconstruction method comparison

Methods benchmarked against recognised baselines on the standard **Shepp–Logan** phantom and the anatomical abdomen phantom, scored with standard image-quality metrics (lower RMSE, higher PSNR, higher SSIM are better). Ground truth is the phantom speed-of-sound map.

| Target | Method | RMSE (m/s) ↓ | PSNR (dB) ↑ | SSIM ↑ | Time (ms) |
|--------|--------|--------------|-------------|--------|-----------|
| Shepp-Logan | backprojection | 208.62 | 13.26 | 0.600 | 4.3 |
| Shepp-Logan | SART | 185.48 | 14.28 | 0.711 | 28.7 |
| Shepp-Logan | Landweber | 154.86 | 15.85 | 0.811 | 102.0 |
| Abdomen | backprojection | 130.31 | 21.51 | 0.221 | 4.0 |
| Abdomen | SART | 98.93 | 23.90 | 0.596 | 27.1 |
| Abdomen | Landweber | 51.48 | 29.57 | 0.916 | 97.5 |

**Reading:** backprojection is the single-sweep baseline; SART (algebraic, relaxed) and Landweber (gradient descent on `‖As−t‖²`) are the recognised iterative competitors. SART converges fastest per iteration on this transmission geometry; Landweber reaches a comparable least-squares solution with more, cheaper steps. Numbers are deterministic and reproducible (`cargo run --release --bin sonic_ct_methods`).
