# ImageNette Teacher entropy feature maps

These figures were generated on cloud node `39729` directly from the frozen experiment data under:

`/linxi/dataset/CV-DD/experiments/imagenette_entropy_selection_v1`

Inputs:

- DINO cache: `cache/imagenette_dinov2.pt`
- Frozen per-image selection entropy: `preflight/per_image_teacher_entropy_and_dino.json`
- Recomputed 16-view mean probabilities: `cache/imagenette_calibration_mean_probabilities.pt`
- Source images: official ImageNette train split

Definitions:

- Point color is the exact selection statistic: the mean of 16 per-view Teacher entropies divided by `log(10)`.
- A red outline means at least one of the 16 calibration views was predicted incorrectly.
- For ordinary representative strips, `pred` and `p` come from the probability vector averaged over all 16 views.
- For the highest-entropy-error strip, `pred` and `p` describe the highest-entropy incorrect calibration view.
- `correct=x/16` reports how many calibration views were classified correctly.

Files:

- `A1_global_dino_pca_entropy_map.{png,pdf}`: one global PCA fitted over all L2-normalized DINO features, then faceted by true class.
- `A2_per_class_dino_pca_entropy_map.{png,pdf}`: a separate deterministic PCA per class; directions and coordinates are not comparable between panels.
- `B_class_entropy_histogram_heatmap.{png,pdf}`: within-class proportions in entropy bins of width 0.05; short black/white marks show class medians.
- `C_representative_strips_overview.{png,pdf}`: all classes and four representative groups.
- `C_class_*.{png,pdf}`: readable per-class representative strips.
- `pca_coordinates_and_entropy.npz`: plotted coordinates, entropy, predictions, correctness, and PCA metadata.
- `figure_metadata.json`: input/output hashes, explained variance, class medians, error counts, and exact representative-image paths.

