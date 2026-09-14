# Pinned teacher backbones

These source trees are vendored snapshots used by the Aircraft teacher-labeler
experiment. They intentionally do not contain nested `.git` directories; the
top-level repository is the sole source of code deployed to compute nodes.

| Directory | Upstream | Pinned commit |
|---|---|---|
| `ViT-pytorch` | `https://github.com/jeonsworld/ViT-pytorch` | `460a162767de1722a014ed2261463dbbc01196b6` |
| `TransFG` | `https://github.com/TACJu/TransFG` | `9336fba46a4ac8ed2e33072c5f74ab459b114e4a` |

The original license and README files remain inside each snapshot. Runtime code
checks each `PINNED_COMMIT` marker before constructing a model.
