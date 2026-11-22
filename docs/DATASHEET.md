# pcpbench datasheet

This datasheet follows *Datasheets for Datasets* (Gebru et al., CACM 64(12), 2021). It
describes the artifact as released with the camera-ready paper (KDD 2027 Datasets and
Benchmarks track). Every count and sample size below is read from the committed per-row
outputs (`outputs/*.json`), the leaderboard (`leaderboard.md`), or the scripts named
beside it. The paper's appendix keeps a short availability paragraph and points here.

## Motivation

**Purpose.** pcpbench measures per-class coverage of conformal prediction under
distribution shift. It fixes one cache format, one protocol, five per-class metrics, and
an efficiency guard, so a new method produces a row that is directly comparable to the
released ones. It proposes no conformal method.

**Gap it fills.** Conformal methods for shift are usually reported on marginal coverage,
on different shifts, and under different protocols. pcpbench scores several methods on
the same cached model outputs and reports worst-class coverage against a matched
finite-sample reference (Section 3 of the paper), not against the nominal level.

## Composition

### Conditions

| Family | Conditions | K | Backbone | Source calibration rows per class | Target pool / evaluation rows per class, per seed |
| --- | --- | --- | --- | --- | --- |
| CIFAR-10-C | 40 (8 corruptions x 5 severities) | 10 | OpenCLIP ViT-B/32 | 5,000 (clean CIFAR-10 train) | 500 / 500 (corrupted CIFAR-10 test) |
| CIFAR-100-C | 40 (8 corruptions x 5 severities) | 100 | OpenCLIP ViT-B/32 | 500 (clean CIFAR-100 train) | 50 / 50 (corrupted CIFAR-100 test) |
| NTU-60 cross-subject | 1 | 60 | CTR-GCN | 325 to 336 | 134 to 138 evaluation rows per class (8,223 pool and 8,226 evaluation rows in total) |
| NTU-60 to N-UCLA | 1 | 5 | CTR-GCN | 324 to 349 | 70 to 150 evaluation rows per class (443 pool and 440 evaluation rows in total) |

The eight corruptions are Gaussian noise, shot noise, defocus blur, Gaussian blur,
contrast, brightness, fog, and JPEG compression.

**Headline.** 492 rows: six methods (split, Source-Mondrian, weighted, PPI, ACP, and the
held-out oracle) over the 82 conditions above.

**Full board** (`leaderboard.md`, outside every headline count):

- the same 80 CIFAR conditions scored by OpenCLIP ViT-L/14 (`laion2b_s32b_b82k`),
  Section 4.1 and Appendix B of the paper;
- ImageNet-1k at K=1000, 10 cells: ViT-B/32 and ViT-L/14 at severity three of Gaussian
  noise, defocus blur, contrast, and fog, plus ViT-B/32 Gaussian noise at severities one
  and five. Source: the clean validation set, 50 images per class. Target: the same
  validation images, corrupted, split 25 / 25 per class into pool and evaluation.
  Section 4.1 and Appendix B.4;
- a seventh, supplementary method, clustered conformal (Ding et al., NeurIPS 2023), on
  every condition. Appendix B.4;
- three legacy CIFAR-100-C caches outside the 8 x 5 grid (an aggregate cache without a
  severity label on each backbone and a mild-severity variant on ViT-B/32), kept for
  provenance. They are outside the headline and the graded grids and enter only the
  counts the paper states over the full board of 175 conditions.

### Instances

One leaderboard row is one (condition, method) pair, averaged over 10 seeds at
alpha = 0.10. Columns of `leaderboard.md`:

| Column | Meaning |
| --- | --- |
| `marginal` | coverage over all evaluation rows |
| `worst_class` | minimum over the K classes of per-class coverage |
| `frac_undercovered` | share of classes with empirical coverage below 0.90 |
| `frac_sig_undercovered` | share of classes whose covered count is significantly low under the beta-binomial law of an exactly valid per-class calibration (see `src/pcpbench/runner.py`); added after submission beside `frac_undercovered` |
| `mean_set_size` | mean prediction-set size |
| `set/K` | mean set size divided by K; the efficiency guard reads it |

The per-row JSON in `outputs/<condition>_<method>.json` adds the per-seed values, the
per-class coverage of every seed, the per-class calibration and evaluation counts, the
score (`lac` or `aps`), alpha, and the seed list.

### Cache schema

Each cache is one `.npz` file read by `pcpbench.datasets.ShiftDataset`
(`src/pcpbench/datasets/shift_dataset.py`):

| Key | Type | Content |
| --- | --- | --- |
| `logits` | float32 [N, K] | class scores of the backbone |
| `labels` | int64 [N] | true class |
| `split` | int8 [N] | role: 0 source train, 1 source calibration, 2 target test |
| `domain` | int32 [N] | 0 source, 1 target |
| `meta` | JSON string | `name`, `shift_family`, `n_classes`, `source` |

The released caches come from `pcpbench.datasets.convert`, which marks every source row
as source calibration and every target row as target test; no released cache uses the
source-train role. The labeled pool and the evaluation set are not stored. For every seed
the runner splits the target-test rows of each class into two disjoint halves: the pool,
whose labels only the target-label methods (the oracle, PPI, and ACP) may read, and the
evaluation half, on which every method is scored.

### Confidentiality and sensitive content

The caches hold class scores, labels, and partition metadata only. They hold no images,
video, or skeleton frames.

## Collection process

**Image caches.** `scripts/build_cifar10c_clip_cache.py`,
`scripts/build_cifar100c_clip_cache.py`, and `scripts/build_imagenetc_clip_cache.py` score
images with a zero-shot OpenCLIP model (prompt `a photo of a <class name>`, the model's
learned logit scale). The headline uses ViT-B/32 `laion2b_s34b_b79k`; the appendix uses
ViT-L/14 `laion2b_s32b_b82k`. Images are read from the Hugging Face datasets
`uoft-cs/cifar10`, `uoft-cs/cifar100`, and `ILSVRC/imagenet-1k` (gated; access must be
granted on the dataset page). No model is fine-tuned.

**Corruptions: read this before comparing with other CIFAR-C results.** The corrupted
images are generated by `src/pcpbench/datasets/corruptions.py`, a pure-Python
implementation (numpy, scipy, PIL) of the Hendrycks and Dietterich (ICLR 2019) suite. It
uses the ImageNet-C parameter tables of the reference implementation (`make_imagenet_c.py`)
and is not bit-identical to the reference package. On CIFAR it applies those tables to the
32 x 32 test images, before CLIP's resize. The released CIFAR-10-C and CIFAR-100-C files
were made with separate tables (`make_cifar_c.py`) that are milder at every severity of all
eight corruptions:

| Corruption | Parameter | pcpbench, severities 1 to 5 | Released CIFAR-C, severities 1 to 5 |
| --- | --- | --- | --- |
| Gaussian noise | noise standard deviation | 0.08, 0.12, 0.18, 0.26, 0.38 | 0.04, 0.06, 0.08, 0.09, 0.10 |
| Shot noise | Poisson rate (lower is noisier) | 60, 25, 12, 5, 3 | 500, 250, 100, 75, 50 |
| Defocus blur | disk radius, pixels | 3, 4, 6, 8, 10 | 0.3, 0.4, 0.5, 1, 1.5 |
| Gaussian blur | sigma, pixels | 1, 2, 3, 4, 6 | 0.4, 0.6, 0.7, 0.8, 1 |
| Contrast | contrast factor (lower is stronger) | 0.4, 0.3, 0.2, 0.1, 0.05 | 0.75, 0.5, 0.4, 0.3, 0.15 |
| Brightness | added value | 0.1, 0.2, 0.3, 0.4, 0.5 | 0.05, 0.1, 0.15, 0.2, 0.3 |
| Fog | fog intensity | 1.5, 2, 2.5, 2.5, 3 | 0.2, 0.5, 0.75, 1, 1.5 |
| JPEG compression | quality (lower is stronger) | 25, 18, 15, 10, 7 | 80, 65, 58, 50, 40 |

So "CIFAR-100-C severity 3" in pcpbench names a stronger shift than severity 3 of the
released files, and coverage values are not comparable across the two. On ImageNet the same
tables are applied to the full-size validation images before CLIP's resize and crop, while
the released ImageNet-C corrupts 224 x 224 center crops, so the size-dependent corruptions
(defocus and Gaussian blur, fog) also differ in relative strength there. The comparisons
inside the paper hold the shift fixed (the controlled sweep varies only K on one cache), so
they do not depend on matching the released files.

**Skeleton caches.** The NTU-60 cross-subject and NTU-60 to N-UCLA caches hold the head
logits of a CTR-GCN classifier trained on NTU-60, converted from the 4-key format of an
earlier action-recognition project with `pcpbench.datasets.convert`. Training that
classifier is not part of this artifact.

**Time frame.** The corruption module last changed on 2026-07-11. The CIFAR 8 x 5 grids
were added from 2026-07-13 on and the ImageNet launch scripts on 2026-07-21. The per-row
outputs were last rerun as a whole on 2026-10-03. The ACP rerun described under
Maintenance rewrote the ACP rows, the controlled K-sweeps and the label sweep on
2026-11-12.

## Preprocessing and labeling

No relabeling. Labels are the upstream class labels (CIFAR-100 fine labels, ImageNet-1k
validation labels, NTU-60 and N-UCLA action labels). Scores are raw model logits; the
methods apply their own softmax and score functions (LAC for five methods, APS for ACP,
LAC for clustered).

## Uses

**Intended.** Comparing conformal methods on per-class coverage under one protocol: a new
method reads the same caches and reports the same metrics, and its row is comparable to
the released ones. Reproducing the paper's results:
`scripts/camera_ready_numbers.py` writes every measured value and table body of the
camera-ready paper from the committed files into `paper/camera_ready/generated/`, and a
test fails when a printed value differs from a fresh run. Protocol constants, such as the 50 calibration points per class or the
82 shift conditions, are part of the design and are typed in the text.

**Not intended.** The logits are not calibrated probabilities outside the conformal
methods that consume them. The artifact is not a source of pretrained classifiers and not
a basis for building or deploying an action-recognition system. A coverage value here is
evidence about these backbones, shifts, and sample sizes, not a certificate for another
deployment.

**Known limits.** Two zero-shot OpenCLIP backbones and one CTR-GCN; synthetic corruption
for the image families; one skeleton condition per shift type; worst-class coverage is a
minimum over noisy per-class estimates and must be read against the matched reference.
Section 5 of the paper discusses what the results cover.

**Thread count.** Weighted and ACP each fit a logistic regression, and what the fit
returns follows the BLAS thread count. Measured on 2026-11-11 with the
released K-sweep of `contrast_s3`: with the 32 threads of the released run every record
outside ACP comes back equal within 1e-9. With 4 threads and with 1 thread, weighted at
K 100 differs in all five seeds, in mean set size by up to 0.042 and in one class's
coverage by up to 0.04, while its worst-class coverage does not move. Weighted at K 10, 25
and 50 and the other five methods are equal at all three counts. ACP's records differ
between the counts as well; by how much was not measured. The job files of the released
runs set these counts: 32 for a controlled K-sweep, 1 for a leaderboard row, 8 for the
label sweep and 1 for the nested sweep. A rerun gives back a released number of weighted
or ACP only with the same count, on a node that offers that many CPUs.

## Distribution

**License.** The code, per-row outputs, and analysis files are released under the MIT
license (`LICENSE`). Each upstream dataset and checkpoint keeps its own license and access
terms; the build scripts download them from their sources and the artifact does not
restate those terms.

**ImageNet-1k.** The archive includes the ten ImageNet-1k caches. They hold class scores,
labels, and partition metadata, no images. ImageNet-1k is a gated dataset, and its terms of
access apply to them.

## Maintenance

Released rows are not replaced silently. A rerun by the maintainers keeps the released
files beside its own, and `scripts/compare_leaderboards.py` lists every printed cell it
changes with the old and new value. New methods extend the board with new rows under the
same protocol.

Rows are replaced in place only to correct a defect, and the ACP rerun is the one such
case. The ACP reimplementation ordered its tied classes by class index, so its per-class
coverage followed the index. The rerun orders the tie at random and rewrote the 175 ACP
rows of the leaderboard, the 80 controlled K-sweeps and the label sweep where they stand.
In every rewritten file only ACP's values and the recorded tie rule changed, and every
rewritten file records `tie_break`. The board as submitted is kept as
`outputs/analysis/leaderboard_submitted.md`, and
`outputs/analysis/leaderboard_rerun_diff.json`, which names it `leaderboard_released.md`,
lists cell by cell and row by row where the committed board differs from it. Appendix D
of the paper gives, for selected ACP results of the reviewed version, the reviewed value
beside the recomputed one. Ten older CIFAR-10-C conditions without a graded
severity (`cifar10c`, `cifar10c_contrast`, `cifar10c_defocus_blur`, `cifar10c_fog` and
`cifar10c_gaussian_noise`, each also with the suffix `_mild`) keep their 60 row files under
`outputs/`. They are not on the board and the archive holds no cache of them; their ACP
rows were not rerun and record no tie rule, and the paper prints no value from them.

After a rerun, `scripts/rebuild_derived.py` rebuilds the board and every analysis that
is computed from the rewritten files, and `tests/test_rebuild_derived.py` fails while a
committed one differs from its rebuild. Three files under `outputs/analysis/` are
records of the rerun of 2026-10-03 and are not rebuilt. `rows_rerun_diff.json` and
`analysis_rerun_diff.json` compare that rerun with a copy of the outputs released in
July 2026, which they name `outputs_released_2026-07` and which is not part of the
repository. `no_shift_summary.json` summarizes the no-shift controls under
`outputs/noshift/`, which the ACP rerun leaves as they are, so the ACP rows of those
controls still carry the index tie order; the paper prints no ACP value from them. The
nested sweep files under `outputs/nested_ksweep/` record as `code_version`, and
`outputs/analysis/nested_trend.json` as `code_versions`, the commit of the code that ran
the sweep, from the history before this release.

## Ethics

NTU RGB+D and Northwestern-UCLA contain recordings of people performing actions. Their
maintainers release them for research use under their own consent and privacy terms.
pcpbench collects no new human-subject data and redistributes only derived class scores,
labels, and partition metadata, not video or skeleton frames.

## References

- T. Gebru, J. Morgenstern, B. Vecchione, J. W. Vaughan, H. Wallach, H. Daumé III, and
  K. Crawford. Datasheets for Datasets. *Communications of the ACM* 64(12):86-92, 2021.
- D. Hendrycks and T. Dietterich. Benchmarking Neural Network Robustness to Common
  Corruptions and Perturbations. *ICLR*, 2019.
- A. Shahroudy, J. Liu, T.-T. Ng, and G. Wang. NTU RGB+D: A Large Scale Dataset for 3D
  Human Activity Analysis. *CVPR*, 2016.
- J. Wang, X. Nie, Y. Xia, Y. Wu, and S.-C. Zhu. Cross-View Action Modeling, Learning and
  Recognition. *CVPR*, 2014.
- M. Cherti et al. Reproducible Scaling Laws for Contrastive Language-Image Learning.
  *CVPR*, 2023.
- A. Radford et al. Learning Transferable Visual Models From Natural Language Supervision.
  *ICML*, 2021.
- Y. Chen, Z. Zhang, C. Yuan, B. Li, Y. Deng, and W. Hu. Channel-Wise Topology Refinement
  Graph Convolution for Skeleton-Based Action Recognition. *ICCV*, 2021.
- T. Ding, A. N. Angelopoulos, S. Bates, M. I. Jordan, and R. J. Tibshirani.
  Class-Conditional Conformal Prediction with Many Classes. *NeurIPS*, 2023.
