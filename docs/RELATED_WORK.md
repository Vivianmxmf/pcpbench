# Extended related work

Section 2 of the camera-ready paper covers the work pcpbench builds on directly: split,
Mondrian and clustered conformal prediction, weighted conformal prediction under
covariate shift, prediction-powered inference, audited conformal prediction, the LAC
and APS scores, and earlier evaluations of coverage under shift. Section 3 cites the
data sources and backbones. This note keeps the wider discussion for which the paper
has no room.
Bracketed numbers refer to the list at the end.

## Beyond class-conditional coverage

**Group-conditional coverage.** Equalized coverage asks for valid coverage within
predefined groups of the population [1]. Per-class coverage is the case where the group
is the true label. pcpbench measures that case only; group structures that cut across
classes, such as demographic attributes, are outside its caches.

**Risk control.** Risk-controlling prediction sets [2] and conformal risk control [3]
extend the guarantee from miscoverage to general bounded losses. A per-class loss would
let these methods target class-conditional error directly. pcpbench reports coverage and
set size, so a risk-controlling method enters the board through the same five metrics.

**Without a single held-out split.** The jackknife+ gives finite-sample predictive
intervals from leave-one-out fits instead of one calibration split [4]. pcpbench fixes
the split protocol so that every method sees the same calibration and evaluation rows;
a method of this kind would need its own cache of leave-one-out scores.

**Training for efficiency.** Conformal training learns a classifier whose conformal sets
are small [5]. pcpbench holds the backbones frozen and caches their scores, so it
compares calibration procedures on fixed scores, not training procedures.

## Distribution shift

**Robust validation** constructs prediction sets that remain valid over a neighborhood of
the calibration distribution [6]. It complements the covariate-shift methods on the
board: it pays for robustness with larger sets instead of estimating the shift.

**Prediction-powered inference** has an efficiency-improved variant, PPI++ [7], that tunes
how much the model's predictions on unlabeled data are trusted. The PPI method on the
board is the rectified-quantile form described in the paper.

**Empirical studies of shift.** A large study of natural distribution shift in image
classification finds that robustness gained on synthetic corruptions rarely transfers to
natural shift [8], which bounds what the corruption families of pcpbench can say. A study
of predictive uncertainty under dataset shift finds that calibration degrades as shift
grows and that good calibration on the test distribution does not carry over [9]. Both
motivate reporting coverage, not only accuracy, across shift conditions; pcpbench adds
the per-class view and the finite-sample yardstick for it.

## Skeleton backbones

Adaptive graph-convolutional models such as 2s-AGCN [10] preceded the CTR-GCN backbone
used for the skeleton conditions. Both belong to the graph-convolutional line opened by
ST-GCN.

## References

1. Y. Romano, R. F. Barber, C. Sabatti, and E. J. Candès. With Malice Toward None:
   Assessing Uncertainty via Equalized Coverage. *Harvard Data Science Review* 2(2), 2020.
2. S. Bates, A. Angelopoulos, L. Lei, J. Malik, and M. I. Jordan. Distribution-Free,
   Risk-Controlling Prediction Sets. *Journal of the ACM* 68(6), article 43, 2021.
3. A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, and T. Schuster. Conformal Risk
   Control. *ICLR*, 2024.
4. R. F. Barber, E. J. Candès, A. Ramdas, and R. J. Tibshirani. Predictive Inference with
   the Jackknife+. *The Annals of Statistics* 49(1):486-507, 2021.
5. D. Stutz, K. Dvijotham, A. T. Cemgil, and A. Doucet. Learning Optimal Conformal
   Classifiers. *ICLR*, 2022.
6. M. Cauchois, S. Gupta, A. Ali, and J. C. Duchi. Robust Validation: Confident
   Predictions Even When Distributions Shift. *Journal of the American Statistical
   Association* 119(548):3033-3044, 2024.
7. A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction-Powered
   Inference. arXiv:2311.01453, 2023.
8. R. Taori, A. Dave, V. Shankar, N. Carlini, B. Recht, and L. Schmidt. Measuring
   Robustness to Natural Distribution Shifts in Image Classification. *NeurIPS* 33,
   2020.
9. Y. Ovadia, E. Fertig, J. Ren, Z. Nado, D. Sculley, S. Nowozin, J. V. Dillon,
   B. Lakshminarayanan, and J. Snoek. Can You Trust Your Model's Uncertainty? Evaluating
   Predictive Uncertainty Under Dataset Shift. *NeurIPS* 32, 2019.
10. L. Shi, Y. Zhang, J. Cheng, and H. Lu. Two-Stream Adaptive Graph Convolutional
    Networks for Skeleton-Based Action Recognition. *CVPR*, 2019.
