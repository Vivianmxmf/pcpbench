# Bootstrap intervals and floor excess for the controlled sweep and the label sweep

Every mean below is a plain average over the stated records. Every interval is a 95 percent percentile bootstrap of 10000 resamples drawn from a random number generator seeded at zero. Every interval other than the five decay-ratio intervals in Table 2 is reported beside its bootstrap standard error, the standard deviation of the same resampled distribution. The decay-ratio intervals carry the caveat given after Table 2 in place of a standard error. The five moderate caches are contrast_s1, contrast_s2, contrast_s3, brightness_s1, brightness_s3, contributing 5 subset seeds each for 25 records per method and K. The controlled sweep draws a fresh random class subset at every K, so the sampled classes differ between the K equal to 10 and the K equal to 100 records of a given subset seed. The pairing used throughout this document is by seed index alone. A two level bootstrap resamples the five caches with replacement and then resamples each drawn cache's five seeds with replacement, which is the procedure behind every controlled sweep interval below. The three floored caches are gaussian_noise_s3, fog_s3, jpeg_compression_s1, each contributing one cache of 5 seeds, so their intervals use a one level bootstrap over seeds alone. The label sweep intervals also use a one level bootstrap, over its 10 seeds.

## Table 1. Worst-class coverage by method and K, with the validity floor

| method | K | n | mean | SE | 95 percent interval | floor mean | floor SD | excess over floor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| split | 10 | 25 | 0.558 | 0.082 | 0.393 to 0.710 | 0.800 | 0.049 | -0.242 |
| split | 25 | 25 | 0.296 | 0.070 | 0.155 to 0.426 | 0.765 | 0.045 | -0.469 |
| split | 50 | 25 | 0.175 | 0.046 | 0.091 to 0.273 | 0.740 | 0.043 | -0.565 |
| split | 100 | 25 | 0.053 | 0.004 | 0.045 to 0.060 | 0.718 | 0.040 | -0.665 |
| mondrian | 10 | 25 | 0.630 | 0.075 | 0.471 to 0.762 | 0.800 | 0.049 | -0.170 |
| mondrian | 25 | 25 | 0.468 | 0.109 | 0.256 to 0.675 | 0.765 | 0.045 | -0.297 |
| mondrian | 50 | 25 | 0.434 | 0.109 | 0.222 to 0.643 | 0.740 | 0.043 | -0.306 |
| mondrian | 100 | 25 | 0.414 | 0.108 | 0.205 to 0.620 | 0.718 | 0.040 | -0.303 |
| weighted | 10 | 25 | 0.663 | 0.046 | 0.566 to 0.741 | 0.800 | 0.049 | -0.137 |
| weighted | 25 | 25 | 0.390 | 0.033 | 0.323 to 0.454 | 0.765 | 0.045 | -0.375 |
| weighted | 50 | 25 | 0.244 | 0.028 | 0.194 to 0.305 | 0.740 | 0.043 | -0.496 |
| weighted | 100 | 25 | 0.125 | 0.032 | 0.070 to 0.196 | 0.718 | 0.040 | -0.593 |
| ppi | 10 | 25 | 0.682 | 0.060 | 0.554 to 0.784 | 0.800 | 0.049 | -0.119 |
| ppi | 25 | 25 | 0.262 | 0.052 | 0.159 to 0.362 | 0.765 | 0.045 | -0.504 |
| ppi | 50 | 25 | 0.083 | 0.030 | 0.032 to 0.150 | 0.740 | 0.043 | -0.657 |
| ppi | 100 | 25 | 0.007 | 0.003 | 0.002 to 0.013 | 0.718 | 0.040 | -0.710 |
| acp | 10 | 25 | 0.969 | 0.006 | 0.957 to 0.979 | 0.800 | 0.049 | 0.169 |
| acp | 25 | 25 | 0.928 | 0.006 | 0.915 to 0.940 | 0.765 | 0.045 | 0.163 |
| acp | 50 | 25 | 0.903 | 0.010 | 0.885 to 0.923 | 0.740 | 0.043 | 0.163 |
| acp | 100 | 25 | 0.884 | 0.016 | 0.851 to 0.914 | 0.718 | 0.040 | 0.166 |
| oracle | 10 | 25 | 0.798 | 0.013 | 0.770 to 0.820 | 0.800 | 0.049 | -0.003 |
| oracle | 25 | 25 | 0.770 | 0.007 | 0.756 to 0.783 | 0.765 | 0.045 | 0.004 |
| oracle | 50 | 25 | 0.742 | 0.007 | 0.728 to 0.756 | 0.740 | 0.043 | 0.002 |
| oracle | 100 | 25 | 0.724 | 0.008 | 0.706 to 0.738 | 0.718 | 0.040 | 0.006 |

### Floored caches, oracle only

| cache | K | n | mean | SE | 95 percent interval | floor mean | floor SD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_noise_s3 | 10 | 5 | 0.796 | 0.018 | 0.764 to 0.836 | 0.800 | 0.049 |
| gaussian_noise_s3 | 100 | 5 | 0.676 | 0.016 | 0.644 to 0.704 | 0.718 | 0.040 |
| fog_s3 | 10 | 5 | 0.792 | 0.020 | 0.752 to 0.828 | 0.800 | 0.049 |
| fog_s3 | 100 | 5 | 0.712 | 0.007 | 0.700 to 0.728 | 0.718 | 0.040 |
| jpeg_compression_s1 | 10 | 5 | 0.808 | 0.018 | 0.772 to 0.844 | 0.800 | 0.049 |
| jpeg_compression_s1 | 100 | 5 | 0.720 | 0.023 | 0.672 to 0.764 | 0.718 | 0.040 |

## Table 2. Decay from K equal to 10 to K equal to 100, ratio to the oracle decay, and the gap to the oracle at K equal to 100

| method | decay | decay SE | decay 95 percent interval | ratio to oracle decay | displayed-endpoint convention | ratio 95 percent interval | ratio interval at or below one | diff to oracle at K=100 | diff 95 percent interval | records below oracle at K=100 of 25 | K=100 interval below floor mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| split | 0.506 | 0.079 | 0.347 to 0.652 | 6.9 | 6.8 | 4.3 to 12.6 | false | -0.671 | -0.690 to -0.650 | 25 | true |
| mondrian | 0.216 | 0.040 | 0.138 to 0.291 | 2.9 | 2.9 | 1.5 to 7.0 | false | -0.310 | -0.522 to -0.099 | 22 | true |
| weighted | 0.538 | 0.070 | 0.390 to 0.660 | 7.3 | 7.3 | 4.7 to 13.8 | false | -0.599 | -0.654 to -0.530 | 25 | true |
| ppi | 0.674 | 0.058 | 0.550 to 0.775 | 9.2 | 9.1 | 6.3 to 17.2 | false | -0.717 | -0.731 to -0.700 | 25 | true |
| acp | 0.085 | 0.018 | 0.050 to 0.120 | 1.2 | 1.1 | 0.7 to 2.2 | true | 0.160 | 0.125 to 0.190 | 0 | false |

The reviewed version of the paper printed the ratios of the displayed three-decimal endpoints for mondrian and ppi as the ends of its range; on these records they are 2.9 and 9.1. On these records the same convention gives 1.1 for acp, outside that range. The same displayed-endpoint convention gives 6.8 for split. The two conventions differ for split, ppi and acp, where the unrounded ratio of means is 6.9, 9.2 and 1.2, and they agree for mondrian and weighted.

The 95 percent percentile interval is the reported uncertainty for each ratio because the bootstrap standard error of a ratio is inflated by the draws in which the oracle's decay falls near zero. In the draws behind the five ratio intervals the oracle's decay fell at or below 0.01 in 6, 5, 5, 5 and 3 of 10000 draws for split, mondrian, weighted, ppi and acp, and at or below zero in 0, 0, 1, 2 and 0.

## Table 3. Label-complexity sweep, worst-class coverage by method and target labels per class

| method | m | labels per class drawn | n | mean | SE | 95 percent interval | floor mean | floor SD | excess over floor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ppi | 0 | 0 | 10 | 0.000 | 0.000 | 0.000 to 0.000 | undefined | undefined | undefined |
| ppi | 10 | 10 | 10 | 0.460 | 0.032 | 0.398 to 0.522 | 0.568 | 0.080 | -0.108 |
| ppi | 25 | 25 | 10 | 0.504 | 0.049 | 0.402 to 0.594 | 0.702 | 0.053 | -0.198 |
| ppi | 50 | 50 | 10 | 0.696 | 0.020 | 0.654 to 0.730 | 0.718 | 0.040 | -0.022 |
| ppi | 100 | 50 | 10 | 0.696 | 0.020 | 0.654 to 0.730 | 0.718 | 0.040 | -0.022 |
| ppi | 200 | 50 | 10 | 0.696 | 0.020 | 0.654 to 0.730 | 0.718 | 0.040 | -0.022 |
| oracle | 0 | undefined | 0 | undefined | undefined | undefined | undefined | undefined | undefined |
| oracle | 10 | 10 | 10 | 0.592 | 0.022 | 0.546 to 0.636 | 0.568 | 0.080 | 0.024 |
| oracle | 25 | 25 | 10 | 0.708 | 0.017 | 0.672 to 0.740 | 0.702 | 0.053 | 0.006 |
| oracle | 50 | 50 | 10 | 0.724 | 0.016 | 0.690 to 0.754 | 0.718 | 0.040 | 0.006 |
| oracle | 100 | 50 | 10 | 0.724 | 0.016 | 0.690 to 0.754 | 0.718 | 0.040 | 0.006 |
| oracle | 200 | 50 | 10 | 0.724 | 0.016 | 0.690 to 0.754 | 0.718 | 0.040 | 0.006 |
| acp | 0 | undefined | 0 | undefined | undefined | undefined | undefined | undefined | undefined |
| acp | 10 | 10 | 10 | 0.918 | 0.012 | 0.892 to 0.938 | 0.568 | 0.080 | 0.350 |
| acp | 25 | 25 | 10 | 0.918 | 0.009 | 0.898 to 0.934 | 0.702 | 0.053 | 0.216 |
| acp | 50 | 50 | 10 | 0.910 | 0.009 | 0.892 to 0.924 | 0.718 | 0.040 | 0.192 |
| acp | 100 | 50 | 10 | 0.910 | 0.009 | 0.892 to 0.924 | 0.718 | 0.040 | 0.192 |
| acp | 200 | 50 | 10 | 0.910 | 0.009 | 0.892 to 0.924 | 0.718 | 0.040 | 0.192 |

The rows at m equal to 100 and 200 repeat the m equal to 50 row exactly, statistics included, because the label budget pool holds 50 labels per class at every larger m.

At K equal to 100 the 95 percent interval sits entirely below the floor mean for split, mondrian, weighted, ppi. The interval reaches the floor mean or above it for acp. The decay ratio interval excludes one for split, mondrian, weighted, ppi, so each of those methods decays faster than the oracle at the 95 percent level. The decay ratio interval reaches one or below for acp. The smallest target label budget per class at which PPI's bootstrap interval contains the floor mean is 50. The smallest target label budget per class at which the oracle's bootstrap interval contains the floor mean is 10. The label budget grid caps at 50 labels per class, so the m equal to 100 and 200 rows repeat the m equal to 50 measurement. At the m equal to 200 point of the label budget grid, which repeats the m equal to 50 point exactly because the pool caps at 50 labels per class, PPI's worst-class mean minus the oracle's paired worst-class mean is -0.028, with a 95 percent interval of -0.082 to 0.016.
