# R2 covered/uncovered five-seed M1 decision

**Final decision: `KEEP_DEFAULT_127_FOR_M1`.**

This is a read-only rescore of existing frozen `baseline_41` M1 binaries. No model was trained, tuned, promoted, or wired. TEST remained `closed_not_read`.

This read-only result does not change M2.
M2 remains `concurrent_43` with `num_leaves=127`.

## 1. Pre-registration

Locked plan: `docs/EXPERIMENT_R2_COVERAGE_DECISION_PLAN.md`, committed at `6fa053e` before predictions were loaded.

Paired delta is always **R2 minus control**. Positive is beneficial for AUC/AP/precision/recall/F1; negative is beneficial for Brier.

## 2. Exact frozen pairs

| Seed | Control (127) | R2 (31) |
| --- | --- | --- |
| 42 | models/runs/2026-07-26_1551__baseline-41-gpa-trend-control | models/runs/2026-07-27_1456__reg-r2-leaves31-baseline-41 |
| 52 | models/runs/2026-07-27_1027__seed52-baseline-41-gpa-trend-control | models/runs/2026-07-27_1600__seed52-regr2-leaves31-baseline-41 |
| 62 | models/runs/2026-07-27_1031__seed62-baseline-41-gpa-trend-control | models/runs/2026-07-27_1817__seed62-regr2-leaves31-baseline-41 |
| 72 | models/runs/2026-07-27_1035__seed72-baseline-41-gpa-trend-control | models/runs/2026-07-27_1820__seed72-regr2-leaves31-baseline-41 |
| 82 | models/runs/2026-07-27_1038__seed82-baseline-41-gpa-trend-control | models/runs/2026-07-27_1823__seed82-regr2-leaves31-baseline-41 |

## 3. Parity gate

All five pairs passed the shared `scripts/r2_parity.py` implementation before any prediction was loaded.

| Seed | Checks | Failed | Result |
| --- | --- | --- | --- |
| 42 | 22 | none | PASS |
| 52 | 22 | none | PASS |
| 62 | 22 | none | PASS |
| 72 | 22 | none | PASS |
| 82 | 22 | none | PASS |

Checks cover immutable TRAIN/VALID hashes, `baseline_41`, exact feature ordering, root/derived seeds, categorical levels, diploma fill, locked threshold, early stopping, TEST policy, serialized parameters, and the fact that only `num_leaves` differs (`127 → 31`). Every binary exists. Every complete-VALID re-score exactly reproduced its saved metrics.

## 4. Segment population

| Segment | n | Fail rate |
| --- | --- | --- |
| complete_valid | 156,097 | 10.34% |
| covered | 129,215 | 9.54% |
| uncovered | 26,882 | 14.19% |
| never_in_train | 25,627 | 14.17% |
| thin_history | 1,255 | 14.66% |

Covered is `course_difficulty_missing == 0`; uncovered is `== 1`. `never_in_train` and `thin_history` remain separate causes. Base rates are identical between paired arms because every model scores the same rows.

## 5.1 Absolute metrics — `complete_valid`

| Seed | Arm | n | Fail rate | AUC | Fail AP | Brier | Fail P | Fail R | Fail F1 | CM TN/FP/FN/TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | control | 156,097 | 10.34% | 0.809189 | 0.321983 | 0.080778 | 0.330729 | 0.419459 | 0.369847 | 6773/9374/13706/126244 |
| 42 | r2 | 156,097 | 10.34% | 0.810708 | 0.324293 | 0.080741 | 0.336915 | 0.407816 | 0.368990 | 6585/9562/12960/126990 |
| 52 | control | 156,097 | 10.34% | 0.809973 | 0.322388 | 0.080684 | 0.321010 | 0.443116 | 0.372307 | 7155/8992/15134/124816 |
| 52 | r2 | 156,097 | 10.34% | 0.809245 | 0.323239 | 0.080713 | 0.325209 | 0.425590 | 0.368689 | 6872/9275/14259/125691 |
| 62 | control | 156,097 | 10.34% | 0.809181 | 0.323275 | 0.080739 | 0.328299 | 0.420016 | 0.368537 | 6782/9365/13876/126074 |
| 62 | r2 | 156,097 | 10.34% | 0.809847 | 0.322577 | 0.080735 | 0.328368 | 0.425961 | 0.370852 | 6878/9269/14068/125882 |
| 72 | control | 156,097 | 10.34% | 0.809547 | 0.322108 | 0.080742 | 0.334079 | 0.425218 | 0.374179 | 6866/9281/13686/126264 |
| 72 | r2 | 156,097 | 10.34% | 0.808964 | 0.320811 | 0.080923 | 0.330922 | 0.410850 | 0.366580 | 6634/9513/13413/126537 |
| 82 | control | 156,097 | 10.34% | 0.809727 | 0.323832 | 0.080800 | 0.336830 | 0.403728 | 0.367257 | 6519/9628/12835/127115 |
| 82 | r2 | 156,097 | 10.34% | 0.809577 | 0.320889 | 0.081027 | 0.337314 | 0.395306 | 0.364015 | 6383/9764/12540/127410 |

Paired deltas:

| Seed | Δ AUC | Δ Fail AP | Δ Brier | Δ Fail P | Δ Fail R | Δ Fail F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | +0.001520 | +0.002310 | -0.000037 | +0.006186 | -0.011643 | -0.000856 |
| 52 | -0.000728 | +0.000851 | +0.000029 | +0.004199 | -0.017526 | -0.003618 |
| 62 | +0.000666 | -0.000698 | -0.000004 | +0.000069 | +0.005945 | +0.002315 |
| 72 | -0.000582 | -0.001298 | +0.000181 | -0.003157 | -0.014368 | -0.007599 |
| 82 | -0.000150 | -0.002943 | +0.000227 | +0.000485 | -0.008423 | -0.003242 |

## 5.2 Absolute metrics — `covered`

| Seed | Arm | n | Fail rate | AUC | Fail AP | Brier | Fail P | Fail R | Fail F1 | CM TN/FP/FN/TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | control | 129,215 | 9.54% | 0.816886 | 0.319622 | 0.074945 | 0.332319 | 0.407558 | 0.366113 | 5026/7306/10098/106785 |
| 42 | r2 | 129,215 | 9.54% | 0.818347 | 0.321342 | 0.074833 | 0.332444 | 0.414045 | 0.368784 | 5106/7226/10253/106630 |
| 52 | control | 129,215 | 9.54% | 0.817893 | 0.319356 | 0.074970 | 0.327370 | 0.411044 | 0.364466 | 5069/7263/10415/106468 |
| 52 | r2 | 129,215 | 9.54% | 0.816064 | 0.319337 | 0.075007 | 0.331470 | 0.399043 | 0.362131 | 4921/7411/9925/106958 |
| 62 | control | 129,215 | 9.54% | 0.817086 | 0.321087 | 0.074913 | 0.330617 | 0.404638 | 0.363902 | 4990/7342/10103/106780 |
| 62 | r2 | 129,215 | 9.54% | 0.816569 | 0.318094 | 0.075027 | 0.328996 | 0.407963 | 0.364248 | 5031/7301/10261/106622 |
| 72 | control | 129,215 | 9.54% | 0.816730 | 0.316429 | 0.075064 | 0.330407 | 0.412018 | 0.366727 | 5081/7251/10297/106586 |
| 72 | r2 | 129,215 | 9.54% | 0.815794 | 0.317157 | 0.075087 | 0.331597 | 0.402530 | 0.363636 | 4964/7368/10006/106877 |
| 82 | control | 129,215 | 9.54% | 0.816172 | 0.319306 | 0.075019 | 0.334219 | 0.397665 | 0.363192 | 4904/7428/9769/107114 |
| 82 | r2 | 129,215 | 9.54% | 0.817132 | 0.319421 | 0.075026 | 0.334627 | 0.398476 | 0.363771 | 4914/7418/9771/107112 |

Paired deltas:

| Seed | Δ AUC | Δ Fail AP | Δ Brier | Δ Fail P | Δ Fail R | Δ Fail F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | +0.001461 | +0.001720 | -0.000112 | +0.000124 | +0.006487 | +0.002671 |
| 52 | -0.001828 | -0.000019 | +0.000037 | +0.004100 | -0.012001 | -0.002335 |
| 62 | -0.000518 | -0.002993 | +0.000113 | -0.001621 | +0.003325 | +0.000347 |
| 72 | -0.000935 | +0.000727 | +0.000024 | +0.001189 | -0.009488 | -0.003090 |
| 82 | +0.000960 | +0.000115 | +0.000007 | +0.000408 | +0.000811 | +0.000579 |

## 5.3 Absolute metrics — `uncovered`

| Seed | Arm | n | Fail rate | AUC | Fail AP | Brier | Fail P | Fail R | Fail F1 | CM TN/FP/FN/TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | control | 26,882 | 14.19% | 0.764525 | 0.331319 | 0.108818 | 0.326237 | 0.457929 | 0.381025 | 1747/2068/3608/19459 |
| 42 | r2 | 26,882 | 14.19% | 0.766460 | 0.343927 | 0.109139 | 0.353321 | 0.387680 | 0.369704 | 1479/2336/2707/20360 |
| 52 | control | 26,882 | 14.19% | 0.765522 | 0.333940 | 0.108148 | 0.306539 | 0.546789 | 0.392844 | 2086/1729/4719/18348 |
| 52 | r2 | 26,882 | 14.19% | 0.767549 | 0.336271 | 0.108140 | 0.310422 | 0.511402 | 0.386337 | 1951/1864/4334/18733 |
| 62 | control | 26,882 | 14.19% | 0.762865 | 0.329949 | 0.108743 | 0.322013 | 0.469725 | 0.382090 | 1792/2023/3773/19294 |
| 62 | r2 | 26,882 | 14.19% | 0.770070 | 0.342155 | 0.108173 | 0.326671 | 0.484142 | 0.390115 | 1847/1968/3807/19260 |
| 72 | control | 26,882 | 14.19% | 0.768872 | 0.348200 | 0.108036 | 0.344994 | 0.467890 | 0.397152 | 1785/2030/3389/19678 |
| 72 | r2 | 26,882 | 14.19% | 0.767059 | 0.336679 | 0.108972 | 0.328934 | 0.437746 | 0.375619 | 1670/2145/3407/19660 |
| 82 | control | 26,882 | 14.19% | 0.770820 | 0.342884 | 0.108586 | 0.345012 | 0.423329 | 0.380179 | 1615/2200/3066/20001 |
| 82 | r2 | 26,882 | 14.19% | 0.764928 | 0.329609 | 0.109872 | 0.346626 | 0.385059 | 0.364833 | 1469/2346/2769/20298 |

Paired deltas:

| Seed | Δ AUC | Δ Fail AP | Δ Brier | Δ Fail P | Δ Fail R | Δ Fail F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | +0.001935 | +0.012609 | +0.000321 | +0.027083 | -0.070249 | -0.011321 |
| 52 | +0.002027 | +0.002331 | -0.000007 | +0.003882 | -0.035387 | -0.006507 |
| 62 | +0.007205 | +0.012206 | -0.000571 | +0.004659 | +0.014417 | +0.008026 |
| 72 | -0.001813 | -0.011521 | +0.000935 | -0.016060 | -0.030144 | -0.021534 |
| 82 | -0.005892 | -0.013275 | +0.001285 | +0.001614 | -0.038270 | -0.015346 |

## 5.4 Absolute metrics — `never_in_train`

| Seed | Arm | n | Fail rate | AUC | Fail AP | Brier | Fail P | Fail R | Fail F1 | CM TN/FP/FN/TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | control | 25,627 | 14.17% | 0.764405 | 0.330615 | 0.108701 | 0.325240 | 0.457450 | 0.380179 | 1661/1970/3446/18550 |
| 42 | r2 | 25,627 | 14.17% | 0.765806 | 0.343666 | 0.109075 | 0.352807 | 0.382539 | 0.367072 | 1389/2242/2548/19448 |
| 52 | control | 25,627 | 14.17% | 0.765295 | 0.333819 | 0.108021 | 0.305118 | 0.549986 | 0.392492 | 1997/1634/4548/17448 |
| 52 | r2 | 25,627 | 14.17% | 0.767596 | 0.335951 | 0.107968 | 0.308593 | 0.515285 | 0.386012 | 1871/1760/4192/17804 |
| 62 | control | 25,627 | 14.17% | 0.762430 | 0.329181 | 0.108632 | 0.320188 | 0.469568 | 0.380750 | 1705/1926/3620/18376 |
| 62 | r2 | 25,627 | 14.17% | 0.769815 | 0.342378 | 0.108052 | 0.325310 | 0.483889 | 0.389061 | 1757/1874/3644/18352 |
| 72 | control | 25,627 | 14.17% | 0.768869 | 0.349350 | 0.107865 | 0.344478 | 0.468191 | 0.396918 | 1700/1931/3235/18761 |
| 72 | r2 | 25,627 | 14.17% | 0.766553 | 0.335604 | 0.108884 | 0.327280 | 0.436794 | 0.374189 | 1586/2045/3260/18736 |
| 82 | control | 25,627 | 14.17% | 0.770927 | 0.342069 | 0.108475 | 0.344797 | 0.422473 | 0.379703 | 1534/2097/2915/19081 |
| 82 | r2 | 25,627 | 14.17% | 0.764107 | 0.328121 | 0.109863 | 0.345623 | 0.379510 | 0.361775 | 1378/2253/2609/19387 |

Paired deltas:

| Seed | Δ AUC | Δ Fail AP | Δ Brier | Δ Fail P | Δ Fail R | Δ Fail F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | +0.001401 | +0.013051 | +0.000375 | +0.027567 | -0.074910 | -0.013107 |
| 52 | +0.002300 | +0.002131 | -0.000053 | +0.003475 | -0.034701 | -0.006480 |
| 62 | +0.007385 | +0.013197 | -0.000580 | +0.005122 | +0.014321 | +0.008311 |
| 72 | -0.002316 | -0.013747 | +0.001020 | -0.017198 | -0.031396 | -0.022729 |
| 82 | -0.006820 | -0.013948 | +0.001388 | +0.000827 | -0.042963 | -0.017928 |

## 5.5 Absolute metrics — `thin_history`

| Seed | Arm | n | Fail rate | AUC | Fail AP | Brier | Fail P | Fail R | Fail F1 | CM TN/FP/FN/TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | control | 1,255 | 14.66% | 0.773601 | 0.355435 | 0.111215 | 0.346774 | 0.467391 | 0.398148 | 86/98/162/909 |
| 42 | r2 | 1,255 | 14.66% | 0.778711 | 0.361097 | 0.110439 | 0.361446 | 0.489130 | 0.415704 | 90/94/159/912 |
| 52 | control | 1,255 | 14.66% | 0.774809 | 0.351370 | 0.110741 | 0.342308 | 0.483696 | 0.400901 | 89/95/171/900 |
| 52 | r2 | 1,255 | 14.66% | 0.776661 | 0.357397 | 0.111660 | 0.360360 | 0.434783 | 0.394089 | 80/104/142/929 |
| 62 | control | 1,255 | 14.66% | 0.775200 | 0.351154 | 0.111024 | 0.362500 | 0.472826 | 0.410377 | 87/97/153/918 |
| 62 | r2 | 1,255 | 14.66% | 0.779331 | 0.354908 | 0.110639 | 0.355731 | 0.489130 | 0.411899 | 90/94/163/908 |
| 72 | control | 1,255 | 14.66% | 0.772039 | 0.346459 | 0.111537 | 0.355649 | 0.461957 | 0.401891 | 85/99/154/917 |
| 72 | r2 | 1,255 | 14.66% | 0.778752 | 0.364717 | 0.110749 | 0.363636 | 0.456522 | 0.404819 | 84/100/147/924 |
| 82 | control | 1,255 | 14.66% | 0.773850 | 0.368090 | 0.110858 | 0.349138 | 0.440217 | 0.389423 | 81/103/151/920 |
| 82 | r2 | 1,255 | 14.66% | 0.779579 | 0.365129 | 0.110036 | 0.362550 | 0.494565 | 0.418391 | 91/93/160/911 |

Paired deltas:

| Seed | Δ AUC | Δ Fail AP | Δ Brier | Δ Fail P | Δ Fail R | Δ Fail F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | +0.005110 | +0.005662 | -0.000776 | +0.014672 | +0.021739 | +0.017556 |
| 52 | +0.001852 | +0.006027 | +0.000919 | +0.018053 | -0.048913 | -0.006812 |
| 62 | +0.004131 | +0.003754 | -0.000384 | -0.006769 | +0.016304 | +0.001522 |
| 72 | +0.006714 | +0.018258 | -0.000788 | +0.007988 | -0.005435 | +0.002928 |
| 82 | +0.005729 | -0.002960 | -0.000822 | +0.013412 | +0.054348 | +0.028968 |

## 6. Five-seed delta summaries

| Segment | Metric | Mean | Median | SD | Min | Max | Beneficial | Harmful | Zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| complete_valid | fail_rate | +0.000000 | +0.000000 | 0.000000 | +0.000000 | +0.000000 | 0 | 0 | 5 |
| complete_valid | roc_auc | +0.000145 | -0.000150 | 0.000841 | -0.000728 | +0.001520 | 2 | 3 | 0 |
| complete_valid | fail_average_precision | -0.000355 | -0.000698 | 0.001803 | -0.002943 | +0.002310 | 2 | 3 | 0 |
| complete_valid | brier | +0.000079 | +0.000029 | 0.000105 | -0.000037 | +0.000227 | 2 | 3 | 0 |
| complete_valid | fail_precision | +0.001556 | +0.000485 | 0.003286 | -0.003157 | +0.006186 | 4 | 1 | 0 |
| complete_valid | fail_recall | -0.009203 | -0.011643 | 0.008149 | -0.017526 | +0.005945 | 1 | 4 | 0 |
| complete_valid | fail_f1 | -0.002600 | -0.003242 | 0.003275 | -0.007599 | +0.002315 | 1 | 4 | 0 |
| covered | fail_rate | +0.000000 | +0.000000 | 0.000000 | +0.000000 | +0.000000 | 0 | 0 | 5 |
| covered | roc_auc | -0.000172 | -0.000518 | 0.001216 | -0.001828 | +0.001461 | 2 | 3 | 0 |
| covered | fail_average_precision | -0.000090 | +0.000115 | 0.001576 | -0.002993 | +0.001720 | 3 | 2 | 0 |
| covered | brier | +0.000014 | +0.000024 | 0.000073 | -0.000112 | +0.000113 | 1 | 4 | 0 |
| covered | fail_precision | +0.000840 | +0.000408 | 0.001871 | -0.001621 | +0.004100 | 4 | 1 | 0 |
| covered | fail_recall | -0.002173 | +0.000811 | 0.007269 | -0.012001 | +0.006487 | 3 | 2 | 0 |
| covered | fail_f1 | -0.000366 | +0.000347 | 0.002094 | -0.003090 | +0.002671 | 3 | 2 | 0 |
| uncovered | fail_rate | +0.000000 | +0.000000 | 0.000000 | +0.000000 | +0.000000 | 0 | 0 | 5 |
| uncovered | roc_auc | +0.000693 | +0.001935 | 0.004368 | -0.005892 | +0.007205 | 3 | 2 | 0 |
| uncovered | fail_average_precision | +0.000470 | +0.002331 | 0.011147 | -0.013275 | +0.012609 | 3 | 2 | 0 |
| uncovered | brier | +0.000393 | +0.000321 | 0.000661 | -0.000571 | +0.001285 | 2 | 3 | 0 |
| uncovered | fail_precision | +0.004236 | +0.003882 | 0.013719 | -0.016060 | +0.027083 | 4 | 1 | 0 |
| uncovered | fail_recall | -0.031927 | -0.035387 | 0.027099 | -0.070249 | +0.014417 | 1 | 4 | 0 |
| uncovered | fail_f1 | -0.009336 | -0.011321 | 0.009983 | -0.021534 | +0.008026 | 1 | 4 | 0 |
| never_in_train | fail_rate | +0.000000 | +0.000000 | 0.000000 | +0.000000 | +0.000000 | 0 | 0 | 5 |
| never_in_train | roc_auc | +0.000390 | +0.001401 | 0.004752 | -0.006820 | +0.007385 | 3 | 2 | 0 |
| never_in_train | fail_average_precision | +0.000137 | +0.002131 | 0.012103 | -0.013948 | +0.013197 | 3 | 2 | 0 |
| never_in_train | brier | +0.000430 | +0.000375 | 0.000710 | -0.000580 | +0.001388 | 2 | 3 | 0 |
| never_in_train | fail_precision | +0.003959 | +0.003475 | 0.014257 | -0.017198 | +0.027567 | 4 | 1 | 0 |
| never_in_train | fail_recall | -0.033930 | -0.034701 | 0.028622 | -0.074910 | +0.014321 | 1 | 4 | 0 |
| never_in_train | fail_f1 | -0.010387 | -0.013107 | 0.010784 | -0.022729 | +0.008311 | 1 | 4 | 0 |
| thin_history | fail_rate | +0.000000 | +0.000000 | 0.000000 | +0.000000 | +0.000000 | 0 | 0 | 5 |
| thin_history | roc_auc | +0.004707 | +0.005110 | 0.001656 | +0.001852 | +0.006714 | 5 | 0 | 0 |
| thin_history | fail_average_precision | +0.006148 | +0.005662 | 0.006864 | -0.002960 | +0.018258 | 4 | 1 | 0 |
| thin_history | brier | -0.000370 | -0.000776 | 0.000664 | -0.000822 | +0.000919 | 4 | 1 | 0 |
| thin_history | fail_precision | +0.009471 | +0.013412 | 0.008742 | -0.006769 | +0.018053 | 4 | 1 | 0 |
| thin_history | fail_recall | +0.007609 | +0.016304 | 0.034131 | -0.048913 | +0.054348 | 3 | 2 | 0 |
| thin_history | fail_f1 | +0.008832 | +0.002928 | 0.012764 | -0.006812 | +0.028968 | 4 | 1 | 0 |

## 7. Locked rule, clause by clause

- `1_uncovered_auc_and_brier`: **FAIL**
  - `uncovered_auc_beneficial_at_least_4_of_5`: fail
  - `uncovered_brier_beneficial_at_least_4_of_5`: fail
  - `uncovered_mean_auc_beneficial`: pass
  - `uncovered_mean_brier_beneficial`: fail
- `2_uncovered_fail_ap`: **PASS**
  - harmful seeds: 2 (maximum 2)
- `3_covered_no_systematic_harm`: **FAIL**
  - `covered_auc_harmful_no_more_than_2`: fail
  - `covered_brier_harmful_no_more_than_2`: fail
- `4_complete_valid_guardrails`: **FAIL**
  - `roc_auc`: harmful breach seeds [52, 72]; mean +0.000145; mean breach False
  - `fail_average_precision`: harmful breach seeds [82]; mean -0.000355; mean breach False
  - `brier`: harmful breach seeds [72, 82]; mean +0.000079; mean breach False
- `5_not_seed42_dependent`: **FAIL**
  - `uncovered_auc_beneficial_at_least_3_of_4`: fail
  - `uncovered_brier_beneficial_at_least_3_of_4`: fail
  - `uncovered_fail_ap_harmful_no_more_than_2_of_4`: pass
  - `covered_auc_harmful_no_more_than_2_of_4`: fail
  - `covered_brier_harmful_no_more_than_2_of_4`: fail
  - `uncovered_mean_auc_beneficial_without_seed42`: pass
  - `uncovered_mean_brier_beneficial_without_seed42`: fail
  - `complete_valid_mean_guardrails_unbreached_without_seed42`: pass

All clauses satisfied: **False**.

Therefore: **`KEEP_DEFAULT_127_FOR_M1`**.

The incumbent wins whenever the rule is not fully met. No result is promoted or wired by this report.

## 8. Scope confirmations

- Existing frozen M1 binaries only; no retraining or retuning.
- `concurrent_43` and `concurrent_44` were not scored for M1.
- M2 was not rescored and its parameter decision was not reopened.
- TEST was never read; policy remained `closed_not_read`.
- No parquet, default, model path, promotion marker, inference, recommendation, API, eligibility, or plan-generation wiring changed.
