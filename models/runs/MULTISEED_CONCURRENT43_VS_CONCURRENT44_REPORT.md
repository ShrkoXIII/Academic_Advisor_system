# Multi-seed stability: concurrent_43 vs concurrent_44

concurrent_43 = concurrent_44 minus `concurrent_peer_difficulty_missing`
(dead: used by any model in only 2 of 5 concurrent_44 seeds, under
0.001% of total gain, zero splits everywhere else; see Decisions_Log.md).
Paired delta is defined as **concurrent_43 minus concurrent_44**.

Dropping an unused feature does NOT guarantee bit-identical models:
column sampling and histogram construction see 43 columns instead of
44, so trees may legitimately differ even where the dropped feature
had zero splits. Identical results and small differences are both
acceptable — the yardstick is NOISE_BAND.md, not zero-delta.

## Verdict

- **M1 verdict: INCONCLUSIVE** — Primary metric(s) outside the band, but on the BENEFICIAL side (not a degradation): ['m1_train_valid_auc_gap']. No metric is outside the band in the harmful direction. Reported as INCONCLUSIVE rather than EQUIVALENT because not every primary delta is strictly inside the band, per the band-is-the-bar rule.
- **M2 verdict: EQUIVALENT** — All 3 primary M2 VALID deltas fall inside the NOISE_BAND.md range.

No statistical significance is claimed from five seeds. The band is the
bar: EQUIVALENT requires every primary VALID delta inside the noise
band with no systematic degradation.

## Ten run paths

- Seed 42 concurrent_44 (comparison arm): `models/runs/2026-07-26_1554__concurrent-44-registration-roster-candidate`
- Seed 42 concurrent_43 (candidate): `models/runs/2026-07-27_1327__seed42-concurrent-43-drop-dead-missing-flag`
- Seed 52 concurrent_44 (comparison arm): `models/runs/2026-07-27_1028__seed52-concurrent-44-registration-roster-candidate`
- Seed 52 concurrent_43 (candidate): `models/runs/2026-07-27_1328__seed52-concurrent-43-drop-dead-missing-flag`
- Seed 62 concurrent_44 (comparison arm): `models/runs/2026-07-27_1033__seed62-concurrent-44-registration-roster-candidate`
- Seed 62 concurrent_43 (candidate): `models/runs/2026-07-27_1329__seed62-concurrent-43-drop-dead-missing-flag`
- Seed 72 concurrent_44 (comparison arm): `models/runs/2026-07-27_1036__seed72-concurrent-44-registration-roster-candidate`
- Seed 72 concurrent_43 (candidate): `models/runs/2026-07-27_1330__seed72-concurrent-43-drop-dead-missing-flag`
- Seed 82 concurrent_44 (comparison arm): `models/runs/2026-07-27_1039__seed82-concurrent-44-registration-roster-candidate`
- Seed 82 concurrent_43 (candidate): `models/runs/2026-07-27_1331__seed82-concurrent-43-drop-dead-missing-flag`

## Contract equality

| Seed | Valid | Effective LightGBM seeds |
|---:|:---:|---|
| 42 | yes | seed=42, data_random_seed=175, feature_fraction_seed=30056, bagging_seed=400, drop_seed=17869 |
| 52 | yes | seed=52, data_random_seed=208, feature_fraction_seed=8545, bagging_seed=9580, drop_seed=32671 |
| 62 | yes | seed=62, data_random_seed=241, feature_fraction_seed=19802, bagging_seed=18760, drop_seed=14704 |
| 72 | yes | seed=72, data_random_seed=273, feature_fraction_seed=31059, bagging_seed=27940, drop_seed=29506 |
| 82 | yes | seed=82, data_random_seed=306, feature_fraction_seed=9548, bagging_seed=4352, drop_seed=11540 |

Every pair used the same train/valid SHA-256 values (8aaff32aeac5b375…, 228719fa492da84b…), 450465 TRAIN rows, 156097 VALID rows, identical categorical levels, threshold 0.80, four threads, 2000-round cap, 50-round VALID-only early stopping, train-only diploma-GPA median fill, and closed TEST (TEST parquet path was nonexistent for every run). The only model-input difference is `concurrent_peer_difficulty_missing`.

## Exact M1 metrics

| Seed | Arm | TRAIN AUC | VALID AUC | VALID fail AP | VALID Brier | AUC gap | Best iter |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | concurrent_44 | 0.86662517566146002 | 0.80977548316641501 | 0.32284489485725293 | 0.080697525754173577 | 0.056849692495045012 | 143 |
| 42 | concurrent_43 | 0.86904731661822021 | 0.80998671304389847 | 0.32328432378749011 | 0.080659720768416765 | 0.059060603574321746 | 155 |
| 52 | concurrent_44 | 0.88265453965238538 | 0.80959020678474003 | 0.3203424057106794 | 0.0808032401083775 | 0.073064332867645354 | 244 |
| 52 | concurrent_43 | 0.87598416628910791 | 0.81006061782365579 | 0.32168750555421266 | 0.080674761991739216 | 0.06592354846545212 | 199 |
| 62 | concurrent_44 | 0.88476988059027839 | 0.81022275714329051 | 0.32228911527858117 | 0.080631595357280386 | 0.074547123446987884 | 264 |
| 62 | concurrent_43 | 0.87207479189936721 | 0.8111789396158946 | 0.32504066432223122 | 0.0804833010533101 | 0.060895852283472607 | 173 |
| 72 | concurrent_44 | 0.86289090671506197 | 0.81056385296104905 | 0.32365227877016317 | 0.080691283071790784 | 0.05232705375401292 | 127 |
| 72 | concurrent_43 | 0.86343574574589343 | 0.81014039907952684 | 0.32415001218075062 | 0.080611800014190882 | 0.053295346666366594 | 128 |
| 82 | concurrent_44 | 0.88297844223990629 | 0.80985872959388194 | 0.32193427587076756 | 0.080707758789744302 | 0.073119712646024349 | 247 |
| 82 | concurrent_43 | 0.86961268874233943 | 0.81084677921028914 | 0.32314535918433407 | 0.080648458480657381 | 0.058765909532050298 | 158 |

## Exact M2 metrics

| Seed | Arm | TRAIN MAE | VALID MAE | VALID RMSE | VALID R2 | Best iter |
|---:|---|---:|---:|---:|---:|---:|
| 42 | concurrent_44 | 8.5848248751728882 | 9.5293119553759826 | 12.813964812242361 | 0.35603105046241112 | 559 |
| 42 | concurrent_43 | 8.9681354029364009 | 9.5783755788310003 | 12.862119935222113 | 0.35118186080742175 | 300 |
| 52 | concurrent_44 | 8.9776159327675842 | 9.5549010097784368 | 12.847472085211535 | 0.35265881471603999 | 295 |
| 52 | concurrent_43 | 8.865228283360457 | 9.566677569057136 | 12.853481140005309 | 0.35205312072018002 | 360 |
| 62 | concurrent_44 | 9.018408377361375 | 9.5852359246154872 | 12.886114296657684 | 0.34875885463221512 | 273 |
| 62 | concurrent_43 | 9.0696004471459926 | 9.5606993674037426 | 12.834704632815408 | 0.3539447939347059 | 247 |
| 72 | concurrent_44 | 8.6187250491290524 | 9.5365669151877075 | 12.813732832730954 | 0.35605436662602363 | 527 |
| 72 | concurrent_43 | 8.7500550033601865 | 9.5555115107053119 | 12.849506951860375 | 0.35245373821582437 | 443 |
| 82 | concurrent_44 | 8.869048856858587 | 9.5503964572417583 | 12.827479438350265 | 0.35467197243412762 | 351 |
| 82 | concurrent_43 | 8.5402298214549415 | 9.5539611231306498 | 12.837399896924117 | 0.3536734246601112 | 612 |

## Paired VALID deltas (concurrent_43 minus concurrent_44)

| Seed | M1 AUC | M1 fail AP | M1 Brier | AUC gap | M2 MAE | M2 RMSE | M2 R2 | Cold-start AUC | Low-support AUC | Level-1 AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.00021122987748345423 | 0.00043942893023718455 | -3.7804985756811904e-05 | 0.0022109110792767339 | 0.049063623455017691 | 0.048155122979752107 | -0.004849189654989372 | -0.0086117744055249634 | -0.0037877746380767796 | 0.00079075487952873225 |
| 52 | 0.00047041103891576341 | 0.0013450998435332595 | -0.00012847811663828368 | -0.0071407844021932343 | 0.011776559278699139 | 0.0060090547937736716 | -0.0006056939958599683 | 0.010442922553549505 | 0.006358495665995556 | -0.00052223656713723532 |
| 62 | 0.00095618247260409461 | 0.0027515490436500478 | -0.00014829430397028665 | -0.013651271163515277 | -0.024536557211744636 | -0.051409663842276743 | 0.0051859393024907829 | -0.00046641583426809685 | 0.00021667768742317239 | 0.00073540062460808464 |
| 72 | -0.00042345388152220664 | 0.00049773341058745313 | -7.9483057599902263e-05 | 0.00096829291235367432 | 0.018944595517604412 | 0.035774119129420967 | -0.0036006284101992581 | -0.0035564902883126193 | -0.00092251569337187256 | -0.0002568244834271205 |
| 82 | 0.00098804961640719391 | 0.0012110833135665078 | -5.9300309086920966e-05 | -0.014353803113974051 | 0.0035646658888914828 | 0.0099204585738519313 | -0.00099854777401642281 | 0.0018479797364368 | 0.0053558847909506158 | 0.00024371336380935027 |

## Five-seed summary vs. NOISE_BAND.md

| Metric | c44 mean | c43 mean | Mean delta | Median delta | SD delta | Min delta | Max delta | Improved | Worsened | Band judgment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| m1_valid_auc | 0.81000220592987537 | 0.81044268975465295 | 0.00044048382477765988 | 0.00047041103891576341 | 0.00058429760447814777 | -0.00042345388152220664 | 0.00098804961640719391 | 4 | 1 | inside_band |
| m1_valid_fail_ap | 0.32221259409748881 | 0.32346157300580375 | 0.0012489789083148905 | 0.0012110833135665078 | 0.00093382724402737624 | 0.00043942893023718455 | 0.0027515490436500478 | 5 | 0 | inside_band |
| m1_valid_brier | 0.080706280616273302 | 0.080615608461662866 | -9.0672154610441097e-05 | -7.9483057599902263e-05 | 4.651324807987511e-05 | -0.00014829430397028665 | -3.7804985756811904e-05 | 5 | 0 | inside_band |
| m1_train_valid_auc_gap | 0.065981583041943098 | 0.059588252104332676 | -0.0063933309376104312 | -0.0071407844021932343 | 0.007823553966528108 | -0.014353803113974051 | 0.0022109110792767339 | 3 | 2 | outside_band |
| m2_valid_mae | 9.5512824524398745 | 9.5630450298255685 | 0.011762577385693617 | 0.011776559278699139 | 0.026588449293060527 | -0.024536557211744636 | 0.049063623455017691 | 1 | 4 | inside_band |
| m2_valid_rmse | 12.837752693038562 | 12.847442511365463 | 0.0096898183269043866 | 0.0099204585738519313 | 0.038427588874547461 | -0.051409663842276743 | 0.048155122979752107 | 1 | 4 | inside_band |
| m2_valid_r2 | 0.35363501177416345 | 0.35266138766764865 | -0.0009736241065148477 | -0.00099854777401642281 | 0.0038728986262613549 | -0.004849189654989372 | 0.0051859393024907829 | 1 | 4 | inside_band |
| cold_start_auc | 0.73404281608331989 | 0.73397406043569613 | -6.8755647623874871e-05 | -0.00046641583426809685 | 0.0070617782672512076 | -0.0086117744055249634 | 0.010442922553549505 | 2 | 3 | inside_band |
| low_difficulty_support_auc | 0.76768239677437655 | 0.76912655033696065 | 0.0014441535625841384 | 0.00021667768742317239 | 0.0042992235520033211 | -0.0037877746380767796 | 0.006358495665995556 | 3 | 2 | inside_band |
| level_1_auc | 0.82126723166680349 | 0.82146539323027989 | 0.00019816156347636228 | 0.00024371336380935027 | 0.00058477632113006239 | -0.00052223656713723532 | 0.00079075487952873225 | 3 | 2 | inside_band |

## Segment stability

`first_semester` and `cold_start_gpa` are the same VALID population in
every run, so they are one piece of evidence.

| Seed | Segment | c44 AUC | c43 AUC | Delta | n |
|---:|---|---:|---:|---:|---:|
| 42 | first_semester | 0.74112095457641147 | 0.73250918017088651 | -0.0086117744055249634 | 14732 |
| 42 | cold_start_gpa | 0.74112095457641147 | 0.73250918017088651 | -0.0086117744055249634 | 14732 |
| 42 | retake_attempt | 0.67646287652778936 | 0.67853959894705806 | 0.0020767224192687062 | 17958 |
| 42 | low_difficulty_support | 0.77111025143701806 | 0.76732247679894128 | -0.0037877746380767796 | 25627 |
| 42 | level_1_difficulty | 0.82042336059177212 | 0.82121411547130085 | 0.00079075487952873225 | 120858 |
| 52 | first_semester | 0.72411837252728661 | 0.73456129508083612 | 0.010442922553549505 | 14732 |
| 52 | cold_start_gpa | 0.72411837252728661 | 0.73456129508083612 | 0.010442922553549505 | 14732 |
| 52 | retake_attempt | 0.67537463671138298 | 0.67689836920452051 | 0.0015237324931375307 | 17958 |
| 52 | low_difficulty_support | 0.76114970754803868 | 0.76750820321403423 | 0.006358495665995556 | 25627 |
| 52 | level_1_difficulty | 0.82164088019050241 | 0.82111864362336517 | -0.00052223656713723532 | 120858 |
| 62 | first_semester | 0.73479398135783347 | 0.73432756552356537 | -0.00046641583426809685 | 14732 |
| 62 | cold_start_gpa | 0.73479398135783347 | 0.73432756552356537 | -0.00046641583426809685 | 14732 |
| 62 | retake_attempt | 0.67574114487654746 | 0.67732939028651251 | 0.00158824540996505 | 17958 |
| 62 | low_difficulty_support | 0.77095243375413536 | 0.77116911144155853 | 0.00021667768742317239 | 25627 |
| 62 | level_1_difficulty | 0.82143463290511098 | 0.82217003352971907 | 0.00073540062460808464 | 120858 |
| 72 | first_semester | 0.73888699591479268 | 0.73533050562648006 | -0.0035564902883126193 | 14732 |
| 72 | cold_start_gpa | 0.73888699591479268 | 0.73533050562648006 | -0.0035564902883126193 | 14732 |
| 72 | retake_attempt | 0.67856990009904505 | 0.67863428293311634 | 6.4382834071285977e-05 | 17958 |
| 72 | low_difficulty_support | 0.77092973365027839 | 0.77000721795690652 | -0.00092251569337187256 | 25627 |
| 72 | level_1_difficulty | 0.82126042544339495 | 0.82100360095996783 | -0.0002568244834271205 | 120858 |
| 82 | first_semester | 0.73129377604027557 | 0.73314175577671237 | 0.0018479797364368 | 14732 |
| 82 | cold_start_gpa | 0.73129377604027557 | 0.73314175577671237 | 0.0018479797364368 | 14732 |
| 82 | retake_attempt | 0.67395894607731555 | 0.67914154877049682 | 0.0051826026931812708 | 17958 |
| 82 | low_difficulty_support | 0.76426985748241238 | 0.769625742273363 | 0.0053558847909506158 | 25627 |
| 82 | level_1_difficulty | 0.82157685920323709 | 0.82182057256704644 | 0.00024371336380935027 | 120858 |

## Concurrent feature evidence (the two remaining features)

| Seed | Model | Feature | c44 rank | c43 rank | rank shift | c44 % gain | c43 % gain |
|---:|---|---|---:|---:|---:|---:|---:|
| 42 | M1 | concurrent_peer_difficulty_mean | 16 | 14 | -2 | 1.2986886991021207 | 1.5504554050639789 |
| 52 | M1 | concurrent_peer_difficulty_mean | 13 | 14 | +1 | 1.7432280186096407 | 1.5147148712554066 |
| 62 | M1 | concurrent_peer_difficulty_mean | 13 | 15 | +2 | 1.8419792930316838 | 1.4770923007211101 |
| 72 | M1 | concurrent_peer_difficulty_mean | 14 | 16 | +2 | 1.4809198047609247 | 1.0149614000317468 |
| 82 | M1 | concurrent_peer_difficulty_mean | 14 | 15 | +1 | 1.8672129964970599 | 1.4019333295361058 |
| 42 | M1 | concurrent_peer_difficulty_max | 14 | 13 | -1 | 1.5585296939707869 | 1.5809057404450169 |
| 52 | M1 | concurrent_peer_difficulty_max | 11 | 11 | +0 | 2.0227223144841546 | 1.879735995014544 |
| 62 | M1 | concurrent_peer_difficulty_max | 11 | 13 | +2 | 2.0605635409077512 | 1.6593854000991688 |
| 72 | M1 | concurrent_peer_difficulty_max | 15 | 12 | -3 | 1.4674629870433824 | 1.6378369037837213 |
| 82 | M1 | concurrent_peer_difficulty_max | 11 | 13 | +2 | 2.2181175039637164 | 1.6788165250004969 |
| 42 | M2 | concurrent_peer_difficulty_mean | 15 | 17 | +2 | 1.6123898122188001 | 1.1234023943830214 |
| 52 | M2 | concurrent_peer_difficulty_mean | 17 | 16 | -1 | 1.1001011596190204 | 1.2305297243793236 |
| 62 | M2 | concurrent_peer_difficulty_mean | 19 | 18 | -1 | 1.0472199096663031 | 0.97466075950243636 |
| 72 | M2 | concurrent_peer_difficulty_mean | 15 | 16 | +1 | 1.6141077391183398 | 1.3798902492793672 |
| 82 | M2 | concurrent_peer_difficulty_mean | 17 | 14 | -3 | 1.1827653153646056 | 1.7666665018267276 |
| 42 | M2 | concurrent_peer_difficulty_max | 12 | 14 | +2 | 1.9303730953683065 | 1.2529926224900225 |
| 52 | M2 | concurrent_peer_difficulty_max | 14 | 13 | -1 | 1.3356125178699374 | 1.5366261704500594 |
| 62 | M2 | concurrent_peer_difficulty_max | 13 | 14 | +1 | 1.27749402809174 | 1.174767373848657 |
| 72 | M2 | concurrent_peer_difficulty_max | 12 | 12 | +0 | 1.8443758099113416 | 1.7205824527247726 |
| 82 | M2 | concurrent_peer_difficulty_max | 14 | 11 | -3 | 1.4951422182656489 | 2.0401582869992039 |

## Best-iteration shift

| Model | c44 best iterations (by seed) | c43 best iterations (by seed) | Mean shift | Max abs shift |
|---|---|---|---:|---:|
| M1 | [143.0, 244.0, 264.0, 127.0, 247.0] | [155.0, 199.0, 173.0, 128.0, 158.0] | -42.399999999999999 | 91 |
| M2 | [559.0, 295.0, 273.0, 527.0, 351.0] | [300.0, 360.0, 247.0, 443.0, 612.0] | -8.5999999999999996 | 261 |

**Flag (M1):** best_iteration decreased in 3/5 seeds, mean shift -42.399999999999999 (-20.682926829268293% of the concurrent_44 mean). This is directionally consistent with the train-valid AUC-gap improvement noted above (fewer boosting rounds before VALID stops improving), and is reported as a real fitting-behavior change, not noise — but it does not by itself move any primary VALID metric outside the noise band.

**Flag (M2):** best_iteration shift is noisy and not systematic (3/5 seeds decreased, 2/5 increased, max abs shift 261.0 rounds against a -2.144638403990025% mean shift) — consistent with ordinary seed-to-seed early-stopping variance, not a directional effect of dropping the feature.

## Separate findings

### M1 — INCONCLUSIVE

{
  "m1_valid_auc": {
    "direction": "higher_is_better",
    "concurrent_44_mean": 0.8100022059298754,
    "concurrent_43_mean": 0.810442689754653,
    "mean_paired_delta": 0.0004404838247776599,
    "median_paired_delta": 0.0004704110389157634,
    "sample_standard_deviation_of_paired_deltas": 0.0005842976044781478,
    "minimum_paired_delta": -0.00042345388152220664,
    "maximum_paired_delta": 0.000988049616407194,
    "paired_delta_range": 0.0014115034979294006,
    "seeds_improved": 4,
    "seeds_worsened": 1,
    "seeds_tied": 0,
    "noise_band": {
      "min": -0.000382,
      "max": 0.001042
    },
    "mean_delta_band_judgment": "inside_band",
    "all_seeds_inside_band": false
  },
  "m1_valid_fail_ap": {
    "direction": "higher_is_better",
    "concurrent_44_mean": 0.3222125940974888,
    "concurrent_43_mean": 0.32346157300580375,
    "mean_paired_delta": 0.0012489789083148905,
    "median_paired_delta": 0.0012110833135665078,
    "sample_standard_deviation_of_paired_deltas": 0.0009338272440273762,
    "minimum_paired_delta": 0.00043942893023718455,
    "maximum_paired_delta": 0.002751549043650048,
    "paired_delta_range": 0.0023121201134128633,
    "seeds_improved": 5,
    "seeds_worsened": 0,
    "seeds_tied": 0,
    "noise_band": {
      "min": -0.002045,
      "max": 0.001544
    },
    "mean_delta_band_judgment": "inside_band",
    "all_seeds_inside_band": false
  },
  "m1_valid_brier": {
    "direction": "lower_is_better",
    "concurrent_44_mean": 0.0807062806162733,
    "concurrent_43_mean": 0.08061560846166287,
    "mean_paired_delta": -9.06721546104411e-05,
    "median_paired_delta": -7.948305759990226e-05,
    "sample_standard_deviation_of_paired_deltas": 4.651324807987511e-05,
    "minimum_paired_delta": -0.00014829430397028665,
    "maximum_paired_delta": -3.7804985756811904e-05,
    "paired_delta_range": 0.00011048931821347474,
    "seeds_improved": 5,
    "seeds_worsened": 0,
    "seeds_tied": 0,
    "noise_band": {
      "min": -0.000108,
      "max": 0.000119
    },
    "mean_delta_band_judgment": "inside_band",
    "all_seeds_inside_band": false
  },
  "m1_train_valid_auc_gap": {
    "direction": "lower_is_better",
    "concurrent_44_mean": 0.0659815830419431,
    "concurrent_43_mean": 0.059588252104332676,
    "mean_paired_delta": -0.006393330937610431,
    "median_paired_delta": -0.007140784402193234,
    "sample_standard_deviation_of_paired_deltas": 0.007823553966528108,
    "minimum_paired_delta": -0.01435380311397405,
    "maximum_paired_delta": 0.002210911079276734,
    "paired_delta_range": 0.016564714193250785,
    "seeds_improved": 3,
    "seeds_worsened": 2,
    "seeds_tied": 0,
    "noise_band": {
      "min": -0.005873,
      "max": 0.02672
    },
    "mean_delta_band_judgment": "outside_band",
    "all_seeds_inside_band": false
  }
}

### M2 — EQUIVALENT

{
  "m2_valid_mae": {
    "direction": "lower_is_better",
    "concurrent_44_mean": 9.551282452439875,
    "concurrent_43_mean": 9.563045029825568,
    "mean_paired_delta": 0.011762577385693617,
    "median_paired_delta": 0.011776559278699139,
    "sample_standard_deviation_of_paired_deltas": 0.026588449293060527,
    "minimum_paired_delta": -0.024536557211744636,
    "maximum_paired_delta": 0.04906362345501769,
    "paired_delta_range": 0.07360018066676233,
    "seeds_improved": 1,
    "seeds_worsened": 4,
    "seeds_tied": 0,
    "noise_band": {
      "min": -0.050423,
      "max": 0.04652
    },
    "mean_delta_band_judgment": "inside_band",
    "all_seeds_inside_band": false
  },
  "m2_valid_rmse": {
    "direction": "lower_is_better",
    "concurrent_44_mean": 12.837752693038562,
    "concurrent_43_mean": 12.847442511365463,
    "mean_paired_delta": 0.009689818326904387,
    "median_paired_delta": 0.009920458573851931,
    "sample_standard_deviation_of_paired_deltas": 0.03842758887454746,
    "minimum_paired_delta": -0.05140966384227674,
    "maximum_paired_delta": 0.04815512297975211,
    "paired_delta_range": 0.09956478682202885,
    "seeds_improved": 1,
    "seeds_worsened": 4,
    "seeds_tied": 0,
    "noise_band": {
      "min": -0.067477,
      "max": 0.07805
    },
    "mean_delta_band_judgment": "inside_band",
    "all_seeds_inside_band": true
  },
  "m2_valid_r2": {
    "direction": "higher_is_better",
    "concurrent_44_mean": 0.35363501177416345,
    "concurrent_43_mean": 0.35266138766764865,
    "mean_paired_delta": -0.0009736241065148477,
    "median_paired_delta": -0.0009985477740164228,
    "sample_standard_deviation_of_paired_deltas": 0.003872898626261355,
    "minimum_paired_delta": -0.004849189654989372,
    "maximum_paired_delta": 0.005185939302490783,
    "paired_delta_range": 0.010035128957480155,
    "seeds_improved": 1,
    "seeds_worsened": 4,
    "seeds_tied": 0,
    "noise_band": {
      "min": -0.007865,
      "max": 0.006807
    },
    "mean_delta_band_judgment": "inside_band",
    "all_seeds_inside_band": true
  }
}

## Integrity confirmations

- No training seed failed and no seed was rerun. Exactly five new persistent training runs were created (concurrent_43 only); concurrent_44 and baseline_41 were not retrained.
- TEST policy is `closed_not_read` in all five new runs; all M1/M2 TEST metric fields are null. All five new runs used a nonexistent TEST path.
- Only `df_train_final.parquet` and `df_valid_final.parquet` were used for evaluation. The TEST parquet was never read.
- No dataset, root/live model artifact, production contract, promotion marker, `CURRENT_VERSION.txt`, recommendation wiring, or inference wiring was changed.
