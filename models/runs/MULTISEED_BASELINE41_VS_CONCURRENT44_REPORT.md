# Multi-seed stability: baseline_41 vs concurrent_44

## Verdict

- **M1 verdict: INCONCLUSIVE** — AUC and Brier improved in 4/5 seeds, but fail AP improved in only 2/5, the train-valid AUC gap worsened in 4/5, and segment direction was mixed.
- **M2 verdict: SUPPORTED** — MAE, RMSE, and R2 improve together in 4/5 seeds; the mean benefit is not caused by a single seed, though it is small.

No statistical significance is claimed from five seeds.

## Repository, tests, and memory

- Initial `git status --short`: clean (no output).
- Initial `git diff --stat`: no diff (no output).
- Initial `git log -3 --oneline`:
  - `5928aaa Add feature contract tests for baseline_41 and concurrent_44`
  - `0291dd2 Enhance concurrent group features and registration roster tests`
  - `e6e2686 Implement concurrent group features for course difficulty analysis`
- Final working-tree status captured before report generation:
```text
 M src/model_training.py
 M tests/test_feature_contracts.py
?? scripts/generate_multiseed_baseline41_vs_concurrent44_report.py
```
- Final diff stat captured before report generation:
```text
 src/model_training.py           | 94 ++++++++++++++++++++++++++++++++++++++---
 tests/test_feature_contracts.py | 68 ++++++++++++++++++++++++++---
 2 files changed, 150 insertions(+), 12 deletions(-)
```
- Final pre-training test gate: `python -m unittest discover -s tests -t .` — 104 tests, 0 failures, 12.891 seconds.
- Test-development note: The first development attempt exposed three failures in the new effective-seed metadata extractor. No experiment run had started. The extractor was corrected to read serialized LightGBM parameters, and the complete mandatory pre-training suite then passed.
- Physical memory: 16855928832 bytes; commit limit: 16855928832 bytes.
- Pagefile configured as `?:\pagefile.sys` but **inactive** (0 active bytes); the commit limit equaled physical memory.
- Available memory immediately before training: 6416273408 bytes.

## Ten run paths

- Seed 42 baseline: `models/runs/2026-07-26_1551__baseline-41-gpa-trend-control`
- Seed 42 candidate: `models/runs/2026-07-26_1554__concurrent-44-registration-roster-candidate`
- Seed 52 baseline: `models/runs/2026-07-27_1027__seed52-baseline-41-gpa-trend-control`
- Seed 52 candidate: `models/runs/2026-07-27_1028__seed52-concurrent-44-registration-roster-candidate`
- Seed 62 baseline: `models/runs/2026-07-27_1031__seed62-baseline-41-gpa-trend-control`
- Seed 62 candidate: `models/runs/2026-07-27_1033__seed62-concurrent-44-registration-roster-candidate`
- Seed 72 baseline: `models/runs/2026-07-27_1035__seed72-baseline-41-gpa-trend-control`
- Seed 72 candidate: `models/runs/2026-07-27_1036__seed72-concurrent-44-registration-roster-candidate`
- Seed 82 baseline: `models/runs/2026-07-27_1038__seed82-baseline-41-gpa-trend-control`
- Seed 82 candidate: `models/runs/2026-07-27_1039__seed82-concurrent-44-registration-roster-candidate`

## Contract equality

| Seed | Valid | Effective LightGBM seeds | Note |
|---:|:---:|---|---|
| 42 | yes | seed=42, data_random_seed=175, feature_fraction_seed=30056, bagging_seed=400, drop_seed=17869 | Directly comparable historical pair; historical code confirms 2000 rounds, 50-round VALID-only early stopping, train-only diploma median fill, and the same seed-only LightGBM derivation. |
| 52 | yes | seed=52, data_random_seed=208, feature_fraction_seed=8545, bagging_seed=9580, drop_seed=32671 | Baseline Git metadata was repaired from the immediately adjacent candidate after a launcher environment-injection failure; model, metric, and data artifacts were unchanged. |
| 62 | yes | seed=62, data_random_seed=241, feature_fraction_seed=19802, bagging_seed=18760, drop_seed=14704 | All recorded non-feature settings match. |
| 72 | yes | seed=72, data_random_seed=273, feature_fraction_seed=31059, bagging_seed=27940, drop_seed=29506 | All recorded non-feature settings match. |
| 82 | yes | seed=82, data_random_seed=306, feature_fraction_seed=9548, bagging_seed=4352, drop_seed=11540 | All recorded non-feature settings match. |

Every pair used the same train/valid SHA-256 values, 450465 TRAIN rows, 156097 VALID rows, identical targets and categorical levels, threshold 0.80, four threads, 2000-round cap, 50-round VALID-only early stopping, train-only diploma-GPA median fill, and closed TEST. The only model-input difference was the three concurrent features. Target hashes were not recorded by the historical artifacts; target definitions and the row-aligned immutable split hashes matched.

## Exact M1 metrics

| Seed | Arm | TRAIN AUC | TRAIN pass AP | TRAIN fail AP | TRAIN Brier | VALID AUC | VALID pass AP | VALID fail AP | VALID Brier | AUC gap | Best iter |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | baseline | 0.8645724829519178 | 0.97009076300831221 | 0.56799584568838457 | 0.096130344671190512 | 0.80918853274907998 | 0.97261215270452817 | 0.3219831593412697 | 0.08077845438389275 | 0.055383950202837817 | 137 |
| 42 | candidate | 0.86662517566146002 | 0.97066551658507783 | 0.5706963342459741 | 0.09573487331805712 | 0.80977548316641501 | 0.97273425685232073 | 0.32284489485725293 | 0.080697525754173577 | 0.056849692495045012 | 143 |
| 52 | baseline | 0.8787079850192796 | 0.97368431192366478 | 0.60436114219426085 | 0.092229015547175205 | 0.80997253661778768 | 0.97279507600979087 | 0.32238769665892147 | 0.080684004794831787 | 0.068735448401491928 | 227 |
| 52 | candidate | 0.88265453965238538 | 0.97464345058713264 | 0.61309545457119641 | 0.091163879503610143 | 0.80959020678474003 | 0.97275016368268663 | 0.3203424057106794 | 0.0808032401083775 | 0.073064332867645354 | 244 |
| 62 | baseline | 0.87075819630425255 | 0.97170347701296444 | 0.58209331366314332 | 0.094596966504173291 | 0.8091809346395975 | 0.97260003333242184 | 0.32327512466144837 | 0.080739139118575012 | 0.061577261664655047 | 172 |
| 62 | candidate | 0.88476988059027839 | 0.97525283438295551 | 0.61804964290209008 | 0.090658082650654268 | 0.81022275714329051 | 0.97278839068796708 | 0.32228911527858117 | 0.080631595357280386 | 0.074547123446987884 | 264 |
| 72 | baseline | 0.86774611911737154 | 0.97090233741595089 | 0.57594300517817065 | 0.095295459458776918 | 0.80954652606314181 | 0.97268830298153663 | 0.32210829879484304 | 0.080742190735292763 | 0.058199593054229726 | 151 |
| 72 | candidate | 0.86289090671506197 | 0.96969024284150951 | 0.56174210445735917 | 0.096745673898295859 | 0.81056385296104905 | 0.97274944302016642 | 0.32365227877016317 | 0.080691283071790784 | 0.05232705375401292 | 127 |
| 82 | baseline | 0.85612696942789923 | 0.96783950916492278 | 0.54751040829136355 | 0.098402065409668088 | 0.80972709467034232 | 0.97259630409036268 | 0.32383240525549156 | 0.080799918448565516 | 0.046399874757556914 | 100 |
| 82 | candidate | 0.88297844223990629 | 0.97475519817645029 | 0.61355333849344817 | 0.09114536593315746 | 0.80985872959388194 | 0.97271866380059069 | 0.32193427587076756 | 0.080707758789744302 | 0.073119712646024349 | 247 |

### VALID threshold metrics at 0.80

| Seed | Arm | Fail P | Fail R | Fail F1 | Pass P | Pass R | Pass F1 | TN | FP | FN | TP |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | baseline | 0.33072903950388205 | 0.4194587229825974 | 0.36984655709059139 | 0.93087938179297736 | 0.90206502322257953 | 0.91624571793531906 | 6773 | 9374 | 13706 | 126244 |
| 42 | candidate | 0.32980369403345905 | 0.4224314114076918 | 0.37041461891444244 | 0.93113022929512979 | 0.90095748481600568 | 0.91579539883427452 | 6821 | 9326 | 13861 | 126089 |
| 52 | baseline | 0.32101036385661086 | 0.44311636836564067 | 0.37230721198876054 | 0.9327992347243812 | 0.89186137906395146 | 0.91187106860804068 | 7155 | 8992 | 15134 | 124816 |
| 52 | candidate | 0.31918861145222177 | 0.4346318201523503 | 0.36807048827817695 | 0.93192901349638357 | 0.89304037156127192 | 0.91207034955849087 | 7018 | 9129 | 14969 | 124981 |
| 62 | baseline | 0.32829896408171166 | 0.42001610206230261 | 0.36853688357560116 | 0.9308544806148894 | 0.90085030367988572 | 0.91560665095555738 | 6782 | 9365 | 13876 | 126074 |
| 62 | candidate | 0.32301310043668124 | 0.45810367250882517 | 0.37887673828975338 | 0.93430782975592541 | 0.88922472311539835 | 0.91120898270894424 | 7397 | 8750 | 15503 | 124447 |
| 72 | baseline | 0.33407940833008953 | 0.42521830680621786 | 0.37417913294640182 | 0.93152827474270539 | 0.90220793140407285 | 0.91663369571135589 | 6866 | 9281 | 13686 | 126264 |
| 72 | candidate | 0.33681893230599891 | 0.4169195516194959 | 0.37261305141971551 | 0.93082800675923882 | 0.90528760271525544 | 0.91788017097732377 | 6732 | 9415 | 13255 | 126695 |
| 82 | baseline | 0.33682959594915779 | 0.40372824673313928 | 0.36725726035886314 | 0.9295905457683391 | 0.90828867452661666 | 0.91881616087143514 | 6519 | 9628 | 12835 | 127115 |
| 82 | candidate | 0.32270786933927248 | 0.43073016659441382 | 0.36897530438473169 | 0.93168085027314285 | 0.89569846373704898 | 0.91333539773037764 | 6955 | 9192 | 14597 | 125353 |

## Exact M2 metrics

| Seed | Arm | TRAIN MAE | TRAIN RMSE | TRAIN R2 | VALID MAE | VALID RMSE | VALID R2 | Best iter |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | baseline | 8.7566428180207385 | 12.466084655266121 | 0.49113589774413258 | 9.5667097306309596 | 12.854908853104188 | 0.35190917004813094 | 438 |
| 42 | candidate | 8.5848248751728882 | 12.295495635860027 | 0.50496745573466617 | 9.5293119553759826 | 12.813964812242361 | 0.35603105046241112 | 559 |
| 52 | baseline | 8.9432660180277139 | 12.644869364505761 | 0.4764352911169405 | 9.5714505170826882 | 12.855157806946629 | 0.35188406737855193 | 308 |
| 52 | candidate | 8.9776159327675842 | 12.673301386952536 | 0.4740781708122489 | 9.5549010097784368 | 12.847472085211535 | 0.35265881471603999 | 295 |
| 62 | baseline | 8.8442814451531664 | 12.563946260886995 | 0.48311514009492296 | 9.5387162884294021 | 12.808063845401405 | 0.35662402297236351 | 379 |
| 62 | candidate | 9.018408377361375 | 12.718857453367994 | 0.4702903666695214 | 9.5852359246154872 | 12.886114296657684 | 0.34875885463221512 | 273 |
| 72 | baseline | 8.5762227261173045 | 12.290547020958662 | 0.5053658508091865 | 9.5491759528971389 | 12.835566407271592 | 0.35385803345502753 | 576 |
| 72 | candidate | 8.6187250491290524 | 12.324539749154683 | 0.50262598644464518 | 9.5365669151877075 | 12.813732832730954 | 0.35605436662602363 | 527 |
| 82 | baseline | 9.1388574685523363 | 12.843883212953127 | 0.45982514125971585 | 9.6008191316346618 | 12.894956155591352 | 0.34786484458287481 | 223 |
| 82 | candidate | 8.869048856858587 | 12.572192843683279 | 0.48243638318177051 | 9.5503964572417583 | 12.827479438350265 | 0.35467197243412762 | 351 |

## Paired VALID deltas (candidate minus baseline)

Raw deltas are shown; lower is better for Brier, AUC gap, MAE, and RMSE.

| Seed | M1 AUC | M1 fail AP | M1 Brier | AUC gap | M2 MAE | M2 RMSE | M2 R2 | Cold-start AUC | Low-support AUC | Level-1 AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.00058695041733503039 | 0.00086173551598323073 | -8.0928629719173584e-05 | 0.001465742292207195 | -0.037397775254977006 | -0.040944040861827347 | 0.0041218804142801879 | 0.0081898718629257461 | 0.0067051198663145017 | -0.00053815982762606129 |
| 52 | -0.00038232983304764545 | -0.0020452909482420734 | 0.00011923531354571248 | 0.0043288844661534265 | -0.016549507304251421 | -0.0076857217350934093 | 0.00077474733748805757 | -0.011618455989708565 | -0.0041456612451357122 | -9.0167243974037525e-05 |
| 62 | 0.0010418225036930018 | -0.00098600938286719231 | -0.00010754376129462617 | 0.012969861782332837 | 0.046519636186085123 | 0.078050451256279629 | -0.0078651683401483874 | 0.0063964035207788594 | 0.0085221423549179942 | 0.00033433650294356632 |
| 72 | 0.0010173268979072336 | 0.0015439799753201311 | -5.0907663501978395e-05 | -0.0058725393002168058 | -0.012609037709431448 | -0.021833574540638168 | 0.0021963331709961009 | 0.0034986925611613096 | 0.0020604006567078725 | 0.00039480757081422624 |
| 82 | 0.00013163492353962525 | -0.0018981293847240011 | -9.2159658821214241e-05 | 0.026719837888467435 | -0.050422674392903488 | -0.067476717241087059 | 0.0068071278512528144 | -0.0043921751753823735 | -0.0066573156762835817 | 0.0011397654587270711 |

## Five-seed summary

The standard deviation below is the sample standard deviation of paired deltas.

| Metric | Baseline mean | Candidate mean | Mean delta | Median delta | SD delta | Min delta | Max delta | Range | Improved | Worsened |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m1_valid_auc | 0.80952312494798984 | 0.81000220592987537 | 0.00047908098188544914 | 0.00058695041733503039 | 0.00060843443712703534 | -0.00038232983304764545 | 0.0010418225036930018 | 0.0014241523367406472 | 4 | 1 |
| m1_valid_fail_ap | 0.32271733694239485 | 0.32221259409748881 | -0.00050474284490598098 | -0.00098600938286719231 | 0.0016287234490156193 | -0.0020452909482420734 | 0.0015439799753201311 | 0.0035892709235622045 | 2 | 3 |
| m1_valid_brier | 0.080748741496231569 | 0.080706280616273302 | -4.2460879958255984e-05 | -8.0928629719173584e-05 | 9.2739588338919691e-05 | -0.00010754376129462617 | 0.00011923531354571248 | 0.00022677907484033866 | 4 | 1 |
| m1_train_valid_auc_gap | 0.058059225616154286 | 0.065981583041943098 | 0.0079223574257888171 | 0.0043288844661534265 | 0.01248722021258236 | -0.0058725393002168058 | 0.026719837888467435 | 0.032592377188684241 | 1 | 4 |
| m2_valid_mae | 9.5653743241349698 | 9.5512824524398745 | -0.014091871695095648 | -0.016549507304251421 | 0.037232667183725913 | -0.050422674392903488 | 0.046519636186085123 | 0.096942310578988611 | 4 | 1 |
| m2_valid_rmse | 12.849730613663032 | 12.837752693038562 | -0.011977920624473271 | -0.021833574540638168 | 0.055090419258745582 | -0.067476717241087059 | 0.078050451256279629 | 0.14552716849736669 | 4 | 1 |
| m2_valid_r2 | 0.35242802768738973 | 0.35363501177416345 | 0.0012069840867737548 | 0.0021963331709961009 | 0.0055526529843431182 | -0.0078651683401483874 | 0.0068071278512528144 | 0.014672296191401202 | 4 | 1 |
| cold_start_auc | 0.73362794872736503 | 0.73404281608331989 | 0.00041486735595499538 | 0.0034986925611613096 | 0.0082711543406914344 | -0.011618455989708565 | 0.0081898718629257461 | 0.019808327852634311 | 3 | 2 |
| low_difficulty_support_auc | 0.76638545958307236 | 0.76768239677437655 | 0.0012969371913042148 | 0.0020604006567078725 | 0.006612997096390294 | -0.0066573156762835817 | 0.0085221423549179942 | 0.015179458031201576 | 3 | 2 |
| level_1_auc | 0.82101911517462656 | 0.82126723166680349 | 0.000248116492176953 | 0.00033433650294356632 | 0.00062383018113616067 | -0.00053815982762606129 | 0.0011397654587270711 | 0.0016779252863531324 | 3 | 2 |

## Segment stability

`first_semester` and `cold_start_gpa` are exactly the same VALID population in every run (n=14732), so they are one piece of evidence.

| Seed | Segment | Baseline AUC | Candidate AUC | Delta | n |
|---:|---|---:|---:|---:|---:|
| 42 | first_semester | 0.73293108271348573 | 0.74112095457641147 | 0.0081898718629257461 | 14732 |
| 42 | cold_start_gpa | 0.73293108271348573 | 0.74112095457641147 | 0.0081898718629257461 | 14732 |
| 42 | retake_attempt | 0.6768272871085127 | 0.67646287652778936 | -0.00036441058072333998 | 17958 |
| 42 | low_difficulty_support | 0.76440513157070356 | 0.77111025143701806 | 0.0067051198663145017 | 25627 |
| 42 | level_1_difficulty | 0.82096152041939818 | 0.82042336059177212 | -0.00053815982762606129 | 120858 |
| 52 | first_semester | 0.73573682851699518 | 0.72411837252728661 | -0.011618455989708565 | 14732 |
| 52 | cold_start_gpa | 0.73573682851699518 | 0.72411837252728661 | -0.011618455989708565 | 14732 |
| 52 | retake_attempt | 0.67337507775696748 | 0.67537463671138298 | 0.0019995589544155035 | 17958 |
| 52 | low_difficulty_support | 0.76529536879317439 | 0.76114970754803868 | -0.0041456612451357122 | 25627 |
| 52 | level_1_difficulty | 0.82173104743447645 | 0.82164088019050241 | -9.0167243974037525e-05 | 120858 |
| 62 | first_semester | 0.72839757783705461 | 0.73479398135783347 | 0.0063964035207788594 | 14732 |
| 62 | cold_start_gpa | 0.72839757783705461 | 0.73479398135783347 | 0.0063964035207788594 | 14732 |
| 62 | retake_attempt | 0.67600729419542127 | 0.67574114487654746 | -0.00026614931887380955 | 17958 |
| 62 | low_difficulty_support | 0.76243029139921736 | 0.77095243375413536 | 0.0085221423549179942 | 25627 |
| 62 | level_1_difficulty | 0.82110029640216742 | 0.82143463290511098 | 0.00033433650294356632 | 120858 |
| 72 | first_semester | 0.73538830335363137 | 0.73888699591479268 | 0.0034986925611613096 | 14732 |
| 72 | cold_start_gpa | 0.73538830335363137 | 0.73888699591479268 | 0.0034986925611613096 | 14732 |
| 72 | retake_attempt | 0.67533424601563019 | 0.67856990009904505 | 0.0032356540834148628 | 17958 |
| 72 | low_difficulty_support | 0.76886933299357052 | 0.77092973365027839 | 0.0020604006567078725 | 25627 |
| 72 | level_1_difficulty | 0.82086561787258072 | 0.82126042544339495 | 0.00039480757081422624 | 120858 |
| 82 | first_semester | 0.73568595121565794 | 0.73129377604027557 | -0.0043921751753823735 | 14732 |
| 82 | cold_start_gpa | 0.73568595121565794 | 0.73129377604027557 | -0.0043921751753823735 | 14732 |
| 82 | retake_attempt | 0.68057837785242215 | 0.67395894607731555 | -0.006619431775106599 | 17958 |
| 82 | low_difficulty_support | 0.77092717315869597 | 0.76426985748241238 | -0.0066573156762835817 | 25627 |
| 82 | level_1_difficulty | 0.82043709374451002 | 0.82157685920323709 | 0.0011397654587270711 | 120858 |

Cold-start direction was mixed (3/5 improved), low-difficulty-support was mixed, and Level-1 did not provide a stable independent pattern. This does not meet strong segment-repeatability support for M1.

## Concurrent feature evidence

| Seed | Model | Feature | Gain | Splits | Gain rank | % total gain | Zero splits |
|---:|---|---|---:|---:|---:|---:|:---:|
| 42 | M1 | concurrent_peer_difficulty_mean | 12808.223442554474 | 692 | 16 | 1.2986886991021207 | no |
| 42 | M1 | concurrent_peer_difficulty_max | 15370.886476516724 | 778 | 14 | 1.5585296939707869 | no |
| 42 | M1 | concurrent_peer_difficulty_missing | 0 | 0 | 40 | 0 | yes |
| 42 | M2 | concurrent_peer_difficulty_mean | 42490.151482343681 | 3094 | 15 | 1.6123898122188001 | no |
| 42 | M2 | concurrent_peer_difficulty_max | 50869.736721277237 | 3488 | 12 | 1.9303730953683065 | no |
| 42 | M2 | concurrent_peer_difficulty_missing | 0 | 0 | 40 | 0 | yes |
| 52 | M1 | concurrent_peer_difficulty_mean | 19297.808964490891 | 1474 | 13 | 1.7432280186096407 | no |
| 52 | M1 | concurrent_peer_difficulty_max | 22391.854878664017 | 1598 | 11 | 2.0227223144841546 | no |
| 52 | M1 | concurrent_peer_difficulty_missing | 8.8362598419189453 | 1 | 40 | 0.00079820542137668915 | no |
| 52 | M2 | concurrent_peer_difficulty_mean | 25114.166133880612 | 1311 | 17 | 1.1001011596190204 | no |
| 52 | M2 | concurrent_peer_difficulty_max | 30490.645674705505 | 1506 | 14 | 1.3356125178699374 | no |
| 52 | M2 | concurrent_peer_difficulty_missing | 0 | 0 | 41 | 0 | yes |
| 62 | M1 | concurrent_peer_difficulty_mean | 20675.72047591209 | 1604 | 13 | 1.8419792930316838 | no |
| 62 | M1 | concurrent_peer_difficulty_max | 23129.269669771194 | 1791 | 11 | 2.0605635409077512 | no |
| 62 | M1 | concurrent_peer_difficulty_missing | 0 | 0 | 41 | 0 | yes |
| 62 | M2 | concurrent_peer_difficulty_mean | 23621.624220848083 | 1156 | 19 | 1.0472199096663031 | no |
| 62 | M2 | concurrent_peer_difficulty_max | 28815.804204463959 | 1387 | 13 | 1.27749402809174 | no |
| 62 | M2 | concurrent_peer_difficulty_missing | 0 | 0 | 40 | 0 | yes |
| 72 | M1 | concurrent_peer_difficulty_mean | 14236.858572006226 | 651 | 14 | 1.4809198047609247 | no |
| 72 | M1 | concurrent_peer_difficulty_max | 14107.491127490995 | 695 | 15 | 1.4674629870433824 | no |
| 72 | M1 | concurrent_peer_difficulty_missing | 0 | 0 | 40 | 0 | yes |
| 72 | M2 | concurrent_peer_difficulty_mean | 41885.588130950928 | 2976 | 15 | 1.6141077391183398 | no |
| 72 | M2 | concurrent_peer_difficulty_max | 47860.972139835358 | 3230 | 12 | 1.8443758099113416 | no |
| 72 | M2 | concurrent_peer_difficulty_missing | 0 | 0 | 41 | 0 | yes |
| 82 | M1 | concurrent_peer_difficulty_mean | 20700.374820947647 | 1529 | 14 | 1.8672129964970599 | no |
| 82 | M1 | concurrent_peer_difficulty_max | 24590.587048769001 | 1765 | 11 | 2.2181175039637164 | no |
| 82 | M1 | concurrent_peer_difficulty_missing | 0 | 0 | 41 | 0 | yes |
| 82 | M2 | concurrent_peer_difficulty_mean | 28130.401743888851 | 1551 | 17 | 1.1827653153646056 | no |
| 82 | M2 | concurrent_peer_difficulty_max | 35559.84498167038 | 1965 | 14 | 1.4951422182656489 | no |
| 82 | M2 | concurrent_peer_difficulty_missing | 13.954299926757812 | 1 | 40 | 0.00058671973844630668 | no |

`concurrent_peer_difficulty_missing` was used by at least one model in 2 of 5 seeds (seeds [52, 82]); therefore it did **not** remain unused in all five seeds. M1 used it in seed 52 only; M2 used it in seed 82 only, each with one split.

## Separate findings

### M1 — INCONCLUSIVE

The candidate shows a consistent but small VALID AUC/Brier direction, yet the operational fail-class AP result is mixed and the generalization gap usually worsens, sometimes by more than the VALID AUC benefit. Cold-start improves in only 3/5 seeds and is identical to first_semester, so it is not independent confirmation. The M1 improvement is inside observed seed variability and does not satisfy strong support.

### M2 — SUPPORTED

Four seeds improve all three VALID regression metrics and one seed (62) worsens all three. Multiple improving seeds contribute to the mean, including seeds 42, 52, 72, and 82. This is a stable direction across seeds and consistent but small, with no claim of statistical significance.

## Integrity and stop-gate confirmations

- No training seed failed and no seed was rerun. Exactly eight new persistent training runs were created.
- Seed 52 had one provenance-only metadata repair after its first launcher failed to inject Git safe-directory state; models and metrics were not changed.
- TEST policy is `closed_not_read` in all ten runs; all M1/M2 TEST metric fields are null. The four new commands used nonexistent TEST paths, and those paths remain nonexistent.
- Only `df_train_final.parquet` and `df_valid_final.parquet` were used for data/model evaluation. The TEST parquet was never read.
- Train/valid hashes still equal the seed-42 immutable artifact hashes.
- No dataset, root/live model artifact, production contract, promotion marker, `CURRENT_VERSION.txt`, recommendation wiring, or inference wiring was changed.
- No commit or push was performed.
- No `concurrent_43`, regularization experiment, promotion, or recommendation change was created.

## Overall next action

Keep M1 on `baseline_41`. Treat the M2 evidence for `concurrent_44` as supported but do not promote or rewire anything in this task; await explicit human review of the model-specific deployment implications.
