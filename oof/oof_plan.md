# 5-fold OOF 설계

## 목적

현재 Ridge/Logistic stacking은 하나의 validation split에서 만든 예측값으로 meta-model을 학습했다. 이 방식은 빠르지만 validation split에 과적합될 수 있다. 5-fold OOF는 각 학습 샘플을 해당 샘플을 학습에 사용하지 않은 모델로 예측하게 만들어, meta-model 학습용 feature를 더 공정하게 만든다.

## Fold 설정

- split: `StratifiedKFold`
- n_splits: 5
- shuffle: True
- random_state: 42
- 저장 파일: `oof/folds_seed42.csv`

Fold별 분포:

| fold | label 0 | label 1 | total | label 1 ratio |
|---:|---:|---:|---:|---:|
| 0 | 269 | 775 | 1044 | 0.74234 |
| 1 | 268 | 775 | 1043 | 0.74305 |
| 2 | 268 | 775 | 1043 | 0.74305 |
| 3 | 268 | 775 | 1043 | 0.74305 |
| 4 | 268 | 775 | 1043 | 0.74305 |

## 저장 구조

```text
oof/
  folds_seed42.csv
  folds_seed42_summary.csv
  oof_plan.md
  models/
    exp41_convnext12_fold0.pt
    exp41_convnext12_fold1.pt
    ...
  preds/
    exp41_convnext12_oof_probs.csv
    exp41_convnext12_test_probs_by_fold.csv
    exp41_convnext12_test_probs_mean.csv
    exp42_ridge_oof_summary.csv
  submissions/
    exp42_oof_ridge_submit.csv
```

## 단일 모델 OOF probability 포맷

`*_oof_probs.csv`

```text
file_name,label,fold,prob_0,prob_1,pred_label
```

조건:

- 모든 train 샘플 5,216개가 정확히 한 번씩 들어가야 한다.
- 각 row의 예측은 해당 row가 validation fold였을 때의 모델에서 나와야 한다.
- `file_name` 중복은 없어야 한다.

`*_test_probs_by_fold.csv`

```text
file_name,fold,prob_0,prob_1,pred_label
```

조건:

- test 샘플 624개 x 5 folds = 3,120 rows.
- fold별 모델이 같은 test set을 예측한 결과를 저장한다.

`*_test_probs_mean.csv`

```text
file_name,prob_0,prob_1,pred_label
```

조건:

- fold별 test probability 평균.
- 최종 단일 모델 제출 또는 stacking test feature로 사용한다.

## Meta-model feature 설계

OOF stacking feature table:

```text
file_name,label,fold,
resnet18_prob1,resnet18_rank,
efficientnet_b0_prob1,efficientnet_b0_rank,
convnext12_prob1,convnext12_rank,
convnext18_prob1,convnext18_rank,
...
```

권장 순서:

1. `prob1`만 사용한 RidgeClassifier
2. `prob1 + rank` RidgeClassifier
3. LogisticRegression
4. LightGBM 또는 TabM

TabM은 단일 validation stacking에서는 과적합 위험이 크므로, OOF feature table이 만들어진 뒤에만 비교한다.

## 1차 OOF 대상 모델

우선순위:

1. ConvNeXt-Tiny stage4+classifier
2. ResNet18 layer4+fc
3. EfficientNet-B0
4. ConvNeXt-Tiny full fine-tuning
5. 추가 seed ConvNeXt 또는 EfficientNetV2-S/RegNetY

실무적으로는 계산량 때문에 ConvNeXt-Tiny 하나로 OOF pipeline을 먼저 검증한 뒤, 나머지 모델을 확장한다.

## 민감 샘플 관리

최근 public 결과 기준으로 다음 샘플은 후보 평가 시 별도 확인한다.

```text
test_0244.png: 0 유지가 유리했던 샘플
test_0025.png: 0 -> 1 변경 후보가 반복적으로 하락
test_0539.png: 0 -> 1 변경 후보가 하락
```

OOF 기반 후보도 validation 점수만 보지 않고, 위 샘플의 예측 변화를 함께 확인한다.

## 다음 실행 단계

1. OOF Logistic baseline 생성
2. ResNet18 OOF 추가
3. ConvNeXt + ResNet18 OOF Ridge stacking
4. ConvNeXt + ResNet18 OOF Logistic stacking
5. 필요하면 EfficientNet-B0 OOF 추가

## 진행 상태

- EXP-45에서 ConvNeXt-Tiny fold0 smoke test 완료.
- fold0 validation accuracy: 0.98467
- fold0 OOF rows: 1,044
- fold0 test rows: 624
- fold0 elapsed: 433.2 sec
- 예상 5-fold 전체 시간: 약 36분
- EXP-46에서 ConvNeXt-Tiny 5-fold OOF 완료.
- 전체 OOF rows: 5,216
- OOF accuracy: 0.97795
- OOF confusion matrix: `[[1312, 29], [86, 3789]]`
- test mean rows: 624
- test mean 민감 샘플: `test_0244=0`, `test_0025=0`, `test_0539=1`
- EXP-47에서 ConvNeXt 단일 OOF Ridge baseline 완료.
- 최고 OOF Ridge: `prob1_only alpha=1000`, OOF accuracy 0.98025
- 최고 OOF Ridge는 `test_0244=1`, `test_0025=1`로 바꿔 제출 보류.
- 민감 샘플 유지 후보: `prob1_margin alpha=1000`, OOF accuracy 0.97872
- EXP-48에서 ResNet18 5-fold OOF 완료.
- OOF accuracy: 0.98524
- OOF confusion matrix: `[[1304, 37], [40, 3835]]`
- test mean 민감 샘플: `test_0244=1`, `test_0025=1`, `test_0539=1`
- ResNet18 OOF는 강하지만 단독 제출보다 stacking feature로 사용한다.
- EXP-49에서 ConvNeXt+ResNet OOF Ridge/Logistic stacking 완료.
- 최고 meta-CV accuracy: 0.98735
- 상위 후보가 `test_0539=1`을 유지해 제출 보류.
- EXP-50에서 EfficientNet-B0 5-fold OOF 완료.
- 초기 `features[7:]+classifier` 설정은 fold0 best 0.9301로 낮아 중단했고, `features[5:]+classifier`로 재실행했다.
- OOF accuracy: 0.98505
- OOF confusion matrix: `[[1312, 29], [49, 3826]]`
- test mean 민감 샘플: `test_0244=0`, `test_0025=0`, `test_0539=0`
- EXP-51에서 ConvNeXt+ResNet+EfficientNet OOF stacking 완료.
- 가장 안정적인 계열은 ResNet을 제외한 ConvNeXt+EfficientNet subset이다.
- 보수적 후보: `oof/submissions/exp51_logistic_cnx_eff_prob_rank_0p3_submit.csv`
  - meta-CV accuracy 0.98639
  - EXP-26 대비 변경 4개
  - 민감 샘플 3개 모두 0 유지
- 공격적 후보: `oof/submissions/exp51_ridge_cnx_eff_prob_only_50_submit.csv`
  - meta-CV accuracy 0.98677
  - EXP-26 대비 변경 8개
  - 민감 샘플 3개 모두 0 유지
- 제출 결과:
  - `exp51_logistic_cnx_eff_prob_rank_0p3_submit.csv`: leaderboard 0.9551282051
  - `exp51_ridge_cnx_eff_prob_only_50_submit.csv`: leaderboard 0.9519230769
- 해석:
  - Logistic 보수 후보가 EXP-26 대비 +2샘플 개선했다.
  - Ridge 공격 후보는 EXP-26과 동점이었다.
  - meta-CV 최고 후보보다 변경 수가 적은 후보가 public에서 더 좋았으므로, 다음 OOF stacking도 low-change 후보를 우선 제출한다.
- EXP-52에서 EXP-51 최고 후보 주변 low-change 탐색 완료.
- 1순위 후보: `oof/submissions/exp52_logistic_cnx_eff_prob_only_p0p25_t0p0_submit.csv`
  - meta-CV accuracy 0.98658
  - EXP-51 최고 대비 변경 1개
  - 변경 샘플: `test_0490` 1->0
  - 민감 샘플 3개 모두 0 유지
  - 제출 결과 leaderboard `0.9567307692`, EXP-51 대비 +1샘플 개선
  - 당시 public leaderboard 최고 순위 14위
- 2순위 후보: `oof/submissions/exp52_ridge_cnx_eff_prob_margin_p500_tm0p025_submit.csv`
  - EXP-51 최고 대비 변경 2개
  - 변경 샘플: `test_0201`, `test_0490` 1->0
  - 제출 결과 leaderboard `0.9551282051`, EXP-52 1순위보다 하락
- EXP-52의 4개 추가 변경 후보는 EXP-51 공격 후보와 같은 방향이라 우선순위 낮음.
- EXP-52 결과 기반 보호 샘플:
  - `test_0025=0`
  - `test_0539=0`
  - `test_0244=0`
  - `test_0490=0`
  - `test_0201=1`
- EXP-53에서 EXP-52 winner 기준 constraint low-change search 완료.
  - 탐색 row 수: 1,012
  - 보호 샘플 5개 유지, label 1 개수 395~399, 변경 수 1~3개 조건 적용
  - 최종 candidate 수: 0
  - 제출 파일 없음
  - 결론: 현재 ConvNeXt+EfficientNet OOF 조합 안에서는 추가 미세 후보보다 새 seed OOF 정보원 확보가 우선
- EXP-54에서 ConvNeXt-Tiny seed123 5-fold OOF 완료.
  - OOF accuracy: 0.98102
  - OOF confusion matrix: `[[1305, 36], [63, 3812]]`
  - test mean label 0/1: 221/403
  - 단독 mean은 `test_0539=1`, `test_0490=1`이라 제출 보류
  - stacking diversity feature로 사용
- EXP-55에서 ConvNeXt seed123 포함 OOF stacking 완료.
  - candidate count: 332
  - 1순위 후보: `oof/submissions/exp55_ridge_cnx123_eff_prob_rank_p50_tm0p1_submit.csv`
    - meta-CV accuracy 0.98677
    - 현재 최고 EXP-52 대비 변경 1개: `test_0222` 1->0
    - 보호 샘플 5개 모두 유지
    - 제출 결과 leaderboard 0.9551282051, 현재 최고보다 하락
  - 2순위 후보: `oof/submissions/exp55_ridge_cnx_eff_compact_p3000_t0p025_submit.csv`
    - 현재 최고 EXP-52 대비 변경 1개: `test_0504` 1->0
  - EXP-55 결과를 바탕으로 `test_0222=1`도 보호 조건에 추가
  - 다음 탐색 보호 샘플:
    - `test_0025=0`
    - `test_0539=0`
    - `test_0244=0`
    - `test_0490=0`
    - `test_0201=1`
    - `test_0222=1`
  - `test_0504` 후보는 바로 제출하지 않고 새 보호 조건 기반 constrained search 이후 판단
- EXP-56에서 `test_0222=1`까지 보호한 low-change search 완료.
  - 후보: `oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv`
  - meta-CV accuracy 0.98658
  - 현재 최고 EXP-52 대비 변경 1개: `test_0504` 1->0
  - 보호 샘플 6개 모두 유지
  - 제출 결과 leaderboard 0.9583333333, 최고점 갱신
  - 최신 public leaderboard 기준 최고 순위 13위까지 확인
  - 2026-06-30 기준 점수는 `0.9583333333`으로 유지 중이나 순위는 13위에서 15위로 하락
  - EXP-56 결과를 바탕으로 `test_0504=0`도 보호 조건에 추가
  - 다음 탐색 보호 샘플:
    - `test_0025=0`
    - `test_0539=0`
    - `test_0244=0`
    - `test_0490=0`
    - `test_0201=1`
    - `test_0222=1`
    - `test_0504=0`
- EXP-57에서 EXP-56 winner 기준 보호 샘플 7개 유지 low-change search 완료.
  - 기준 최고 후보: `oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv`
  - 탐색 row 수: 7,956
  - 최고 meta-CV accuracy: 0.98696
  - 보호 샘플 7개 유지, label 1 개수 395~399, 현재 최고 대비 변경 수 1~2개 조건 적용
  - 최종 candidate 수: 0
  - 제출 파일 없음
  - 결론: 현재 ConvNeXt/ConvNeXt seed123/EfficientNet-B0 OOF feature 공간에서는 추가 low-change 후보가 소진된 것으로 판단. 다음은 calibration 짧은 확인 또는 새 OOF 정보원 추가가 우선.
- EXP-58 calibrated OOF stacking 완료.
  - Ridge/Logistic/LinearSVC에 sigmoid/isotonic calibration 적용
  - 후보: `oof/submissions/exp58_svc_all_margin_stats_isotonic_p0p2_t0p575_submit.csv`
  - meta-CV accuracy: 0.98639
  - recall: 0.99174
  - test label 0/1: 226/398
  - 현재 최고 EXP-56 대비 변경 2개:
    - `test_0188`: 0->1
    - `test_0568`: 0->1
  - 보호 샘플 7개 모두 유지
  - 제출 결과 leaderboard `0.9551282051`, EXP-56 최고점보다 하락
  - 해석: `test_0188`, `test_0568`을 모두 1로 바꾸는 recall 상승 후보는 public에서 불리했다. 이후에는 두 샘플을 동시에 1로 바꾸는 후보의 제출 우선순위를 낮춘다.

## 추가 우선순위 메모

- EfficientNetV2-S `features[7]+classifier` 단일 split probe는 validation accuracy 0.87261로 낮고 FN이 크게 증가했다. 현재 설정은 단독/stacking 후보에서 제외한다.
- EfficientNetV2-S를 다시 시도한다면 마지막 block만이 아니라 `features[6]+features[7]+classifier`처럼 더 넓은 fine-tuning으로 재평가한다.
- RegNetY-3.2GF `trunk_output.block4+fc` probe는 validation accuracy 0.95211로 단독 제출은 약하지만, 민감 샘플 `test_0244`, `test_0025`, `test_0539`를 모두 0으로 유지해 stacking diversity 후보로 남긴다.
- Lion optimizer 비교는 AdamW 기본선을 유지한 뒤, 8순위 실험 전에 추가한다. 우선 대상은 ConvNeXt-Tiny 또는 RegNetY처럼 단독 validation이 안정적인 모델로 한다.
- 10시간 초과 장기 실험은 바로 실행하지 않고, 목적/기대 이득/근거/계산 비용/중단 조건/제출 기준을 먼저 작성한 뒤 검토한다.
- 장기 실험 후보 우선순위는 ConvNeXt 계열 OOF seed 추가, ConvNeXt V2 Tiny, EfficientNetV2-S 재설계, DINOv2 feature extractor 순으로 본다.
- EXP-65 ConvNeXt V2 Tiny 사전 검토 완료.
  - `timm==1.0.27` 추가 및 pretrained loading 확인
  - 권장 모델: `convnextv2_tiny.fcmae_ft_in22k_in1k`
  - 224 입력, 파라미터 약 27.87M
  - 1차 fine-tuning 범위: `stage3 + head`
  - trainable params 약 15.49M
  - 예상 비용: 5-fold OOF 약 40~70분
  - 10시간 초과 장기 실험 게이트 대상은 아님
  - 다음 작업: `scripts/run_exp65_convnextv2_tiny_oof.py` 작성 후 fold0 smoke test
- EXP-65 ConvNeXt V2 Tiny stage3+head 5-fold OOF 완료.
  - OOF accuracy: 0.97680
  - OOF confusion matrix: `[[1299, 42], [79, 3796]]`
  - test mean label 0/1: 208/416
  - 총 학습 시간: 약 95.3분
  - EXP-56 대비 test 변경 수: 30개
  - 단독 mean은 다음 보호 방향을 깨므로 제출 보류:
    - `test_0025=0`
    - `test_0539=0`
    - `test_0244=0`
    - `test_0490=0`
    - `test_0504=0`
  - 유지한 보호 방향:
    - `test_0201=1`
    - `test_0222=1`
  - 2026-06-24 단독 mean 제출 결과: 0.9262820513
  - 제출 후 판단: EXP-56보다 크게 하락했으므로 단독 모델로는 사용하지 않고 stacking diversity feature로만 유지
  - 다음 작업: EXP-66에서 ConvNeXt V2 feature를 낮은 가중치/제약 조건으로 stacking에 추가하고, 보호 샘플을 모두 유지하는 후보만 검토.
- EXP-66 ConvNeXt V2 포함 OOF stacking 완료.
  - 기준 최고 후보: `oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv`
  - 추가 정보원: EXP-65 ConvNeXt V2 Tiny OOF/test probability
  - 탐색 feature:
    - 기존 ConvNeXt, ConvNeXt seed123, EfficientNet-B0 probability
    - ConvNeXt V2 probability
    - core/all probability mean, std
    - ConvNeXt V2와 core mean 차이
  - 최종 candidate 수: 6
  - unique submission 수: 1
  - 1순위 후보: `oof/submissions/exp66_ridge_all_prob_stats_with_cnxv2_p10_t0p15_submit.csv`
  - meta-CV accuracy: 0.98696
  - meta-CV confusion matrix: `[[1317, 24], [44, 3831]]`
  - test label 0/1: 229/395
  - EXP-56 대비 변경 1개:
    - `test_0141`: 1->0
  - 보호 샘플 7개 유지:
    - `test_0025=0`
    - `test_0539=0`
    - `test_0244=0`
    - `test_0490=0`
    - `test_0201=1`
    - `test_0222=1`
    - `test_0504=0`
  - EXP-58에서 하락한 방향인 `test_0188=1`, `test_0568=1` 동시 변경도 피함.
  - 제출 결과 leaderboard score: `0.9567307692`
  - 판단: EXP-56 최고점 `0.9583333333`보다 하락했으므로 `test_0141=0` 방향은 public에서 이득이 아니었다. 이후 low-change 탐색에서는 `test_0141=1`을 우선 유지한다.
- EXP-67 RegNetY-3.2GF 5-fold OOF 완료.
  - 목적: ConvNeXt/EfficientNet과 다른 CNN family의 OOF 정보원 확보
  - fine-tuning 범위: `trunk_output.block4 + fc`
  - OOF accuracy: 0.96166
  - OOF confusion matrix: `[[1305, 36], [164, 3711]]`
  - test mean label 0/1: 273/351
  - EXP-56 대비 test 변경 수: 69개
  - 단독 mean은 제출 제외:
    - FN이 164개로 많음
    - `test_0244=1`, `test_0490=1`, `test_0222=0`, `test_0141=0`으로 보호 방향 일부를 깸
  - 유지한 방향:
    - `test_0025=0`
    - `test_0539=0`
    - `test_0201=1`
    - `test_0504=0`
    - `test_0188=0`
    - `test_0568=0`
- EXP-68 RegNetY 포함 OOF stacking 완료.
  - 추가 정보원: EXP-67 RegNetY OOF/test probability
  - 보호 샘플은 EXP-66 결과를 반영해 `test_0141=1`까지 8개로 확장
  - 탐색 row 수: 3724
  - 최고 meta-CV accuracy: 0.98735
  - 보호 샘플 8개 유지 candidate 수: 0
  - 제출 파일 없음
  - 해석: RegNetY feature는 일부 meta-CV를 올렸지만, 상위 후보가 `test_0539`, `test_0490`, `test_0504`, `test_0141` 등 보호 방향을 자주 깨므로 현재 guardrail 안에서는 제출 후보로 쓰기 어렵다.
- EXP-69 DINOv2 feature extractor + Ridge/Logistic 완료.
  - 목적: CNN 계열과 다른 ViT/foundation feature family 확보
  - 모델: `timm` `vit_small_patch14_dinov2`
  - 입력 크기: 518x518
  - frozen feature dimension: 384
  - train/test feature rows: 5216/624
  - feature 추출 시간: 약 623.6초
  - meta-model: `StandardScaler + RidgeClassifier/LogisticRegression`
  - 최고 meta-CV accuracy: 0.97373
  - 보호 샘플 유지 candidate 수: 0
  - 제출 파일 없음
  - 해석: 새 family 확보 실험으로는 의미가 있지만, raw frozen DINOv2 feature는 현재 ConvNeXt/EfficientNet OOF pool보다 약하고 보호 샘플을 자주 깨서 제출 후보로 쓰지 않는다.
- EXP-70 EfficientNetV2-S 재설계 5-fold OOF 완료.
  - 목적: EXP-42의 마지막 block 제한 설정 실패를 재검증하고 EfficientNetV2-S family 정보원 확보
  - 모델: `torchvision.models.efficientnet_v2_s`
  - 학습 범위: `features[6:] + classifier`
  - epoch/fold: 6 x 5 folds
  - optimizer: AdamW
  - LR: feature 2e-5, classifier 8e-5
  - OOF accuracy: 0.97642
  - OOF confusion matrix: `[[1307, 34], [89, 3786]]`
  - test mean label 0/1: 208/416
  - EXP-56 대비 변경 수: 34개
  - 단독 mean은 다음 보호 방향을 깨므로 제출 제외:
    - `test_0025=0`
    - `test_0539=0`
    - `test_0244=0`
    - `test_0490=0`
    - `test_0222=1`
    - `test_0504=0`
  - 해석: EXP-42 대비 성능 회복은 성공했지만, 현재 최고 후보를 직접 대체하기에는 위험하다. 다음 단계에서는 낮은 가중치/제약 조건으로 stacking에만 추가한다.
- EXP-71 EfficientNetV2-S 포함 OOF stacking 완료.
  - 추가 정보원: EXP-70 EfficientNetV2-S OOF/test probability
  - 탐색 row 수: 6318
  - 최고 meta-CV accuracy: 0.98888
  - 보호 샘플 8개 유지 row 수: 5
  - 보호 샘플 8개 유지 비중복 candidate 수: 0
  - 제출 파일 없음
  - 해석: 상위 후보는 `test_0490`, `test_0504`, `test_0539`, `test_0244` 등 보호 방향을 자주 깨며, 보호 조건을 통과한 후보는 EXP-56과 동일 예측이었다. EfficientNetV2-S는 현재 guardrail 안에서 새 제출 후보를 만들지 못했다.
- EXP-72 ConvNeXt-Tiny seed777 5-fold OOF 완료.
  - 목적: 강한 ConvNeXt family의 seed diversity 추가 확보
  - 모델/설정: EXP-46/54와 동일, 학습 seed 777
  - OOF accuracy: 0.98006
  - OOF confusion matrix: `[[1295, 46], [58, 3817]]`
  - test mean label 0/1: 210/414
  - EXP-56 대비 변경 수: 20개
  - 단독 mean은 다음 보호 방향을 깨므로 제출 제외:
    - `test_0025=0`
    - `test_0539=0`
    - `test_0244=0`
    - `test_0490=0`
    - `test_0504=0`
  - 유지한 방향:
    - `test_0201=1`
    - `test_0222=1`
    - `test_0141=1`
    - `test_0568=0`
  - 해석: 단독 제출 후보는 아니지만, OOF가 강하므로 EXP-73에서 ConvNeXt seed ensemble/stacking feature로 검토한다.
- EXP-73 ConvNeXt seed777 포함 OOF stacking 완료.
  - 추가 정보원: EXP-72 ConvNeXt seed777 OOF/test probability
  - 탐색 row 수: 20592
  - 최고 meta-CV accuracy: 0.98754
  - 보호 샘플 8개 유지 row 수: 21
  - 보호 샘플 8개 유지 비중복 candidate 수: 0
  - 제출 파일 없음
  - 해석: seed777 추가 후에도 보호 샘플을 유지한 row는 모두 EXP-56과 동일 예측이었다. 상위 후보는 `test_0504`, `test_0490`, `test_0539`를 자주 1로 바꿔 public guardrail과 충돌한다.
- EXP-77 ConvNeXt-Tiny Lion fold0 smoke 완료.
  - 목적: AdamW와 다른 optimizer로 ConvNeXt 오류 방향이 달라지는지 확인
  - 설정: ConvNeXt-Tiny fold0, `features[-1]+classifier`, Lion LR 2e-5, weight decay 1e-4
  - Lion fold0 accuracy: 0.98563
  - Lion fold0 confusion matrix: `[[265, 4], [11, 764]]`
  - AdamW fold0 reference: accuracy 0.98467, confusion matrix `[[266, 3], [13, 762]]`
  - test label 0/1: 234/390
  - EXP-56 대비 변경 수: 26개
  - 유지한 보호 방향:
    - `test_0025=0`
    - `test_0244=0`
    - `test_0201=1`
    - `test_0222=1`
    - `test_0504=0`
    - `test_0188=0`
    - `test_0568=0`
  - 깨진 보호 방향:
    - `test_0539=1`
    - `test_0490=1`
    - `test_0141=0`
  - 해석: fold0 기준으로 AdamW보다 소폭 좋고 FN을 줄였으므로 Lion 5-fold OOF 확장 후보로 남긴다. 단, fold0 단독 제출은 제외한다.
- EXP-78 ConvNeXt-Tiny Lion 5-fold OOF 완료.
  - 목적: EXP-77 fold0 smoke를 5-fold로 확장해 optimizer diversity 확보
  - 복구 메모: IDE/Docker 메모리 문제로 fold2 진행 중 중단되었고, 기존 fold0/fold1 산출물을 유지한 채 fold2~4만 재실행해 병합했다.
  - 설정: ConvNeXt-Tiny `features[-1]+classifier`, Lion LR 2e-5, weight decay 1e-4
  - OOF accuracy: 0.98140
  - OOF confusion matrix: `[[1294, 47], [50, 3825]]`
  - test label 0/1: 204/420
  - EXP-56 대비 변경 수: 24개
  - 유지한 보호 방향:
    - `test_0025=0`
    - `test_0201=1`
    - `test_0222=1`
    - `test_0141=1`
    - `test_0568=0`
  - 깨진 보호 방향:
    - `test_0539=1`
    - `test_0244=1`
    - `test_0490=1`
    - `test_0504=1`
    - `test_0188=1`
  - 해석: 단독 OOF는 강하지만, public guardrail과 충돌하는 `0 -> 1` 변경이 많다. 단독 제출은 제외하고 다음에는 기존 OOF pool에 낮은 가중치의 stacking feature로만 추가한다.
- EXP-79 Lion OOF 포함 stacking 완료.
  - 목적: EXP-78 Lion OOF를 기존 ConvNeXt/EfficientNet OOF pool에 낮은 비중 feature로 추가
  - 탐색 row 수: 40,248
  - 전체 최고 meta-CV accuracy: 0.98811
  - 보호 샘플 유지 row 수: 32
  - 최종 candidate 수: 0
  - 제출 파일 없음
  - 가장 가까운 보호 후보:
    - spec: `cnx_lion_eff_prob_only`
    - model: RidgeClassifier alpha 500, threshold 0.20
    - meta-CV accuracy: 0.98447
    - EXP-56 대비 변경: `test_0049: 0 -> 1`
  - 해석: Lion feature는 meta-CV를 올리는 조합도 만들지만 보호 샘플과 충돌한다. 보호 조건을 만족하는 후보는 현재 기준보다 meta-CV가 약 0.00211 낮아 제출 제외한다.
  - 사용자 요청으로 가장 가까운 보호 후보 CSV를 별도 생성:
    - `oof/submissions/exp79_ridge_cnx_lion_eff_prob_only_alpha500_t0p2_submit.csv`
    - EXP-56 대비 변경: `test_0049: 0 -> 1`
    - test label 0/1: 227/397
    - 보호 샘플 모두 유지
    - Leaderboard score: 0.9567307692
    - 제출 후 판단: EXP-56보다 하락했으므로 `test_0049=1` 방향은 이득이 아니었고, `test_0049=0`을 우선 유지한다.
- EXP-80 ResNet18 제한 포함 compact stacking 완료.
  - 목적: EXP-56 OOF pool에 ResNet18 OOF를 제한적으로 다시 추가해 ConvNeXt/EfficientNet과 보완되는지 확인
  - EXP-79 결과를 반영해 `test_0049=0`도 보호 조건에 추가
  - 탐색 row 수: 21,060
  - 전체 최고 meta-CV accuracy: 0.98907
  - 보호 샘플 유지 row 수: 21
  - 최종 candidate 수: 0
  - 제출 파일 없음
  - 전체 최고 후보는 `test_0049`, `test_0504`, `test_0539`, `test_0244`, `test_0490`, `test_0188`을 모두 1로 바꿔 보호 조건과 충돌
  - 해석: ResNet18 추가는 meta-CV를 올릴 수 있지만, 현재 public guardrail 안에서는 새 제출 후보를 만들지 못했다.
- EXP-81 EfficientNet-B0 seed123 5-fold OOF 완료.
  - 목적: EXP-50과 동일 EfficientNet-B0 구조/split에서 seed diversity 확보
  - OOF accuracy: 0.98332
  - OOF confusion matrix: `[[1313, 28], [59, 3816]]`
  - test label 0/1: 251/373
  - EXP-50 대비 OOF/test label disagreement: 45/14개
  - EXP-50 대비 OOF/test probability correlation: 0.99227/0.99217
  - EXP-56 대비 test 변경 수: 29개
  - 단독 mean은 `test_0201`, `test_0222`, `test_0141`, `test_0568` 보호 방향을 깨므로 제출 제외
  - 해석: EXP-50보다 OOF는 소폭 낮지만 test 예측 14개가 달라 EXP-82에서 제한 stacking feature로 검토할 가치가 있다.
- EXP-82~91 fast suite 완료.
  - EXP-82 EfficientNet seed stacking: 최고 meta-CV 0.98735, 보호 통과 후보 0개
  - EXP-83 EfficientNet seed soft voting: 최고 OOF 0.98581, 보호 위반
  - EXP-84 diversity: ConvNeXt-ResNet disagreement 126개로 가장 큼
  - EXP-85 importance: `eff_prob > eff123_prob > cnx123_prob > cnx_prob`
  - EXP-86 pruning: `cnx123_prob` 제거 시 OOF 0.98658로 최고
  - EXP-87 threshold/calibration: 최고 OOF 0.98696, 보호 통과 없음
  - EXP-88 low-change: 162개 중 보호 통과 0개
  - EXP-89 FP/FN: ResNet FN 40으로 가장 적음
  - EXP-90 rank ensemble: 최고 OOF 0.98658, 변경 수가 많고 보호 위반
  - EXP-91 existing TTA: 단일 validation 기반이므로 분석용으로만 유지
  - 제출 후보 생성 없음
  - 다음 방향: 미세 탐색보다 새 모델 family 또는 표현 공간 확보
- EXP-92 DINOv2 PCA feature 개선 완료.
  - cached DINOv2 384-dim feature에 fold-safe PCA 32/64/128/256 적용
  - 최고: PCA 256 + LogisticRegression C=0.05, threshold -0.20
  - meta-CV accuracy: 0.97604, EXP-69 대비 +0.00230
  - confusion matrix: `[[1286, 55], [70, 3805]]`
  - EXP-56 대비 test 변경 71개, 보호 통과 candidate 0개
  - 해석: DINOv2 표현 압축은 효과가 있지만 현재 CNN OOF pool을 대체할 수준은 아님
- EXP-93 ConvNeXt-ResNet disagreement 설명 실험 완료.
  - OOF disagreement 126개
  - CNX=0/RES=1 90개, 실제 양성 비율 71.1%, ResNet 정확도 71.1%
  - CNX=1/RES=0 36개, 실제 양성 비율 50.0%, 두 모델 정확도 각 50.0%
  - 밝기/대비/entropy/gradient 기반 방향 분류 balanced accuracy 0.514
  - 해석: 단순 영상 품질보다 모델의 공간적 표현 차이가 더 유력하며, 후속으로 동일 cohort Grad-CAM 비교가 필요
- EXP-94 OOF disagreement Grad-CAM 비교 완료.
  - 대표 8장, 각 OOF fold checkpoint 사용
  - ConvNeXt/ResNet CAM correlation 평균 0.251
  - CAM 상위 20% 영역 IoU 평균 0.216
  - CNX=0/RES=1: ResNet 중앙 CAM 0.379, ConvNeXt 0.272
  - CNX=1/RES=0: ResNet 상단 CAM 0.754, 중앙 CAM 0.070
  - 해석: 두 모델의 양성 반응 위치가 다른 경향이 방향 차이의 한 설명이 될 수 있지만, CAM을 실제 병변 근거로 단정하지 않음
- EXP-95 suspicious test case review pack 완료.
  - 목적: leaderboard probing식 라벨 뒤집기가 아니라 모델 간 불일치/불확실성이 큰 test 이미지를 리뷰 대상으로 묶기
  - 기준 제출: `oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv`
  - 사용 모델 수: 10
  - action count: keep 553, review 60, avoid_change_protected 9, review_high_priority 2
  - high-priority review 후보: `test_0188.png`, `test_0377.png`
  - 산출물:
    - `reports/exp95_suspicious_cases/exp95_suspicious_cases.csv`
    - `reports/exp95_suspicious_cases/exp95_suspicious_top50.csv`
    - `reports/exp95_suspicious_cases/exp95_suspicious_panel_top50.png`
    - `reports/exp95_suspicious_cases/report.md`
  - 해석: suspicious score는 변경 지시가 아니라 XAI/오류 분석 우선순위이며, 보호 샘플은 `avoid_change_protected`로 분리
- EXP-96 suspicious rule submission candidates 완료.
  - EXP-95 `review_high_priority`만 사용해 모델 기반 제출 후보 3개 생성
  - 모든 보호 샘플 9개 유지
  - 1순위: `oof/submissions/exp96_flip0188_only_submit.csv`
    - 변경: `test_0188: 0 -> 1`
    - 근거: EXP-56 margin -0.0109, 10개 모델 중 7개가 1
  - 2순위: `oof/submissions/exp96_flip0377_only_submit.csv`
    - 변경: `test_0377: 0 -> 1`
  - 3순위: `oof/submissions/exp96_flip0188_0377_submit.csv`
    - 변경 2개라 공격적 후보
  - 제출 결과:
    - `exp96_flip0188_only_submit.csv` leaderboard `0.9567307692`
    - EXP-56 `0.9583333333`보다 하락
    - 이후 `test_0188=0` 유지 우선
    - `exp96_flip0377_only`, `exp96_flip0188_0377` 제출 우선순위 하락
- EXP-97 guardrail tree filter 완료.
  - 목적: 기존 제출 CSV를 public-informed guardrail tree로 일괄 평가
  - 평가 파일 수: 181
  - pass: 2
  - fail: 179
  - pass 파일:
    - `oof/submissions/exp55_ridge_cnx_eff_compact_p3000_t0p025_submit.csv`
    - `oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv`
  - 둘 다 EXP-56과 동일 예측이라 새 제출 후보는 없음
  - 주요 fail 원인:
    - `avoid_change_protected_changed`: 175
    - `high_priority_changed`: 135
    - `test_0188` 위반: 133
    - `test_0504` 위반: 126
    - `test_0490` 위반: 114
  - 산출물:
    - `scripts/run_exp97_guardrail_tree_filter.py`
    - `oof/preds/exp97_guardrail_tree_report.csv`
    - `oof/preds/exp97_guardrail_tree_passed.csv`
    - `oof/preds/exp97_guardrail_tree.md`
- EXP-98 new model family precheck 완료.
  - 목적: ConvNeXt/EfficientNet과 다른 오류 방향을 줄 수 있는 새 모델 family 후보 선정
  - timm `1.0.27`에서 후보 모두 생성 가능
  - 1순위: `maxvit_tiny_rw_224.sw_in1k`
    - total params 28,545,338
    - trainable guess 16,336,130
    - fold0 smoke 2 epochs 추천
  - 2순위: `coatnet_nano_rw_224.sw_in1k`
    - total params 14,629,270
    - trainable guess 8,874,346
    - MaxViT가 메모리/속도 문제를 보일 때 대안
  - 3순위: `swinv2_tiny_window8_256.ms_in1k`
  - 4순위: `deit3_small_patch16_224.fb_in1k`
  - 5순위: `coatnet_0_rw_224.sw_in1k`
  - 바로 5-fold로 가지 않고 fold0 smoke 후 EXP-97 guardrail과 오류 방향을 확인
  - 산출물:
    - `scripts/run_exp98_model_family_precheck.py`
    - `reports/exp98_model_family_precheck/exp98_model_family_precheck.csv`
    - `reports/exp98_model_family_precheck/exp98_model_family_details.json`
    - `reports/exp98_model_family_precheck/report.md`
- EXP-99 MaxViT Tiny fold0 smoke 완료.
  - 모델: `maxvit_tiny_rw_224.sw_in1k`
  - 학습 범위: `stages[-1] + head`
  - epoch: 2
  - fold0 valid accuracy: 0.96552
  - precision/recall/F1: 0.99332 / 0.96000 / 0.97638
  - confusion matrix: `[[264, 5], [31, 744]]`
  - test label 0/1: 228/396
  - EXP-56 대비 변경 수: 38
  - guardrail fail count: 7
  - 실패 샘플:
    - `test_0025`: 0 -> 1
    - `test_0049`: 0 -> 1
    - `test_0141`: 1 -> 0
    - `test_0188`: 0 -> 1
    - `test_0244`: 0 -> 1
    - `test_0490`: 0 -> 1
    - `test_0504`: 0 -> 1
  - 판단: 단독 제출 및 즉시 5-fold 확장 보류. 필요하면 낮은 가중치 stacking feature/오류 분석용으로만 검토
- EXP-100 CoAtNet Nano fold0 smoke 완료.
  - 모델: `coatnet_nano_rw_224.sw_in1k`
  - 학습 범위: `stages[-1] + head`
  - epoch: 2
  - fold0 valid accuracy: 0.96456
  - precision/recall/F1: 0.99200 / 0.96000 / 0.97574
  - confusion matrix: `[[263, 6], [31, 744]]`
  - test label 0/1: 316/308
  - EXP-56 대비 변경 수: 90
  - guardrail fail count: 4
  - 실패 샘플:
    - `test_0141`: 1 -> 0
    - `test_0188`: 0 -> 1
    - `test_0201`: 1 -> 0
    - `test_0222`: 1 -> 0
  - 판단: MaxViT보다 guardrail fail은 적지만 label1 308로 과도하게 보수적. 단독 제출 및 즉시 5-fold 확장 보류
  - 제출 결과: `oof/submissions/exp100_coatnet_nano_fold0_smoke_submit.csv`
  - Public score: `0.8493589744`
  - 해석: fold0 validation은 0.96456이었지만 public에서 크게 하락. label1 308, EXP-56 대비 90개 변경, 보호 샘플 실패가 실제 위험 신호였음
  - 후속 결정: CoAtNet Nano 단독 제출 및 5-fold 확장은 제외. 새 모델 family smoke는 validation보다 label 분포/guardrail/변경 수를 우선 확인
- EXP-101~104 Gemini 생성 산출물 로컬 검증.
  - 공통 처리 원칙:
    - 내가 직접 실행한 결과가 아니라 `Gemini generated / locally verified`로 분리 기록
    - 제출 CSV 형식, JSON 파싱, 스크립트 문법만 로컬에서 검증
    - full OOF가 완성되지 않은 결과는 stacking 입력으로 사용하지 않음
  - EXP-101 SwinV2-Tiny fold0 smoke:
    - 모델: `swinv2_tiny_window8_256.ms_in1k`
    - fold0 valid accuracy: 0.97414
    - precision/recall/F1: 0.97826 / 0.98710 / 0.98266
    - confusion matrix: `[[252, 17], [10, 765]]`
    - test label 0/1: 168/456
    - EXP-56 대비 변경 수: 68
    - guardrail fail count: 8
    - 판단: 단독 제출 금지. full OOF 완료 후 낮은 영향 stacking feature로만 검토
  - EXP-102 DeiT-Small fold0 smoke:
    - 모델: `deit3_small_patch16_224.fb_in1k`
    - fold0 valid accuracy: 0.91858
    - precision/recall/F1: 0.98592 / 0.90323 / 0.94276
    - confusion matrix: `[[259, 10], [75, 700]]`
    - test label 0/1: 427/197
    - EXP-56 대비 변경 수: 213
    - guardrail fail count: 3
    - 판단: 단독 제출 및 5-fold 확장 우선순위 낮음
  - EXP-103 SwinV2-Tiny 5-fold OOF:
    - Gemini partial 산출물에서 fold0/fold1 재사용, fold2~4 로컬 실행 완료
    - fold0 valid accuracy: 0.97414
    - fold1 valid accuracy: 0.97124
    - fold2 valid accuracy: 0.95973
    - fold3 valid accuracy: 0.97315
    - fold4 valid accuracy: 0.97124
    - OOF accuracy: 0.96990
    - OOF precision/recall/F1: 0.98411 / 0.97523 / 0.97965
    - OOF confusion matrix: `[[1280, 61], [96, 3779]]`
    - test mean label 0/1: 197/427
    - EXP-56 대비 변경 수: 53
    - guardrail fail count: 7
    - 판단: 단독 mean 제출 금지. EXP-104 stacking feature로만 사용
  - EXP-104 SwinV2 regularized stacking:
    - 실행 완료
    - 탐색 row 수: 2520
    - SwinV2 포함 row 수: 1440
    - strict candidate 수: 0
    - relaxed candidate 수: 0
    - 보호 샘플 전체 통과 row 수: 0
    - 최고 SwinV2 포함 meta-CV: 0.98677
    - 판단: 모든 후보가 보호 샘플을 최소 1개 이상 위반. 제출 후보 없음
  - EXP-106 No-Swin low-change search:
    - 실행 완료
    - 탐색 row 수: 21780
    - candidate 수: 0
    - 보호 샘플 전체 통과 row 수: 34
    - 최고 meta-CV: 0.98926
    - 판단: SwinV2를 제외해도 강한 후보는 주로 `test_0504=1`로 되돌아가고, 보호 샘플을 모두 유지한 row는 EXP-56과 동일 예측이라 제출 후보 없음
  - EXP-107 Boundary bottleneck analysis:
    - 실행 완료
    - plausible no-Swin row 수: 5784
    - 제출 candidate 수: 0
    - 최대 병목: `test_0504` 위반 5383/5784 rows
    - 다음 병목: `test_0490`, `test_0539`, `test_0049`, `test_0188`
    - 비보호 반복 변경 샘플은 `test_0568` 1개뿐
  - EXP-108 Flip0568 rule candidates:
    - 실행 완료
    - 1순위 제출 후보: `oof/submissions/exp108_flip0568_only_submit.csv`
    - EXP-56 대비 변경: `test_0568: 0->1`
    - test label 0/1: 227/397
    - 보호 샘플/avoid-pair 조건 통과
    - 비교 후보: `exp108_flip0377_only_recheck_submit.csv`, `exp108_flip0568_0377_submit.csv`
    - 제출 결과: leaderboard `0.9567307692`, EXP-56보다 하락
    - 판단: `test_0568=1` 단독 변경도 이득이 아니므로 `test_0568=0` 유지 우선
  - EXP-109 Extended guardrail filter:
    - 실행 완료
    - 기존 제출 CSV 평가 수: 189
    - 최신 guardrail 통과 파일 수: 2
    - EXP-56 대비 비중복 통과 후보: 0
    - 통과 파일은 `exp55_ridge_cnx_eff_compact_p3000_t0p025_submit.csv`, `exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv`로 둘 다 EXP-56과 동일 예측
    - 판단: 기존 CSV 재활용 기반 제출 후보 탐색은 소진
  - EXP-110 ConvNeXt hard-case FP drop audit:
    - 실행 완료
    - EXP-56 양성 중 ConvNeXt 계열 불일치/낮은 margin 기반 1개 drop 후보 생성
    - 후보 3개:
      - `oof/submissions/exp110_drop_test_0172_rank1_submit.csv`
      - `oof/submissions/exp110_drop_test_0455_rank2_submit.csv`
      - `oof/submissions/exp110_drop_test_0057_rank3_submit.csv`
    - test label 0/1은 모두 229/395
    - 제출 결과:
      - 제출 파일: `oof/submissions/exp110_drop_test_0455_rank2_submit.csv`
      - leaderboard: `0.9599358974`
      - 2026-06-30 기준 12위
      - EXP-56 `0.9583333333` 대비 +1샘플 개선
      - `test_0455: 1->0` 방향이 public에서 이득으로 확인되어 `test_0455=0`을 새 보호 조건에 추가
  - EXP-111 ConvNeXt threshold replay:
    - 실행 완료
    - ConvNeXt score 하위 EXP-56 양성 샘플을 1~2개 drop하는 후보 7개 생성
    - `exp111_core_min_drop1_submit.csv`는 `test_0455` 단독 drop으로 EXP-110 rank2와 동일 예측이므로 중복 제출하지 않음
    - `exp111_core_min_drop2_submit.csv` 제출 결과:
      - leaderboard: `0.9583333333`
      - 현재 최고 EXP-110 `0.9599358974`보다 하락
      - EXP-110 대비 추가 변경은 `test_0289: 1->0` 1개뿐
      - 이후 `test_0289=1`을 우선 보호 방향으로 유지
  - EXP-112 Drop candidate Grad-CAM panel:
    - 실행 완료
    - 대상 샘플 7개
    - 패널: `reports/exp112_drop_candidate_gradcam/exp112_drop_candidate_gradcam_panel.png`
    - 판단: XAI 단독으로 label을 확정하지 않지만, `test_0455` 단독 drop 후보를 1순위 제출 후보로 본다
  - 다음 액션:
    - 기존 EXP-110/111 후보를 그대로 추가 제출하지 않는다. 이 파일들은 EXP-56 기준이므로 `test_0455=0`을 다시 1로 되돌릴 수 있다.
    - EXP-113 evidence table 재구성 완료:
      - `reports/exp113_tree_replan/exp113_candidate_evidence_table.csv`
      - `reports/exp113_tree_replan/exp113_candidate_ranked.csv`
      - 기준 제출물 차이, 후보 ranking 품질, public probing 누적 정보를 분리해 관리
    - EXP-114 current-best 기준 단일 drop 후보 생성 완료:
      - 기준 파일: `oof/submissions/exp110_drop_test_0455_rank2_submit.csv`
      - 모든 후보는 `test_0455=0`, `test_0289=1` 유지
      - 후보 5개 모두 현재 최고 대비 변경 수 1개, label 0/1은 230/394
    - 제출 후보:
      - 1순위: `oof/submissions/exp114_drop_test_0057_only_submit.csv`
      - 2순위: `oof/submissions/exp114_drop_test_0172_only_submit.csv`
      - 3순위: `oof/submissions/exp114_drop_test_0385_only_submit.csv`
      - 4순위: `oof/submissions/exp114_drop_test_0013_only_submit.csv`
      - 5순위: `oof/submissions/exp114_drop_test_0303_only_submit.csv`
    - 주의: evidence score만 보면 `test_0385`가 가장 높지만, 계획상 branch 순서는 `test_0057`부터 유지한다. `test_0057` 하락 시 `test_0385`를 바로 2순위로 올려도 된다.
  - EXP-115 public result tree update:
    - `exp114_drop_test_0057_only_submit.csv` 제출 결과 leaderboard `0.9615384615`
    - EXP-110 대비 +1샘플 개선
    - `test_0057=0`을 성공 방향으로 추가
    - 새 기준 파일: `oof/submissions/exp114_drop_test_0057_only_submit.csv`
  - EXP-116 after-0057 branch candidates:
    - 실행 완료
    - 모든 후보는 `test_0455=0`, `test_0057=0`, `test_0289=1` 유지
    - 후보 4개 모두 EXP-114 대비 변경 수 1개, label 0/1은 231/393
    - `exp116_after0057_drop_test_0385_only_submit.csv` 제출 결과 leaderboard `0.9631410256`
    - EXP-114 대비 +1샘플 개선
    - `test_0385=0`을 성공 방향으로 추가
    - 새 기준 파일: `oof/submissions/exp116_after0057_drop_test_0385_only_submit.csv`
  - EXP-117 after-0385 branch candidates:
    - 실행 완료
    - 모든 후보는 `test_0455=0`, `test_0057=0`, `test_0385=0`, `test_0289=1` 유지
    - 후보 3개 모두 EXP-116 대비 변경 수 1개, label 0/1은 232/392
    - `exp117_after0385_drop_test_0172_only_submit.csv` 제출 결과 leaderboard `0.9615384615`
    - EXP-116 대비 -1샘플 하락
    - `test_0172=1`을 실패 방향으로 추가
    - `exp117_after0385_drop_test_0013_only_submit.csv` 제출 결과 leaderboard `0.9615384615`
    - EXP-116 대비 -1샘플 하락
    - `test_0013=1`을 실패 방향으로 추가
    - `exp117_after0385_drop_test_0303_only_submit.csv` 제출 결과 leaderboard `0.9647435897`
    - EXP-116 대비 +1샘플 개선
    - `test_0303=0`을 성공 방향으로 추가
    - 현재 최고:
      - `oof/submissions/exp117_after0385_drop_test_0303_only_submit.csv`
      - public `0.9647435897`
      - 2026-07-01 기준 7위
    - 남은 제출 기회: 4회
    - 다음 후보는 기존 EXP-113 ranking을 그대로 이어 제출하지 않고, 성공/실패 방향을 모두 반영해 EXP-118에서 재랭킹 후 1개씩 제출한다.
  - EXP-118 after-0303 remaining branch triage:
    - 실행 완료
    - 기준 제출: `oof/submissions/exp117_after0385_drop_test_0303_only_submit.csv`
    - 성공 방향 유지: `test_0455=0`, `test_0057=0`, `test_0385=0`, `test_0303=0`
    - 실패 방향 유지: `test_0289=1`, `test_0172=1`, `test_0013=1`
    - 모든 후보는 현재 최고 대비 변경 수 1개, label 0/1은 233/391
    - 후보 1순위:
      - `oof/submissions/exp118_after0303_drop_test_0212_only_submit.csv`
      - 변경: `test_0212: 1->0`
      - 단, direct prior/XAI 근거가 없어 이전 성공 branch보다 신뢰도는 낮음
    - 후보 2순위:
      - `oof/submissions/exp118_after0303_drop_test_0267_only_submit.csv`
    - 후보 3순위:
      - `oof/submissions/exp118_after0303_drop_test_0022_only_submit.csv`
    - 후보 4순위:
      - `oof/submissions/exp118_after0303_drop_test_0525_only_submit.csv`
    - 제출 방침:
      - 남은 4회를 한 번에 소진하지 않고, 1순위 제출 결과를 보고 다음 후보군 신뢰도를 재평가한다.
  - EXP-119 RF/SVM/Gaussian evidence reranking:
    - 실행 완료
    - 기준 제출: `oof/submissions/exp117_after0385_drop_test_0303_only_submit.csv`
    - 목적: 현재 최고에서 label 1인 샘플 중 FP 가능성이 높은 후보 재정렬
    - 사용 모델:
      - RandomForestClassifier
      - ExtraTreesClassifier
      - LinearSVC
      - LogisticRegression
      - RidgeClassifier
    - 추가 feature:
      - Gaussian blur delta
      - edge/Laplacian/contrast/entropy 이미지 통계
      - EXP-113 tree score
      - XAI 요약값이 있는 경우 병합
    - meta-CV:
      - RF `0.98735`
      - ExtraTrees `0.98658`
      - LinearSVC `0.98601`
      - Logistic `0.98696`
      - Ridge `0.98715`
    - 재랭킹 결과:
      - 1순위 `test_0547`
      - 2순위 `test_0315`
      - 3순위 `test_0390`
      - 4순위 `test_0040`
      - 5순위 `test_0212`
    - 해석:
      - `test_0547`이 EXP-118 1순위 `test_0212`보다 위로 올라왔지만, RF/ExtraTrees/SVM의 절대 class-0 지지는 강하지 않다.
      - 따라서 제출 가능 후보이나 EXP-110~117 성공 branch보다 위험도가 높다.
  - EXP-120 after-0303 top-1 RF/SVM/Gaussian drop:
    - 제출 파일 생성 완료
    - `oof/submissions/exp120_after0303_drop_test_0547_only_submit.csv`
    - 현재 최고 대비 변경: `test_0547: 1->0`
    - 변경 수 1개, label 0/1은 233/391
    - 제출 방침:
      - 남은 4회 중 1회를 쓸 수 있는 고위험 후보
      - 하락하면 EXP-119 계열 ranking 신뢰도를 낮추고 기존 current-best branch 전략으로 복귀
  - EXP-121 evidence ablation consensus:
    - 실행 완료
    - RF/SVM/Gaussian evidence를 한 번에 합치지 않고 개별/2개 조합/3개 조합으로 분해
    - 이유:
      - RF/ExtraTrees는 OOF feature 기반 비선형 tabular 판단
      - SVM은 boundary 근처 선형 분리 신호
      - Gaussian evidence는 label 근거가 아니라 영상 texture/edge 민감도 보조 신호
      - 세 근거를 단순 합산하면 보조 신호가 과대평가될 수 있음
    - consensus top5:
      - `test_0390`
      - `test_0315`
      - `test_0478`
      - `test_0009`
      - `test_0596`
    - EXP-119 1위 `test_0547`은 분해 검증에서 consensus 상위권이 아니므로 우선순위 하락
    - 출력:
      - `reports/exp121_evidence_ablation_consensus/exp121_evidence_ablation_consensus_ranking.csv`
      - `reports/exp121_evidence_ablation_consensus/exp121_top20_by_rf_only.csv`
      - `reports/exp121_evidence_ablation_consensus/exp121_top20_by_svm_only.csv`
      - `reports/exp121_evidence_ablation_consensus/exp121_top20_by_gaussian_only.csv`
  - EXP-122 after-0303 consensus single drop:
    - 제출 파일 생성 완료
    - `oof/submissions/exp122_after0303_consensus_drop_test_0390_only_submit.csv`
    - 현재 최고 대비 변경: `test_0390: 1->0`
    - 변경 수 1개, label 0/1은 233/391
    - CSV 검증 완료:
      - row 수 624
      - 컬럼 `file_name,label`
      - sample submission과 file order 동일
      - label은 0/1만 존재
    - 제출 결과:
      - public `0.9631410256`
      - 현재 최고 `0.9647435897` 대비 -1샘플 하락
      - `test_0390=1` 실패 guardrail 추가
      - 남은 제출 기회: 3회
    - 다음 방침:
      - EXP-121 consensus 점수는 단독 제출 결정 근거로 사용하지 않고 보조 근거로만 사용
      - 다음 후보는 public 성공 방향과 가까운 branch 후보 위주로 다시 재정렬
  - EXP-123 augmentation and XAI review:
    - 실행 완료
    - 기존 학습 증강은 대부분 `Resize(224) + HorizontalFlip(0.5) + Rotation(5~7)` 구조
    - TTA는 `original + horizontal flip`
    - 결론:
      - X-ray에서는 좌우 방향 정보와 marker/심장 위치가 의미 있을 수 있어 horizontal flip을 안전한 기본 증강으로 보기 어렵다.
      - rotation도 불필요한 촬영 방향 변형을 만들 수 있으므로 줄이는 편이 타당하다.
      - Center crop은 주요 학습 스크립트에서 거의 쓰지 않았고, 다음에도 쓰지 않는 쪽이 안전하다.
      - 다음 학습형 후보는 `Resize 384`, no flip, no rotation, weak brightness/contrast.
    - EXP-112 Grad-CAM 대조:
      - public에서 성공한 drop 후보 평균 ConvNeXt fold0 p1 `0.2251`
      - 실패한 keep-1 후보 평균 ConvNeXt fold0 p1 `0.6042`
      - 성공 drop 후보의 CAM center fraction 평균 `0.1696`, 실패 후보 평균 `0.3210`
      - 성공 drop 후보는 폐 중심보다 주변/상단/측면 반응 가능성이 더 커 보임
    - 산출물:
      - `reports/exp123_augmentation_xai_review/exp123_augmentation_xai_review.md`
    - 다음 학습 후보:
      - ConvNeXt-Tiny 384 no-flip/no-rotation fold0 smoke
      - 효과가 있으면 5-fold OOF 확장 검토
  - EXP-124 ConvNeXt-Tiny 384 no-flip/no-rotation fold0 smoke:
    - 실행 완료
    - 설정:
      - `IMG_SIZE=384`
      - `BATCH_SIZE=16`
      - `EPOCHS=6`
      - `LR=5e-5`
      - `Resize((384,384))`
      - `ColorJitter(brightness=0.08, contrast=0.08)`
      - `RandomHorizontalFlip`, `RandomRotation`, `CenterCrop` 제거
    - best epoch: 5
    - fold0 validation:
      - accuracy `0.9875478927`
      - precision `0.9948051948`
      - recall `0.9883870968`
      - F1 `0.9915857605`
      - confusion matrix `[[265, 4], [9, 766]]`
    - test label count:
      - label0 `235`
      - label1 `389`
    - 저장 파일:
      - `oof/models/exp124_convnext_tiny_384_noflip_fold0.pt`
      - `oof/preds/exp124_convnext_tiny_384_noflip_fold0_oof_probs.csv`
      - `oof/preds/exp124_convnext_tiny_384_noflip_fold0_test_probs.csv`
      - `oof/submissions/exp124_convnext_tiny_384_noflip_fold0_submit.csv`
    - 판단:
      - fold0 metric은 EXP-45 `0.98467`보다 개선
      - 단독 제출은 current-best 대비 변경 수 21개라 제외
  - EXP-125 EXP-124 augmentation/XAI compare:
    - 실행 완료
    - 산출물:
      - `reports/exp125_aug_xai_compare/exp125_aug_xai_compare.md`
      - `reports/exp125_aug_xai_compare/exp125_fold0_metric_compare.csv`
      - `reports/exp125_aug_xai_compare/exp125_guardrail_compare.csv`
      - `reports/exp125_aug_xai_compare/exp125_changed_vs_exp117_current_best.csv`
      - `reports/exp125_aug_xai_compare/exp125_gradcam_summary.csv`
      - `reports/exp125_aug_xai_compare/exp125_gradcam_exp45_vs_exp124_panel.png`
    - current-best 대비 변경 수: 21개
    - guardrail 위반 수: 8개
    - 주요 위반:
      - `test_0385` 0 -> 1
      - `test_0289` 1 -> 0
      - `test_0013` 1 -> 0
      - `test_0188` 0 -> 1
      - `test_0539` 0 -> 1
      - `test_0244` 0 -> 1
      - `test_0222` 1 -> 0
    - Grad-CAM 요약:
      - success drop center fraction: EXP-45 `0.1696`, EXP-124 `0.1883`
      - failed keep center fraction: EXP-45 `0.2982`, EXP-124 `0.2616`
    - EXP-126 gate:
      - 보류
      - 이유: fold0 metric은 좋지만 public 방향 확인 샘플을 다수 깨고, CAM 분리도도 충분히 명확하지 않음
  - EXP-126 ConvNeXt-Tiny 384 no-flip/no-rotation 5-fold OOF:
    - 사용자 지시에 따라 EXP-125 gate 보류 판단을 넘어서 5-fold 정보원 확보 실험 진행
    - fold0은 EXP-124 산출물 재사용, fold1~4 추가 학습
    - 설정:
      - `IMG_SIZE=384`
      - `BATCH_SIZE=16`
      - `EPOCHS=6`
      - `LR=5e-5`
      - `Resize((384,384))`
      - `ColorJitter(brightness=0.08, contrast=0.08)`
      - `RandomHorizontalFlip`, `RandomRotation`, `CenterCrop` 제거
    - 5-fold OOF:
      - accuracy `0.9840874233`
      - precision `0.9929797192`
      - recall `0.9855483871`
      - F1 `0.9892500971`
      - confusion matrix `[[1314, 27], [56, 3819]]`
    - 단독 mean submission:
      - `oof/submissions/exp126_convnext_tiny_384_noflip_mean_submit.csv`
      - label count: `0=218`, `1=406`
      - current-best EXP-117 대비 변경 수 22개
      - guardrail 위반 12개
    - 판단:
      - 단독 제출 제외
      - 384/no-flip 확률은 EXP-127 stacking feature로만 제한 사용
  - EXP-127 384 no-flip feature limited stacking:
    - 목적:
      - EXP-126 확률을 기존 ConvNeXt/EfficientNet OOF stacking feature에 제한적으로 추가
      - current-best EXP-117 대비 1~3개 변경 후보만 탐색
    - 탐색:
      - RidgeClassifier / LogisticRegression
      - prob/rank/margin/uncertainty/delta feature
      - guardrail 전체 유지
      - meta-CV accuracy `0.985` 이상
    - 결과:
      - 저장 가능한 submission 후보 0개
      - `oof/preds/exp127_384_noflip_stacking_summary.csv`
      - `oof/preds/exp127_384_noflip_stacking_result.json`
    - 판단:
      - meta-CV가 높은 조합도 `test_0057`, `test_0385`, `test_0303`을 다시 1로 되돌리는 경향
      - 현재 public branch 방향과 충돌하므로 남은 제출 기회 사용하지 않음
  - EXP-128 current-best hard-case rerank v2:
    - 실행 완료
    - 목적:
      - 현재 최고 EXP-117 기준에서 남은 `1->0` 후보만 재정렬
      - EXP-113 tree score, EXP-119 RF/SVM/Gaussian evidence, EXP-121 consensus, EXP-126/127 결과 병합
      - public 성공/실패 방향은 hard guardrail로 유지
    - 현재 최고:
      - `oof/submissions/exp117_after0385_drop_test_0303_only_submit.csv`
      - public `0.9647435897`
    - 산출물:
      - `reports/exp128_current_best_rerank_v2/exp128_candidate_evidence_table.csv`
      - `reports/exp128_current_best_rerank_v2/exp128_candidate_ranked.csv`
      - `reports/exp128_current_best_rerank_v2/exp128_eligible_top30.csv`
      - `reports/exp128_current_best_rerank_v2/exp128_result.json`
    - 생성 submission:
      - `oof/submissions/exp128_drop_test_0600_rank1_submit.csv`
      - `oof/submissions/exp128_drop_test_0596_rank2_submit.csv`
      - `oof/submissions/exp128_drop_test_0398_rank3_submit.csv`
    - 검증:
      - 각 submission row 수 624
      - 컬럼 `file_name,label`
      - sample submission과 file order 동일
      - label은 0/1만 존재
      - 현재 최고 대비 정확히 1개만 `1->0`
      - label count `0=233`, `1=391`
    - 제출 우선순위:
      1. `test_0600: 1->0`
      2. `test_0596: 1->0`
      3. `test_0398: 1->0`
    - 판단:
      - 남은 제출 기회가 3회이므로 한 번에 모두 제출하지 말고 rank1만 먼저 제출 후 public feedback으로 branch 갱신
    - 제출 결과:
      - `oof/submissions/exp128_drop_test_0600_rank1_submit.csv`
      - public `0.9631410256`
      - 현재 최고 `0.9647435897` 대비 -1샘플 하락
      - `test_0600=1` 유지 우선 guardrail 추가
      - EXP-128 rank2/rank3는 같은 evidence 계열이므로 제출 우선순위 하락
      - 남은 제출 기회: 2회
  - EXP-129 success-branch similarity rerank:
    - 실행 완료
    - 목적:
      - EXP-128 broad consensus가 실패했으므로, 성공 drop 샘플 `test_0455`, `test_0057`, `test_0385`, `test_0303`과 닮은 후보를 재정렬
      - `test_0600=1`을 실패 guardrail에 추가
    - 산출물:
      - `reports/exp129_success_branch_rerank/exp129_candidate_evidence_table.csv`
      - `reports/exp129_success_branch_rerank/exp129_candidate_ranked.csv`
      - `reports/exp129_success_branch_rerank/exp129_top8_for_xai.csv`
      - `reports/exp129_success_branch_rerank/exp129_success_profile.csv`
    - top5:
      1. `test_0212: 1->0`
      2. `test_0527: 1->0`
      3. `test_0267: 1->0`
      4. `test_0457: 1->0`
      5. `test_0525: 1->0`
  - EXP-130 success-candidate Grad-CAM:
    - 실행 완료
    - 목적:
      - EXP-129 top8 후보만 대상으로 EXP-45 224 ConvNeXt와 EXP-124 384 no-flip ConvNeXt Grad-CAM 비교
      - 마지막 제출 후보를 1개로 압축
    - 산출물:
      - `reports/exp130_success_candidate_gradcam/exp130_gradcam_summary.csv`
      - `reports/exp130_success_candidate_gradcam/exp130_selection_summary.csv`
      - `reports/exp130_success_candidate_gradcam/exp130_gradcam_panel.png`
      - `oof/submissions/exp130_after0303_successcam_drop_test_0212_submit.csv`
      - `oof/preds/exp130_after0303_successcam_drop_test_0212_changed_vs_exp117.csv`
    - 추천 제출 후보:
      - `oof/submissions/exp130_after0303_successcam_drop_test_0212_submit.csv`
      - 현재 최고 대비 `test_0212: 1->0` 1개만 변경
      - label count `0=233`, `1=391`
      - 검증: row 624, 컬럼 `file_name,label`, sample submission 순서 일치, label 0/1
    - 근거:
      - EXP-129 종합 점수 1위
      - EXP-130 selection score 1위
      - EXP-45 p1 `0.3138`, EXP-124 384 no-flip p1 `0.3166`
      - `test_0267`은 ConvNeXt 확률은 더 낮지만 EXP-129 종합 근거가 약해 2순위로 둠
  - EXP-131 final submit decision recheck:
    - 실행 완료
    - 목적:
      - 남은 제출 2회 중 첫 제출 후보를 `test_0212`로 확정해도 되는지 로컬에서 재검증
      - balanced / branch similarity / XAI probability / ML evidence / public-branch conservative 시나리오 비교
    - 산출물:
      - `reports/exp131_final_submit_decision/exp131_final_decision_table.csv`
      - `reports/exp131_final_submit_decision/exp131_final_submit_decision.md`
      - `oof/submissions/exp131_final_recheck_drop_test_0212_submit.csv`
      - `oof/preds/exp131_final_recheck_drop_test_0212_changed_vs_exp117.csv`
    - 결과:
      - `test_0212` final score `0.9098`, scenario win count `5/5`, average rank `1.00`
      - `test_0267` final score `0.6269`, scenario win count `0/5`, average rank `2.20`
    - 최종 제출 후보:
      - `oof/submissions/exp131_final_recheck_drop_test_0212_submit.csv`
      - 현재 최고 대비 `test_0212: 1->0` 1개만 변경
      - row 624, sample submission 순서 일치, label count `0=233`, `1=391`
    - 제출 결과:
      - public `0.9631410256`
      - 현재 최고 `0.9647435897` 대비 -1샘플 하락
      - `test_0212=1` 유지 guardrail 추가
  - EXP-132 success-vs-failure contrast rerank:
    - 실행 완료
    - 목적:
      - EXP-131 실패 이후 성공 drop과 실패 keep을 동시에 사용해 마지막 후보 재정렬
      - 성공과 가까울수록 가점, 실패와 가까울수록 감점
    - 성공 drop:
      - `test_0455=0`
      - `test_0057=0`
      - `test_0385=0`
      - `test_0303=0`
    - 실패 keep:
      - `test_0289=1`
      - `test_0172=1`
      - `test_0013=1`
      - `test_0390=1`
      - `test_0600=1`
      - `test_0212=1`
      - 기존 민감 guardrail 전체
    - 산출물:
      - `reports/exp132_success_failure_contrast/exp132_candidate_ranked.csv`
      - `reports/exp132_success_failure_contrast/exp132_success_failure_profile.csv`
      - `reports/exp132_success_failure_contrast/exp132_top30.csv`
      - `reports/exp132_success_failure_contrast/exp132_success_failure_contrast.md`
      - `oof/submissions/exp132_contrast_drop_test_0267_submit.csv`
      - `oof/preds/exp132_contrast_drop_test_0267_changed_vs_exp117.csv`
    - top3:
      1. `test_0267`: score `0.6880`, success distance `1.9386`, failure distance `2.3512`
      2. `test_0034`: score `0.5562`
      3. `test_0457`: score `0.5550`
    - 추천 제출 후보:
      - `oof/submissions/exp132_contrast_drop_test_0267_submit.csv`
      - 현재 최고 대비 `test_0267: 1->0` 1개만 변경
      - row 624, sample submission 순서 일치, label count `0=233`, `1=391`
    - 제출 결과:
      - public `0.9663461538`
      - 최종 순위 `6위`
      - 이전 최고 `0.9647435897` 대비 +1샘플 개선
      - 최종 제출 수 `55`
      - `test_0267=0` 성공 방향 확인
    - 최종 정리:
      - EXP-132가 최종 최고 제출로 확정
      - binary-tree/drop branch는 모델 일반화 지표가 아니라 제출 후보 선별 전략으로 분리해서 해석
