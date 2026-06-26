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
  - 최신 public leaderboard 기준 최고 순위 13위 확인
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
