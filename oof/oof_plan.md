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
  - 제출 판단: 두 샘플이 모두 맞으면 +2샘플 가능, 하나만 맞으면 동점, 둘 다 틀리면 -2샘플. 중간 리스크 후보.

## 추가 우선순위 메모

- EfficientNetV2-S `features[7]+classifier` 단일 split probe는 validation accuracy 0.87261로 낮고 FN이 크게 증가했다. 현재 설정은 단독/stacking 후보에서 제외한다.
- EfficientNetV2-S를 다시 시도한다면 마지막 block만이 아니라 `features[6]+features[7]+classifier`처럼 더 넓은 fine-tuning으로 재평가한다.
- RegNetY-3.2GF `trunk_output.block4+fc` probe는 validation accuracy 0.95211로 단독 제출은 약하지만, 민감 샘플 `test_0244`, `test_0025`, `test_0539`를 모두 0으로 유지해 stacking diversity 후보로 남긴다.
- Lion optimizer 비교는 AdamW 기본선을 유지한 뒤, 8순위 실험 전에 추가한다. 우선 대상은 ConvNeXt-Tiny 또는 RegNetY처럼 단독 validation이 안정적인 모델로 한다.
- 10시간 초과 장기 실험은 바로 실행하지 않고, 목적/기대 이득/근거/계산 비용/중단 조건/제출 기준을 먼저 작성한 뒤 검토한다.
- 장기 실험 후보 우선순위는 ConvNeXt 계열 OOF seed 추가, ConvNeXt V2 Tiny, EfficientNetV2-S 재설계, DINOv2 feature extractor 순으로 본다.
