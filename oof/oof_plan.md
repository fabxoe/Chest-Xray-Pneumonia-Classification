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

1. ConvNeXt-Tiny OOF 학습 스크립트 작성
2. fold 0만 먼저 smoke test
3. fold 0 결과 파일 포맷 검증
4. 5 folds 전체 학습
5. OOF Ridge/Logistic baseline 생성

## 추가 우선순위 메모

- EfficientNetV2-S `features[7]+classifier` 단일 split probe는 validation accuracy 0.87261로 낮고 FN이 크게 증가했다. 현재 설정은 단독/stacking 후보에서 제외한다.
- EfficientNetV2-S를 다시 시도한다면 마지막 block만이 아니라 `features[6]+features[7]+classifier`처럼 더 넓은 fine-tuning으로 재평가한다.
- Lion optimizer 비교는 AdamW 기본선을 유지한 뒤, 8순위 실험 전에 추가한다. 우선 대상은 ConvNeXt-Tiny 또는 RegNetY처럼 단독 validation이 안정적인 모델로 한다.
