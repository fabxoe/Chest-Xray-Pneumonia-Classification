# Chest X-ray Pneumonia Classification

> Generalization Gap · Backbone Comparison · 5-Fold OOF · Stacking · Explainable AI

흉부 X-ray 정상/폐렴 이진 분류 프로젝트입니다.  
초기 CNN이 **Validation Accuracy 0.9732**를 기록했지만 **Public Score 0.8189**에 머문 현상에서 출발해, 모델 복잡도를 높이는 것보다 **검증 신뢰성, 모델 간 오류 차이, 실패 사례의 원인**을 추적하는 데 초점을 맞췄습니다.

최종적으로 모델·검증 파이프라인을 통해 Public Score를 **0.9583**까지 개선했고, 이후 competition-specific submission optimization을 별도 단계로 수행해 **0.96635, 최종 6위**를 기록했습니다.

## 1. Problem

초기 SimpleCNN은 validation에서 높은 정확도를 보였지만 leaderboard에서는 큰 성능 하락이 발생했습니다.

| Stage | Validation / CV | Public Score |
|---|---:|---:|
| SimpleCNN | Accuracy 0.97318 | 0.81891 |
| ResNet18 | Accuracy 0.98659 | 0.92788 |
| ConvNeXt-Tiny | Accuracy 0.98084 | 0.94711 |
| Ridge Stacking | Accuracy 0.98755 | 0.95032 |
| OOF Logistic | Meta-CV 0.98639 | 0.95513 |
| OOF compact ensemble | Meta-CV 0.98658 | **0.95833** |
| Final competition submission | - | **0.96635 / 6th** |

Validation score만 높이는 방향으로는 실제 leaderboard 성능을 안정적으로 설명하기 어려웠습니다. 이에 단일 holdout 의존도를 낮추고, backbone별 prediction과 failure pattern을 비교하는 방향으로 실험을 재설계했습니다.

## 2. Approach

### Backbone comparison

다음 pretrained CNN 계열을 비교했습니다.

- ResNet18
- DenseNet121
- EfficientNet-B0
- ConvNeXt-Tiny
- 이후 ConvNeXt V2, RegNet, EfficientNet V2 등 추가 후보도 검토

ConvNeXt-Tiny가 단일 모델 기준 Public Score **0.94711**을 기록해 주요 backbone으로 사용됐습니다.

### 5-Fold OOF validation

단일 validation split에서 관측된 높은 성능이 test distribution에서도 유지되는지 확인하기 위해 5-Fold Out-of-Fold prediction을 구성했습니다.

OOF probability를 기반으로 Logistic Regression, Ridge 계열 meta-model 및 여러 probability/rank feature 조합을 비교했습니다.

이 과정에서 meta-CV가 가장 높은 조합이 항상 Public Score도 가장 높은 것은 아니라는 점을 확인했고, validation metric 하나만으로 모델 선택을 결정하지 않았습니다.

### Ensemble / Stacking

단순 soft voting부터 Ridge/Logistic stacking까지 단계적으로 비교했습니다.

대표적인 결과:

- ResNet18 + DenseNet121 soft voting: Public 0.93590
- ConvNeXt-Tiny: Public 0.94711
- Ridge stacking: Public 0.95032
- OOF Logistic combination: Public 0.95513
- OOF compact ensemble: Public **0.95833**

앙상블 복잡도를 높이는 것 자체보다 실제 OOF와 leaderboard에서 일관된 개선이 발생하는지를 기준으로 실험을 평가했습니다.

## 3. Failure Analysis & XAI

성능 차이를 숫자로만 비교하지 않고 Grad-CAM과 LIME을 이용해 모델이 어떤 영역에 반응하는지 확인했습니다.

분석 대상에는 다음이 포함됩니다.

- Validation false positive / false negative
- Backbone 간 prediction disagreement
- ConvNeXt disagreement samples
- augmentation 변경 전후 activation 변화
- marker / artifact 등 비병변 영역에 대한 반응

대표 XAI 결과는 `reports/final/assets/`에 포함되어 있습니다.

![ConvNeXt disagreement](reports/final/assets/gradcam_convnext_disagreement_comparison.png)

![Multi-model disagreement](reports/final/assets/gradcam_multimodel_disagreement_comparison.png)

### Augmentation review

X-ray의 해부학적 구조와 artifact 가능성을 고려해 Flip/Rotation을 기계적으로 적용하는 대신 augmentation 가정을 재검토했습니다.

후반 실험에서는 no-flip 설정과 XAI 비교를 통해 augmentation 변경이 prediction 및 activation에 미치는 영향을 별도로 분석했습니다.

## 4. Model Improvement vs. Competition Optimization

이 프로젝트에서는 **모델/검증 개선**과 **competition-specific submission optimization**을 구분합니다.

### Model / validation pipeline

Backbone comparison, OOF validation, stacking을 통해 Public Score를:

**0.81891 → 0.95833**

까지 개선했습니다.

이 구간을 모델 선택 및 validation pipeline 개선의 핵심 결과로 봅니다.

### Competition-specific final stage

후반에는 OOF probability, model disagreement, XAI evidence와 제한적인 public feedback을 이용해 hard-case test sample을 재검토했습니다.

그 결과 최종 submission은:

- Public Score: **0.9663461538**
- Final Rank: **6th**
- 이전 최고 submission 0.9647435897 대비 약 +1 sample 개선

을 기록했습니다.

이 단계는 모델의 일반화 성능 자체가 향상됐다는 근거로 해석하지 않습니다. 실험 기록에서도 후반 sample-level reranking을 **마지막 제출 후보를 선별하기 위한 competition-specific strategy**로 분리해 기록했습니다.

## 5. Experiment Progression

주요 progression은 다음과 같습니다.

```text
SimpleCNN
  ↓ generalization gap 발견
ResNet18 / DenseNet121 / EfficientNet-B0
  ↓
ConvNeXt-Tiny
  ↓
Soft Voting
  ↓
Ridge / Logistic Stacking
  ↓
5-Fold OOF
  ↓
OOF Ensemble
  ↓
XAI + Failure / Disagreement Analysis
  ↓
Competition-specific Hard-case Review
```

전체 실험 이력과 각 submission 결과는 [`experiment_history.md`](experiment_history.md)에 기록되어 있습니다.

## 6. Repository Structure

```text
.
├── [Baseline]_CNN...ipynb
├── experiment_history.md
├── oof/
│   └── oof_plan.md
├── reports/
│   └── final/
│       ├── final_report.md
│       ├── mission1_data_baseline.md
│       ├── mission2_improvement_strategy.md
│       ├── mission3_experiments_validation.md
│       ├── mission4_result_analysis.md
│       ├── mission5_xai_service_design.md
│       └── assets/
├── scripts/
│   ├── run_exp124_convnext_384_noflip_fold0.py
│   ├── run_exp125_aug_xai_compare.py
│   ├── run_exp126_convnext_384_noflip_oof.py
│   ├── run_exp127_384_noflip_stacking.py
│   ├── run_exp131_final_submit_decision.py
│   └── run_exp132_success_failure_contrast.py
├── pyproject.toml
└── uv.lock
```

Raw X-ray images, model checkpoints, bulk predictions/submissions 및 대용량 생성 파일은 repository에서 제외했습니다.

## 7. Key Takeaways

1. **높은 validation accuracy가 높은 generalization을 보장하지 않았습니다.**  
   초기 Validation 0.9732와 Public 0.8189의 차이가 이후 실험 설계의 출발점이 됐습니다.

2. **OOF는 성능 향상 수단인 동시에 검증 구조를 개선하는 수단이었습니다.**  
   단일 split보다 안정적인 prediction을 확보하고 stacking 입력을 구성하는 데 사용했습니다.

3. **Meta-CV 최고점과 leaderboard 최고점은 일치하지 않았습니다.**  
   하나의 metric만을 최적화하기보다 모델 간 disagreement와 failure case를 함께 분석했습니다.

4. **XAI를 결과 시각화에만 사용하지 않았습니다.**  
   Grad-CAM/LIME을 augmentation 및 failure hypothesis를 검토하는 분석 도구로 사용했습니다.

5. **Competition optimization과 generalization improvement를 분리했습니다.**  
   최종 0.96635는 competition-specific sample review까지 포함한 결과이며, 모델/검증 파이프라인의 핵심 성과는 0.95833까지로 구분합니다.

## 8. Limitations

- Public leaderboard는 제한된 test set에 대한 결과이므로 외부 의료기관 데이터에서의 일반화 성능을 의미하지 않습니다.
- 후반 submission optimization은 public feedback을 일부 사용했으므로 독립적인 external validation으로 볼 수 없습니다.
- 본 프로젝트는 의료 진단 시스템의 임상적 유효성을 검증한 연구가 아닙니다.
- 실제 임상 적용을 위해서는 external validation, domain shift 평가, calibration 및 임상적 decision threshold 검증이 추가로 필요합니다.

## Tech Stack

`Python` · `PyTorch` · `torchvision` · `scikit-learn` · `pandas` · `NumPy` · `Grad-CAM` · `LIME`

---

### Result

**Model / Validation Pipeline:** Public `0.81891 → 0.95833`  
**Final Competition Submission:** Public `0.9663461538` · **6th place**
