# Mission 3. 실험 수행 및 검증

## 1. 실험 관리 방식

실험은 `experiment_history.md`에 기록하며 진행했다. 각 실험마다 모델, 주요 변경점, epoch, optimizer, learning rate, loss, validation accuracy, 리더보드 점수를 남겼다. 초반에는 단일 validation split으로 비교했고, 중반 이후에는 5-fold OOF 기반으로 검증 방식을 확장했다.

## 2. Baseline 대비 주요 성능 변화

대표 실험 흐름은 다음과 같다.

| 단계 | 대표 모델/전략 | 리더보드 점수 | 해석 |
|---|---|---:|---|
| Baseline | 개선 SimpleCNN | 0.81891 | 구조가 단순해 테스트 분포 일반화 한계 |
| Transfer learning 1 | ResNet18 fine-tuning | 0.9278846154 | pretrained CNN의 효과 확인 |
| Model family 확장 | DenseNet121 / EfficientNet-B0 | 0.9262820513 / 0.8894230769 | 단독 성능보다 보완 정보 가능성 확인 |
| 강한 CNN backbone | ConvNeXt-Tiny | 0.94711 | 단일 모델 최고 수준으로 상승 |
| 초기 stacking | Ridge / Logistic stacking | 0.9503205128 | 모델 확률 조합의 효과 확인 |
| OOF stacking | ConvNeXt + EfficientNet OOF | 0.9551282051 이상 | 단일 split보다 안정적인 meta feature 확보 |
| 최종 결과 | 최종 제출 | 0.9663461538 | 전체 실험 흐름의 최종 성과 |

## 3. 주요 실험 결과 해석

### 3.1 ResNet18 fine-tuning

ResNet18은 SimpleCNN 대비 크게 개선되었다. pretrained feature를 활용하면서 마지막 layer 중심으로 학습한 것이 작은 데이터셋에서 효과적이었다. 다만 epoch를 10으로 늘리거나 layer3까지 학습 범위를 넓혔을 때 리더보드 성능이 하락했다. 이는 더 많은 학습과 더 넓은 fine-tuning이 항상 일반화 성능 개선으로 이어지지 않음을 보여준다.

### 3.2 ConvNeXt-Tiny

ConvNeXt-Tiny는 단일 모델 기준으로 가장 큰 개선을 보였다. ImageNet pretrained ConvNeXt의 표현력이 X-ray의 질감과 국소 패턴을 잘 포착한 것으로 해석할 수 있다. 다만 full fine-tuning이나 class weight 변경은 validation에서는 좋아 보이는 경우가 있었지만, 실제 리더보드에서는 일관된 개선을 보이지 않았다.

### 3.3 OOF 기반 검증

단일 validation split만 사용할 때는 validation과 리더보드의 방향이 어긋나는 경우가 있었다. 이를 줄이기 위해 ConvNeXt, ResNet18, EfficientNet-B0에 대해 5-fold OOF prediction을 만들었다. OOF 기반 stacking은 각 이미지가 학습에 사용되지 않은 fold에서 예측된 값으로 meta-model을 학습하기 때문에, 단일 split보다 신뢰도가 높다.

### 3.4 384 no-flip 실험

입력 크기를 384로 키우고 좌우 반전과 회전을 제거한 ConvNeXt 실험도 수행했다. fold0에서는 accuracy가 `0.98755`로 개선되었고, 5-fold OOF에서는 `0.98409`를 기록했다. 그러나 전체 관점에서는 기존 강한 제출 방향과 충돌하는 예측도 있었으므로, 단독 제출보다는 정보원으로 제한적으로 해석했다.

## 4. 실패한 실험에서 얻은 검증 기준

실험이 실패했을 때도 단순히 버리지 않고 다음 기준을 얻었다.

- epoch 증가가 항상 성능 향상을 의미하지 않는다.
- fine-tuning 범위 확대는 과적합 또는 분포 불일치를 만들 수 있다.
- augmentation은 의료 영상의 방향성과 촬영 조건을 고려해야 한다.
- validation metric과 리더보드 metric이 다를 때는 OOF 기반 재검증이 필요하다.

## 5. 핵심 인사이트

validation 성능이 높아도 실제 제출 성능과 항상 일치하지 않았다. 따라서 최종 성능 개선의 핵심은 모델을 계속 바꾸는 것보다, OOF prediction을 저장하고 여러 모델의 오류 방향을 비교할 수 있는 검증 구조를 만든 것이었다.

