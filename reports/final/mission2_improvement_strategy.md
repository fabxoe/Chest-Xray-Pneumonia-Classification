# Mission 2. 개선 전략 설계

## 1. Baseline 한계에서 출발한 가설

SimpleCNN baseline은 validation에서는 높은 accuracy를 보였지만, 리더보드에서는 충분히 일반화되지 않았다. 이를 바탕으로 다음 가설을 세웠다.

- pretrained CNN은 자연 이미지에서 학습한 저수준 edge/texture 표현을 활용할 수 있어 작은 의료 이미지 데이터에서도 더 안정적일 수 있다.
- 단일 모델보다 서로 다른 구조의 모델을 조합하면 오류 방향이 달라져 보완 효과가 생길 수 있다.
- 클래스 불균형이 있으므로 loss와 threshold, class weight를 함께 검토해야 한다.
- 의료 이미지에서는 augmentation이 항상 유리하지 않으며, 좌우 반전이나 과한 회전은 영상의 방향 정보를 흔들 수 있다.

## 2. 선택한 개선 전략

### 2.1 Pretrained CNN fine-tuning

ResNet18, DenseNet121, EfficientNet-B0, ConvNeXt-Tiny를 순차적으로 적용했다. 처음부터 전체 모델을 강하게 학습하기보다, 마지막 block과 classifier 중심으로 fine-tuning을 시작했다.

| 모델 | 선택 이유 | 주요 실험 결과 |
|---|---|---:|
| ResNet18 | 안정적인 baseline transfer learning 모델 | `0.9278846154` |
| DenseNet121 | feature reuse 구조로 X-ray 분류 사례가 많음 | `0.9262820513` |
| EfficientNet-B0 | 파라미터 효율이 좋고 앙상블 보완 가능성 있음 | `0.8894230769` |
| ConvNeXt-Tiny | CNN 기반이면서 현대적 구조, 강한 표현력 기대 | `0.94711` |

### 2.2 Loss와 class weight 조정

초기 클래스 비율이 `0: 25.71%`, `1: 74.29%`로 불균형했기 때문에 class weight를 적용했다. 이후 ConvNeXt 실험에서는 class weight를 sqrt 방식으로 약화하는 실험도 진행했다. validation은 개선되는 경우가 있었지만, 리더보드 성능은 항상 같이 오르지 않았다.

### 2.3 Augmentation 조정

초기에는 좌우 반전과 약한 회전을 사용했다. 이후 X-ray의 방향 정보가 의미 있을 수 있다고 판단해 no-flip 실험도 진행했다. 최종적으로는 augmentation을 많이 넣는 것보다, 모델이 어떤 변화에 민감한지 확인하는 방식으로 접근했다.

### 2.4 OOF와 stacking

단일 validation split의 한계를 줄이기 위해 5-fold OOF를 설계했다. ConvNeXt, ResNet18, EfficientNet-B0 등 서로 다른 backbone의 OOF probability를 저장하고, Ridge/Logistic 기반 meta-model을 적용했다.

## 3. 적용하지 않거나 우선순위를 낮춘 전략

Swin, CoAtNet, MaxViT, DINOv2, RegNetY 등도 검토했지만, 대부분 OOF 성능이나 제출 안정성 측면에서 기존 ConvNeXt/EfficientNet 계열보다 우선순위가 낮았다. 큰 모델을 무작정 추가하기보다, 기존 강한 모델들의 오류 차이를 이해하고 조합하는 쪽이 더 실용적이었다.

## 4. 핵심 인사이트

단일 모델의 accuracy만 높이는 것보다, 서로 다른 backbone이 어떤 상황에서 다르게 판단하는지 확인하는 것이 더 중요했다. 특히 ConvNeXt, ResNet, EfficientNet은 같은 이미지를 보더라도 확률과 반응 위치가 달랐고, 이 차이가 OOF stacking의 근거가 되었다.

