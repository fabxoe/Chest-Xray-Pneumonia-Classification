# Mission 1. 데이터 이해 및 Baseline 구축

## 1. 문제 정의

이번 과제는 흉부 X-ray 이미지를 입력받아 정상과 폐렴을 이진 분류하는 문제이다. 공식 평가는 Accuracy이지만, 의료 이미지 분류에서는 단순 정확도만으로 모델을 해석하기 어렵다. 실제 폐렴을 정상으로 예측하는 FN은 위험도가 높을 수 있으므로, validation 단계에서는 accuracy와 함께 precision, recall, F1, confusion matrix를 함께 확인했다.

## 2. 데이터 구조와 클래스 분포

전체 학습 데이터는 5,216장이며, 테스트 데이터는 624장이다. 학습 데이터의 클래스 분포는 다음과 같았다.

| label | 개수 | 비율 |
|---:|---:|---:|
| 0 | 1,341 | 25.71% |
| 1 | 3,875 | 74.29% |

클래스 1이 약 74%를 차지하므로, 모델이 폐렴 쪽으로 치우쳐도 accuracy가 높게 보일 수 있다. 따라서 train/validation 분리는 `stratified split`으로 수행했고, validation에서도 클래스별 오류를 따로 확인했다.

## 3. Baseline 모델 구성

초기 baseline은 간단한 CNN 구조를 사용했다. 이후 개선 baseline에서는 다음 요소를 추가했다.

- stratified train/validation split
- class weight를 적용한 `Weighted CrossEntropyLoss`
- BatchNorm과 Dropout
- 기본 이미지 augmentation
- validation metric과 confusion matrix 출력

개선 SimpleCNN은 validation accuracy `0.97318`을 기록했지만, 리더보드 점수는 `0.81891`에 머물렀다. 이는 validation 분할에서 잘 보이는 성능이 실제 테스트 분포에서도 그대로 유지되지 않을 수 있음을 보여준다.

## 4. Baseline 성능과 초기 한계

개선 SimpleCNN의 validation 결과는 표면적으로 높았지만, 다음 한계가 있었다.

1. 모델 용량이 작아 흉부 X-ray의 복잡한 질감과 국소 패턴을 충분히 학습하기 어렵다.
2. 클래스 불균형으로 인해 accuracy만 보면 모델의 오류 방향을 놓치기 쉽다.
3. validation set이 전체 데이터의 일부이므로, 한 번의 split만으로 일반화 성능을 단정하기 어렵다.
4. Grad-CAM 같은 해석 도구가 없으면 모델이 실제로 어느 영역에 반응했는지 확인하기 어렵다.

## 5. 다음 개선 방향

Baseline 이후에는 세 가지 방향으로 개선을 진행하기로 했다.

- ImageNet pretrained CNN을 활용한 transfer learning
- class weight, fine-tuning 범위, optimizer, augmentation 조정
- 단일 validation이 아닌 OOF 기반 검증과 앙상블

## 6. 핵심 인사이트

의료 이미지 분류에서는 accuracy가 높아도 안전한 모델이라고 단정할 수 없다. 특히 클래스 불균형이 있는 상황에서는 FN/FP의 의미를 함께 확인해야 하며, baseline 단계부터 confusion matrix와 recall을 같이 보는 구조를 만들어야 이후 실험을 올바르게 비교할 수 있다.

