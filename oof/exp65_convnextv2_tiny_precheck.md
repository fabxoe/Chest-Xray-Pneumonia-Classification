# EXP-65 ConvNeXt V2 Tiny 사전 검토

## 결론

ConvNeXt V2 Tiny는 다음 OOF 모델 자산 후보로 타당하다. 현재 환경에서 `timm` 기반 pretrained weight 로딩과 224 입력 forward smoke test가 성공했고, 기존 ConvNeXt OOF 파이프라인과 구조적으로 잘 맞는다.

다만 바로 full fine-tuning으로 시작하기보다 `stage3 + head` fine-tuning을 1차 OOF로 진행하는 것이 적절하다. 목적은 단일 최고점 갱신이 아니라 ConvNeXt V1, EfficientNet-B0, ResNet18과 다른 오류 방향을 가진 일반화된 모델을 확보하는 것이다.

## 환경 확인

- `torch`: 2.12.0
- `torchvision`: 0.27.0
- `timm`: 1.0.27
- device 후보: `mps`
- pretrained 로딩 성공 모델: `convnextv2_tiny.fcmae_ft_in22k_in1k`

사용 가능한 ConvNeXt V2 Tiny pretrained 모델:

| timm model name | 입력 | 파라미터 | 비고 |
|---|---:|---:|---|
| `convnextv2_tiny.fcmae` | 224 | 27.87M | FCMAE pretrain |
| `convnextv2_tiny.fcmae_ft_in1k` | 224 | 27.87M | ImageNet-1K fine-tuned |
| `convnextv2_tiny.fcmae_ft_in22k_in1k` | 224 | 27.87M | ImageNet-22K -> 1K fine-tuned |
| `convnextv2_tiny.fcmae_ft_in22k_in1k_384` | 384 | 27.87M | 고해상도, 비용 증가 |

## 권장 시작 모델

`convnextv2_tiny.fcmae_ft_in22k_in1k`

이유:

- 224 입력이라 기존 ConvNeXt-Tiny OOF 파이프라인과 동일한 이미지 크기를 유지할 수 있다.
- ImageNet-22K를 거친 pretrained weight라 일반화 feature가 더 넓을 가능성이 있다.
- 384 모델은 비용이 커지고 기존 224 파이프라인과 비교가 어려워 1차 후보로는 후순위다.

## Fine-tuning 범위

| 설정 | trainable params | 판단 |
|---|---:|---|
| head only | 0.003M | 너무 약함. 이전 Swin/EfficientNetV2-S probe처럼 underfit 위험 |
| stage3 + head | 15.49M | 1차 권장. 기존 ConvNeXt-Tiny stage4 학습과 유사한 위치 |
| stage2 + stage3 + head | 26.63M | 2차 후보. 1차가 underfit이면 확대 |
| full fine-tuning | 27.86M | 장기 후보. 첫 시도부터는 과함 |

## 예상 계산 비용

기존 기록:

- EXP-46 ConvNeXt-Tiny OOF: fold당 평균 약 9.1분, 5-fold 총 약 36.5분
- EXP-54 ConvNeXt-Tiny seed123 OOF: fold당 평균 약 6.8분, 5-fold 총 약 34.2분

예상:

- ConvNeXt V2 Tiny `stage3 + head`, 224, 6 epochs: 약 40~70분
- `stage2 + stage3 + head`: 약 60~120분
- 384 입력 또는 full fine-tuning: 2시간 이상 가능

따라서 EXP-65 1차 OOF는 10시간 초과 장기 실험 게이트 대상은 아니다.

## 실행 기준

1차 실행안:

- `EXP_ID`: `exp65`
- model: `convnextv2_tiny.fcmae_ft_in22k_in1k`
- image size: 224
- trainable: `stage3 + head`
- epochs: 6
- optimizer: AdamW
- LR: `5e-5`
- loss: weighted CrossEntropyLoss
- augmentation: 기존 ConvNeXt OOF와 동일
  - Resize 224
  - RandomHorizontalFlip 0.5
  - RandomRotation 7
  - ImageNet mean/std normalize

## 중단 조건

fold0 smoke 또는 1개 fold 기준 다음이면 중단/재설계한다.

- validation accuracy가 `0.970` 미만
- FN이 기존 ConvNeXt OOF보다 뚜렷하게 증가
- test label 1 비율이 과도하게 증가하거나 감소
- 보호 샘플 중 `test_0244`, `test_0025`, `test_0539`, `test_0490`이 동시에 나쁜 방향으로 흔들림

## 제출 기준

단독 mean 제출은 우선순위가 낮다. 제출 여부는 다음 순서로 판단한다.

1. OOF accuracy와 confusion matrix 확인
2. test mean label 분포 확인
3. 보호 샘플 방향 확인
4. 기존 ConvNeXt/EfficientNet OOF와 prediction correlation 확인
5. EXP-56 기준 protected stacking에 추가

즉, EXP-65의 목적은 단독 제출이 아니라 EXP-66 이후 stacking feature로 쓰는 것이다.

## 다음 작업

1. `scripts/run_exp65_convnextv2_tiny_oof.py` 작성
2. fold0 smoke test 실행
3. fold0 결과가 기준을 넘으면 5-fold OOF 진행
4. `EXP-66`: ConvNeXt V2 Tiny OOF 포함 stacking
