# EXP-123 Augmentation and XAI Review

## Summary

기존 학습 증강은 대부분 다음 형태였다.

- `Resize(224, 224)`
- `RandomHorizontalFlip(p=0.5)`
- `RandomRotation(5~7)`
- `ToTensor`
- `Normalize(ImageNet mean/std)`

TTA는 `original + horizontal flip` 중심이었다. Center crop은 주요 학습 스크립트에서 거의 사용하지 않았다.

현재 public 결과와 XAI 관찰을 함께 보면, 다음 방향이 더 타당하다.

- 좌우반전은 기본 증강에서 제거하거나 최소화한다.
- 회전도 제거하거나 아주 약하게만 둔다.
- 입력 크기는 `320` 또는 `384`를 우선 검토한다.
- 강한 crop 대신 전체 흉부 구조를 보존하는 resize를 사용한다.
- brightness/contrast는 약하게만 사용한다.

## Why Reduce Flip and Rotation

흉부 X-ray는 이진 폐렴 분류라도 좌우 방향 정보가 완전히 무의미하지 않다. `R` marker, 심장 위치, 촬영 방향, 폐야 비대칭 같은 요소가 존재한다. 따라서 horizontal flip은 일반 자연 이미지처럼 항상 안전한 증강으로 보기 어렵다.

기존 결과에서도 flip/TTA 계열은 안정적인 이득을 주지 못했다.

- EXP-37 TTA: original+hflip을 적용했지만 모든 모델에서 개선되지 않음
- EXP-38 TTA stacking: public `0.9487179487`로 하락
- EXP-15 no horizontal flip: public `0.9391025641`로 낮았지만, 이 실험은 flip 제거 외에 학습 흔들림/rotation 축소/단일 split 영향이 섞여 있어 단독 결론으로 쓰기 어려움

따라서 결론은 “좌우반전이 반드시 나쁘다”가 아니라, “이 데이터에서는 좌우반전을 안전한 기본 증강으로 보기 어렵다”가 더 정확하다.

## Image Size

`224`는 빠르고 안정적이지만, X-ray의 미세 음영과 폐야 texture를 보기에는 작을 수 있다. 다음 후보는 `320` 또는 `384`다.

- `320`: 계산 비용 증가가 비교적 작고, 224보다 세부 정보가 늘어남
- `384`: 폐야 texture와 작은 침윤 패턴을 더 잘 보존할 가능성이 있음
- `448`: 정보량은 크지만 현재 마감/메모리/과적합 위험을 고려하면 우선순위 낮음

다음 학습형 실험을 한다면 `ConvNeXt-Tiny` 또는 `EfficientNet-B0/V2`에 대해 `384` 입력을 우선 검토한다.

## Grad-CAM Evidence

EXP-112 drop candidate Grad-CAM 요약을 public 결과와 대조했다.

| group | ConvNeXt fold0 p1 mean | CAM center fraction mean | CAM top fraction mean | mean vote1 |
|---|---:|---:|---:|---:|
| success drop to 0: `0455`, `0057`, `0385`, `0303` | 0.2251 | 0.1696 | 0.3690 | 7.50 |
| failed keep 1: `0289`, `0172`, `0013` | 0.6042 | 0.3210 | 0.2164 | 6.67 |

해석:

- 성공한 drop 후보는 ConvNeXt fold0 class-1 확률이 낮았다.
- 성공한 drop 후보는 CAM center fraction도 낮았다.
- 특히 `test_0385`는 CAM이 폐 중심보다 상단/가장자리 쪽에 강하게 반응했다.
- `test_0455`, `test_0057`, `test_0303`도 class-1 CAM이 폐 중심 병변 근거라기보다 주변부/하부/측면 반응으로 보인다.

주의:

Grad-CAM은 실제 병변 위치를 확정하는 도구가 아니다. 여기서는 “모델이 class 1을 판단할 때 상대적으로 어디에 반응했는지”를 보는 보조 근거로만 사용한다.

## Practical Next Transform

다음 학습 실험 후보:

```python
train_tf = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ColorJitter(brightness=0.08, contrast=0.08),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_tf = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

보류할 증강:

- `RandomHorizontalFlip`
- 큰 `RandomRotation`
- vertical flip
- 강한 crop
- 강한 color jitter

선택적으로 검토할 약한 증강:

- `RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02))`
- 단, 마감 직전 제출용 후보에는 새 학습보다 current-best branch를 우선한다.

## Decision

남은 제출 기회가 적은 현재 상황에서는 새 384 학습을 즉시 제출 전략으로 쓰기보다, 다음 장기/보조 실험으로 둔다. 마감 전 추가 학습 시간이 허용된다면 `ConvNeXt-Tiny 384 no-flip/no-rotation`을 OOF가 아닌 fold0 smoke로 먼저 확인하고, OOF 확장 여부를 결정한다.
