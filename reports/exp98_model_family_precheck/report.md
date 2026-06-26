# EXP-98 Model Family Precheck

목적: 새 제출 후보를 바로 만들기 전에 ConvNeXt/EfficientNet과 다른 오류 방향을 줄 수 있는 모델 family를 사전 검토한다.

## 결론

1순위는 `maxvit_tiny_rw_224.sw_in1k` fold0 smoke다. hybrid attention 계열이라 기존 CNN OOF pool과 다른 정보를 줄 가능성이 있고, tiny급이라 계산 리스크가 비교적 작다.

2순위는 `coatnet_nano_rw_224.sw_in1k`다. MaxViT smoke가 메모리/속도 문제를 보이면 더 가벼운 hybrid 후보로 진행한다.

SwinV2/DeiT는 diversity 후보지만, 기존 Swin/DINOv2 결과를 감안하면 바로 5-fold로 가지 않고 smoke 결과를 본 뒤 결정한다.

## Candidate Summary

| rank | family | model | params | trainable guess | first run | cost |
|---:|---|---|---:|---:|---|---|
| 1 | MaxViT | `maxvit_tiny_rw_224.sw_in1k` | 28,545,338 | 16,336,130 | fold0 smoke 2 epochs | 4~8h for 5-fold |
| 2 | CoAtNet | `coatnet_nano_rw_224.sw_in1k` | 14,629,270 | 8,874,346 | fold0 smoke 2 epochs | 2~4h for 5-fold if stable |
| 3 | SwinV2 | `swinv2_tiny_window8_256.ms_in1k` | 27,579,692 | 15,384,626 | fold0 smoke 2 epochs | 4~8h for 5-fold |
| 4 | DeiT | `deit3_small_patch16_224.fb_in1k` | 21,675,266 | 1,776,002 | fold0 smoke 2 epochs | 4~8h for 5-fold |
| 5 | CoAtNet larger | `coatnet_0_rw_224.sw_in1k` | 26,668,100 | 12,854,342 | fold0 smoke 2 epochs | 4~8h for 5-fold |

## 실행 게이트

- 바로 5-fold 학습하지 않는다.
- 먼저 fold0 smoke 2 epoch로 OOF 성능, confusion matrix, guardrail 샘플 방향을 확인한다.
- smoke가 EXP-65처럼 보호 샘플을 대량으로 깨면 단독 OOF 확장은 중단한다.
- smoke가 약하더라도 오류 방향이 기존 모델과 다르면 stacking feature 후보로 보류한다.

## Outputs

- `reports/exp98_model_family_precheck/exp98_model_family_precheck.csv`
- `reports/exp98_model_family_precheck/exp98_model_family_details.json`
- `reports/exp98_model_family_precheck/report.md`