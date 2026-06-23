# 흉부 X-ray 이미지 분류 실험 기록

## 프로젝트 개요

- 대회: 흉부 X-ray 이미지 정상/폐렴 이진 분류
- 공식 평가 지표: Accuracy
- 학습 데이터: 5,216장
- 테스트 데이터: 624장
- 클래스 분포:
  - label 0: 1,341장, 25.71%
  - label 1: 3,875장, 74.29%

## 실험 관리 기준

- validation split: train 데이터의 20%
- split 방식: `stratified split`
- seed: 42
- 주요 검증 지표:
  - Accuracy: 공식 평가 지표
  - Precision: 폐렴이라고 예측한 것 중 실제 폐렴인 비율
  - Recall: 실제 폐렴 중 모델이 폐렴으로 맞춘 비율
  - F1-score: precision과 recall의 균형
  - Confusion matrix: FP/FN 오류 양상 확인

의료 이미지 분류에서는 accuracy만으로 모델을 평가하기 어렵다. 특히 실제 폐렴을 정상으로 예측하는 FN은 위험도가 높을 수 있으므로, recall과 confusion matrix를 함께 확인한다.

## 실험 요약표

| 실험 ID | 모델 | 주요 변경점 | Epoch | Optimizer | LR | Loss | Validation Accuracy | Leaderboard Score | 당시 저장 파일명 | 메모 |
|---|---|---|---:|---|---:|---|---:|---:|---|---|
| EXP-00 | 기본 SimpleCNN | 원본 baseline, train 전체 학습, validation 없음 | 2 | Adam | 1e-3 | CrossEntropyLoss | - | 미제출 | - | 원본 baseline은 리더보드에 제출하지 않음 |
| EXP-01 | 개선 SimpleCNN | stratified validation split, augmentation, class weight, BatchNorm, Dropout 추가 | 8 | AdamW | 1e-3 | Weighted CrossEntropyLoss | 0.97318 | 0.81891 | submissions/baseline_submit.csv | validation 기준 최고 epoch 6 선택 후 제출 |
| EXP-02 | ResNet18 fine-tuning | pretrained ResNet18, layer4+fc 학습, 3채널 변환, class weight 적용 | 6 | AdamW | 1e-4 | Weighted CrossEntropyLoss | 0.98659 | 0.9278846154 | submissions/resnet18_submit.csv | 현재 최고 점수 |
| EXP-03 | ResNet18 fine-tuning | EXP-02와 동일, epoch만 10으로 증가 | 10 | AdamW | 1e-4 | Weighted CrossEntropyLoss | 0.98659 | 0.8782051282 | submissions/resnet18_epoch10_submit.csv | epoch 증가 후 leaderboard 하락 |
| EXP-04 | ResNet18 fine-tuning | layer3+layer4+fc 학습으로 fine-tuning 범위 확대 | 6 | AdamW | 5e-5 | Weighted CrossEntropyLoss | 0.98563 | 0.8894230769 | submissions/resnet18_layer34_submit.csv | fine-tuning 범위 확대 후 leaderboard 하락 |
| EXP-05 | ResNet18 threshold search | layer4+fc 재학습 후 threshold 0.30~0.75 비교 | 6 | AdamW | 1e-4 | Weighted CrossEntropyLoss | 0.98276 | 0.891025641 | submissions/resnet18_threshold_0p7_submit.csv | threshold 조정 후 최고 점수 갱신 실패 |
| EXP-06 | EfficientNet-B0 fine-tuning | pretrained EfficientNet-B0, features[-2:]+classifier 학습, 앙상블 대비 확률 저장 | 6 | AdamW | 1e-4 | Weighted CrossEntropyLoss | 0.97605 | 0.8894230769 | submissions/efficientnet_b0_submit.csv | 단독 성능은 ResNet보다 낮지만 앙상블 후보 |
| EXP-07 | ResNet18 재현 및 확률 저장 | EXP-02 최고 구조 재현, 앙상블 대비 model/prob 저장 | 6 | AdamW | 1e-4 | Weighted CrossEntropyLoss | 0.98659 | 0.9278846154 | submissions/exp07_resnet18_submit.csv | EXP-02와 동일 성능 재현, 앙상블 기준 모델 |
| EXP-08 | Soft voting 후보 생성 | EXP-07 ResNet18 + EXP-06 EfficientNet-B0 probability weighted average | - | - | - | - | 0.98563 | 0.9230769231 | submissions/exp08_softvote_r70_e30_submit.csv | soft voting은 최고 점수 갱신 실패 |
| EXP-09 | DenseNet121 fine-tuning | pretrained DenseNet121, denseblock4+classifier 학습, 앙상블 대비 확률 저장 | 6 | AdamW | 1e-4 | Weighted CrossEntropyLoss | 0.97797 | 0.9262820513 | submissions/exp09_densenet121_submit.csv | ResNet 최고와 근접, 앙상블 유력 후보 |
| EXP-10 | ResNet18 + DenseNet121 soft voting | EXP-07 ResNet18과 EXP-09 DenseNet121 probability weighted average | - | - | - | - | 0.99042 | 0.9358974359 | submissions/exp10_softvote_res40_den60_submit.csv | 현재 최고 점수 갱신 |
| EXP-11 | ResNet18 + DenseNet121 fine soft voting | DenseNet 비중 0.62~0.75 미세 탐색 | - | - | - | - | 0.98946 | 0.9342948718 | submissions/exp11_softvote_res35_den65_submit.csv | DenseNet 0.65는 최고점보다 소폭 하락 |
| EXP-12 | ConvNeXt-Tiny fine-tuning | pretrained ConvNeXt-Tiny, 마지막 stage+classifier 학습, 앙상블 대비 확률 저장 | 6 | AdamW | 5e-5 | Weighted CrossEntropyLoss | 0.98084 | 0.94711 | submissions/exp12_convnext_tiny_submit.csv | 현재 최고 점수 갱신 |
| EXP-13 | ConvNeXt 중심 soft voting 후보 | ConvNeXt + ResNet/DenseNet/3-model probability weighted average | - | - | - | - | 0.98659 | 0.9455128205 | submissions/exp13_softvote_cnx50_den50_submit.csv | validation은 개선됐지만 leaderboard는 ConvNeXt 단독보다 하락 |
| EXP-14 | ConvNeXt-Tiny sqrt class weight | 기존 class weight를 sqrt 방식으로 약화, 마지막 stage+classifier 학습 | 6 | AdamW | 5e-5 | Sqrt Weighted CrossEntropyLoss | 0.98563 | 0.9439102564 | submissions/exp14_convnext_tiny_sqrt_weight_submit.csv | validation은 개선됐지만 leaderboard 하락 |
| EXP-15 | ConvNeXt-Tiny no horizontal flip | HorizontalFlip 제거, Rotation 5도로 축소, 기존 class weight 유지 | 6 | AdamW | 5e-5 | Weighted CrossEntropyLoss | 0.98084 | 0.9391025641 | submissions/exp15_convnext_tiny_no_flip_submit.csv | validation은 EXP-12와 동일했지만 leaderboard 하락 |
| EXP-16 | ConvNeXt-Tiny seed ensemble | EXP-12 설정 유지, seed7 추가 학습 후 EXP-12와 soft voting | 6 | AdamW | 5e-5 | Weighted CrossEntropyLoss | 0.98467 | 0.9455128205 | submissions/exp16_softvote_cnx12_70_seed7_30_submit.csv | validation은 개선됐지만 leaderboard는 EXP-12보다 하락 |
| EXP-17 | ConvNeXt-Tiny threshold 상향 | EXP-12 probability 사용, threshold 0.52~0.70 탐색 | - | - | - | - | 0.97989 | 0.9455128205 | submissions/exp17_convnext_tiny_threshold_0p54_submit.csv | threshold 0.54는 EXP-12보다 하락 |
| EXP-18 | ConvNeXt-Tiny full fine-tuning | 전체 ConvNeXt unfreeze, backbone/classifier differential LR 적용 | 6 | AdamW | 1e-5 / 5e-5 | Weighted CrossEntropyLoss | 0.97989 | 0.9471153846 | submissions/exp18_convnext_tiny_full_finetune_submit.csv | validation은 EXP-12보다 낮지만 leaderboard는 EXP-12와 동점 |
| EXP-19 | EXP-12 + EXP-18 soft voting | ConvNeXt stage4 모델 0.7 + full fine-tuning 모델 0.3 확률 평균 | - | - | - | - | 0.98276 | 0.9439102564 | submissions/exp19_softvote_cnx12_70_full30_submit.csv | validation은 개선됐지만 leaderboard 하락 |
| EXP-20 | RidgeClassifier stacking | ResNet/DenseNet/EfficientNet/ConvNeXt 확률+rank feature로 RidgeClassifier 학습 | - | - | - | RidgeClassifier alpha=100 | 0.98755 | 0.9503205128 | submissions/exp20_ridge_stacking_alpha100_submit.csv | 현재 최고 점수 갱신 |
| EXP-21 | LogisticRegression stacking | EXP-20과 동일 feature, LogisticRegression C=1.0 적용 | - | - | - | LogisticRegression C=1.0 | 0.98755 | 0.9503205128 | submissions/exp21_logistic_stacking_C1_submit.csv | Ridge와 leaderboard 동점 |
| EXP-22 | RidgeClassifier feature v2 | 기존 prob/rank에 mean/std/range와 모델 간 차이 feature 추가 | - | - | - | RidgeClassifier alpha=3000 | 0.98276 | 0.9471153846 | submissions/exp22_ridge_feature_v2_alpha3000_submit.csv | feature 추가 후 leaderboard 하락 |
| EXP-23 | RidgeClassifier feature pruning | ConvNeXt12 + DenseNet121 + ResNet18의 prob/rank만 사용 | - | - | - | RidgeClassifier alpha=100 | 0.98755 | 0.9487179487 | submissions/exp23_ridge_pruned_cnx_den_res_alpha100_submit.csv | EXP-20보다 하락, EXP-12보다는 개선 |
| EXP-24 | Ridge feature importance 분석 | EXP-20 Ridge의 coefficient와 permutation importance 확인 | - | - | - | 분석 | 0.98755 | 제출 없음 | - | EfficientNet/ResNet이 보완 정보로 작동하는 신호 확인 |
| EXP-25 | Ridge alpha 미세 조정 | EXP-20 feature 유지, alpha 40~300 탐색 | - | - | - | RidgeClassifier | 0.98755 | 제출 없음 | - | alpha 변경에도 test 예측은 EXP-20과 동일 |
| EXP-26 | Ridge without DenseNet | EXP-20에서 DenseNet 제거, ResNet/EfficientNet/ConvNeXt 계열만 사용 | - | - | - | RidgeClassifier alpha=30 | 0.98563 | 0.9519230769 | submissions/exp26_ridge_without_densenet_alpha30_submit.csv | 현재 최고 점수 갱신 |
| EXP-27 | DenseNet 영향 제한 분석 | DenseNet prob/rank/half feature를 분리 비교 | - | - | - | 분석 | 0.98755 | 제출 없음 | - | DenseNet prob가 들어가면 EXP-20, rank만 쓰면 EXP-26과 동일 |
| EXP-29 | LIME XAI 분석 | EXP-12 ConvNeXt disagreement 샘플 LIME 시각화 | - | - | - | 분석 | - | 제출 없음 | - | Grad-CAM 보조 해석용 LIME 이미지 생성 |
| EXP-30 | Grad-CAM + LIME 비교 리포트 | XAI 결과, 모델별 확률, stacking label을 샘플별 패널로 정리 | - | - | - | 분석 | - | 제출 없음 | - | 과제/보고서용 XAI 비교 패널 생성 |
| EXP-31 | 다중 모델 Grad-CAM 비교 | ResNet/DenseNet/EfficientNet/ConvNeXt Grad-CAM을 disagreement 샘플에서 비교 | - | - | - | 분석 | - | 제출 없음 | - | test_0244 등 모델별 반응 영역 비교 |
| EXP-32 | Validation FP/FN XAI 후보 분석 | EXP-20/EXP-26 validation 오분류 샘플과 모델별 확률 비교 | - | - | - | 분석 | - | 제출 없음 | - | 정답이 있는 validation 기반 XAI 후보 정리 |
| EXP-33 | Validation 판단 변경 샘플 XAI | EXP-20/EXP-26 판단이 갈린 validation 샘플 Grad-CAM/LIME 비교 | - | - | - | 분석 | - | 제출 없음 | - | ConvNeXt 계열 FN 증가 가능성 확인 |
| EXP-34 | EXP-26 Ridge Threshold Search | EXP-26 decision score 기준선 -0.20~0.20 탐색 | - | - | - | threshold | 0.98946 | 0.9503205128 | submissions/exp34_ridge_threshold_m0p02_submit.csv | 안전 후보 1개 변경 제출도 EXP-26보다 하락 |
| EXP-35 | ConvNeXt 영향 제한 | EXP-26 기반 ConvNeXt prob/rank 제거 및 모델 제거 후보 비교 | - | - | - | RidgeClassifier | 0.99042 | 제출 없음 | - | validation 개선 후보는 test_0244를 다시 1로 바꿔 제출 보류 |
| EXP-36 | test_0244 고정 stacking 탐색 | test_0244=0 유지, EXP-26 대비 low-change Ridge 후보 탐색 | - | - | - | RidgeClassifier | 0.98946 | 0.9503205128 | submissions/exp36_fixed0244_core_prob_only_alpha100_submit.csv | test_0025만 0->1 변경했지만 EXP-26보다 하락 |
| EXP-37 | TTA probability 생성 | ResNet/EfficientNet/ConvNeXt 계열 original+hflip TTA 확률 저장 | - | - | - | 분석 | - | 제출 없음 | - | ConvNeXt12만 validation 개선 |
| EXP-38 | TTA 기반 Ridge stacking | TTA probability를 사용한 Ridge stacking 후보 생성 | - | - | - | RidgeClassifier | 0.98851 | 0.9487179487 | submissions/exp38_tta_all_prob_only_alpha50_submit.csv | test_0244는 유지했지만 test_0025 변경 영향으로 하락 |
| EXP-39 | ConvNeXt-Tiny seed123 추가 학습 | EXP-12와 동일 구조, split seed 42 유지, 학습 seed만 123으로 변경 | 6 | AdamW | 5e-5 | Weighted CrossEntropyLoss | 0.97893 | 미제출 | submissions/exp39_convnext_tiny_seed123_submit.csv | 단독 제출 비추천, stacking diversity 후보 |
| EXP-40 | Swin-Tiny 추가 실험 | ImageNet pretrained Swin-Tiny head-only probe 및 stacking 추가 확인 | 4 | AdamW | 5e-5 | Weighted CrossEntropyLoss | 0.89943 | 미제출 | submissions/exp40_swin_tiny_submit.csv | 단독 성능 낮고 stacking 후보도 test_0244를 1로 변경 |
| EXP-41 | 5-fold OOF 설계 | StratifiedKFold 5분할 생성, OOF 저장 포맷과 meta-model 입력 구조 정의 | - | - | - | 설계 | - | 제출 없음 | - | OOF Ridge/Logistic/TabM 기반 마련 |
| EXP-42 | EfficientNetV2-S 단일 split | ImageNet pretrained EfficientNetV2-S, features[7]+classifier probe | 6 | AdamW | 1e-5 / 5e-5 | Weighted CrossEntropyLoss | 0.87261 | 미제출 | submissions/exp42_efficientnet_v2_s_submit.csv | FN이 크게 증가해 단독/stacking 후보 비추천 |
| EXP-43 | RegNetY-3.2GF 단일 split | ImageNet pretrained RegNetY-3.2GF, trunk_output.block4+fc 학습 | 6 | AdamW | 1e-5 / 5e-5 | Weighted CrossEntropyLoss | 0.95211 | 미제출 | submissions/exp43_regnet_y_3_2gf_submit.csv | 단독 제출은 약하지만 민감 샘플 3개 유지, stacking 후보 |
| EXP-44 | RegNetY 포함 Ridge stacking | EXP-26 core에서 ResNet18 제거, EfficientNet/ConvNeXt/RegNetY prob+rank 사용 | - | - | - | RidgeClassifier alpha=150 | 0.98659 | 0.9503205128 | submissions/exp44_regnet_drop_resnet18_add_reg_alpha150_submit.csv | 민감 샘플 3개 유지했지만 EXP-26보다 하락 |
| EXP-45 | ConvNeXt OOF fold0 smoke test | 5-fold OOF pipeline 검증, fold0만 학습 및 OOF/test probability 저장 | 6 | AdamW | 5e-5 | Weighted CrossEntropyLoss | 0.98467 | 제출 없음 | oof/submissions/exp45_convnext_tiny_fold0_submit.csv | OOF 파일 포맷 검증 완료, fold당 약 7.2분 |
| EXP-46 | ConvNeXt 5-fold OOF 완성 | fold1~4 학습 후 fold0 포함 OOF/test mean probability 병합 | 6 x 5 folds | AdamW | 5e-5 | Weighted CrossEntropyLoss | 0.97795 OOF | 미제출 | oof/submissions/exp46_convnext_tiny_mean_submit.csv | OOF 5216 rows, test mean 624 rows 생성 |
| EXP-47 | OOF Ridge baseline | ConvNeXt OOF prob 기반 RidgeClassifier alpha/feature 탐색 | - | - | - | RidgeClassifier | 0.98025 OOF | 미제출 | oof/submissions/exp47_prob1_only_alpha1000_submit.csv | OOF score는 개선됐지만 민감 샘플 0244/0025를 1로 변경 |
| EXP-48 | ResNet18 5-fold OOF | pretrained ResNet18 layer4+fc를 fold별 학습, OOF/test mean 저장 | 6 x 5 folds | AdamW | 1e-4 | Weighted CrossEntropyLoss | 0.98524 OOF | 미제출 | oof/submissions/exp48_resnet18_mean_submit.csv | OOF는 강하지만 민감 샘플 0244/0025/0539를 모두 1로 예측해 단독 제출 보류 |
| EXP-49 | ConvNeXt+ResNet OOF stacking | EXP-46/48 OOF feature로 Ridge/Logistic meta-CV 비교 | - | - | - | Ridge/Logistic | 0.98735 meta-CV | 미제출 | oof/submissions/exp49_ridge_prob_only_1_submit.csv | meta-CV는 높지만 상위 후보가 test_0539를 1로 유지해 제출 보류 |
| EXP-50 | EfficientNet-B0 5-fold OOF | features[5:]+classifier fine-tuning으로 B0 OOF/test mean 저장 | 6 x 5 folds | AdamW | 5e-5 / 1e-4 | Weighted CrossEntropyLoss | 0.98505 OOF | 미제출 | oof/submissions/exp50_efficientnet_b0_mean_submit.csv | 민감 샘플 0244/0025/0539를 모두 0으로 유지, stacking 보완축 |
| EXP-51 | ConvNeXt+ResNet+EfficientNet OOF stacking | 3-model OOF 기반 Ridge/Logistic meta-CV, subset 포함 비교 | - | - | - | Ridge/Logistic | 0.98677 meta-CV | 0.9551282051 | oof/submissions/exp51_logistic_cnx_eff_prob_rank_0p3_submit.csv | 보수 후보가 EXP-26 대비 +2샘플 개선, 공격 후보는 EXP-26과 동점 |
| EXP-52 | EXP-51 low-change search | ConvNeXt+EfficientNet OOF 후보 주변 threshold/regularization 미세 탐색 | - | - | - | Ridge/Logistic | 0.98658 meta-CV | 0.9567307692 | oof/submissions/exp52_logistic_cnx_eff_prob_only_p0p25_t0p0_submit.csv | EXP-51 대비 test_0490만 1->0 변경해 +1샘플 개선 |
| EXP-53 | EXP-52 constraint low-change search | EXP-52 winner 기준 보호 샘플 5개 유지, 변경 1~3개 후보 탐색 | - | - | - | Ridge/Logistic | 0.98696 meta-CV | 제출 없음 | - | 보호 규칙을 만족하는 비중복 후보 0개 |
| EXP-54 | ConvNeXt seed123 5-fold OOF | ConvNeXt-Tiny 동일 설정, seed123으로 5-fold OOF/test mean 생성 | 6 x 5 folds | AdamW | 5e-5 | Weighted CrossEntropyLoss | 0.98102 OOF | 미제출 | oof/submissions/exp54_convnext_tiny_seed123_mean_submit.csv | 단독은 test_0539/test_0490을 1로 예측해 보류, stacking feature로 사용 |
| EXP-55 | ConvNeXt seed OOF stacking | EXP-54 seed123 + EfficientNet/기존 ConvNeXt OOF 기반 Ridge/Logistic 탐색 | - | - | - | Ridge/Logistic | 0.98677 meta-CV | 0.9551282051 | oof/submissions/exp55_ridge_cnx123_eff_prob_rank_p50_tm0p1_submit.csv | test_0222만 1->0 변경했지만 하락 |
| EXP-56 | 보호 샘플 기반 low-change search | EXP-55 결과 반영, test_0222=1까지 보호하고 후보 재탐색 | - | - | - | Ridge/Logistic | 0.98658 meta-CV | 0.9583333333 | oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv | test_0504만 1->0 변경해 +1샘플 개선 |
| EXP-57 | EXP-56 기준 보호 샘플 low-change search | EXP-56 winner 기준 test_0504=0까지 보호하고 후보 재탐색 | - | - | - | Ridge/Logistic | 0.98696 meta-CV | 제출 없음 | - | 보호 샘플 7개를 모두 유지하는 비중복 후보 0개 |
| EXP-58 | Calibrated OOF stacking | Ridge/Logistic/LinearSVC에 sigmoid/isotonic calibration 적용, 보호 샘플 유지 후보 탐색 | - | - | - | Calibrated SVC | 0.98639 meta-CV | 0.9551282051 | oof/submissions/exp58_svc_all_margin_stats_isotonic_p0p2_t0p575_submit.csv | test_0188/test_0568 0->1 변경 후 하락 |
| EXP-65 | ConvNeXt V2 Tiny 5-fold OOF | timm ConvNeXt V2 Tiny, stage3+head fine-tuning, 5-fold OOF 생성 | 6 x 5 folds | AdamW | 5e-5 | Weighted CrossEntropyLoss | 0.97680 OOF | 미제출 | oof/submissions/exp65_convnextv2_tiny_stage3_mean_submit.csv | 단독 mean은 보호 샘플 다수 1로 변경해 제출 보류, stacking diversity 후보 |
| EXP-66 | ConvNeXt V2 포함 OOF stacking | EXP-65 ConvNeXt V2 OOF를 기존 ConvNeXt/EfficientNet OOF feature에 제한적으로 추가 | - | - | - | RidgeClassifier alpha=10 | 0.98696 meta-CV | 0.9567307692 | oof/submissions/exp66_ridge_all_prob_stats_with_cnxv2_p10_t0p15_submit.csv | test_0141만 1->0 변경했지만 EXP-56보다 하락 |
| EXP-67 | RegNetY-3.2GF 5-fold OOF | torchvision RegNetY-3.2GF block4+fc fine-tuning, 5-fold OOF 생성 | 6 x 5 folds | AdamW | 1e-5 / 5e-5 | Weighted CrossEntropyLoss | 0.96166 OOF | 미제출 | oof/submissions/exp67_regnet_y_3_2gf_mean_submit.csv | FN 164개, EXP-56 대비 69개 변경으로 단독 제출 제외 |
| EXP-68 | RegNetY 포함 OOF stacking | EXP-67 RegNetY OOF를 기존 ConvNeXt/EfficientNet OOF feature에 추가 | - | - | - | Ridge/Logistic | 0.98735 meta-CV | 제출 없음 | - | 보호 샘플 8개 유지 후보 0개 |
| EXP-69 | DINOv2 feature extractor + Ridge/Logistic | timm `vit_small_patch14_dinov2` frozen feature 384-dim 추출 후 Ridge/Logistic meta-model 탐색 | - | - | - | Ridge/Logistic | 0.97373 meta-CV | 제출 없음 | - | ViT/foundation feature family 확보 실험. 보호 샘플 유지 후보 0개 |
| EXP-70 | EfficientNetV2-S 재설계 5-fold OOF | `features[6:] + classifier`를 학습해 EXP-42보다 넓게 fine-tuning | 6 x 5 folds | AdamW | 2e-5 / 8e-5 | Weighted CrossEntropyLoss | 0.97642 OOF | 미제출 | oof/submissions/exp70_efficientnet_v2_s_wide_mean_submit.csv | EXP-42 대비 회복. 단독 mean은 보호 샘플 다수를 깨서 제출 제외 |
| EXP-71 | EfficientNetV2-S 포함 OOF stacking | EXP-70 EfficientNetV2-S OOF를 기존 ConvNeXt/EfficientNet-B0 OOF feature에 추가 | - | - | - | Ridge/Logistic | 0.98888 meta-CV | 제출 없음 | - | 보호 샘플 유지 비중복 후보 0개. 안전 후보는 EXP-56과 동일 |
| EXP-72 | ConvNeXt-Tiny seed777 5-fold OOF | EXP-46과 동일 구조, 학습 seed만 777로 변경해 ConvNeXt seed diversity 확보 | 6 x 5 folds | AdamW | 5e-5 | Weighted CrossEntropyLoss | 0.98006 OOF | 미제출 | oof/submissions/exp72_convnext_tiny_seed777_mean_submit.csv | OOF는 강하지만 단독 mean은 보호 샘플 다수를 깨서 제출 제외 |
| EXP-73 | ConvNeXt seed777 포함 OOF stacking | EXP-72 seed777 OOF를 ConvNeXt seed123/EfficientNet-B0 OOF pool에 추가 | - | - | - | Ridge/Logistic | 0.98754 meta-CV | 제출 없음 | - | 보호 샘플 유지 비중복 후보 0개. 안전 후보는 EXP-56과 동일 |
| EXP-77 | ConvNeXt-Tiny Lion fold0 smoke | EXP-45와 같은 fold0 조건에서 optimizer만 Lion으로 변경 | 6 fold0 | Lion | 2e-5 | Weighted CrossEntropyLoss | 0.98563 fold0 | 미제출 | oof/submissions/exp77_convnext_tiny_lion_fold0_submit.csv | AdamW fold0보다 소폭 개선. 5-fold 확장 후보 |
| EXP-78 | ConvNeXt-Tiny Lion 5-fold OOF | EXP-77을 5-fold로 확장, optimizer만 Lion 유지 | 6 x 5 folds | Lion | 2e-5 | Weighted CrossEntropyLoss | 0.98140 OOF | 미제출 | oof/submissions/exp78_convnext_tiny_lion_mean_submit.csv | 단독 mean은 보호 샘플 다수를 1로 바꿔 제출 제외. stacking 보조 feature 후보 |
| EXP-79 | Lion OOF 포함 stacking | EXP-78 Lion OOF를 기존 ConvNeXt/EfficientNet OOF pool에 추가해 Ridge/Logistic 탐색 | - | - | - | Ridge/Logistic | 0.98811 meta-CV | 0.9567307692 | oof/submissions/exp79_ridge_cnx_lion_eff_prob_only_alpha500_t0p2_submit.csv | test_0049만 0->1 변경했지만 EXP-56보다 하락 |

## EXP-00. 원본 Baseline

### 설정

- 모델: SimpleCNN
- 입력 크기: 128 x 128
- 입력 채널: 1채널 grayscale
- augmentation: 없음
- validation split: 없음
- loss: `CrossEntropyLoss`
- optimizer: `Adam(lr=1e-3)`
- epoch: 2

### 한계

- validation set이 없어 학습 성능과 일반화 성능을 구분할 수 없다.
- 클래스 불균형이 있지만 loss에 반영되지 않았다.
- accuracy 외 precision, recall, confusion matrix를 확인하지 않았다.
- 제출 결과가 낮을 경우 원인을 분석하기 어렵다.

### 기록

- Leaderboard score: 미제출
- 원본 baseline은 리더보드에 제출하지 않았다.
- 이후 비교 기준으로 구조와 한계만 기록한다.

## EXP-01. 개선 SimpleCNN

### 변경한 하이퍼파라미터 및 구조

| 항목 | 기존 | 변경 |
|---|---|---|
| 입력 크기 | 128 x 128 | 224 x 224 |
| validation split | 없음 | train 80%, valid 20% |
| split 방식 | 없음 | stratified split |
| augmentation | 없음 | RandomHorizontalFlip, RandomRotation(7도) |
| normalization | ToTensor만 사용 | Normalize(mean=0.5, std=0.5) |
| CNN 구조 | Conv 2개 | Conv 3개 |
| BatchNorm | 없음 | Conv 뒤 BatchNorm 추가 |
| Dropout | 없음 | FC 앞 Dropout(0.3) |
| loss | CrossEntropyLoss | class weight 적용 CrossEntropyLoss |
| optimizer | Adam | AdamW |
| weight decay | 없음 | 1e-4 |
| scheduler | 없음 | ReduceLROnPlateau |
| epoch | 2 | 8 |

### Class Weight

클래스 분포가 label 1에 치우쳐 있어 class weight를 적용했다.

```text
class weights: [1.944082, 0.6731204]
```

label 0의 샘플 수가 더 적으므로 label 0에 더 큰 weight가 적용된다. 이는 모델이 다수 클래스인 label 1에만 치우쳐 예측하는 문제를 줄이기 위한 설정이다.

### Epoch별 Validation 결과

| Epoch | Train Loss | Valid Loss | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.3366 | 0.3131 | 0.9502 | 0.9665 | 0.9665 | 0.9665 |
| 2 | 0.3784 | 0.2290 | 0.9626 | 0.9706 | 0.9794 | 0.9750 |
| 3 | 0.2956 | 0.3568 | 0.9521 | 0.9481 | 0.9897 | 0.9685 |
| 4 | 0.2918 | 0.1928 | 0.9665 | 0.9720 | 0.9832 | 0.9776 |
| 5 | 0.3583 | 0.2091 | 0.9377 | 0.9944 | 0.9214 | 0.9565 |
| 6 | 0.2822 | 0.1778 | 0.9732 | 0.9820 | 0.9820 | 0.9820 |
| 7 | 0.2892 | 0.2383 | 0.9205 | 0.9957 | 0.8969 | 0.9437 |
| 8 | 0.1821 | 0.1256 | 0.9607 | 0.9933 | 0.9536 | 0.9730 |

### Best Model

- 기준: validation accuracy
- Best epoch: 6
- Best validation accuracy: 0.97318
- Precision: 0.9820
- Recall: 0.9820
- F1-score: 0.9820

### Confusion Matrix

```text
[[254  14]
 [ 14 762]]
```

해석:

- 실제 label 0을 label 0으로 맞춘 경우: 254
- 실제 label 0을 label 1로 예측한 경우: 14
- 실제 label 1을 label 0으로 예측한 경우: 14
- 실제 label 1을 label 1로 맞춘 경우: 762

FP와 FN이 각각 14개로 동일하게 나타났다. 특정 클래스에만 오류가 집중되지는 않았지만, 의료 이미지 분류에서는 실제 폐렴을 정상으로 예측하는 FN이 더 위험할 수 있으므로 recall을 함께 확인해야 한다.

### 제출 파일 분포

```text
submissions/baseline_submit.csv
label 0: 297
label 1: 327
```

제출 파일의 예측 분포는 한쪽 클래스로 심하게 몰리지는 않았다. 다만 validation 성능과 리더보드 점수 사이에 차이가 클 수 있으므로, 제출 파일 갱신 여부와 test 데이터 일반화 성능을 함께 확인해야 한다.

### 리더보드 결과

- 제출 파일: `submissions/baseline_submit.csv`
- Leaderboard score: 0.81891
- 순위: 34위 확인
- 제출 횟수: 1회

Validation accuracy는 0.97318이었지만 리더보드 score는 0.81891로 낮게 나타났다. 이는 단일 validation split이 test 데이터 분포를 충분히 대표하지 못했거나, 모델이 validation set에 과적합되었을 가능성을 시사한다. 따라서 다음 실험에서는 pretrained ResNet18과 threshold 조정을 통해 일반화 성능을 추가로 확인한다.

### 과제용 해석

개선 SimpleCNN은 기존 baseline에 비해 validation split, class weight, augmentation, BatchNorm, Dropout을 추가하였다. Validation 기준 최고 성능은 epoch 6에서 나타났으며 accuracy 0.9732, precision 0.9820, recall 0.9820, F1-score 0.9820을 기록하였다. 그러나 리더보드 score는 0.81891로 validation 결과보다 낮게 나타났다. 따라서 단일 validation split 결과만으로 실제 test 성능을 단정하기 어렵고, 추가 모델과 threshold 조정을 통해 일반화 성능을 확인할 필요가 있다.

## EXP-02. ResNet18 Fine-tuning

### 목적

SimpleCNN은 구조가 단순하여 이미지의 복잡한 특징을 충분히 학습하지 못할 수 있다. ResNet18은 사전학습된 convolution feature를 활용할 수 있으므로, X-ray 이미지 분류에서도 더 안정적인 feature extraction을 기대할 수 있다.

### 설정

- 모델: pretrained ResNet18
- 입력 채널: grayscale 이미지를 3채널로 변환
- 입력 크기: 224 x 224
- normalization: ImageNet mean/std 사용
- fine-tuning 범위:
  - backbone 전체는 우선 freeze
  - `layer4`와 `fc`만 학습
- optimizer: `AdamW`
- learning rate: 1e-4
- weight decay: 1e-4
- loss: class weight 적용 `CrossEntropyLoss`
- epoch: 6
- 제출 파일: `submissions/resnet18_submit.csv`

### 학습 파라미터 수

```text
trainable params: 8,394,754
total params: 11,177,538
```

### 기대 효과

- pretrained feature를 사용해 SimpleCNN보다 일반화 성능이 좋아질 가능성이 있다.
- 전체 layer를 처음부터 학습하지 않고 `layer4 + fc`만 학습하므로 overfitting 위험을 줄일 수 있다.
- 만약 성능이 부족하면 다음 단계에서 `layer3 + layer4 + fc` 또는 full fine-tuning으로 확장할 수 있다.

### Epoch별 Validation 결과

| Epoch | Train Loss | Valid Loss | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1372 | 0.0843 | 0.9636 | 0.9920 | 0.9588 | 0.9751 |
| 2 | 0.0805 | 0.0642 | 0.9751 | 0.9960 | 0.9704 | 0.9830 |
| 3 | 0.0476 | 0.0491 | 0.9808 | 0.9974 | 0.9768 | 0.9870 |
| 4 | 0.0377 | 0.0527 | 0.9828 | 0.9948 | 0.9820 | 0.9883 |
| 5 | 0.0365 | 0.0479 | 0.9799 | 0.9974 | 0.9755 | 0.9863 |
| 6 | 0.0260 | 0.0558 | 0.9866 | 0.9910 | 0.9910 | 0.9910 |

### Best Model

- 기준: validation accuracy
- Best epoch: 6
- Best validation accuracy: 0.98659
- Precision: 0.9910
- Recall: 0.9910
- F1-score: 0.9910

### Confusion Matrix

```text
[[261   7]
 [  7 769]]
```

개선 SimpleCNN의 confusion matrix는 `[[254, 14], [14, 762]]`였고, ResNet18에서는 FP와 FN이 각각 7개로 줄었다. Validation 기준으로는 SimpleCNN보다 오분류 수가 감소했고, precision과 recall도 균형 있게 개선되었다.

### 제출 파일 분포

```text
submissions/resnet18_submit.csv
label 0: 219
label 1: 405
```

SimpleCNN 제출 파일은 label 0이 297개, label 1이 327개였고, ResNet18 제출 파일은 label 1 예측 비율이 더 높아졌다. 실제 test 정답 분포는 알 수 없으므로, 리더보드 점수 확인 후 이 분포가 유리했는지 판단해야 한다.

### 리더보드 결과

- 제출 파일: `submissions/resnet18_submit.csv`
- Leaderboard score: 0.9278846154
- 순위: 26위 확인
- 제출 횟수: 2회

개선 SimpleCNN의 leaderboard score 0.81891보다 0.10897 상승했다. Validation accuracy도 0.97318에서 0.98659로 증가했고, leaderboard score도 함께 상승했으므로 ResNet18 fine-tuning이 실제 test 성능 개선에도 기여한 것으로 볼 수 있다. 다만 validation과 leaderboard 사이에는 여전히 차이가 있으므로, 단일 validation split 결과만으로 최종 성능을 단정하기보다는 추가 실험으로 안정성을 확인할 필요가 있다.

### 과제용 해석

ResNet18 fine-tuning 실험에서는 ImageNet 사전학습 가중치를 사용하고, `layer4`와 `fc`만 학습하였다. Validation 기준 최고 성능은 epoch 6에서 나타났으며 accuracy 0.9866, precision 0.9910, recall 0.9910, F1-score 0.9910을 기록하였다. Confusion matrix에서는 FP와 FN이 각각 7개로 나타나 개선 SimpleCNN보다 오류 수가 감소하였다. Leaderboard score도 0.81891에서 0.9278846154로 상승하여, pretrained ResNet18 적용이 test 성능 개선에도 효과가 있었음을 확인했다.

### 주의점

Validation 성능이 높아도 리더보드 점수가 반드시 높아지는 것은 아니다. 특히 validation split이 한 번만 수행되었기 때문에 데이터 분할에 따른 성능 변동 가능성이 있다.

## EXP-03. ResNet18 Epoch 10 확인

### 목적

EXP-02의 epoch 6은 early stopping으로 결정된 값이 아니라 사전에 정한 고정 epoch 수였다. 추가 학습이 성능을 더 개선하는지 확인하기 위해 같은 설정에서 epoch만 10으로 늘려 다시 학습했다.

### 설정

- 모델: pretrained ResNet18
- fine-tuning 범위: `layer4 + fc`
- optimizer: `AdamW`
- learning rate: 1e-4
- weight decay: 1e-4
- loss: class weight 적용 `CrossEntropyLoss`
- epoch: 10
- 제출 파일: `submissions/resnet18_epoch10_submit.csv`

### Epoch별 Validation 결과

| Epoch | Train Loss | Valid Loss | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1398 | 0.0778 | 0.9713 | 0.9934 | 0.9678 | 0.9804 |
| 2 | 0.0642 | 0.0767 | 0.9713 | 0.9882 | 0.9729 | 0.9805 |
| 3 | 0.0601 | 0.0850 | 0.9617 | 0.9960 | 0.9523 | 0.9736 |
| 4 | 0.0411 | 0.0679 | 0.9818 | 0.9871 | 0.9884 | 0.9878 |
| 5 | 0.0343 | 0.0796 | 0.9818 | 0.9822 | 0.9936 | 0.9878 |
| 6 | 0.0304 | 0.0647 | 0.9741 | 0.9987 | 0.9665 | 0.9823 |
| 7 | 0.0371 | 0.0630 | 0.9732 | 0.9987 | 0.9652 | 0.9817 |
| 8 | 0.0211 | 0.0548 | 0.9847 | 0.9910 | 0.9884 | 0.9897 |
| 9 | 0.0106 | 0.0609 | 0.9866 | 0.9897 | 0.9923 | 0.9910 |
| 10 | 0.0123 | 0.0483 | 0.9866 | 0.9935 | 0.9884 | 0.9910 |

### Best Model

- 기준: validation accuracy
- Best validation accuracy: 0.98659
- 최종 선택된 best state 기준:
  - Precision: 0.9897
  - Recall: 0.9923
  - F1-score: 0.9910

### Confusion Matrix

```text
[[260   8]
 [  6 770]]
```

EXP-02의 confusion matrix는 `[[261, 7], [7, 769]]`였다. EXP-03에서는 FN이 7개에서 6개로 1개 줄었지만, FP는 7개에서 8개로 1개 증가했다. Accuracy는 동일하므로, 의료 이미지 관점에서 recall을 조금 더 우선한다면 EXP-03도 제출 후보가 될 수 있다.

### 제출 파일 분포

```text
submissions/resnet18_epoch10_submit.csv
label 0: 166
label 1: 458
```

EXP-02의 제출 분포는 label 0이 219개, label 1이 405개였다. EXP-03은 label 1 예측이 458개로 더 많아졌으므로 폐렴으로 예측하는 경향이 강해졌다. 이 변화는 recall을 높이는 데 유리할 수 있지만, FP 증가로 leaderboard accuracy가 낮아질 가능성도 있다.

### 리더보드 결과

- 제출 파일: `submissions/resnet18_epoch10_submit.csv`
- Leaderboard score: 0.8782051282
- 최고 순위: 26위 유지
- 제출 횟수: 3회

Epoch을 10으로 늘렸지만 leaderboard score는 EXP-02의 0.9278846154보다 낮은 0.8782051282를 기록했다. Validation 기준으로 FN은 1개 감소했지만 test 기준 accuracy 개선으로 이어지지 않았고, 오히려 label 1 예측이 과도하게 많아진 것이 성능 하락에 영향을 주었을 가능성이 있다.

### 과제용 해석

Epoch을 6에서 10으로 늘렸을 때 validation accuracy는 0.98659로 동일하게 유지되었다. Confusion matrix에서는 FN이 7개에서 6개로 줄었지만 FP가 7개에서 8개로 증가해 전체 accuracy 개선은 나타나지 않았다. Leaderboard score는 0.8782051282로 오히려 하락했으므로, epoch 증가만으로는 추가적인 성능 향상이 확인되지 않았다.

### 1. Threshold Search

기본 분류 기준은 `probability >= 0.5`이다. threshold를 낮추면 label 1로 예측하는 기준이 완화되어 recall이 증가하고 FN이 줄어들 수 있다. 반대로 FP가 증가하여 precision이나 accuracy가 낮아질 수 있다.

실험 범위:

```text
threshold: 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70
```

기록할 지표:

- threshold별 accuracy
- threshold별 precision
- threshold별 recall
- threshold별 F1
- threshold별 confusion matrix

### 2. Fine-tuning 범위 확대

ResNet18 `layer4 + fc` 결과가 부족하면 fine-tuning 범위를 넓힌다.

우선순위:

1. `layer4 + fc`
2. `layer3 + layer4 + fc`
3. full fine-tuning

전체 layer를 학습하면 성능이 향상될 수 있지만, 데이터 수가 많지 않기 때문에 overfitting 가능성도 함께 증가한다.

## EXP-04. ResNet18 Layer3+Layer4 Fine-tuning

### 목적

Epoch 증가만으로는 leaderboard 최고 점수가 갱신되지 않았기 때문에, 다음 실험에서는 fine-tuning 범위를 `layer4 + fc`에서 `layer3 + layer4 + fc`로 확장했다. 더 많은 고수준 feature를 X-ray 데이터에 맞게 조정하는 것이 목적이다.

### 설정

- 모델: pretrained ResNet18
- fine-tuning 범위: `layer3 + layer4 + fc`
- optimizer: `AdamW`
- learning rate: 5e-5
- weight decay: 1e-4
- loss: class weight 적용 `CrossEntropyLoss`
- epoch: 6
- 제출 파일: `submissions/resnet18_layer34_submit.csv`

### 학습 파라미터 수

```text
trainable params: 10,494,466
total params: 11,177,538
```

### Epoch별 Validation 결과

| Epoch | Train Loss | Valid Loss | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1480 | 0.0905 | 0.9540 | 0.9959 | 0.9420 | 0.9682 |
| 2 | 0.0633 | 0.0636 | 0.9808 | 0.9909 | 0.9832 | 0.9871 |
| 3 | 0.0534 | 0.0871 | 0.9636 | 0.9973 | 0.9536 | 0.9750 |
| 4 | 0.0390 | 0.0540 | 0.9847 | 0.9910 | 0.9884 | 0.9897 |
| 5 | 0.0267 | 0.0618 | 0.9856 | 0.9897 | 0.9910 | 0.9903 |
| 6 | 0.0301 | 0.0603 | 0.9741 | 0.9973 | 0.9678 | 0.9823 |

### Best Model

- 기준: validation accuracy
- Best validation accuracy: 0.98563
- Precision: 0.9897
- Recall: 0.9910
- F1-score: 0.9903

### Confusion Matrix

```text
[[260   8]
 [  7 769]]
```

EXP-02의 confusion matrix는 `[[261, 7], [7, 769]]`였다. EXP-04는 FN은 동일하지만 FP가 1개 증가하여 validation accuracy가 소폭 낮아졌다. Fine-tuning 범위를 넓혔지만 validation 기준으로는 명확한 개선이 나타나지 않았다.

### 제출 파일 분포

```text
submissions/resnet18_layer34_submit.csv
label 0: 175
label 1: 449
```

EXP-04는 label 1 예측이 449개로 많아졌다. EXP-02의 label 1 예측 수 405개보다 더 공격적으로 폐렴을 예측하는 경향이 있다. 공식 metric이 accuracy이므로 FP 증가 가능성을 고려하면 제출 우선순위는 EXP-02보다 낮다.

### 리더보드 결과

- 제출 파일: `submissions/resnet18_layer34_submit.csv`
- Leaderboard score: 0.8894230769
- 최고 순위: 26위 유지
- 제출 횟수: 4회

Fine-tuning 범위를 넓힌 EXP-04는 leaderboard score가 0.8894230769로 나타났다. EXP-02의 0.9278846154보다 낮으므로, 현재까지는 `layer4 + fc`만 학습한 EXP-02가 가장 좋은 설정이다.

### 과제용 해석

Fine-tuning 범위를 `layer3 + layer4 + fc`로 확장했지만 validation accuracy는 0.98563으로, 기존 `layer4 + fc` 실험의 0.98659보다 소폭 낮았다. Confusion matrix에서도 FN은 동일하고 FP가 1개 증가했으며, leaderboard score도 0.8894230769로 하락했다. 따라서 더 많은 layer를 학습하는 것이 항상 성능 향상으로 이어지지는 않았고, 현재 데이터 규모에서는 `layer4 + fc` 설정이 더 적절한 것으로 판단된다.

## EXP-05. ResNet18 Threshold Search

### 목적

EXP-03과 EXP-04에서 label 1 예측이 많아질수록 leaderboard score가 하락했다. 따라서 같은 ResNet18 `layer4 + fc` 구조에서 예측 확률 threshold를 조정하여 label 1 예측 비율과 성능 변화를 확인했다.

### 설정

- 모델: pretrained ResNet18
- fine-tuning 범위: `layer4 + fc`
- optimizer: `AdamW`
- learning rate: 1e-4
- weight decay: 1e-4
- loss: class weight 적용 `CrossEntropyLoss`
- epoch: 6
- threshold 범위: 0.30 ~ 0.75
- 확률 저장 파일: `resnet18_exp05_probs.npz`
- threshold 결과 파일: `resnet18_exp05_threshold_results.json`

### Epoch별 Validation 결과

| Epoch | Train Loss | Valid Loss | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1398 | 0.0778 | 0.9713 | 0.9934 | 0.9678 | 0.9804 |
| 2 | 0.0642 | 0.0767 | 0.9713 | 0.9882 | 0.9729 | 0.9805 |
| 3 | 0.0601 | 0.0850 | 0.9617 | 0.9960 | 0.9523 | 0.9736 |
| 4 | 0.0411 | 0.0679 | 0.9818 | 0.9871 | 0.9884 | 0.9878 |
| 5 | 0.0343 | 0.0796 | 0.9818 | 0.9822 | 0.9936 | 0.9878 |
| 6 | 0.0304 | 0.0647 | 0.9741 | 0.9987 | 0.9665 | 0.9823 |

### Threshold별 Validation 결과

| Threshold | Accuracy | Precision | Recall | F1 | Valid Pred 0 | Valid Pred 1 | Confusion Matrix |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.30 | 0.9808 | 0.9821 | 0.9923 | 0.9872 | 260 | 784 | `[[254, 14], [6, 770]]` |
| 0.35 | 0.9808 | 0.9834 | 0.9910 | 0.9872 | 262 | 782 | `[[255, 13], [7, 769]]` |
| 0.40 | 0.9828 | 0.9859 | 0.9910 | 0.9884 | 264 | 780 | `[[257, 11], [7, 769]]` |
| 0.45 | 0.9818 | 0.9859 | 0.9897 | 0.9878 | 265 | 779 | `[[257, 11], [8, 768]]` |
| 0.50 | 0.9818 | 0.9871 | 0.9884 | 0.9878 | 267 | 777 | `[[258, 10], [9, 767]]` |
| 0.55 | 0.9808 | 0.9884 | 0.9858 | 0.9871 | 270 | 774 | `[[259, 9], [11, 765]]` |
| 0.60 | 0.9799 | 0.9896 | 0.9832 | 0.9864 | 273 | 771 | `[[260, 8], [13, 763]]` |
| 0.65 | 0.9780 | 0.9896 | 0.9807 | 0.9851 | 275 | 769 | `[[260, 8], [15, 761]]` |
| 0.70 | 0.9751 | 0.9896 | 0.9768 | 0.9831 | 278 | 766 | `[[260, 8], [18, 758]]` |
| 0.75 | 0.9761 | 0.9908 | 0.9768 | 0.9838 | 279 | 765 | `[[261, 7], [18, 758]]` |

### 생성된 제출 파일

| 파일명 | Threshold | Test Pred 0 | Test Pred 1 | 제출 우선순위 |
|---|---:|---:|---:|---|
| `submissions/resnet18_threshold_0p4_submit.csv` | 0.40 | 141 | 483 | 낮음 |
| `submissions/resnet18_threshold_0p5_submit.csv` | 0.50 | 148 | 476 | 낮음 |
| `submissions/resnet18_threshold_0p55_submit.csv` | 0.55 | 155 | 469 | 낮음 |
| `submissions/resnet18_threshold_0p6_submit.csv` | 0.60 | 161 | 463 | 보류 |
| `submissions/resnet18_threshold_0p65_submit.csv` | 0.65 | 169 | 455 | 후보 |
| `submissions/resnet18_threshold_0p7_submit.csv` | 0.70 | 176 | 448 | 후보 |

### 리더보드 결과

| 제출 파일 | Threshold | Leaderboard Score | 결과 |
|---|---:|---:|---|
| `submissions/resnet18_threshold_0p7_submit.csv` | 0.70 | 0.891025641 | 최고 점수 갱신 실패 |
| `submissions/resnet18_threshold_0p65_submit.csv` | 0.65 | 0.8798076923 | 최고 점수 갱신 실패 |

Threshold 조정 제출은 모두 EXP-02의 0.9278846154보다 낮았다. 특히 이번 threshold search run은 재학습된 모델 자체의 validation accuracy가 EXP-02보다 낮았고, test 예측도 label 1에 많이 치우쳤다. 따라서 현재 기준으로는 threshold 조정보다 EXP-02의 기본 argmax 제출이 더 적절하다.

### 해석

Validation 기준 best threshold는 0.40이었지만, test 예측 분포가 label 1에 매우 치우쳤다. 실제로 label 1 예측을 줄인 0.65와 0.70 threshold도 leaderboard score가 각각 0.8798076923, 0.891025641에 그쳐 최고 점수를 갱신하지 못했다. 따라서 이번 실험에서는 threshold 조정이 성능 개선으로 이어지지 않았다.

### 과제용 해석

Threshold를 조정한 결과 validation set에서는 threshold 0.40에서 가장 높은 accuracy 0.9828을 기록하였다. 그러나 실제 제출에서는 threshold 0.70이 0.891025641, threshold 0.65가 0.8798076923으로 나타나 기존 최고 점수 0.9278846154보다 낮았다. 따라서 validation threshold 최적화가 test 성능 향상으로 바로 이어지지는 않았으며, 현재 설정에서는 기본 argmax를 사용한 EXP-02가 가장 안정적인 결과를 보였다.

## EXP-06. EfficientNet-B0 Fine-tuning

### 목적

ResNet18 계열의 epoch 증가, fine-tuning 범위 확대, threshold 조정이 최고 점수를 갱신하지 못했다. 다음 단계에서는 다른 pretrained CNN 계열인 EfficientNet-B0를 학습하여 모델 다양성을 확보하고, 이후 soft voting 또는 stacking ensemble에 사용할 probability를 저장한다.

### 설정

- 모델: pretrained EfficientNet-B0
- fine-tuning 범위: `features[-2:] + classifier`
- 입력 채널: grayscale 이미지를 3채널로 변환
- 입력 크기: 224 x 224
- optimizer: `AdamW`
- learning rate: 1e-4
- weight decay: 1e-4
- loss: class weight 적용 `CrossEntropyLoss`
- epoch: 6
- threshold: 0.5

### 저장 파일

```text
models/exp06_efficientnet_b0_best.pt
preds/exp06_efficientnet_b0_valid_probs.csv
preds/exp06_efficientnet_b0_test_probs.csv
preds/exp06_efficientnet_b0_result.json
submissions/exp06_efficientnet_b0_submit.csv
submissions/efficientnet_b0_submit.csv
```

### 학습 파라미터 수

```text
trainable params: 1,131,954
total params: 4,010,110
```

### Epoch별 Validation 결과

| Epoch | Train Loss | Valid Loss | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2828 | 0.1611 | 0.9291 | 0.9861 | 0.9175 | 0.9506 |
| 2 | 0.1501 | 0.1175 | 0.9511 | 0.9892 | 0.9446 | 0.9664 |
| 3 | 0.1270 | 0.1017 | 0.9569 | 0.9919 | 0.9497 | 0.9704 |
| 4 | 0.1051 | 0.0903 | 0.9703 | 0.9882 | 0.9716 | 0.9799 |
| 5 | 0.0887 | 0.0837 | 0.9751 | 0.9908 | 0.9755 | 0.9831 |
| 6 | 0.0844 | 0.0847 | 0.9761 | 0.9883 | 0.9794 | 0.9838 |

### Best Model

- 기준: validation accuracy
- Best epoch: 6
- Validation accuracy: 0.97605
- Precision: 0.9883
- Recall: 0.9794
- F1-score: 0.9838

### Confusion Matrix

```text
[[259   9]
 [ 16 760]]
```

EfficientNet-B0는 ResNet18 EXP-02보다 validation accuracy가 낮고 FN이 더 많았다. 단독 제출 성능은 ResNet18보다 낮을 가능성이 있지만, 모델 구조가 다르기 때문에 앙상블에서 보완적인 확률 정보로 활용할 수 있다.

### 제출 파일 분포

```text
submissions/efficientnet_b0_submit.csv
label 0: 263
label 1: 361
```

EXP-02의 ResNet18 제출 분포는 label 0이 219개, label 1이 405개였다. EfficientNet-B0는 label 1 예측이 상대적으로 적어, ResNet18과 예측 경향이 다르다. 이 차이는 soft voting이나 stacking ensemble에서 유용할 수 있다.

### 리더보드 결과

- 제출 파일: `submissions/efficientnet_b0_submit.csv`
- Leaderboard score: 0.8894230769
- 제출 횟수: 7회

EfficientNet-B0 단독 제출은 EXP-02 ResNet18 최고 점수 0.9278846154보다 낮았다. 단독 모델로는 최고 성능 후보가 아니지만, ResNet18과 test 예측 분포가 다르기 때문에 앙상블용 probability로 활용할 수 있다.

### 과제용 해석

EfficientNet-B0 실험에서는 validation accuracy 0.9761을 기록하여 ResNet18의 0.9866보다 낮은 성능을 보였다. Leaderboard score도 0.8894230769로 ResNet18 최고 점수보다 낮았으므로 단독 모델로는 최고 성능 후보가 아니었다. 그러나 test 예측 분포가 ResNet18과 다르게 나타났고, probability 파일을 저장했기 때문에 이후 앙상블에서 서로 다른 모델의 판단 경향을 결합하는 데 활용할 수 있다.

## EXP-07. ResNet18 재현 및 Probability 저장

### 목적

EXP-02의 ResNet18 `layer4 + fc` 모델이 현재 최고 leaderboard score를 기록했지만, 당시에는 앙상블에 사용할 validation/test probability를 저장하지 않았다. 따라서 같은 학습 흐름을 재현하여 모델 가중치와 probability 파일을 저장했다.

### 설정

- 모델: pretrained ResNet18
- fine-tuning 범위: `layer4 + fc`
- optimizer: `AdamW`
- learning rate: 1e-4
- weight decay: 1e-4
- loss: class weight 적용 `CrossEntropyLoss`
- epoch: 6
- threshold: 0.5

### 저장 파일

```text
models/exp07_resnet18_layer4_fc_best.pt
preds/exp07_resnet18_valid_probs.csv
preds/exp07_resnet18_test_probs.csv
preds/exp07_resnet18_result.json
submissions/exp07_resnet18_submit.csv
```

### Epoch별 Validation 결과

| Epoch | Train Loss | Valid Loss | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1372 | 0.0843 | 0.9636 | 0.9920 | 0.9588 | 0.9751 |
| 2 | 0.0805 | 0.0642 | 0.9751 | 0.9960 | 0.9704 | 0.9830 |
| 3 | 0.0476 | 0.0491 | 0.9808 | 0.9974 | 0.9768 | 0.9870 |
| 4 | 0.0377 | 0.0527 | 0.9828 | 0.9948 | 0.9820 | 0.9883 |
| 5 | 0.0365 | 0.0479 | 0.9799 | 0.9974 | 0.9755 | 0.9863 |
| 6 | 0.0260 | 0.0558 | 0.9866 | 0.9910 | 0.9910 | 0.9910 |

### Best Model

- Best epoch: 6
- Validation accuracy: 0.98659
- Precision: 0.9910
- Recall: 0.9910
- F1-score: 0.9910

### Confusion Matrix

```text
[[261   7]
 [  7 769]]
```

EXP-07은 EXP-02와 동일한 validation 성능과 test 예측 분포를 재현했다. 따라서 이후 soft voting ensemble에서 기준 ResNet18 모델로 사용한다.

### 제출 파일 분포

```text
submissions/exp07_resnet18_submit.csv
label 0: 219
label 1: 405
```

### 과제용 해석

EXP-07은 새로운 성능 개선 실험이라기보다, 앙상블을 위한 재현성 확보 실험이다. 기존 최고 성능을 보인 ResNet18 `layer4 + fc` 설정을 동일하게 재현했고, validation/test probability를 저장하여 EfficientNet-B0와의 soft voting 또는 추후 stacking ensemble에 활용할 수 있도록 했다.

## EXP-08. ResNet18 + EfficientNet-B0 Soft Voting 후보

### 목적

EXP-07 ResNet18과 EXP-06 EfficientNet-B0의 test probability를 가중 평균하여 soft voting 제출 후보를 생성했다. EfficientNet-B0 단독 성능은 낮지만 ResNet18과 예측 분포가 다르므로, 일부 샘플에서 보완 효과가 있는지 확인하는 것이 목적이다.

### 사용 파일

```text
preds/exp07_resnet18_valid_probs.csv
preds/exp07_resnet18_test_probs.csv
preds/exp06_efficientnet_b0_valid_probs.csv
preds/exp06_efficientnet_b0_test_probs.csv
preds/exp08_softvote_summary.csv
```

### Soft Voting 결과

| ResNet Weight | EfficientNet Weight | Valid Accuracy | Valid Precision | Valid Recall | Valid F1 | Test Pred 0 | Test Pred 1 | ResNet 대비 변경 샘플 수 | 제출 파일 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.95 | 0.05 | 0.98659 | 0.9910 | 0.9910 | 0.9910 | 219 | 405 | 2 | `submissions/exp08_softvote_r95_e05_submit.csv` |
| 0.90 | 0.10 | 0.98563 | 0.9910 | 0.9897 | 0.9903 | 219 | 405 | 2 | `submissions/exp08_softvote_r90_e09_submit.csv` |
| 0.85 | 0.15 | 0.98467 | 0.9897 | 0.9897 | 0.9897 | 219 | 405 | 2 | `submissions/exp08_softvote_r85_e15_submit.csv` |
| 0.80 | 0.20 | 0.98372 | 0.9897 | 0.9884 | 0.9890 | 219 | 405 | 2 | `submissions/exp08_softvote_r80_e19_submit.csv` |
| 0.75 | 0.25 | 0.98372 | 0.9897 | 0.9884 | 0.9890 | 219 | 405 | 2 | `submissions/exp08_softvote_r75_e25_submit.csv` |
| 0.70 | 0.30 | 0.98563 | 0.9910 | 0.9897 | 0.9903 | 222 | 402 | 5 | `submissions/exp08_softvote_r70_e30_submit.csv` |
| 0.65 | 0.35 | 0.98563 | 0.9910 | 0.9897 | 0.9903 | 222 | 402 | 7 | `submissions/exp08_softvote_r65_e35_submit.csv` |
| 0.60 | 0.40 | 0.98467 | 0.9897 | 0.9897 | 0.9897 | 222 | 402 | 11 | `submissions/exp08_softvote_r60_e40_submit.csv` |
| 0.50 | 0.50 | 0.98467 | 0.9897 | 0.9897 | 0.9897 | 227 | 397 | 18 | `submissions/exp08_softvote_r50_e50_submit.csv` |

### 해석

EfficientNet-B0 weight 0.1은 매우 보수적인 설정이었다. test 예측 분포가 ResNet18 단독과 동일하게 `label 0: 219`, `label 1: 405`였고, hard label이 바뀐 샘플도 2개뿐이었다. 따라서 0.1 앙상블은 leaderboard 점수가 ResNet18 단독과 같게 나올 가능성이 크다. 앙상블 효과를 확인하려면 `0.70:0.30` 또는 `0.65:0.35`처럼 EfficientNet-B0 비중을 더 높인 후보가 더 적절하다.

### 리더보드 결과

- 제출 파일: `submissions/exp08_softvote_r70_e30_submit.csv`
- Leaderboard score: 0.9230769231
- 제출 횟수: 9회

ResNet18 0.70, EfficientNet-B0 0.30 soft voting은 기존 최고 점수 0.9278846154보다 낮았다. 단독 EfficientNet-B0보다는 높았지만, ResNet18 단독보다 낮았으므로 현재 조합에서는 앙상블이 추가 개선으로 이어지지 않았다.

### 과제용 해석

Soft voting에서 EfficientNet-B0의 비중을 0.1로 설정했을 때는 ResNet18 단독 예측과 거의 동일한 hard label이 생성되었다. 이후 EfficientNet-B0 비중을 0.3으로 늘린 soft voting을 제출했지만 leaderboard score는 0.9230769231로 ResNet18 단독보다 낮았다. 따라서 단순히 서로 다른 모델을 평균내는 것만으로는 성능 향상이 보장되지 않으며, 앙상블에 포함할 모델의 단독 성능과 오류 보완 관계를 함께 확인해야 한다.

## EXP-09. DenseNet121 Fine-tuning

### 목적

ResNet18과 다른 CNN 구조를 가진 DenseNet121을 실험하여 모델 다양성을 확보한다. DenseNet은 feature reuse 구조를 가지므로 의료 이미지 분류에서 안정적인 특징 추출을 기대할 수 있다.

### 설정

- 모델: pretrained DenseNet121
- fine-tuning 범위: `denseblock4 + norm5 + classifier`
- 입력 채널: grayscale 이미지를 3채널로 변환
- 입력 크기: 224 x 224
- optimizer: `AdamW`
- learning rate: 1e-4
- weight decay: 1e-4
- loss: class weight 적용 `CrossEntropyLoss`
- epoch: 6
- threshold: 0.5

### 저장 파일

```text
models/exp09_densenet121_best.pt
preds/exp09_densenet121_valid_probs.csv
preds/exp09_densenet121_test_probs.csv
preds/exp09_densenet121_result.json
submissions/exp09_densenet121_submit.csv
```

### 학습 파라미터 수

```text
trainable params: 2,162,178
total params: 6,955,906
```

### Epoch별 Validation 결과

| Epoch | Train Loss | Valid Loss | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1940 | 0.1042 | 0.9607 | 0.9907 | 0.9562 | 0.9731 |
| 2 | 0.0941 | 0.0860 | 0.9732 | 0.9870 | 0.9768 | 0.9819 |
| 3 | 0.0645 | 0.0853 | 0.9780 | 0.9833 | 0.9871 | 0.9852 |
| 4 | 0.0580 | 0.0724 | 0.9684 | 0.9960 | 0.9613 | 0.9784 |
| 5 | 0.0456 | 0.0833 | 0.9598 | 0.9946 | 0.9510 | 0.9723 |
| 6 | 0.0416 | 0.0649 | 0.9713 | 0.9973 | 0.9639 | 0.9803 |

### Best Model

- Best epoch: 3
- Validation accuracy: 0.97797
- Precision: 0.9833
- Recall: 0.9871
- F1-score: 0.9852

### Confusion Matrix

```text
[[255  13]
 [ 10 766]]
```

DenseNet121은 EfficientNet-B0보다 validation accuracy가 높았지만, ResNet18 EXP-02/EXP-07보다는 낮았다. FN은 10개로 EfficientNet-B0보다 적지만 ResNet18의 7개보다는 많다.

### 제출 파일 분포

```text
submissions/exp09_densenet121_submit.csv
label 0: 220
label 1: 404
```

DenseNet121의 test 예측 분포는 ResNet18 최고 모델의 `label 0: 219`, `label 1: 405`와 거의 동일하다. 단독 성능은 ResNet18보다 낮을 가능성이 있지만, probability 저장 파일을 기반으로 후속 soft voting 또는 stacking에 활용할 수 있다.

### 리더보드 결과

- 제출 파일: `submissions/exp09_densenet121_submit.csv`
- Leaderboard score: 0.9262820513

DenseNet121 단독 점수는 ResNet18 최고 점수 0.9278846154보다 약간 낮지만, 차이가 매우 작다. ResNet18과 다른 구조를 사용하면서도 비슷한 leaderboard 성능을 보였기 때문에, EfficientNet-B0보다 더 강한 앙상블 후보로 판단된다.

### 과제용 해석

DenseNet121은 validation accuracy 0.9780을 기록하여 EfficientNet-B0보다는 높았지만, ResNet18 최고 모델보다는 낮았다. 그러나 leaderboard score는 0.9262820513으로 ResNet18 최고 점수와 매우 근접했다. ResNet18과 다른 구조의 CNN이면서도 단독 성능이 높게 나타났기 때문에, 후속 soft voting 또는 stacking ensemble에서 중요한 후보로 활용할 수 있다.

## EXP-10. ResNet18 + DenseNet121 Soft Voting 후보

### 목적

DenseNet121 단독 leaderboard score가 ResNet18 최고 점수와 매우 근접했기 때문에, 두 모델의 probability를 가중 평균하여 soft voting을 시도했다. EfficientNet-B0와 달리 DenseNet121은 단독 성능이 높아 앙상블 개선 가능성이 더 크다.

### 사용 파일

```text
preds/exp07_resnet18_valid_probs.csv
preds/exp07_resnet18_test_probs.csv
preds/exp09_densenet121_valid_probs.csv
preds/exp09_densenet121_test_probs.csv
preds/exp10_resnet_densenet_softvote_summary.csv
```

### Soft Voting 결과

| ResNet Weight | DenseNet Weight | Valid Accuracy | Valid Precision | Valid Recall | Valid F1 | Test Pred 0 | Test Pred 1 | ResNet 대비 변경 샘플 수 | 제출 파일 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.95 | 0.05 | 0.98659 | 0.9910 | 0.9910 | 0.9910 | 219 | 405 | 2 | `submissions/exp10_softvote_res95_den05_submit.csv` |
| 0.90 | 0.10 | 0.98659 | 0.9910 | 0.9910 | 0.9910 | 219 | 405 | 2 | `submissions/exp10_softvote_res90_den10_submit.csv` |
| 0.85 | 0.15 | 0.98659 | 0.9910 | 0.9910 | 0.9910 | 218 | 406 | 3 | `submissions/exp10_softvote_res85_den15_submit.csv` |
| 0.80 | 0.20 | 0.98563 | 0.9897 | 0.9910 | 0.9903 | 217 | 407 | 4 | `submissions/exp10_softvote_res80_den20_submit.csv` |
| 0.75 | 0.25 | 0.98659 | 0.9897 | 0.9923 | 0.9910 | 217 | 407 | 6 | `submissions/exp10_softvote_res75_den25_submit.csv` |
| 0.70 | 0.30 | 0.98755 | 0.9910 | 0.9923 | 0.9916 | 216 | 408 | 7 | `submissions/exp10_softvote_res70_den30_submit.csv` |
| 0.65 | 0.35 | 0.98755 | 0.9910 | 0.9923 | 0.9916 | 215 | 409 | 8 | `submissions/exp10_softvote_res65_den35_submit.csv` |
| 0.60 | 0.40 | 0.98851 | 0.9910 | 0.9936 | 0.9923 | 215 | 409 | 10 | `submissions/exp10_softvote_res60_den40_submit.csv` |
| 0.55 | 0.45 | 0.98851 | 0.9910 | 0.9936 | 0.9923 | 216 | 408 | 11 | `submissions/exp10_softvote_res55_den45_submit.csv` |
| 0.50 | 0.50 | 0.99042 | 0.9923 | 0.9948 | 0.9936 | 218 | 406 | 13 | `submissions/exp10_softvote_res50_den50_submit.csv` |
| 0.45 | 0.55 | 0.98946 | 0.9923 | 0.9936 | 0.9929 | 217 | 407 | 18 | `submissions/exp10_softvote_res45_den55_submit.csv` |
| 0.40 | 0.60 | 0.98946 | 0.9923 | 0.9936 | 0.9929 | 218 | 406 | 23 | `submissions/exp10_softvote_res40_den60_submit.csv` |

### 해석

ResNet18과 DenseNet121 soft voting은 validation 기준으로 ResNet18 단독보다 개선되었다. 특히 50:50 조합은 validation accuracy 0.99042, confusion matrix `[[262, 6], [4, 772]]` 수준으로 FN과 FP가 모두 줄었다. Test 예측 분포도 `label 0: 218`, `label 1: 406`으로 기존 ResNet18 최고 모델과 거의 비슷하다. 따라서 `res50_den50`이 1순위 제출 후보이고, 보수적으로는 `res60_den40`도 제출 후보가 될 수 있다.

### 리더보드 결과

- 제출 파일: `submissions/exp10_softvote_res50_den50_submit.csv`
- Leaderboard score: 0.9294871795
- 제출 일시: 2026-06-19 01:03:19

EXP-10의 50:50 soft voting은 기존 최고였던 ResNet18 단독 0.9278846154보다 높은 0.9294871795를 기록했다. 이후 DenseNet 비중을 더 높인 `res45_den55` 제출에서 0.9342948718, `res40_den60` 제출에서 0.9358974359를 기록하며 추가 개선이 확인되었다. 현재까지 가장 좋은 제출 파일은 `submissions/exp10_softvote_res40_den60_submit.csv`이다.

### 과제용 해석

DenseNet121은 단독 validation 성능은 ResNet18보다 낮았지만 leaderboard score는 거의 비슷했다. 두 모델의 probability를 soft voting으로 결합하자 validation accuracy가 0.99042까지 상승했고, leaderboard score도 최종적으로 0.9358974359까지 개선되었다. 이는 두 모델이 서로 다른 오류를 일부 보완했기 때문으로 해석할 수 있다. 특히 DenseNet 비중을 0.60까지 높였을 때 leaderboard가 더 개선되어, test 데이터에서는 DenseNet의 예측이 일부 샘플에서 더 유리하게 작용한 것으로 볼 수 있다.

## EXP-11. ResNet18 + DenseNet121 Fine Soft Voting

### 목적

EXP-10에서 DenseNet 비중을 0.60까지 높였을 때 leaderboard가 상승했다. 따라서 DenseNet 비중을 0.62~0.75까지 더 높인 fine search 후보를 생성했다.

### 사용 파일

```text
preds/exp07_resnet18_test_probs.csv
preds/exp09_densenet121_test_probs.csv
preds/exp11_resnet_densenet_fine_softvote_summary.csv
```

### Fine Search 결과

| ResNet Weight | DenseNet Weight | Valid Accuracy | Valid Precision | Valid Recall | Valid F1 | Test Pred 0 | Test Pred 1 | ResNet 대비 변경 샘플 수 | 제출 파일 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.38 | 0.62 | 0.98946 | 0.9923 | 0.9936 | 0.9929 | 218 | 406 | 23 | `submissions/exp11_softvote_res38_den62_submit.csv` |
| 0.35 | 0.65 | 0.98946 | 0.9923 | 0.9936 | 0.9929 | 219 | 405 | 24 | `submissions/exp11_softvote_res35_den65_submit.csv` |
| 0.32 | 0.68 | 0.98755 | 0.9923 | 0.9910 | 0.9916 | 220 | 404 | 27 | `submissions/exp11_softvote_res32_den68_submit.csv` |
| 0.30 | 0.70 | 0.98755 | 0.9923 | 0.9910 | 0.9916 | 221 | 403 | 28 | `submissions/exp11_softvote_res30_den70_submit.csv` |
| 0.28 | 0.72 | 0.98659 | 0.9922 | 0.9897 | 0.9910 | 222 | 402 | 29 | `submissions/exp11_softvote_res28_den72_submit.csv` |
| 0.25 | 0.75 | 0.98467 | 0.9910 | 0.9884 | 0.9897 | 223 | 401 | 30 | `submissions/exp11_softvote_res25_den75_submit.csv` |

### 해석

DenseNet 비중을 0.62~0.65까지 높였을 때 validation accuracy는 0.98946으로 유지되었다. 이는 EXP-10의 50:50 validation accuracy 0.99042보다는 낮지만, leaderboard에서는 40:60이 더 좋았기 때문에 test 기준 최적점이 validation 최적점보다 DenseNet 쪽에 더 있을 가능성이 있다. 우선 제출 후보는 `res35_den65`이고, 그다음 후보는 `res38_den62`이다.

### 리더보드 결과

- 제출 파일: `submissions/exp11_softvote_res35_den65_submit.csv`
- Leaderboard score: 0.9342948718

DenseNet 비중을 0.65까지 높인 결과, 현재 최고인 `res40_den60`의 0.9358974359보다 낮은 0.9342948718을 기록했다. 따라서 DenseNet 비중을 더 높이는 방향은 0.65부터 성능이 감소하는 것으로 보이며, 현재 최적점은 `res40_den60` 근처로 판단된다.

### 과제용 해석

Soft voting에서 DenseNet 비중을 0.60까지 높였을 때 leaderboard가 개선되었으므로, 추가로 0.62~0.75 범위를 탐색하였다. Validation 기준으로는 0.38:0.62와 0.35:0.65 조합이 안정적이었지만, 실제 제출 결과 0.35:0.65는 0.9342948718로 0.40:0.60보다 낮았다. 따라서 DenseNet 비중을 과도하게 높이는 것보다 0.60 근처가 더 적절한 것으로 판단된다.

## EXP-12. ConvNeXt-Tiny Fine-tuning

### 목적

ResNet18과 DenseNet121 앙상블이 현재 최고 점수를 기록했지만, 추가적인 모델 다양성을 확보하기 위해 현대 CNN 구조인 ConvNeXt-Tiny를 실험했다.

### 설정

- 모델: pretrained ConvNeXt-Tiny
- fine-tuning 범위: 마지막 ConvNeXt stage + classifier
- 입력 채널: grayscale 이미지를 3채널로 변환
- 입력 크기: 224 x 224
- optimizer: `AdamW`
- learning rate: 5e-5
- weight decay: 1e-4
- loss: class weight 적용 `CrossEntropyLoss`
- epoch: 6
- threshold: 0.5

### 저장 파일

```text
models/exp12_convnext_tiny_best.pt
preds/exp12_convnext_tiny_valid_probs.csv
preds/exp12_convnext_tiny_test_probs.csv
preds/exp12_convnext_tiny_result.json
submissions/exp12_convnext_tiny_submit.csv
```

### 학습 파라미터 수

```text
trainable params: 14,290,946
total params: 27,821,666
```

### Epoch별 Validation 결과

| Epoch | Train Loss | Valid Loss | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2184 | 0.1028 | 0.9540 | 0.9919 | 0.9459 | 0.9683 |
| 2 | 0.1014 | 0.0839 | 0.9588 | 0.9959 | 0.9485 | 0.9716 |
| 3 | 0.0746 | 0.0663 | 0.9770 | 0.9909 | 0.9781 | 0.9844 |
| 4 | 0.0599 | 0.0692 | 0.9646 | 0.9946 | 0.9575 | 0.9757 |
| 5 | 0.0531 | 0.0587 | 0.9789 | 0.9948 | 0.9768 | 0.9857 |
| 6 | 0.0422 | 0.0596 | 0.9808 | 0.9948 | 0.9794 | 0.9870 |

### Best Model

- Best epoch: 6
- Validation accuracy: 0.98084
- Precision: 0.9948
- Recall: 0.9794
- F1-score: 0.9870

### Confusion Matrix

```text
[[264   4]
 [ 16 760]]
```

ConvNeXt-Tiny는 precision이 높고 FP가 4개로 적었지만, FN이 16개로 ResNet18/DenseNet121보다 많았다. 의료 이미지 관점에서는 폐렴을 놓치는 FN이 중요하므로 단독 모델로는 최고 후보가 아니다.

### 제출 파일 분포

```text
submissions/exp12_convnext_tiny_submit.csv
label 0: 233
label 1: 391
```

ConvNeXt-Tiny는 기존 최고 앙상블보다 label 1 예측이 적다. Validation에서는 ResNet+DenseNet 앙상블보다 낮았지만, leaderboard에서는 오히려 가장 높은 점수를 기록했다. 이는 현재 validation split이 test 분포를 완전히 대표하지 못하며, ConvNeXt-Tiny의 보수적인 예측 분포가 test 데이터에는 더 잘 맞았을 가능성을 시사한다.

### 리더보드 결과

- 제출 파일: `submissions/exp12_convnext_tiny_submit.csv`
- Leaderboard score: 0.94711
- 순위: 23위 확인
- 제출 횟수: 15회

ConvNeXt-Tiny 단독 제출은 기존 최고였던 ResNet18+DenseNet121 soft voting의 0.9358974359보다 크게 상승했다. Validation accuracy는 0.98084로 최고가 아니었지만, leaderboard에서는 가장 좋은 결과를 보였으므로 이후 실험은 ConvNeXt-Tiny를 중심으로 확장하는 것이 적절하다.

### 과제용 해석

ConvNeXt-Tiny는 validation accuracy 0.9808로 ResNet+DenseNet 앙상블보다 낮았지만, leaderboard score는 0.94711로 가장 높게 나타났다. Confusion matrix에서는 FP가 4개로 적고 FN이 16개로 많아 validation 기준 recall은 낮았지만, test 데이터에서는 이러한 예측 경향이 더 유리하게 작용한 것으로 보인다. 따라서 단일 validation split만으로 모델을 판단하기 어렵고, leaderboard 결과까지 함께 고려하면 ConvNeXt-Tiny가 현재 가장 유효한 모델이다.

## EXP-13. ConvNeXt 중심 Soft Voting 후보

### 목적

EXP-12 ConvNeXt-Tiny가 leaderboard에서 크게 상승했으므로, ConvNeXt를 중심으로 ResNet18 또는 DenseNet121을 섞는 soft voting 후보를 생성했다.

### 사용 파일

```text
preds/exp12_convnext_tiny_test_probs.csv
preds/exp07_resnet18_test_probs.csv
preds/exp09_densenet121_test_probs.csv
preds/exp13_convnext_ensemble_summary.csv
```

### 주요 후보 결과

| 조합 | Valid Accuracy | Valid Precision | Valid Recall | Valid F1 | Test Pred 0 | Test Pred 1 | ConvNeXt 대비 변경 샘플 수 | 제출 파일 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ConvNeXt 0.5 + DenseNet 0.5 | 0.98659 | 0.9948 | 0.9871 | 0.9909 | 228 | 396 | 11 | `submissions/exp13_softvote_cnx50_den50_submit.csv` |
| ConvNeXt 0.5 + ResNet 0.5 | 0.98659 | 0.9948 | 0.9871 | 0.9909 | 221 | 403 | 22 | `submissions/exp13_softvote_cnx50_res50_submit.csv` |
| ConvNeXt 0.7 + ResNet 0.3 | 0.98467 | 0.9948 | 0.9845 | 0.9896 | 228 | 396 | 11 | `submissions/exp13_softvote_cnx70_res30_submit.csv` |
| ConvNeXt 0.6 + ResNet 0.4 | 0.98372 | 0.9935 | 0.9845 | 0.9890 | 225 | 399 | 16 | `submissions/exp13_softvote_cnx60_res40_submit.csv` |
| ConvNeXt 0.5 + ResNet 0.2 + DenseNet 0.3 | 0.98372 | 0.9935 | 0.9845 | 0.9890 | 225 | 399 | 14 | `submissions/exp13_softvote_cnx50_res20_den30_submit.csv` |

### 해석

ConvNeXt 중심 soft voting은 validation accuracy를 ConvNeXt 단독 0.98084보다 높였다. 특히 ConvNeXt 0.5 + DenseNet 0.5와 ConvNeXt 0.5 + ResNet 0.5는 validation accuracy 0.98659를 기록했다. 다만 ConvNeXt 단독이 leaderboard에서 강했던 이유가 test 분포와 관련되어 있을 수 있으므로, ConvNeXt의 예측을 너무 많이 바꾸는 조합은 조심해야 한다. 제출 후보는 `cnx50_den50` 또는 `cnx50_res50`이며, 더 보수적으로는 ConvNeXt 단독 유지가 기준이다.

### 리더보드 결과

- 제출 파일: `submissions/exp13_softvote_cnx50_den50_submit.csv`
- Leaderboard score: 0.9455128205

ConvNeXt 0.5 + DenseNet 0.5 soft voting은 ConvNeXt 단독 점수 0.9471153846보다 낮았다. Validation에서는 개선되었지만 leaderboard에서는 하락했으므로, ConvNeXt 단독의 예측 분포를 크게 바꾸는 앙상블은 현재 test set에 불리한 것으로 판단된다.

### 과제용 해석

ConvNeXt-Tiny는 단독 validation 성능은 최고가 아니었지만 leaderboard에서 가장 높은 점수를 기록했다. 이에 따라 ConvNeXt를 중심으로 DenseNet121과 soft voting을 수행했으나, 0.5:0.5 조합은 leaderboard score 0.9455128205로 ConvNeXt 단독보다 낮았다. Validation 기준 개선이 test 성능 개선으로 이어지지 않았으며, 현재 test set에서는 ConvNeXt 단독 예측 분포가 더 적합한 것으로 보인다.

## EXP-14. ConvNeXt-Tiny Sqrt Class Weight

### 목적

EXP-12 ConvNeXt-Tiny는 leaderboard에서 가장 높은 점수를 기록했지만, validation 기준 FN이 16개로 많았다. 기존 class weight가 강하게 적용되어 있었기 때문에, class weight를 sqrt 방식으로 약화하여 더 안정적인 학습이 가능한지 확인했다.

### 설정

- 모델: pretrained ConvNeXt-Tiny
- fine-tuning 범위: 마지막 ConvNeXt stage + classifier
- optimizer: `AdamW`
- learning rate: 5e-5
- weight decay: 1e-4
- loss: sqrt class weight 적용 `CrossEntropyLoss`
- epoch: 6

### Class Weight

```text
기존 class weights: [1.94408201, 0.67312036]
sqrt class weights: [1.39430341, 0.82043913]
```

### Epoch별 Validation 결과

| Epoch | Train Loss | Valid Loss | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2137 | 0.0957 | 0.9626 | 0.9830 | 0.9665 | 0.9747 |
| 2 | 0.0977 | 0.0703 | 0.9722 | 0.9934 | 0.9691 | 0.9811 |
| 3 | 0.0719 | 0.0593 | 0.9847 | 0.9897 | 0.9897 | 0.9897 |
| 4 | 0.0596 | 0.0686 | 0.9713 | 0.9960 | 0.9652 | 0.9804 |
| 5 | 0.0517 | 0.0548 | 0.9799 | 0.9948 | 0.9781 | 0.9864 |
| 6 | 0.0397 | 0.0537 | 0.9856 | 0.9935 | 0.9871 | 0.9903 |

### Best Model

- Best epoch: 6
- Validation accuracy: 0.98563
- Precision: 0.9935
- Recall: 0.9871
- F1-score: 0.9903

### Confusion Matrix

```text
[[263   5]
 [ 10 766]]
```

기존 ConvNeXt-Tiny EXP-12의 confusion matrix는 `[[264, 4], [16, 760]]`이었다. Sqrt class weight를 적용한 EXP-14에서는 FP가 1개 증가했지만 FN이 16개에서 10개로 줄었다. 의료 이미지 관점에서는 FN 감소가 의미 있는 개선이다.

### 제출 파일 분포

```text
submissions/exp14_convnext_tiny_sqrt_weight_submit.csv
label 0: 231
label 1: 393
```

기존 ConvNeXt-Tiny EXP-12는 `label 0: 233`, `label 1: 391`이었다. EXP-14는 label 1 예측이 2개 증가했지만 전체 분포는 크게 달라지지 않았다.

### 리더보드 결과

- 제출 파일: `submissions/exp14_convnext_tiny_sqrt_weight_submit.csv`
- Leaderboard score: 0.9439102564

EXP-14는 validation accuracy가 EXP-12보다 높았지만, leaderboard score는 EXP-12의 0.9471153846보다 낮았다. 따라서 ConvNeXt-Tiny에서는 sqrt class weight가 validation 기준으로는 좋아 보였지만 실제 test 성능 개선으로 이어지지 않았다.

### 과제용 해석

ConvNeXt-Tiny에 sqrt class weight를 적용하자 validation accuracy가 0.98084에서 0.98563으로 상승했고, FN이 16개에서 10개로 감소하였다. 그러나 leaderboard score는 0.9439102564로 기존 ConvNeXt-Tiny보다 낮게 나타났다. 이는 validation에서의 FN 감소가 실제 test accuracy 개선으로 이어지지 않았음을 보여주며, 현재 기준으로는 기존 class weight를 사용한 EXP-12가 더 적절하다.

## EXP-15. ConvNeXt-Tiny No Horizontal Flip

### 목적

EXP-12 ConvNeXt-Tiny가 현재 최고 leaderboard score를 기록했으므로, 모델 구조와 class weight는 유지하고 augmentation만 보수적으로 변경했다. 흉부 X-ray에서는 좌우 반전이 실제 해부학적 방향 정보를 바꿀 수 있으므로 `RandomHorizontalFlip`을 제거하고, 회전 강도도 7도에서 5도로 낮췄다.

### 설정

- 모델: pretrained ConvNeXt-Tiny
- fine-tuning 범위: 마지막 ConvNeXt stage + classifier
- optimizer: `AdamW`
- learning rate: 5e-5
- weight decay: 1e-4
- loss: 기존 class weight 적용 `CrossEntropyLoss`
- augmentation: `RandomHorizontalFlip` 제거, `RandomRotation(5)` 적용
- epoch: 6

### Epoch별 Validation 결과

| Epoch | Train Loss | Valid Loss | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2185 | 0.1152 | 0.9435 | 0.9959 | 0.9278 | 0.9606 |
| 2 | 0.0967 | 0.1655 | 0.9148 | 0.9985 | 0.8866 | 0.9392 |
| 3 | 0.0666 | 0.0822 | 0.9626 | 0.9986 | 0.9510 | 0.9743 |
| 4 | 0.0587 | 0.0708 | 0.9693 | 0.9960 | 0.9626 | 0.9790 |
| 5 | 0.0496 | 0.0563 | 0.9780 | 0.9947 | 0.9755 | 0.9850 |
| 6 | 0.0355 | 0.0558 | 0.9808 | 0.9948 | 0.9794 | 0.9870 |

### Best Model

- Best epoch: 6
- Validation accuracy: 0.98084
- Precision: 0.9948
- Recall: 0.9794
- F1-score: 0.9870

### Confusion Matrix

```text
[[264   4]
 [ 16 760]]
```

EXP-15의 validation confusion matrix는 EXP-12와 동일했다. 따라서 validation set 기준으로는 HorizontalFlip 제거가 명확한 성능 개선을 만들지는 않았다.

### 제출 파일 분포

```text
submissions/exp15_convnext_tiny_no_flip_submit.csv
label 0: 220
label 1: 404
```

EXP-12의 제출 분포는 `label 0: 233`, `label 1: 391`이었다. EXP-15는 validation 결과는 같지만 test set에서는 label 1 예측이 13개 증가했다. 이 차이는 leaderboard에서 달라질 수 있으므로 제출 후 실제 점수를 확인해야 한다.

### 저장 파일

```text
scripts/run_exp15_convnext_no_flip.py
models/exp15_convnext_tiny_no_flip_best.pt
preds/exp15_convnext_tiny_no_flip_valid_probs.csv
preds/exp15_convnext_tiny_no_flip_test_probs.csv
preds/exp15_convnext_tiny_no_flip_result.json
submissions/exp15_convnext_tiny_no_flip_submit.csv
```

### 리더보드 결과

- 제출 파일: `submissions/exp15_convnext_tiny_no_flip_submit.csv`
- Leaderboard score: 0.9391025641

EXP-15는 validation 결과가 EXP-12와 동일했지만, leaderboard score는 0.9391025641로 하락했다. Test 예측에서 label 1이 391개에서 404개로 증가했기 때문에, 현재 test set에서는 label 1 쪽으로 더 많이 예측하는 방향이 불리했던 것으로 해석할 수 있다.

### 과제용 해석

ConvNeXt-Tiny에서 의료 이미지에 더 보수적인 augmentation을 적용하기 위해 HorizontalFlip을 제거하고 회전 강도를 낮췄다. Validation accuracy와 confusion matrix는 기존 ConvNeXt-Tiny와 동일했지만, test 예측 분포는 label 1 쪽으로 더 이동했다. 따라서 validation 기준으로는 성능 차이가 없으며, 실제 test set 일반화 여부는 leaderboard 결과를 통해 추가 확인이 필요하다.

## EXP-16. ConvNeXt-Tiny Seed Ensemble

### 목적

EXP-12 ConvNeXt-Tiny가 현재 최고 leaderboard score를 기록했기 때문에, 같은 구조와 같은 train/validation split을 유지하고 학습 seed만 바꾼 ConvNeXt-Tiny를 추가로 학습했다. 목적은 서로 다른 초기화와 augmentation 순서에서 생기는 예측 변동을 soft voting으로 평균내는 것이다.

### 설정

- 기준 모델: EXP-12 ConvNeXt-Tiny
- 추가 모델: ConvNeXt-Tiny seed7
- train/validation split: EXP-12와 동일
- augmentation: EXP-12와 동일, `RandomHorizontalFlip(p=0.5)`, `RandomRotation(7)`
- fine-tuning 범위: 마지막 ConvNeXt stage + classifier
- optimizer: `AdamW`
- learning rate: 5e-5
- weight decay: 1e-4
- loss: 기존 class weight 적용 `CrossEntropyLoss`
- epoch: 6

### Seed7 단일 모델 Validation 결과

| Epoch | Train Loss | Valid Loss | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2553 | 0.1003 | 0.9693 | 0.9895 | 0.9691 | 0.9792 |
| 2 | 0.1097 | 0.0781 | 0.9684 | 0.9921 | 0.9652 | 0.9784 |
| 3 | 0.0813 | 0.0652 | 0.9837 | 0.9897 | 0.9884 | 0.9890 |
| 4 | 0.0614 | 0.0582 | 0.9741 | 0.9973 | 0.9678 | 0.9823 |
| 5 | 0.0584 | 0.0708 | 0.9674 | 0.9973 | 0.9588 | 0.9777 |
| 6 | 0.0481 | 0.0605 | 0.9828 | 0.9897 | 0.9871 | 0.9884 |

### Seed7 Best Model

- Best epoch: 3
- Validation accuracy: 0.98372
- Precision: 0.9897
- Recall: 0.9884
- F1-score: 0.9890

### Seed7 Confusion Matrix

```text
[[260   8]
 [  9 767]]
```

Seed7 단일 모델은 validation에서는 EXP-12보다 accuracy가 높았지만, test 예측 분포가 `label 0: 214`, `label 1: 410`으로 label 1 쪽에 크게 치우쳤다. EXP-15에서 label 1 증가가 leaderboard 하락으로 이어졌기 때문에 seed7 단독 제출은 위험도가 높다고 판단했다.

### Soft Voting 후보

| EXP-12 weight | Seed7 weight | Validation Accuracy | Precision | Recall | F1 | Test label 0 | Test label 1 | EXP-12 대비 변경 수 | 제출 파일 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.95 | 0.05 | 0.98180 | 0.9948 | 0.9807 | 0.9877 | 233 | 391 | 0 | `submissions/exp16_softvote_cnx12_95_seed7_05_submit.csv` |
| 0.90 | 0.10 | 0.98180 | 0.9948 | 0.9807 | 0.9877 | 233 | 391 | 0 | `submissions/exp16_softvote_cnx12_90_seed7_10_submit.csv` |
| 0.85 | 0.15 | 0.98180 | 0.9948 | 0.9807 | 0.9877 | 232 | 392 | 1 | `submissions/exp16_softvote_cnx12_85_seed7_15_submit.csv` |
| 0.80 | 0.20 | 0.98180 | 0.9948 | 0.9807 | 0.9877 | 232 | 392 | 1 | `submissions/exp16_softvote_cnx12_80_seed7_20_submit.csv` |
| 0.75 | 0.25 | 0.98276 | 0.9948 | 0.9820 | 0.9883 | 231 | 393 | 2 | `submissions/exp16_softvote_cnx12_75_seed7_25_submit.csv` |
| 0.70 | 0.30 | 0.98467 | 0.9948 | 0.9845 | 0.9896 | 230 | 394 | 3 | `submissions/exp16_softvote_cnx12_70_seed7_30_submit.csv` |
| 0.60 | 0.40 | 0.98372 | 0.9935 | 0.9845 | 0.9890 | 225 | 399 | 8 | `submissions/exp16_softvote_cnx12_60_seed7_40_submit.csv` |
| 0.50 | 0.50 | 0.98372 | 0.9935 | 0.9845 | 0.9890 | 223 | 401 | 10 | `submissions/exp16_softvote_cnx12_50_seed7_50_submit.csv` |

### 제출 후보

```text
submissions/exp16_softvote_cnx12_70_seed7_30_submit.csv
```

이 후보는 validation accuracy가 0.98467로 가장 높고, EXP-12 대비 test 예측 변경은 3개뿐이다. 변경된 3개 샘플은 모두 `0 -> 1` 방향이다.

### 저장 파일

```text
scripts/run_exp16_convnext_seed7.py
models/exp16_convnext_tiny_seed7_best.pt
preds/exp16_convnext_tiny_seed7_valid_probs.csv
preds/exp16_convnext_tiny_seed7_test_probs.csv
preds/exp16_convnext_tiny_seed7_result.json
preds/exp16_convnext_seed_ensemble_summary.csv
submissions/exp16_softvote_cnx12_70_seed7_30_submit.csv
```

### 리더보드 결과

- 제출 파일: `submissions/exp16_softvote_cnx12_70_seed7_30_submit.csv`
- Leaderboard score: 0.9455128205

EXP-16은 validation accuracy가 EXP-12보다 높았고 test 예측 변경도 3개뿐이었지만, leaderboard score는 EXP-12의 0.9471153846보다 낮았다. 변경된 3개가 모두 `0 -> 1` 방향이었으므로, 현재 test set에서는 EXP-12가 0으로 유지한 일부 샘플이 더 적절했을 가능성이 있다.

### 과제용 해석

같은 ConvNeXt-Tiny 구조에서 seed를 바꿔 학습한 모델은 validation 기준으로 기존 EXP-12보다 높은 성능을 보였다. 그러나 seed7 단일 모델은 test 예측이 label 1 쪽으로 과하게 이동했기 때문에 단독 제출보다는 EXP-12와 soft voting하는 방식이 더 안정적이다. EXP-12 0.70, seed7 0.30 조합은 validation 성능을 높이면서 test 예측 변경 수를 3개로 제한했으므로, 현재 기준에서 제출 후보로 가장 적절하다.

## EXP-17. ConvNeXt-Tiny Threshold 상향

### 목적

EXP-15와 EXP-16에서 test 예측이 label 1 쪽으로 이동했을 때 leaderboard가 하락했다. 따라서 이번에는 EXP-12 ConvNeXt-Tiny의 probability를 그대로 사용하되 threshold를 0.5보다 높여 label 1 예측을 조금 줄이는 방향을 확인했다.

### 설정

- 기준 probability: `preds/exp12_convnext_tiny_valid_probs.csv`, `preds/exp12_convnext_tiny_test_probs.csv`
- 모델 재학습 없음
- threshold 후보: 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70

### Threshold별 결과

| Threshold | Validation Accuracy | Precision | Recall | F1 | Test label 0 | Test label 1 | EXP-12 대비 변경 수 | 1->0 | 제출 파일 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 0.98084 | 0.9948 | 0.9794 | 0.9870 | 233 | 391 | 0 | 0 | `submissions/exp17_convnext_tiny_threshold_0p5_submit.csv` |
| 0.52 | 0.97989 | 0.9948 | 0.9781 | 0.9864 | 233 | 391 | 0 | 0 | `submissions/exp17_convnext_tiny_threshold_0p52_submit.csv` |
| 0.54 | 0.97989 | 0.9948 | 0.9781 | 0.9864 | 234 | 390 | 1 | 1 | `submissions/exp17_convnext_tiny_threshold_0p54_submit.csv` |
| 0.56 | 0.97989 | 0.9961 | 0.9768 | 0.9863 | 235 | 389 | 2 | 2 | `submissions/exp17_convnext_tiny_threshold_0p56_submit.csv` |
| 0.58 | 0.97893 | 0.9961 | 0.9755 | 0.9857 | 237 | 387 | 4 | 4 | `submissions/exp17_convnext_tiny_threshold_0p58_submit.csv` |
| 0.60 | 0.97701 | 0.9960 | 0.9729 | 0.9844 | 238 | 386 | 5 | 5 | `submissions/exp17_convnext_tiny_threshold_0p6_submit.csv` |
| 0.62 | 0.97605 | 0.9960 | 0.9716 | 0.9837 | 238 | 386 | 5 | 5 | `submissions/exp17_convnext_tiny_threshold_0p62_submit.csv` |
| 0.65 | 0.97510 | 0.9960 | 0.9704 | 0.9830 | 239 | 385 | 6 | 6 | `submissions/exp17_convnext_tiny_threshold_0p65_submit.csv` |
| 0.70 | 0.97414 | 0.9960 | 0.9691 | 0.9824 | 241 | 383 | 8 | 8 | `submissions/exp17_convnext_tiny_threshold_0p7_submit.csv` |

### 제출 후보

```text
submissions/exp17_convnext_tiny_threshold_0p54_submit.csv
```

Threshold 0.54는 EXP-12 대비 test 예측을 1개만 바꾼다. 변경 샘플은 `test_0351.png`이며, EXP-12에서는 `prob_1=0.523306`으로 label 1이었지만 threshold 0.54에서는 label 0으로 바뀐다.

### 리더보드 결과

- 제출 파일: `submissions/exp17_convnext_tiny_threshold_0p54_submit.csv`
- Leaderboard score: 0.9455128205

EXP-17은 EXP-12 대비 `test_0351.png` 1개만 `1 -> 0`으로 바꿨지만 leaderboard score가 하락했다. 따라서 해당 샘플은 EXP-12처럼 label 1로 유지하는 쪽이 더 적절했을 가능성이 높다.

### 과제용 해석

ConvNeXt-Tiny의 확률 출력에서 threshold를 높이면 precision은 유지되거나 소폭 증가하지만 recall은 감소하는 경향을 보였다. 의료 이미지 분류에서는 recall 감소가 위험할 수 있으므로 threshold를 크게 올리기보다는, test 예측을 1개만 바꾸는 0.54 후보를 우선 확인하는 것이 안전하다. 이 실험은 validation 성능 개선보다는 leaderboard에서 EXP-12의 애매한 양성 예측 1개를 줄이는 목적이다.

## EXP-18. ConvNeXt-Tiny Full Fine-tuning

### 목적

EXP-12는 ConvNeXt-Tiny의 마지막 stage와 classifier만 학습했다. 이번 실험에서는 전체 ConvNeXt-Tiny를 unfreeze하여 X-ray 데이터에 더 맞게 전체 feature extractor를 조정할 수 있는지 확인했다.

### 설정

- 모델: pretrained ConvNeXt-Tiny
- fine-tuning 범위: 전체 ConvNeXt-Tiny + classifier
- trainable params: 27,821,666
- optimizer: `AdamW`
- backbone learning rate: 1e-5
- classifier learning rate: 5e-5
- weight decay: 1e-4
- loss: 기존 class weight 적용 `CrossEntropyLoss`
- epoch: 6
- augmentation: EXP-12와 동일

### Epoch별 Validation 결과

| Epoch | Train Loss | Valid Loss | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2222 | 0.1121 | 0.9435 | 0.9931 | 0.9304 | 0.9607 |
| 2 | 0.0938 | 0.0780 | 0.9646 | 0.9946 | 0.9575 | 0.9757 |
| 3 | 0.0689 | 0.0591 | 0.9751 | 0.9921 | 0.9742 | 0.9831 |
| 4 | 0.0544 | 0.0563 | 0.9741 | 0.9960 | 0.9691 | 0.9824 |
| 5 | 0.0379 | 0.0561 | 0.9713 | 0.9960 | 0.9652 | 0.9804 |
| 6 | 0.0395 | 0.0463 | 0.9799 | 0.9961 | 0.9768 | 0.9863 |

### Best Model

- Best epoch: 6
- Validation accuracy: 0.97989
- Precision: 0.9961
- Recall: 0.9768
- F1-score: 0.9863

### Confusion Matrix

```text
[[265   3]
 [ 18 758]]
```

EXP-12의 validation confusion matrix는 `[[264, 4], [16, 760]]`이었다. Full fine-tuning은 FP를 1개 줄였지만 FN이 2개 증가하여 validation accuracy가 EXP-12보다 낮았다.

### 제출 파일 분포

```text
submissions/exp18_convnext_tiny_full_finetune_submit.csv
label 0: 219
label 1: 405
```

EXP-12의 제출 분포는 `label 0: 233`, `label 1: 391`이었다. EXP-18은 EXP-12 대비 18개 예측이 바뀌었고, 그중 16개가 `0 -> 1` 방향이었다. 그럼에도 leaderboard score가 EXP-12와 동일했기 때문에, 변경된 샘플들 안에서 맞은 예측과 틀린 예측이 균형을 이룬 것으로 해석할 수 있다.

### 저장 파일

```text
scripts/run_exp18_convnext_full_finetune.py
models/exp18_convnext_tiny_full_finetune_best.pt
preds/exp18_convnext_tiny_full_finetune_valid_probs.csv
preds/exp18_convnext_tiny_full_finetune_test_probs.csv
preds/exp18_convnext_tiny_full_finetune_result.json
submissions/exp18_convnext_tiny_full_finetune_submit.csv
```

### 리더보드 결과

- 제출 파일: `submissions/exp18_convnext_tiny_full_finetune_submit.csv`
- Leaderboard score: 0.9471153846

### 과제용 해석

ConvNeXt-Tiny 전체를 fine-tuning하면 ImageNet 기반 feature extractor까지 X-ray 데이터에 맞게 조정할 수 있다. 이번 실험에서는 validation accuracy가 EXP-12보다 낮았지만, leaderboard score는 EXP-12와 동일했다. 따라서 full fine-tuning이 즉각적인 점수 개선으로 이어지지는 않았지만, EXP-12와 다른 예측을 내면서도 같은 공개 점수를 유지했으므로 이후 앙상블 후보로 활용할 수 있다.

## EXP-19. EXP-12 + EXP-18 Soft Voting

### 목적

EXP-12와 EXP-18은 leaderboard score가 동일하지만 test 예측이 서로 달랐다. 따라서 같은 ConvNeXt-Tiny 계열이지만 fine-tuning 범위가 다른 두 모델을 soft voting하여, 서로 다른 예측 경향을 보완할 수 있는지 확인했다.

### 설정

- EXP-12 ConvNeXt-Tiny stage4 fine-tuning weight: 0.7
- EXP-18 ConvNeXt-Tiny full fine-tuning weight: 0.3
- threshold: 0.5
- 모델 재학습 없음

### Validation 결과

- Accuracy: 0.98276
- Precision: 0.9961
- Recall: 0.9807
- F1-score: 0.9883

### Confusion Matrix

```text
[[265   3]
 [ 15 761]]
```

EXP-12의 validation confusion matrix는 `[[264, 4], [16, 760]]`이었다. Soft voting 후 FP와 FN이 각각 1개씩 줄어 validation accuracy가 상승했다.

### 제출 파일 분포

```text
submissions/exp19_softvote_cnx12_70_full30_submit.csv
label 0: 227
label 1: 397
```

EXP-12 대비 test 예측은 6개 바뀌었고, 모두 `0 -> 1` 방향이다.

```text
test_0057.png
test_0201.png
test_0248.png
test_0267.png
test_0385.png
test_0455.png
```

### 저장 파일

```text
preds/exp19_softvote_cnx12_70_full30_valid_probs.csv
preds/exp19_softvote_cnx12_70_full30_test_probs.csv
preds/exp19_softvote_cnx12_70_full30_changed_vs_exp12.csv
preds/exp19_softvote_cnx12_70_full30_result.json
submissions/exp19_softvote_cnx12_70_full30_submit.csv
```

### 리더보드 결과

- 제출 파일: `submissions/exp19_softvote_cnx12_70_full30_submit.csv`
- Leaderboard score: 0.9439102564

EXP-19는 validation에서는 EXP-12보다 개선되었지만 leaderboard score는 하락했다. EXP-12 대비 변경된 6개가 모두 `0 -> 1` 방향이었기 때문에, 이 추가 양성 예측이 공개 test set에서는 불리하게 작용한 것으로 보인다.

### 과제용 해석

EXP-12와 EXP-18은 같은 leaderboard score를 기록했지만 서로 다른 test 예측을 냈다. 두 모델을 soft voting한 결과 validation에서는 FP와 FN이 모두 줄어 개선 경향을 보였다. 다만 test에서는 label 1 예측이 EXP-12보다 6개 증가했으므로, 실제 일반화 성능은 leaderboard 결과를 통해 확인해야 한다.

## EXP-20. RidgeClassifier Stacking

### 목적

Soft voting은 모델별 확률을 고정 비율로 평균하지만, stacking은 validation 데이터를 이용해 어떤 모델의 예측을 더 신뢰할지 학습할 수 있다. 이번 실험에서는 수업에서 효과가 좋았던 RidgeClassifier를 먼저 사용하여 딥러닝 모델들의 확률 출력을 결합했다.

### 사용 모델

- EXP-07 ResNet18
- EXP-09 DenseNet121
- EXP-06 EfficientNet-B0
- EXP-12 ConvNeXt-Tiny
- EXP-18 ConvNeXt-Tiny full fine-tuning

### Feature

각 모델마다 다음 2개 feature를 사용했다.

- `prob_1`
- validation/test 내부 `rank(prob_1)`

총 feature 수는 10개다.

### 설정

- meta model: `RidgeClassifier`
- alpha: 100.0
- class_weight: `balanced`
- preprocessing: `StandardScaler`
- 학습 데이터: 현재 validation split 1,044개

### Validation 결과

- Accuracy: 0.98755
- Precision: 0.9948
- Recall: 0.9884
- F1-score: 0.9916

### Confusion Matrix

```text
[[264   4]
 [  9 767]]
```

Validation 기준으로는 지금까지 가장 높은 성능이다. 특히 FN이 9개로 줄어 recall이 개선되었다.

### 제출 파일 분포

```text
submissions/exp20_ridge_stacking_alpha100_submit.csv
label 0: 221
label 1: 403
```

EXP-12 대비 test 예측은 22개 바뀌었다.

```text
0 -> 1: 17개
1 -> 0: 5개
```

### 저장 파일

```text
preds/exp20_ridge_stacking_summary.csv
preds/exp20_ridge_stacking_alpha100_valid_scores.csv
preds/exp20_ridge_stacking_alpha100_test_scores.csv
preds/exp20_ridge_stacking_alpha100_changed_vs_exp12.csv
preds/exp20_ridge_stacking_alpha100_result.json
submissions/exp20_ridge_stacking_alpha100_submit.csv
```

### 리더보드 결과

- 제출 파일: `submissions/exp20_ridge_stacking_alpha100_submit.csv`
- Leaderboard score: 0.9503205128

EXP-20은 기존 최고였던 EXP-12/EXP-18의 0.9471153846을 넘어 0.9503205128을 기록했다. 단일 validation 기반 stacking이라 과적합 위험은 있었지만, 이번 공개 leaderboard에서는 RidgeClassifier가 모델 간 예측 차이를 효과적으로 활용한 것으로 볼 수 있다.

### 과제용 해석

RidgeClassifier stacking은 여러 딥러닝 모델의 확률과 순위 정보를 이용해 validation 성능을 크게 개선했다. 특히 FN이 9개로 줄어 의료 이미지 분류에서 중요한 recall 측면의 개선 경향을 보였다. 다만 현재 방식은 하나의 validation split에 meta model을 학습했기 때문에 과적합 가능성이 있으며, 더 신뢰성 있는 stacking을 위해서는 OOF 기반 예측을 사용하는 것이 바람직하다.

## EXP-21. LogisticRegression Stacking

### 목적

EXP-20 RidgeClassifier stacking이 최고점을 갱신했으므로, 같은 feature를 사용해 LogisticRegression stacking을 비교했다. 목적은 RidgeClassifier와 LogisticRegression의 차이가 leaderboard에 어떤 영향을 주는지 확인하는 것이다.

### 사용 모델 및 Feature

EXP-20과 동일하다.

- EXP-07 ResNet18
- EXP-09 DenseNet121
- EXP-06 EfficientNet-B0
- EXP-12 ConvNeXt-Tiny
- EXP-18 ConvNeXt-Tiny full fine-tuning

Feature는 각 모델의 `prob_1`과 `rank(prob_1)`를 사용했다.

### 설정

- meta model: `LogisticRegression`
- C: 1.0
- class_weight: `balanced`
- solver: `liblinear`
- preprocessing: `StandardScaler`

### Validation 결과

- Accuracy: 0.98755
- Precision: 0.9948
- Recall: 0.9884
- F1-score: 0.9916

### Confusion Matrix

```text
[[264   4]
 [  9 767]]
```

Validation 결과는 EXP-20 RidgeClassifier와 동일하다.

### 제출 파일 분포

```text
submissions/exp21_logistic_stacking_C1_submit.csv
label 0: 221
label 1: 403
```

전체 label 분포도 EXP-20 RidgeClassifier와 동일하지만, 실제 test 예측은 2개 다르다.

```text
test_0267.png: Ridge 1 -> Logistic 0
test_0535.png: Ridge 0 -> Logistic 1
```

### 저장 파일

```text
preds/exp21_logistic_stacking_summary.csv
preds/exp21_logistic_stacking_C1_valid_probs.csv
preds/exp21_logistic_stacking_C1_test_probs.csv
preds/exp21_logistic_stacking_C1_changed_vs_ridge.csv
preds/exp21_logistic_stacking_C1_changed_vs_exp12.csv
preds/exp21_logistic_stacking_C1_result.json
submissions/exp21_logistic_stacking_C1_submit.csv
```

### 리더보드 결과

- 제출 파일: `submissions/exp21_logistic_stacking_C1_submit.csv`
- Leaderboard score: 0.9503205128

EXP-21은 EXP-20 RidgeClassifier와 leaderboard score가 동일했다. 두 제출 파일은 test 샘플 2개에서만 예측이 달랐기 때문에, 두 샘플 중 하나씩 맞고 틀렸거나 해당 샘플들이 public score 계산에 영향을 주지 않았을 가능성이 있다.

### 과제용 해석

LogisticRegression stacking은 RidgeClassifier와 동일한 validation 성능을 기록했다. Test set의 전체 label 분포도 같지만, 개별 샘플 2개에서 예측이 달라졌다. 따라서 이 실험은 stacking 방식 자체의 차이가 공개 leaderboard에서 얼마나 영향을 주는지 확인하는 비교 실험으로 볼 수 있다.

## EXP-22. RidgeClassifier Feature V2

### 목적

EXP-20 RidgeClassifier가 최고점을 갱신했으므로 meta model은 RidgeClassifier로 유지하고 feature만 확장했다. 목적은 단순 `prob_1`과 rank 외에 모델 간 불일치 정보를 추가했을 때 성능이 개선되는지 확인하는 것이다.

### Feature

기존 feature:

- 각 모델의 `prob_1`
- 각 모델의 `rank(prob_1)`

추가 feature:

- `mean_prob`
- `std_prob`
- `max_prob`
- `min_prob`
- `range_prob`
- `convnext12 - convnext18_full`
- `convnext12 - densenet121`
- `convnext18_full - densenet121`

총 feature 수는 18개다.

### 설정

- meta model: `RidgeClassifier`
- alpha: 3000.0
- class_weight: `balanced`
- preprocessing: `StandardScaler`

### Validation 결과

- Accuracy: 0.98276
- Precision: 0.9961
- Recall: 0.9807
- F1-score: 0.9883

### Confusion Matrix

```text
[[265   3]
 [ 15 761]]
```

Validation 성능은 EXP-20 RidgeClassifier보다 낮았다. 다만 test 예측은 EXP-20과 4개만 달라 비교 실험으로 제출할 수 있는 수준이다.

### 제출 파일 분포

```text
submissions/exp22_ridge_feature_v2_alpha3000_submit.csv
label 0: 221
label 1: 403
```

EXP-20 Ridge 대비 다른 샘플은 4개다.

```text
test_0049.png: Ridge 0 -> V2 1
test_0244.png: Ridge 1 -> V2 0
test_0342.png: Ridge 1 -> V2 0
test_0539.png: Ridge 0 -> V2 1
```

### 저장 파일

```text
preds/exp22_ridge_feature_v2_summary.csv
preds/exp22_ridge_feature_v2_alpha3000_valid_scores.csv
preds/exp22_ridge_feature_v2_alpha3000_test_scores.csv
preds/exp22_ridge_feature_v2_alpha3000_changed_vs_ridge.csv
preds/exp22_ridge_feature_v2_alpha3000_changed_vs_exp12.csv
preds/exp22_ridge_feature_v2_alpha3000_result.json
submissions/exp22_ridge_feature_v2_alpha3000_submit.csv
```

### 리더보드 결과

- 제출 파일: `submissions/exp22_ridge_feature_v2_alpha3000_submit.csv`
- Leaderboard score: 0.9471153846

EXP-22는 EXP-20 RidgeClassifier보다 validation과 leaderboard가 모두 낮았다. 따라서 현재 기준에서는 mean/std/range 및 모델 간 차이 feature를 추가하는 것보다, EXP-20의 단순 `prob_1 + rank` feature 구성이 더 적절하다.

### 과제용 해석

RidgeClassifier의 feature를 확장하여 모델 평균, 분산, 최대/최소 확률, 모델 간 차이 정보를 추가했다. Validation 기준으로는 기존 RidgeClassifier보다 성능이 낮아졌으므로, feature 추가가 항상 개선으로 이어지지는 않았다. 다만 test 예측이 기존 Ridge와 4개만 다르기 때문에, 개별 샘플 판단 차이를 확인하는 비교 실험으로 의미가 있다.

## EXP-23. RidgeClassifier Feature Pruning

### 목적

EXP-22에서 파생 feature 추가가 leaderboard 하락으로 이어졌으므로, 이번에는 반대로 feature를 줄였다. 기존 EXP-20의 `prob_1 + rank` 구조는 유지하되, 모델 조합을 줄여 noise가 될 수 있는 모델을 제거하는 방향을 확인했다.

### 후보 비교 요약

여러 조합을 비교한 결과, `ConvNeXt12 + DenseNet121 + ResNet18` 조합이 EXP-20과 동일한 validation 성능을 내면서 test 예측 변경 수가 작았다. 이 조합은 EfficientNet-B0와 ConvNeXt full fine-tuning 모델을 제외한다.

### 설정

- meta model: `RidgeClassifier`
- alpha: 100.0
- class_weight: `balanced`
- preprocessing: `StandardScaler`
- 사용 모델:
  - EXP-12 ConvNeXt-Tiny
  - EXP-09 DenseNet121
  - EXP-07 ResNet18
- feature:
  - 각 모델의 `prob_1`
  - 각 모델의 `rank(prob_1)`

### Validation 결과

- Accuracy: 0.98755
- Precision: 0.9948
- Recall: 0.9884
- F1-score: 0.9916

### Confusion Matrix

```text
[[264   4]
 [  9 767]]
```

Validation 결과는 EXP-20 RidgeClassifier와 동일하다.

### 제출 파일 분포

```text
submissions/exp23_ridge_pruned_cnx_den_res_alpha100_submit.csv
label 0: 222
label 1: 402
```

EXP-20 Ridge 대비 다른 샘플은 3개다.

```text
test_0267.png: Ridge 1 -> Pruned 0
test_0371.png: Ridge 1 -> Pruned 0
test_0535.png: Ridge 0 -> Pruned 1
```

### 저장 파일

```text
preds/exp23_ridge_feature_pruning_summary.csv
preds/exp23_ridge_pruned_cnx_den_res_alpha100_valid_scores.csv
preds/exp23_ridge_pruned_cnx_den_res_alpha100_test_scores.csv
preds/exp23_ridge_pruned_cnx_den_res_alpha100_changed_vs_ridge.csv
preds/exp23_ridge_pruned_cnx_den_res_alpha100_changed_vs_exp12.csv
preds/exp23_ridge_pruned_cnx_den_res_alpha100_result.json
submissions/exp23_ridge_pruned_cnx_den_res_alpha100_submit.csv
```

### 리더보드 결과

- 제출 파일: `submissions/exp23_ridge_pruned_cnx_den_res_alpha100_submit.csv`
- Leaderboard score: 0.9487179487

EXP-23은 EXP-20 RidgeClassifier보다 낮았지만, EXP-12 ConvNeXt 단일 모델보다는 높았다. 따라서 EfficientNet-B0와 full fine-tuning ConvNeXt를 제거한 pruning 조합도 효과는 있었지만, 최고 성능에는 전체 모델을 사용한 EXP-20 구성이 더 적절했다.

### 과제용 해석

파생 feature를 추가하는 대신 모델 조합을 줄였을 때, validation 성능은 기존 RidgeClassifier와 동일하게 유지되었다. 이는 모든 모델을 사용하는 것보다 일부 모델을 제거해도 stacking 성능이 유지될 수 있음을 보여준다. 다만 test 예측은 3개 샘플에서 달라졌기 때문에 leaderboard 결과를 통해 실제 일반화 여부를 확인해야 한다.

## EXP-24. Ridge Feature Importance 분석

### 목적

EXP-20 RidgeClassifier가 현재 최고 score를 기록했기 때문에, 어떤 stacking feature가 최종 판단에 기여하는지 확인했다. 이 분석은 이후 feature pruning 또는 모델 조합 선택의 근거로 사용한다.

### 분석 대상

- 기준 모델: EXP-20 RidgeClassifier
- feature: 각 모델의 `prob_1`, `rank(prob_1)`
- 사용 모델:
  - ResNet18
  - DenseNet121
  - EfficientNet-B0
  - ConvNeXt-Tiny EXP-12
  - ConvNeXt-Tiny full fine-tuning EXP-18

### Coefficient 기준 중요도

Ridge coefficient는 `StandardScaler` 이후 값을 기준으로 비교했다.

| Feature | Coef | 해석 |
|---|---:|---|
| `convnext18_full_prob` | 0.2138 | 가장 큰 양의 계수 |
| `convnext12_prob` | 0.1990 | EXP-12 ConvNeXt 단일 모델 기여 |
| `resnet18_prob` | 0.1970 | ResNet도 강한 양의 기여 |
| `densenet121_prob` | 0.1116 | 중간 정도 기여 |
| `efficientnet_b0_prob` | 0.1062 | 단독 성능은 낮지만 stacking에서는 양의 기여 |

Rank feature의 coefficient는 전반적으로 작았다. 즉 EXP-20에서는 rank보다 각 모델의 `prob_1` 값 자체가 더 큰 역할을 했다.

### Permutation Importance 기준

Permutation importance는 feature 하나를 섞었을 때 validation 성능이 얼마나 떨어지는지 본다.

| Feature | Accuracy Importance Mean | 해석 |
|---|---:|---|
| `resnet18_prob` | 0.000785 | 가장 안정적인 성능 기여 |
| `efficientnet_b0_rank` | 0.000670 | EfficientNet의 순위 정보가 보완 역할 |
| `efficientnet_b0_prob` | 0.000498 | 단독 성능은 낮아도 일부 샘플에서 기여 |
| `convnext18_full_prob` | 0.000364 | full fine-tuning ConvNeXt도 보완 정보 제공 |
| `densenet121_rank` | 0.000153 | 작은 보완 효과 |

### 모델 단위 요약

| Model | Abs Coef Sum | Accuracy Importance Sum | 해석 |
|---|---:|---:|---|
| EfficientNet-B0 | 0.1146 | 0.001169 | 단독 성능은 낮지만 stacking 보완 정보가 있음 |
| ResNet18 | 0.2147 | 0.000441 | 강한 기본 feature 역할 |
| ConvNeXt12 | 0.2038 | 0.000153 | 주력 모델 |
| ConvNeXt18 full | 0.2276 | 0.000077 | coefficient는 크지만 permutation 효과는 작음 |
| DenseNet121 | 0.1174 | -0.000249 | validation 기준 중요도는 낮거나 불안정 |

### 해석

EXP-23에서 EfficientNet-B0와 full fine-tuning ConvNeXt를 제거했을 때 leaderboard가 하락했다. EXP-24 분석에서도 EfficientNet-B0는 단독 성능은 낮지만 permutation importance 기준으로는 보완 정보를 제공하는 것으로 나타났다. 따라서 단독 성능이 낮은 모델이라도 stacking에서는 일부 어려운 샘플을 보완할 수 있으므로, 단순히 단독 leaderboard만 보고 제거하면 안 된다.

### 저장 파일

```text
preds/exp24_ridge_feature_importance_coefficients.csv
preds/exp24_ridge_feature_importance_permutation.csv
preds/exp24_ridge_feature_importance_model_summary.csv
preds/exp24_ridge_feature_importance_result.json
```

### 과제용 해석

Stacking 단계에서 feature importance를 분석한 결과, 단독 성능이 낮았던 EfficientNet-B0도 일부 보완 정보를 제공하는 것으로 나타났다. 이는 딥러닝 모델 앙상블에서 단일 모델 성능만이 아니라 모델 간 오류 패턴과 예측 다양성이 중요함을 보여준다. 다만 importance 값은 현재 validation split에 기반하므로, 절대적인 모델 우열로 해석하기보다는 다음 실험 설계를 위한 참고 지표로 사용해야 한다.

### 다음 분석 메모

딥러닝 모델 내부 feature importance도 별도로 확인해야 한다. 이후 Grad-CAM 또는 occlusion 기반 분석을 통해 모델이 폐 영역에 반응하는지 확인한다.

## EXP-25. Ridge Alpha 미세 조정

### 목적

EXP-20 RidgeClassifier가 현재 최고 score를 기록했으므로, 같은 feature 구성을 유지한 채 alpha만 미세 조정했다. 목적은 alpha=100 주변에서 decision boundary가 바뀌어 추가 개선 후보가 생기는지 확인하는 것이다.

### 설정

- 기준 feature: EXP-20과 동일
  - 각 모델의 `prob_1`
  - 각 모델의 `rank(prob_1)`
- meta model: `RidgeClassifier`
- class_weight: `balanced`
- alpha 후보:

```text
40, 50, 60, 70, 75, 80, 90, 100, 110, 125, 150, 175, 200, 250, 300
```

### 결과 요약

Alpha를 바꿔도 test 예측은 모두 EXP-20과 동일했다.

```text
test label 0: 221
test label 1: 403
diff_vs_EXP20_Ridge: 0
```

Validation 기준으로는 alpha 100~175 구간이 가장 높았다.

```text
Validation accuracy: 0.98755
Confusion matrix:
[[264   4]
 [  9 767]]
```

### 해석

Alpha 미세 조정은 validation score에는 일부 차이를 만들었지만, 실제 test 제출 label은 EXP-20과 동일했다. 따라서 별도 제출 파일을 만들 필요가 없으며, 현재 alpha=100 설정을 유지하는 것이 충분하다.

### 저장 파일

```text
preds/exp25_ridge_alpha_finetune_summary.csv
```

### 과제용 해석

RidgeClassifier의 alpha를 조정해도 최종 test label은 바뀌지 않았다. 이는 현재 stacking decision boundary가 alpha 40~300 범위에서는 비교적 안정적이라는 의미로 해석할 수 있다. 따라서 추가 성능 개선은 alpha 조정보다는 feature 구성 또는 새로운 보완 모델 추가에서 나올 가능성이 높다.

## EXP-26. Ridge without DenseNet

### 목적

EXP-24 feature importance에서 DenseNet121의 permutation importance가 낮게 나타났다. 따라서 EXP-20 RidgeClassifier 구성에서 DenseNet121만 제거하고, 나머지 ResNet18, EfficientNet-B0, ConvNeXt12, ConvNeXt18 full fine-tuning 조합을 확인했다.

### 설정

- meta model: `RidgeClassifier`
- alpha: 30.0
- class_weight: `balanced`
- preprocessing: `StandardScaler`
- 사용 모델:
  - EXP-07 ResNet18
  - EXP-06 EfficientNet-B0
  - EXP-12 ConvNeXt-Tiny
  - EXP-18 ConvNeXt-Tiny full fine-tuning
- feature:
  - 각 모델의 `prob_1`
  - 각 모델의 `rank(prob_1)`

### Validation 결과

- Accuracy: 0.98563
- Precision: 0.9961
- Recall: 0.9845
- F1-score: 0.9903

### Confusion Matrix

```text
[[265   3]
 [ 12 764]]
```

Validation 기준으로는 EXP-20보다 낮다. 다만 test 예측은 EXP-20 Ridge와 1개만 다르다.

### 제출 파일 분포

```text
submissions/exp26_ridge_without_densenet_alpha30_submit.csv
label 0: 222
label 1: 402
```

EXP-20 Ridge 대비 다른 샘플은 1개다.

```text
test_0244.png: Ridge 1 -> without DenseNet 0
```

### 저장 파일

```text
preds/exp26_ridge_without_densenet_summary.csv
preds/exp26_ridge_without_densenet_alpha30_valid_scores.csv
preds/exp26_ridge_without_densenet_alpha30_test_scores.csv
preds/exp26_ridge_without_densenet_alpha30_changed_vs_ridge.csv
preds/exp26_ridge_without_densenet_alpha30_changed_vs_exp12.csv
preds/exp26_ridge_without_densenet_alpha30_result.json
submissions/exp26_ridge_without_densenet_alpha30_submit.csv
```

### 리더보드 결과

- 제출 파일: `submissions/exp26_ridge_without_densenet_alpha30_submit.csv`
- Leaderboard score: 0.9519230769

EXP-26은 EXP-20 대비 `test_0244.png` 하나만 `1 -> 0`으로 변경했고 leaderboard가 상승했다. 따라서 공개 test set 기준으로는 해당 샘플을 label 0으로 예측한 것이 더 적절했을 가능성이 높다. DenseNet121은 전체적으로는 유용할 수 있지만, 이 샘플에서는 label 1 방향으로 과하게 밀었을 가능성이 있다.

### 과제용 해석

DenseNet121을 제거한 Ridge stacking은 validation 성능이 EXP-20보다 낮았지만, test 예측은 단 1개만 달랐다. 이 실험은 DenseNet이 특정 샘플에서 최종 판단에 어떤 영향을 주는지 확인하는 비교 실험으로 볼 수 있다.

## EXP-27. DenseNet 영향 제한 분석

### 목적

EXP-26에서 DenseNet121을 제거하자 leaderboard가 상승했다. 이번 분석에서는 DenseNet을 완전히 제거하는 대신 `prob`, `rank`, half scale feature로 나눠 DenseNet의 어떤 feature가 최종 예측을 EXP-20 방향으로 되돌리는지 확인했다.

### 비교한 모드

- DenseNet 제거
- DenseNet `prob`만 사용
- DenseNet `rank`만 사용
- DenseNet `prob + rank` 모두 사용
- DenseNet `prob` half scale
- DenseNet `rank` half scale
- DenseNet `prob + rank` half scale

### 결과 요약

DenseNet `prob`가 포함되면 test 예측이 EXP-20과 동일해졌다.

```text
DenseNet prob 포함:
test label 0: 221
test label 1: 403
diff_vs_EXP20: 0
diff_vs_EXP26: 1
```

DenseNet을 제거하거나 `rank`만 사용하면 EXP-26과 동일한 예측이 나왔다.

```text
DenseNet 제거 또는 rank only:
test label 0: 222
test label 1: 402
diff_vs_EXP20: 1
diff_vs_EXP26: 0
```

### 해석

EXP-20과 EXP-26의 차이를 만든 핵심은 DenseNet의 `prob_1` feature였다. DenseNet의 rank 정보만으로는 `test_0244.png`를 label 1로 밀지 못했고, DenseNet probability가 들어갔을 때 EXP-20처럼 label 1로 바뀌었다. 따라서 현재 공개 leaderboard 기준으로는 DenseNet probability가 특정 샘플에서 과한 양성 신호를 준 것으로 볼 수 있다.

### 저장 파일

```text
preds/exp27_densenet_limited_ridge_summary.csv
```

### 과제용 해석

Stacking feature를 세분화해 분석한 결과, DenseNet121의 probability feature가 최종 예측에 직접적인 영향을 주는 것으로 나타났다. 반면 rank feature만 사용할 경우 현재 최고 제출인 EXP-26과 동일한 예측을 유지했다. 이는 stacking에서 단순히 모델을 포함/제외하는 것뿐 아니라, probability와 rank처럼 feature 유형별로 영향이 다를 수 있음을 보여준다.

## EXP-29. LIME XAI 분석

### 목적

Grad-CAM은 모델 내부 activation 기반 설명이고, LIME은 입력 이미지의 일부 superpixel을 가리거나 유지했을 때 예측 변화가 어떻게 나타나는지 보는 설명 방법이다. 이번 실험에서는 EXP-12 ConvNeXt-Tiny를 대상으로 disagreement 샘플에 LIME을 적용해 Grad-CAM과 비교할 수 있는 보조 시각화를 생성했다.

### 대상 샘플

```text
test_0244.png
test_0267.png
test_0371.png
test_0535.png
```

### 대상 모델

- EXP-12 ConvNeXt-Tiny

### 설정

- LIME explainer: `lime_image.LimeImageExplainer`
- perturbation samples: 600
- target class: 0, 1 각각 시각화
- 저장 위치: `reports/xai/`

### 저장 파일

```text
scripts/run_exp29_lime_convnext.py
reports/xai/lime_convnext_disagreement_comparison.png
reports/xai/lime_convnext_disagreement_summary.csv
```

### 해석 주의점

LIME은 superpixel 단위로 이미지를 나누어 설명하므로, X-ray처럼 경계가 명확하지 않은 이미지에서는 분할 방식에 따라 결과가 달라질 수 있다. 따라서 LIME 결과는 단독 근거로 사용하지 않고, Grad-CAM과 함께 모델이 어떤 영역에 반응했는지 보조적으로 확인하는 용도로 사용한다.

### 과제용 해석

ConvNeXt-Tiny의 판단 근거를 Grad-CAM뿐 아니라 LIME으로도 확인했다. LIME은 이미지 일부 영역을 perturbation했을 때 예측이 어떻게 바뀌는지 기반으로 설명을 제공한다. 다만 의료 영상에서는 superpixel 경계가 실제 병변 경계와 일치한다고 볼 수 없으므로, 결과는 “해당 영역이 예측에 영향을 주었다” 정도로 신중하게 해석해야 한다.

## EXP-30. Grad-CAM + LIME 비교 리포트

### 목적

EXP-28 Grad-CAM과 EXP-29 LIME 결과를 개별 이미지 파일로만 두지 않고, disagreement 샘플별로 원본 이미지, 모델별 확률, stacking label, Grad-CAM, LIME을 한 번에 볼 수 있도록 정리했다.

### 대상 샘플

```text
test_0244.png
test_0267.png
test_0371.png
test_0535.png
```

### 포함 내용

- 원본 X-ray 이미지
- EXP-12, EXP-20, EXP-26 label 비교
- ResNet18, DenseNet121, EfficientNet-B0, ConvNeXt12, ConvNeXt18 probability 비교
- Grad-CAM 결과
- LIME 결과
- 샘플별 해석 메모

### 저장 파일

```text
reports/xai/xai_comparison_panel.png
reports/xai/xai_comparison_summary.csv
reports/xai/xai_comparison_report.md
```

### 핵심 관찰

- `test_0244.png`: EXP-20 Ridge는 1, EXP-26 best는 0으로 판단했고, EXP-26 제출에서 점수가 상승했다. 공개 leaderboard 기준으로는 0 판단이 더 적절했을 가능성이 높다.
- `test_0267.png`, `test_0371.png`: EXP-12 ConvNeXt는 0 쪽이지만 EXP-18 full fine-tuning은 1 쪽으로 강하게 이동했다. full fine-tuning이 일부 샘플에서 양성 방향 신호를 크게 키운 것으로 보인다.
- `test_0535.png`: ConvNeXt 계열은 0에 가깝지만 stacking 모델 간 판단 차이가 있었던 샘플이다.

### 과제용 해석

Grad-CAM과 LIME을 함께 사용하여 모델의 예측 근거를 비교했다. 두 방법은 서로 다른 원리로 설명을 생성하기 때문에 결과가 완전히 일치하지 않을 수 있다. 따라서 특정 영역을 실제 병변이라고 단정하지 않고, 모델이 해당 영역에 더 반응했거나 해당 영역이 예측에 영향을 주었다는 수준으로 해석했다.

## EXP-31. 다중 모델 Grad-CAM 비교

### 목적

EXP-26에서 DenseNet을 제거하자 `test_0244.png`의 최종 label이 1에서 0으로 바뀌고 leaderboard score가 상승했다. 따라서 이번 분석에서는 ResNet18, DenseNet121, EfficientNet-B0, ConvNeXt-Tiny가 같은 disagreement 샘플에서 어느 영역에 반응하는지 Grad-CAM으로 비교했다.

### 대상 샘플

```text
test_0244.png
test_0267.png
test_0371.png
test_0535.png
```

### 대상 모델

- EXP-07 ResNet18
- EXP-09 DenseNet121
- EXP-06 EfficientNet-B0
- EXP-12 ConvNeXt-Tiny

### 저장 파일

```text
scripts/run_exp31_multimodel_gradcam.py
reports/xai/gradcam_multimodel_disagreement_comparison.png
reports/xai/gradcam_multimodel_disagreement_summary.csv
```

### 해석 방향

각 모델은 같은 이미지를 보더라도 서로 다른 feature map과 inductive bias를 사용한다. 따라서 Grad-CAM 반응 영역이 다를 수 있으며, 이는 stacking에서 모델 다양성이 생기는 원인 중 하나로 볼 수 있다. 다만 Grad-CAM은 모델의 반응 영역을 보여주는 도구일 뿐 실제 병변 위치를 확정하는 근거는 아니다.

### 과제용 해석

다중 모델 Grad-CAM 비교를 통해 ResNet, DenseNet, EfficientNet, ConvNeXt가 같은 X-ray 이미지에서도 서로 다른 영역에 반응할 수 있음을 확인했다. 이는 stacking이 단일 모델보다 좋은 성능을 보인 이유를 설명하는 보조 근거가 될 수 있다. 다만 의료 영상 해석에서는 Grad-CAM 결과를 진단 근거로 단정하지 않고, 모델이 특정 영역에 상대적으로 더 반응했다는 수준에서 해석해야 한다.

## EXP-32. Validation FP/FN XAI 후보 분석

### 목적

test set은 정답을 알 수 없기 때문에 XAI 결과를 해석할 때 실제 오류인지 확인할 수 없다. 이번 분석에서는 정답이 있는 validation set에서 EXP-20 Ridge stacking과 현재 최고 제출인 EXP-26 Ridge without DenseNet의 FP/FN을 비교해, 이후 Grad-CAM/LIME을 적용할 후보 샘플을 선정했다.

### 저장 파일

```text
scripts/run_exp32_validation_error_xai.py
reports/xai/validation_error_summary.csv
reports/xai/validation_error_counts.json
reports/xai/validation_error_grid.png
```

### 결과 요약

| 모델 | FP | FN | 해석 |
|---|---:|---:|---|
| EXP-20 Ridge stacking | 4 | 9 | validation 기준 총 오류 13개 |
| EXP-26 Ridge without DenseNet | 3 | 12 | FP는 1개 줄었지만 FN은 3개 증가 |

EXP-26은 leaderboard score가 `0.9519230769`로 EXP-20보다 높았지만, validation 기준으로는 FN이 증가했다. 따라서 EXP-26이 무조건 더 안전한 모델이라고 단정하기보다는, public test 분포에서 특정 샘플 판단이 유리하게 작동했을 가능성을 함께 고려해야 한다.

### 주요 샘플

```text
train_0770.png
train_1329.png
train_0841.png
train_1229.png
train_1831.png
train_4685.png
train_3253.png
train_1381.png
train_2325.png
train_4170.png
```

`train_1229.png`는 EXP-20에서 FP였지만 EXP-26에서는 정상 분류로 바뀐 샘플이다. 반대로 `train_2325.png`, `train_4170.png`는 EXP-20에서는 맞췄지만 EXP-26에서는 FN으로 바뀐 샘플이다. 이런 샘플은 단순 점수 비교보다 모델별 확률과 XAI 반응 영역을 함께 확인할 가치가 있다.

### 과제용 해석

Validation 오분류 분석 결과, leaderboard 점수가 더 높은 EXP-26도 validation 기준에서는 FN이 증가하는 경향을 보였다. 의료 이미지 분류에서는 실제 양성을 음성으로 예측하는 FN이 중요한 오류이므로, 단순 accuracy뿐 아니라 FP/FN 양상과 모델별 확률 차이를 함께 확인해야 한다. 이 분석은 이후 Grad-CAM/LIME으로 모델이 어떤 영역에 반응했는지 확인할 후보를 선정하는 기준으로 사용할 수 있다.

## EXP-33. Validation 판단 변경 샘플 XAI

### 목적

EXP-20과 EXP-26의 validation 판단이 서로 달라진 샘플을 대상으로 Grad-CAM/LIME을 확인했다. test set과 달리 validation set은 정답을 알 수 있으므로, 모델 판단 변화가 실제로 FP를 줄였는지 또는 FN을 늘렸는지 해석하기 더 적합하다.

### 대상 샘플

```text
train_1229.png: label 0, EXP-20 FP, EXP-26 OK
train_2325.png: label 1, EXP-20 OK, EXP-26 FN
train_4170.png: label 1, EXP-20 OK, EXP-26 FN
```

### 저장 파일

```text
scripts/run_exp33_validation_changed_xai.py
reports/xai/validation_changed_xai_panel.png
reports/xai/validation_changed_xai_summary.csv
```

### 결과 요약

- `train_1229.png`는 ResNet, DenseNet, EfficientNet, ConvNeXt 단일 모델이 모두 label 1 쪽으로 예측했지만, EXP-26 stacking은 최종적으로 label 0을 선택해 FP를 줄였다.
- `train_2325.png`, `train_4170.png`는 ResNet, DenseNet, EfficientNet이 label 1 쪽 확률을 높게 보였지만, ConvNeXt12는 label 0 쪽으로 강하게 기울었다.
- 따라서 EXP-26의 개선은 일부 FP를 줄이는 데 유리했지만, ConvNeXt 계열이 음성 쪽으로 강하게 작동하는 샘플에서는 FN 증가 가능성이 있다.

### 과제용 해석

Validation 판단 변경 샘플을 XAI로 확인한 결과, stacking 모델의 최종 판단은 단일 모델의 단순 다수결과 다르게 나타날 수 있었다. 특히 일부 샘플에서는 ConvNeXt 계열의 음성 판단이 강하게 반영되어 FN이 발생했다. Grad-CAM/LIME 결과는 실제 병변 위치를 확정하는 근거가 아니라, 모델이 어떤 영역에 더 반응했는지 확인하는 보조 해석 도구로 사용했다.

## EXP-34. EXP-26 Ridge Threshold Search

### 목적

EXP-26 Ridge without DenseNet의 decision score 기준선을 조정해 validation 기준 FP/FN 균형을 확인했다. 기본 기준은 `decision_score >= 0`이면 label 1이다.

### 저장 파일

```text
scripts/run_exp34_ridge_threshold_search.py
preds/exp34_ridge_threshold_summary.csv
submissions/exp34_ridge_threshold_m0p20_submit.csv
submissions/exp34_ridge_threshold_m0p15_submit.csv
submissions/exp34_ridge_threshold_m0p02_submit.csv
submissions/exp34_ridge_threshold_m0p06_submit.csv
```

### 주요 결과

| threshold | Validation Accuracy | FP | FN | Recall | test 변경 수 | 메모 |
|---:|---:|---:|---:|---:|---:|---|
| -0.20 | 0.98946 | 4 | 7 | 0.99098 | 5 | validation 최고, recall 개선 |
| -0.15 | 0.98851 | 4 | 8 | 0.98969 | 4 | recall 개선, 변경 수 4 |
| -0.06 | 0.98563 | 3 | 12 | 0.98454 | 3 | valid metric은 EXP-26과 동일 |
| -0.02 | 0.98563 | 3 | 12 | 0.98454 | 1 | `test_0539.png`만 0->1 변경, leaderboard 하락 |
| 0.00 | 0.98563 | 3 | 12 | 0.98454 | 0 | EXP-26 기본 |
| 0.02 | 0.98563 | 3 | 12 | 0.98454 | 1 | 더 보수적, label 1 감소 |

### 해석

validation만 보면 threshold를 낮추는 `-0.20`이 가장 좋다. FN이 12개에서 7개로 줄어 의료 이미지 분류 관점에서는 더 안전해 보일 수 있다. 그러나 test에서는 `test_0244.png`가 다시 0에서 1로 바뀐다. 이 샘플은 EXP-26에서 0으로 바뀌며 leaderboard 점수가 상승했던 핵심 샘플이므로, public score 기준으로는 위험한 후보일 수 있다.

실제 제출은 가장 안전한 threshold 하향 후보인 `submissions/exp34_ridge_threshold_m0p02_submit.csv`로 진행했다. 이 후보는 `test_0539.png` 1개만 `0 -> 1`로 바꿨지만 leaderboard score가 `0.9503205128`로 하락했다. 따라서 EXP-26 기준에서 threshold를 낮춰 label 1을 늘리는 방향은 public test 기준으로 유리하지 않은 것으로 판단한다.

### 제출 우선순위

실제 제출 결과 `m0p02`도 하락했기 때문에 추가 threshold 하향 제출 우선순위는 낮춘다.

1. EXP-26 기본 제출 유지
   - 현재 최고 점수 `0.9519230769`.
2. `submissions/exp34_ridge_threshold_m0p15_submit.csv`
   - validation 개선은 크지만 `test_0244.png`를 다시 1로 바꾸므로 public 위험이 큼.
3. `submissions/exp34_ridge_threshold_m0p20_submit.csv`
   - validation 최고지만 가장 공격적인 후보라 제출 우선순위는 낮음.

### 과제용 해석

Threshold 조정은 모델 재학습 없이 FP/FN 균형을 조절하는 방법이다. 이번 실험에서는 threshold를 낮출수록 validation FN이 감소했지만, public test에서 이미 유리하게 작동한 샘플이 다시 바뀔 위험도 확인했다. 따라서 validation 성능만으로 최종 제출을 결정하기보다는, 변경 샘플의 성격과 기존 leaderboard 결과를 함께 고려해야 한다.

`m0p02` 제출 결과, test 예측을 1개만 바꾼 보수적인 threshold 하향도 leaderboard에서 하락했다. 이는 현재 public set에서는 EXP-26의 기존 기준선이 더 적절했음을 시사한다.

## EXP-35. ConvNeXt 영향 제한

### 목적

EXP-33에서 validation 일부 샘플의 FN이 ConvNeXt 계열의 음성 판단과 연결될 수 있음을 확인했다. 따라서 EXP-35에서는 EXP-26의 기본 방향을 유지하되 ConvNeXt 계열 feature의 영향을 제한하는 후보를 비교했다.

### 비교 설정

- `base_rebuilt`: ResNet18, EfficientNet-B0, ConvNeXt12, ConvNeXt18 full의 prob/rank 사용
- `conv_rank_only`: ConvNeXt 계열 probability 제거, rank만 유지
- `conv12_rank_only`: ConvNeXt12 probability만 제거
- `conv18_rank_only`: ConvNeXt18 full probability만 제거
- `without_conv12`: ConvNeXt12 제거
- `without_conv18`: ConvNeXt18 full 제거
- `res_eff_only`: ResNet18 + EfficientNet-B0만 사용

### 저장 파일

```text
scripts/run_exp35_convnext_influence_limit.py
preds/exp35_convnext_influence_limit_summary.csv
submissions/exp35_base_rebuilt_alpha300_submit.csv
submissions/exp35_base_rebuilt_alpha30_submit.csv
submissions/exp35_conv18_rank_only_alpha100_submit.csv
submissions/exp35_without_conv18_alpha100_submit.csv
```

### 주요 결과

| 후보 | alpha | Validation Accuracy | FP | FN | test 변경 수 | 메모 |
|---|---:|---:|---:|---:|---:|---|
| base_rebuilt | 1000 | 0.99042 | 10 | 0 | 28 | validation recall 최고지만 test 변경 과다 |
| base_rebuilt | 300 | 0.99042 | 4 | 6 | 6 | validation 개선, `test_0244` 0->1 |
| conv18_rank_only | 300 | 0.98851 | 6 | 6 | 7 | ConvNeXt18 prob 제거, `test_0244` 0->1 |
| without_conv18 | 300 | 0.98851 | 6 | 6 | 7 | ConvNeXt18 제거, `test_0244` 0->1 |
| base_rebuilt | 30 | 0.98755 | 4 | 9 | 1 | `test_0244` 0->1 |

### 해석

ConvNeXt 영향을 제한하거나 Ridge regularization을 강하게 조정하면 validation FN은 줄어드는 경향이 있었다. 그러나 public에서 중요하게 보이는 `test_0244.png`를 다시 label 1로 바꾸는 후보가 많았다. EXP-26은 이 샘플을 0으로 바꾸며 leaderboard 점수가 상승했기 때문에, EXP-35 후보들은 validation만 보고 제출하기에는 위험하다.

### 결론

EXP-35에서는 EXP-26을 대체할 만한 제출 후보를 찾지 못했다. 특히 `base_rebuilt alpha30`은 test 변경이 1개뿐이지만 그 1개가 `test_0244.png`라서 EXP-34와 같은 위험을 가진다. 따라서 현재 최고 제출은 계속 EXP-26으로 유지한다.

### 과제용 해석

ConvNeXt feature의 영향을 제한하면 validation set에서는 FN 감소가 나타날 수 있었지만, public test에서 성능 개선에 기여했던 샘플 판단이 되돌아가는 문제가 있었다. 이는 validation 성능과 public leaderboard 성능이 항상 같은 방향으로 움직이지 않음을 보여준다. 따라서 stacking feature 조정은 단순 validation accuracy뿐 아니라 변경 샘플의 성격까지 함께 확인해야 한다.

## EXP-39. ConvNeXt-Tiny Seed123 추가 학습

### 목적

TTA와 threshold 조정은 public leaderboard에서 확실한 개선을 만들지 못했다. 따라서 기존 강한 계열인 ConvNeXt-Tiny의 seed diversity를 추가로 확보해, 이후 stacking 후보로 사용할 probability를 저장했다.

### 설정

- 모델: `ConvNeXt-Tiny`
- pretrained weight: ImageNet-1K
- 학습 범위: 마지막 stage + classifier
- split seed: 42 유지
- 학습 seed: 123
- epoch: 6
- batch size: 32
- optimizer: AdamW
- learning rate: `5e-5`
- loss: weighted CrossEntropyLoss

### 결과

| 항목 | 값 |
|---|---:|
| best epoch | 4 |
| validation accuracy | 0.97893 |
| precision | 0.99346 |
| recall | 0.97809 |
| F1 | 0.98571 |
| confusion matrix | `[[263, 5], [17, 759]]` |
| test label 0 | 215 |
| test label 1 | 409 |
| EXP-26 대비 test 변경 | 19개 |

### 저장 파일

```text
scripts/run_exp39_convnext_seed123.py
models/exp39_convnext_tiny_seed123_best.pt
preds/exp39_convnext_tiny_seed123_valid_probs.csv
preds/exp39_convnext_tiny_seed123_test_probs.csv
preds/exp39_convnext_tiny_seed123_summary.csv
preds/exp39_convnext_tiny_seed123_changed_vs_exp26.csv
submissions/exp39_convnext_tiny_seed123_submit.csv
```

### 해석

단독 validation accuracy는 EXP-12 ConvNeXt-Tiny보다 낮다. 또한 EXP-26 대비 `test_0025.png`, `test_0539.png`를 모두 `0 -> 1`로 바꾼다. 이 두 샘플은 최근 public 제출에서 하락과 연결된 민감 샘플이므로, EXP-39 단독 제출 우선순위는 낮다.

다만 목적은 단독 제출이 아니라 seed diversity 확보다. 따라서 EXP-39의 probability는 다음 stacking 실험에서 후보 feature로 사용한다.

### 과제용 해석

동일한 ConvNeXt-Tiny 구조라도 학습 seed가 달라지면 validation 성능과 test 예측이 달라질 수 있다. 이번 seed123 모델은 단독 성능은 기존 최고 ConvNeXt보다 낮았지만, 다른 seed 모델과 다른 예측 패턴을 제공하므로 stacking diversity 확보 관점에서는 활용 가치가 있다.

## EXP-40. Swin-Tiny 추가 실험

### 목적

지금까지는 ResNet, DenseNet, EfficientNet, ConvNeXt처럼 CNN 또는 CNN 성격이 강한 모델 위주로 실험했다. EXP-40에서는 다른 inductive bias를 가진 Transformer 계열 모델인 Swin-Tiny를 추가해, 단독 성능과 stacking 보완 feature 가능성을 확인했다.

### 설정

- 모델: `torchvision.models.swin_t`
- pretrained weight: ImageNet-1K
- 입력 크기: 224 x 224
- 학습 범위: head-only
- epoch: 4
- batch size: 24
- optimizer: AdamW
- learning rate: head `5e-5`
- loss: weighted CrossEntropyLoss

처음에는 마지막 Swin stage와 head를 함께 fine-tuning하려 했으나, 현재 MPS 환경에서 첫 epoch 출력까지 3분 이상 걸려 실험 효율이 낮았다. 따라서 빠른 probe 목적으로 backbone을 고정하고 classification head만 학습했다.

### 저장 파일

```text
scripts/run_exp40_swin_tiny.py
models/exp40_swin_tiny_best.pt
preds/exp40_swin_tiny_result.json
preds/exp40_swin_tiny_valid_probs.csv
preds/exp40_swin_tiny_test_probs.csv
submissions/exp40_swin_tiny_submit.csv

scripts/run_exp40_swin_stacking_probe.py
preds/exp40_swin_stacking_probe_summary.csv
```

### 단독 모델 결과

| Metric | 값 |
|---|---:|
| Validation Accuracy | 0.89943 |
| Precision | 0.98273 |
| Recall | 0.88015 |
| F1 | 0.92862 |
| Confusion Matrix | `[[256, 12], [93, 683]]` |
| Test label count | `{0: 235, 1: 389}` |

Swin-Tiny head-only는 precision은 높지만 recall이 낮고 FN이 93개로 많았다. 따라서 단독 제출 후보로는 적절하지 않다.

### Stacking 추가 확인

Swin-Tiny의 probability/rank를 EXP-26 계열 Ridge stacking 후보에 추가했다. 그러나 상위 후보들은 validation 성능이 좋아 보여도 대부분 `test_0244.png`를 다시 label 1로 바꿨다.

| 후보 | alpha | Validation Accuracy | FP | FN | test 변경 수 | test_0244 |
|---|---:|---:|---:|---:|---:|---:|
| no_den_plus_swin | 1000 | 0.99042 | 10 | 0 | 27 | 1 |
| no_den_plus_swin | 300 | 0.99042 | 4 | 6 | 6 | 1 |
| no_den_plus_swin | 30 | 0.98755 | 4 | 9 | 1 | 1 |

### 결론

Swin-Tiny head-only는 현재 단계에서 단독 제출 후보가 아니며, stacking에 추가해도 EXP-26을 대체할 만한 안정적인 후보를 만들지 못했다. 특히 `test_0244.png`가 1로 바뀌는 패턴이 반복되어 public leaderboard 관점에서는 위험하다.

### 과제용 해석

Transformer 계열인 Swin-Tiny를 추가해 기존 CNN 계열 모델과 다른 특징을 활용할 수 있는지 확인했다. 그러나 head-only 학습만으로는 validation 성능이 낮았고, stacking에 추가해도 기존 최고 모델을 안정적으로 개선하지 못했다. 이는 모델 구조가 다양하더라도, 충분한 fine-tuning과 데이터 규모가 뒷받침되지 않으면 성능 개선으로 이어지지 않을 수 있음을 보여준다.

## EXP-36. test_0244 고정 Stacking 탐색

### 목적

EXP-34, EXP-35, EXP-40에서 validation 성능이 좋아 보여도 `test_0244.png`를 다시 label 1로 바꾸는 후보는 public leaderboard 기준으로 위험했다. 따라서 EXP-36에서는 `test_0244.png`를 반드시 label 0으로 유지하는 조건을 걸고, EXP-26 대비 변경 수가 작은 stacking 후보를 탐색했다.

### 필터 기준

- `test_0244_pred == 0`
- EXP-26 대비 test 변경 수 1~5개
- validation accuracy가 EXP-26 이상

### 저장 파일

```text
scripts/run_exp36_fixed_0244_stacking_search.py
preds/exp36_fixed_0244_stacking_summary.csv
submissions/exp36_fixed0244_core_prob_only_alpha100_submit.csv
preds/exp36_fixed0244_core_prob_only_alpha100_changed_vs_exp26.csv
```

### 주요 후보

| 후보 | alpha | Validation Accuracy | FP | FN | test 변경 수 | test_0244 | 변경 샘플 |
|---|---:|---:|---:|---:|---:|---:|---|
| core_prob_only | 100 | 0.98946 | 4 | 7 | 1 | 0 | `test_0025: 0 -> 1` |
| efficientnet_b0+convnext18_full | 150 | 0.98563 | 6 | 9 | 5 | 0 | `test_0025`, `test_0046`, `test_0049`, `test_0104`, `test_0410` |
| drop_resnet18_all_core_rank | 150 | 0.98563 | 4 | 11 | 4 | 0 | `test_0049`, `test_0342`, `test_0377`, `test_0539` |

### 1순위 제출 후보

```text
submissions/exp36_fixed0244_core_prob_only_alpha100_submit.csv
```

이 후보는 ResNet18, EfficientNet-B0, ConvNeXt12, ConvNeXt18 full의 `prob_1`만 사용하고 rank feature는 제외했다. EXP-26 대비 `test_0025.png` 하나만 label 0에서 1로 바뀌며, `test_0244.png`는 0으로 유지된다. validation accuracy는 `0.98946`으로 EXP-26의 `0.98563`보다 높고, FN도 12개에서 7개로 감소했다.

### 해석

rank feature를 제외하고 probability만 사용한 후보가 가장 안정적인 low-change 후보로 나타났다. 이는 rank feature가 일부 경계 샘플을 불필요하게 뒤집었을 가능성을 시사한다. 다만 public test에서 `test_0025.png`의 실제 정답을 알 수 없기 때문에, 리더보드 제출 전에는 EXP-26을 기준 모델로 유지하고 이 후보를 1순위 실험 제출로 본다.

### 과제용 해석

기존 public 결과에서 중요하게 보였던 샘플을 고정한 상태로 stacking 후보를 탐색했다. 그 결과 rank feature를 제외한 probability-only Ridge 후보가 validation 성능을 개선하면서도 test 변경 수를 1개로 제한했다. 이는 stacking feature를 많이 추가하는 것보다, 신뢰도가 높은 확률 정보만 사용하는 것이 더 안정적일 수 있음을 보여준다.

## EXP-37. TTA Probability 생성

### 목적

재학습 없이 기존 강한 모델의 예측 확률을 안정화하기 위해 Test-Time Augmentation을 적용했다. 첫 실험에서는 의료 이미지에서 과한 변환을 피하고, `original + horizontal flip`만 사용했다.

### 대상 모델

- EXP-07 ResNet18
- EXP-06 EfficientNet-B0
- EXP-12 ConvNeXt-Tiny
- EXP-18 ConvNeXt-Tiny full fine-tuning

### 저장 파일

```text
scripts/run_exp37_tta_probs.py
preds/exp37_tta_summary.csv
preds/exp37_tta_resnet18_valid_probs.csv
preds/exp37_tta_resnet18_test_probs.csv
preds/exp37_tta_efficientnet_b0_valid_probs.csv
preds/exp37_tta_efficientnet_b0_test_probs.csv
preds/exp37_tta_convnext12_valid_probs.csv
preds/exp37_tta_convnext12_test_probs.csv
preds/exp37_tta_convnext18_full_valid_probs.csv
preds/exp37_tta_convnext18_full_test_probs.csv
```

### 결과 요약

| 모델 | 기존 Valid Acc | TTA Valid Acc | 기존 Recall | TTA Recall | test 변경 수 |
|---|---:|---:|---:|---:|---:|
| ResNet18 | 0.98659 | 0.98563 | 0.99098 | 0.98969 | 11 |
| EfficientNet-B0 | 0.97605 | 0.97510 | 0.97938 | 0.98067 | 30 |
| ConvNeXt12 | 0.98084 | 0.98372 | 0.97938 | 0.98196 | 6 |
| ConvNeXt18 full | 0.97989 | 0.97989 | 0.97680 | 0.97680 | 3 |

### 해석

TTA가 모든 모델에 일관되게 좋은 것은 아니었다. ConvNeXt12는 validation accuracy와 recall이 함께 개선됐지만, ResNet18과 EfficientNet-B0는 accuracy가 소폭 하락했다. 따라서 TTA를 무조건 전체 모델에 적용하기보다는 모델별로 선택적으로 쓰는 것이 적절하다.

## EXP-38. TTA 기반 Ridge Stacking

### 목적

EXP-37에서 생성한 TTA probability를 stacking feature로 사용해 EXP-26을 넘는 후보가 있는지 확인했다.

### 저장 파일

```text
scripts/run_exp38_tta_stacking.py
preds/exp38_tta_stacking_summary.csv
submissions/exp38_tta_all_prob_only_alpha50_submit.csv
submissions/exp38_tta_all_prob_only_alpha75_submit.csv
preds/exp38_tta_all_prob_only_alpha50_changed_vs_exp26.csv
```

### 주요 후보

| 후보 | alpha | Validation Accuracy | FP | FN | test 변경 수 | test_0244 | 변경 샘플 |
|---|---:|---:|---:|---:|---:|---:|---|
| tta_all prob_only | 50 | 0.98851 | 4 | 8 | 2 | 0 | `test_0025: 0->1`, `test_0342: 1->0` |
| tta_all prob_only | 75 | 0.98851 | 4 | 8 | 2 | 0 | `test_0025: 0->1`, `test_0342: 1->0` |

### 해석

EXP-38 후보는 `test_0244.png`를 0으로 유지하면서 validation 성능도 EXP-26보다 높았다. 다만 EXP-36에서 `test_0025.png`를 1로 바꾸는 후보가 public score 하락으로 이어졌고, EXP-38도 같은 방향으로 `test_0025.png`를 1로 바꿨다. 실제 제출 결과 leaderboard score는 `0.9487179487`로 하락했다. 함께 바뀐 `test_0342.png`의 `1 -> 0` 변경은 `test_0025.png` 손실을 상쇄하지 못한 것으로 보인다.

### 과제용 해석

TTA는 일부 모델, 특히 ConvNeXt12의 확률 안정화에는 도움이 됐다. 그러나 TTA probability를 stacking에 넣었을 때 public에서 이미 위험 신호를 보였던 샘플이 다시 바뀌는 문제가 있었다. 따라서 TTA는 유용한 보조 기법이지만, 모든 모델에 일괄 적용하거나 validation 성능만 보고 제출을 결정하는 것은 위험하다.

실제 제출 결과도 이 해석을 뒷받침했다. `test_0244.png`를 유지하더라도 `test_0025.png`를 `0 -> 1`로 바꾸는 후보는 public leaderboard에서 하락했다. 따라서 이후 후보 탐색에서는 `test_0244`뿐 아니라 `test_0025`도 민감 샘플로 관리한다.

## EXP-41. 5-fold OOF 설계

### 목적

현재까지의 Ridge/Logistic stacking은 하나의 validation split을 기준으로 meta-model을 학습했다. 이 방식은 빠르지만 validation split에 과적합될 수 있다. EXP-41에서는 5-fold OOF 구조를 설계해, 각 train 샘플을 해당 샘플을 학습에 사용하지 않은 모델로 예측하는 기반을 마련했다.

### Fold 설정

- split: `StratifiedKFold`
- n_splits: 5
- shuffle: True
- random_state: 42
- 저장 파일: `oof/folds_seed42.csv`

### Fold 분포

| fold | label 0 | label 1 | total | label 1 ratio |
|---:|---:|---:|---:|---:|
| 0 | 269 | 775 | 1044 | 0.74234 |
| 1 | 268 | 775 | 1043 | 0.74305 |
| 2 | 268 | 775 | 1043 | 0.74305 |
| 3 | 268 | 775 | 1043 | 0.74305 |
| 4 | 268 | 775 | 1043 | 0.74305 |

### 저장 파일

```text
oof/folds_seed42.csv
oof/folds_seed42_summary.csv
oof/oof_plan.md
scripts/run_exp41_oof_design_check.py
```

### OOF prediction 포맷

단일 모델 OOF probability:

```text
file_name,label,fold,prob_0,prob_1,pred_label
```

fold별 test probability:

```text
file_name,fold,prob_0,prob_1,pred_label
```

fold 평균 test probability:

```text
file_name,prob_0,prob_1,pred_label
```

### Meta-model 방향

OOF feature table이 만들어지면 다음 순서로 meta-model을 비교한다.

1. RidgeClassifier `prob1` only
2. RidgeClassifier `prob1 + rank`
3. LogisticRegression
4. LightGBM 또는 TabM

TabM은 단일 validation stacking에서는 과적합 위험이 크므로, OOF feature가 만들어진 뒤 비교한다.

### 다음 실행 단계

1. ConvNeXt-Tiny OOF 학습 스크립트 작성
2. fold 0 smoke test
3. fold 0 OOF/test probability 포맷 검증
4. 5 folds 전체 학습
5. OOF Ridge/Logistic baseline 생성

### 과제용 해석

OOF는 train 데이터를 여러 fold로 나누고, 각 샘플을 해당 샘플을 학습하지 않은 모델로 예측해 meta-model 학습용 feature를 만드는 방식이다. 단일 validation stacking보다 과적합 위험을 줄일 수 있으며, Ridge/Logistic/TabM 같은 meta-model을 더 공정하게 비교할 수 있다.

## EXP-42. EfficientNetV2-S 단일 split

### 목적

기존 EfficientNet-B0보다 큰 EfficientNetV2-S를 사용해 새 CNN 계열의 diversity를 확보할 수 있는지 확인했다. 첫 시도에서는 계산량을 줄이기 위해 마지막 feature block과 classifier만 학습했다.

### 설정

- 모델: `torchvision.models.efficientnet_v2_s(weights=IMAGENET1K_V1)`
- 학습 범위: `features[7] + classifier`
- trainable params: 332,802 / 20,180,050
- epoch: 6
- optimizer: AdamW
- lr: backbone last block `1e-5`, classifier `5e-5`
- weight decay: `1e-4`
- loss: Weighted CrossEntropyLoss
- augmentation: resize 224, horizontal flip, rotation 5도
- split seed: 42
- train seed: 42

### 결과

| metric | value |
|---|---:|
| best epoch | 5 |
| validation accuracy | 0.87261 |
| precision | 0.98346 |
| recall | 0.84278 |
| f1 | 0.90770 |
| confusion matrix | `[[257, 11], [122, 654]]` |
| test label 0 | 288 |
| test label 1 | 336 |
| EXP-26 대비 test 변경 수 | 114 |

민감 샘플 예측:

```text
test_0244: 1
test_0025: 1
test_0539: 0
```

### 저장 파일

```text
scripts/run_exp42_efficientnet_v2_s.py
models/exp42_efficientnet_v2_s_best.pt
preds/exp42_efficientnet_v2_s_valid_probs.csv
preds/exp42_efficientnet_v2_s_test_probs.csv
preds/exp42_efficientnet_v2_s_result.json
preds/exp42_efficientnet_v2_s_summary.csv
preds/exp42_efficientnet_v2_s_changed_vs_exp26.csv
submissions/exp42_efficientnet_v2_s_submit.csv
```

### 해석

EfficientNetV2-S는 이번 제한 fine-tuning 설정에서 pneumonia class recall이 크게 낮아졌고, FN이 122개로 증가했다. 의료 이미지 분류 관점에서는 정상으로 놓친 샘플이 많아지는 방향이므로 단독 제출 후보로 적합하지 않다.

또한 public에서 위험 신호였던 `test_0244`, `test_0025`를 모두 1로 바꿨다. 따라서 현재 설정의 EfficientNetV2-S probability는 stacking에 바로 넣기보다 제외하는 편이 안전하다. 다시 시도한다면 `features[6] + features[7] + classifier`처럼 더 넓은 fine-tuning을 하거나, 우선 RegNetY 계열로 diversity를 확보하는 것이 낫다.

### 과제용 해석

EfficientNetV2-S는 더 큰 pretrained CNN 구조이지만, 마지막 block만 학습한 설정에서는 validation accuracy가 0.87261로 낮았고 FN이 크게 증가했다. 이는 모델 크기 자체보다 fine-tuning 범위와 데이터 적응이 중요하다는 점을 보여준다. 따라서 이번 결과만으로 EfficientNetV2-S가 부적합하다고 단정하기보다는, 현재 학습 설정에서는 제출 후보로 사용하기 어렵다고 해석하는 것이 적절하다.

## EXP-43. RegNetY-3.2GF 단일 split

### 목적

RegNetY 계열을 추가해 ResNet, DenseNet, EfficientNet, ConvNeXt와 다른 CNN 구조의 diversity를 확보할 수 있는지 확인했다. 원래 후보는 RegNetY-4GF였지만 현재 torchvision에는 4GF weight가 없어서 가장 가까운 `RegNetY-3.2GF`를 사용했다.

### 설정

- 모델: `torchvision.models.regnet_y_3_2gf(weights=IMAGENET1K_V1)`
- 학습 범위: `trunk_output.block4 + fc`
- trainable params: 4,806,794 / 17,926,364
- epoch: 6
- optimizer: AdamW
- lr: final block `1e-5`, fc `5e-5`
- weight decay: `1e-4`
- loss: Weighted CrossEntropyLoss
- augmentation: resize 224, horizontal flip, rotation 5도
- split seed: 42
- train seed: 42

### 결과

| metric | value |
|---|---:|
| best epoch | 6 |
| validation accuracy | 0.95211 |
| precision | 0.99054 |
| recall | 0.94459 |
| f1 | 0.96702 |
| confusion matrix | `[[261, 7], [43, 733]]` |
| test label 0 | 301 |
| test label 1 | 323 |
| EXP-26 대비 test 변경 수 | 83 |

민감 샘플 예측:

```text
test_0244: 0
test_0025: 0
test_0539: 0
```

### 저장 파일

```text
scripts/run_exp43_regnet_y_3_2gf.py
models/exp43_regnet_y_3_2gf_best.pt
preds/exp43_regnet_y_3_2gf_valid_probs.csv
preds/exp43_regnet_y_3_2gf_test_probs.csv
preds/exp43_regnet_y_3_2gf_result.json
preds/exp43_regnet_y_3_2gf_summary.csv
preds/exp43_regnet_y_3_2gf_changed_vs_exp26.csv
submissions/exp43_regnet_y_3_2gf_submit.csv
```

### 해석

RegNetY-3.2GF는 EXP-42 EfficientNetV2-S보다 훨씬 안정적으로 학습되었지만, validation accuracy 0.95211은 기존 최고 단독 모델인 ConvNeXt-Tiny나 Ridge stacking보다 낮다. confusion matrix 기준 FN은 43개로 EXP-42의 122개보다 크게 줄었지만, 여전히 단독 제출 후보로 보기에는 부족하다.

긍정적인 점은 public에서 민감하게 관리하던 `test_0244`, `test_0025`, `test_0539`를 모두 0으로 유지했다는 것이다. 따라서 단독 제출보다는 EXP-26/ConvNeXt 계열과 결합하는 stacking diversity 후보로 평가하는 것이 적절하다.

### 과제용 해석

RegNetY-3.2GF는 EfficientNetV2-S보다 validation 성능과 FN 측면에서 더 안정적인 결과를 보였다. 그러나 기존 최고 모델들에 비해 전체 accuracy는 낮으므로 단독 분류 모델로 채택하기보다는, 서로 다른 CNN 구조의 예측 차이를 활용하는 앙상블 또는 stacking 후보로 보는 것이 타당하다. 의료 이미지 분류에서는 accuracy뿐 아니라 FN 증가 여부를 함께 확인해야 하며, 이번 실험에서는 FN이 EXP-42보다 줄었다는 점이 의미 있다.

## EXP-44. RegNetY 포함 Ridge stacking

### 목적

EXP-43 RegNetY-3.2GF 단독 성능은 낮았지만, 민감 샘플을 모두 0으로 유지했다. 따라서 단독 제출이 아니라 EXP-26 stacking feature에 RegNetY 정보를 추가했을 때 보완 정보로 작동하는지 확인했다.

### 설정

- meta-model: RidgeClassifier
- 기준 후보: EXP-26 Ridge without DenseNet
- 사용 feature: `efficientnet_b0`, `convnext12`, `convnext18_full`, `regnet_y_3_2gf`의 `prob + rank`
- 제외 feature: `resnet18`
- alpha: 150
- 제출 파일: `submissions/exp44_regnet_drop_resnet18_add_reg_alpha150_submit.csv`

### 결과

| metric | value |
|---|---:|
| validation accuracy | 0.98659 |
| precision | 0.99609 |
| recall | 0.98582 |
| f1 | 0.99093 |
| confusion matrix | `[[265, 3], [11, 765]]` |
| EXP-26 대비 test 변경 수 | 5 |

민감 샘플 예측:

```text
test_0244: 0
test_0025: 0
test_0539: 0
```

EXP-26 대비 변경 샘플:

```text
test_0049: 0 -> 1
test_0060: 1 -> 0
test_0172: 1 -> 0
test_0342: 1 -> 0
test_0504: 1 -> 0
```

### 저장 파일

```text
scripts/run_exp44_regnet_stacking_search.py
preds/exp44_regnet_stacking_summary.csv
preds/exp44_regnet_drop_resnet18_add_reg_alpha150_test_scores.csv
preds/exp44_regnet_drop_resnet18_add_reg_alpha150_changed_vs_exp26.csv
submissions/exp44_regnet_drop_resnet18_add_reg_alpha150_submit.csv
```

### 해석

EXP-44는 validation accuracy가 EXP-26의 0.98563보다 약간 높은 0.98659이며, 민감 샘플 3개를 모두 0으로 유지했다. 변경 수가 5개로 과하지 않고, 기존에 위험했던 `test_0025`, `test_0539`를 건드리지 않는 점은 긍정적이다.

하지만 실제 leaderboard score는 0.9503205128로 EXP-26의 0.9519230769보다 낮았다. 민감 샘플 3개를 유지해도 나머지 5개 변경이 public 정답과 맞지 않았을 가능성이 있다. 특히 이 결과는 validation 점수의 소폭 개선만으로 public 성능 향상을 단정하기 어렵다는 점을 다시 보여준다.

### 과제용 해석

RegNetY 단독 모델은 성능이 낮았지만, stacking에서는 일부 보완 정보를 제공할 가능성을 보였다. 특히 validation 기준으로 EXP-26보다 소폭 높은 성능을 보였고, public에서 민감하게 관리하던 샘플의 예측을 유지했다. 다만 test 정답은 알 수 없으므로, 이 결과는 성능 개선 가능성을 보인 후보로 해석하고 실제 일반화 성능은 leaderboard 제출 결과로 확인해야 한다.

## EXP-45. ConvNeXt OOF fold0 smoke test

### 목적

5-fold OOF 전체 학습을 바로 돌리기 전에, ConvNeXt-Tiny로 fold 0 하나만 학습해 OOF pipeline이 정상 작동하는지 확인했다. 이 실험의 목적은 leaderboard 제출이 아니라 fold split, 저장 파일 포맷, row 수, 학습 시간을 검증하는 것이다.

### 설정

- 모델: ConvNeXt-Tiny ImageNet pretrained
- 학습 범위: 마지막 stage `features[-1]` + classifier
- fold: 0
- train size: 4,172
- valid size: 1,044
- epoch: 6
- optimizer: AdamW
- lr: `5e-5`
- loss: Weighted CrossEntropyLoss
- augmentation: resize 224, horizontal flip, rotation 7도

### 결과

| metric | value |
|---|---:|
| best epoch | 6 |
| validation accuracy | 0.98467 |
| precision | 0.99608 |
| recall | 0.98323 |
| f1 | 0.98961 |
| confusion matrix | `[[266, 3], [13, 762]]` |
| elapsed sec | 433.2 |
| fold당 예상 시간 | 약 7.2분 |
| 5-fold 예상 시간 | 약 36분 |

### 포맷 검증

```text
oof rows: 1044
oof duplicated file_name: 0
oof fold value: {0: 1044}
test rows: 624
test duplicated file_name: 0
test fold value: {0: 624}
```

민감 샘플 예측:

```text
test_0244: 0, prob_1=0.10542
test_0025: 0, prob_1=0.12887
test_0539: 1, prob_1=0.94032
```

### 저장 파일

```text
scripts/run_exp45_convnext_oof_fold0_smoke.py
oof/models/exp45_convnext_tiny_fold0.pt
oof/preds/exp45_convnext_tiny_fold0_oof_probs.csv
oof/preds/exp45_convnext_tiny_fold0_test_probs.csv
oof/preds/exp45_convnext_tiny_fold0_result.json
oof/preds/exp45_convnext_tiny_fold0_summary.csv
oof/submissions/exp45_convnext_tiny_fold0_submit.csv
```

### 해석

OOF fold0 smoke test는 정상적으로 통과했다. OOF probability 파일은 fold0 validation 샘플 1,044개를 중복 없이 포함했고, test probability 파일도 624개 샘플을 정상 저장했다. 따라서 다음 단계에서는 fold 1~4까지 같은 구조로 확장해 ConvNeXt-Tiny 5-fold OOF probability를 완성할 수 있다.

주의할 점은 fold0 단일 결과가 기존 단일 validation 결과와 직접 비교 가능한 최종 성능은 아니라는 점이다. OOF의 목적은 모든 train 샘플에 대해 out-of-fold prediction을 만들어 meta-model의 과적합을 줄이는 것이다.

### 과제용 해석

OOF smoke test를 통해 5-fold 학습 구조가 정상적으로 작동하는지 확인했다. fold0 기준 validation accuracy는 0.98467이었고, OOF probability와 test probability가 기대한 row 수로 저장되었다. 이는 이후 Ridge/Logistic/TabM meta-model을 단일 validation이 아닌 OOF 기반 feature로 학습하기 위한 준비 단계이다.

## EXP-46. ConvNeXt 5-fold OOF 완성

### 목적

EXP-45에서 fold0 smoke test가 통과했으므로, fold1~4를 추가로 학습해 ConvNeXt-Tiny 5-fold OOF probability를 완성했다. 이 결과는 이후 OOF Ridge/Logistic/TabM meta-model의 첫 입력 feature로 사용한다.

### 설정

- 모델: ConvNeXt-Tiny ImageNet pretrained
- 학습 범위: 마지막 stage `features[-1]` + classifier
- folds: 0~4
- epoch: 각 fold 6
- optimizer: AdamW
- lr: `5e-5`
- loss: Weighted CrossEntropyLoss
- augmentation: resize 224, horizontal flip, rotation 7도

### Fold별 결과

| fold | best epoch | valid accuracy | precision | recall | f1 | confusion matrix |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 6 | 0.98467 | 0.99608 | 0.98323 | 0.98961 | `[[266, 3], [13, 762]]` |
| 1 | 6 | 0.97700 | 0.99474 | 0.97419 | 0.98436 | - |
| 2 | 5 | 0.97028 | 0.98819 | 0.97161 | 0.97983 | - |
| 3 | 5 | 0.98082 | 0.98839 | 0.98581 | 0.98710 | - |
| 4 | 5 | 0.97700 | 0.99474 | 0.97419 | 0.98436 | - |

### OOF 병합 결과

| metric | value |
|---|---:|
| OOF rows | 5,216 |
| OOF duplicate file_name | 0 |
| test by fold rows | 3,120 |
| test mean rows | 624 |
| OOF accuracy | 0.97795 |
| OOF precision | 0.99240 |
| OOF recall | 0.97781 |
| OOF f1 | 0.98505 |
| OOF confusion matrix | `[[1312, 29], [86, 3789]]` |
| test mean label 0 | 227 |
| test mean label 1 | 397 |

민감 샘플 test mean:

```text
test_0244: pred=0, prob_1=0.42179
test_0025: pred=0, prob_1=0.35498
test_0539: pred=1, prob_1=0.84474
```

### 저장 파일

```text
scripts/run_exp46_convnext_oof_remaining_folds.py
oof/preds/exp46_convnext_tiny_oof_probs.csv
oof/preds/exp46_convnext_tiny_test_probs_by_fold.csv
oof/preds/exp46_convnext_tiny_test_probs_mean.csv
oof/preds/exp46_convnext_tiny_fold_summary.csv
oof/preds/exp46_convnext_tiny_combined_result.json
oof/submissions/exp46_convnext_tiny_mean_submit.csv
```

### 해석

ConvNeXt-Tiny 5-fold OOF 결과는 전체 train 5,216개 샘플에 대해 out-of-fold prediction을 완성했다는 점이 핵심이다. OOF accuracy는 0.97795로 단일 validation 최고값보다 낮지만, 이는 각 샘플을 해당 샘플을 학습하지 않은 모델로 예측했기 때문에 더 보수적인 추정에 가깝다.

test mean 예측은 `test_0244`, `test_0025`를 0으로 유지했고 `test_0539`는 1로 예측했다. `test_0539`는 과거 EXP-34에서 0->1 변경이 하락했던 민감 샘플이므로, OOF mean 단독 제출은 바로 추천하지 않는다. 다음 단계는 이 OOF feature를 기준으로 Ridge/Logistic OOF meta-model을 만드는 것이다.

### 과제용 해석

5-fold OOF는 각 학습 샘플을 해당 샘플을 학습하지 않은 모델로 예측하게 하므로, 단일 validation 기반 stacking보다 과적합 위험을 줄일 수 있다. 이번 실험에서는 ConvNeXt-Tiny의 OOF probability를 전체 train set 5,216개에 대해 생성했고, OOF accuracy는 0.97795였다. 이는 최종 제출 성능을 직접 보장하는 값은 아니지만, 이후 meta-model 학습을 더 공정하게 수행하기 위한 기반 데이터로 의미가 있다.

## EXP-47. OOF Ridge baseline

### 목적

EXP-46에서 만든 ConvNeXt 5-fold OOF probability를 사용해 OOF 기반 meta-model 파이프라인을 검증했다. 아직 OOF feature가 ConvNeXt 하나뿐이므로 본격적인 multi-model stacking이라기보다는 OOF Ridge baseline 성격이다.

### 설정

- 입력: `oof/preds/exp46_convnext_tiny_oof_probs.csv`
- test 입력: `oof/preds/exp46_convnext_tiny_test_probs_mean.csv`
- meta-model: RidgeClassifier
- alpha 탐색: `0.3~3000`
- feature 후보:
  - `prob_1`
  - `prob_1 + rank`
  - `prob_0 + prob_1 + rank`
  - `prob_1 + margin`
  - `prob_1 + rank + margin + uncertainty`

### 기준 direct ConvNeXt OOF mean

| metric | value |
|---|---:|
| OOF accuracy | 0.97795 |
| precision | 0.99240 |
| recall | 0.97781 |
| f1 | 0.98505 |
| confusion matrix | `[[1312, 29], [86, 3789]]` |
| test label 0 | 227 |
| test label 1 | 397 |
| test_0244 | 0 |
| test_0025 | 0 |
| test_0539 | 1 |

### 최고 OOF Ridge 후보

| spec | alpha | OOF accuracy | precision | recall | f1 | test label 0 | test label 1 | 민감 샘플 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `prob1_only` | 1000 | 0.98025 | 0.98759 | 0.98581 | 0.98670 | 194 | 430 | 0244=1, 0025=1, 0539=1 |
| `prob1_margin` | 3000 | 0.98006 | 0.98987 | 0.98323 | 0.98654 | 205 | 419 | 0244=1, 0025=1, 0539=1 |
| `prob1_rank_margin_unc` | 30 | 0.98006 | 0.99012 | 0.98297 | 0.98653 | 207 | 417 | 0244=1, 0025=1, 0539=1 |

민감 샘플을 유지하는 가장 높은 후보:

| spec | alpha | OOF accuracy | precision | recall | f1 | test label 0 | test label 1 | 민감 샘플 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `prob1_margin` | 1000 | 0.97872 | 0.99164 | 0.97961 | 0.98559 | 218 | 406 | 0244=0, 0025=0, 0539=1 |

### 저장 파일

```text
scripts/run_exp47_oof_ridge_baseline.py
oof/preds/exp47_oof_ridge_baseline_summary.csv
oof/submissions/exp47_prob1_only_alpha1000_submit.csv
oof/submissions/exp47_prob1_margin_alpha3000_submit.csv
oof/submissions/exp47_prob1_rank_alpha1000_submit.csv
oof/submissions/exp47_prob1_rank_margin_unc_alpha30_submit.csv
oof/submissions/exp47_prob1_rank_margin_unc_alpha3000_submit.csv
oof/submissions/exp47_prob1_margin_alpha1000_submit.csv
```

### 해석

OOF Ridge는 direct ConvNeXt mean보다 OOF accuracy를 0.97795에서 0.98025까지 올렸다. 하지만 최고 후보들은 `test_0244`와 `test_0025`를 모두 1로 바꾼다. 두 샘플은 이전 public 결과에서 0 유지가 더 안전했던 샘플이므로, 최고 OOF score 후보를 바로 제출하기는 위험하다.

민감 샘플을 유지하는 후보는 `prob1_margin alpha1000`이며 OOF accuracy는 0.97872다. direct보다 약간 높지만 여전히 `test_0539`는 1로 유지한다. 따라서 EXP-47은 제출용 실험이라기보다 OOF Ridge 파이프라인 검증 성공으로 보는 것이 맞다. 다음 단계에서는 OOF feature에 ResNet18 또는 EfficientNet-B0를 추가해 multi-model OOF stacking으로 확장해야 한다.

### 과제용 해석

ConvNeXt 단일 OOF probability만으로 RidgeClassifier를 학습했을 때 OOF accuracy는 소폭 개선되었다. 그러나 test 민감 샘플의 예측이 불리한 방향으로 바뀌는 후보가 많아, validation 또는 OOF 점수만으로 제출 성능을 단정하기 어렵다. 이 실험은 OOF 기반 meta-model 구조가 작동함을 확인한 baseline이며, 실제 성능 개선을 위해서는 서로 다른 모델의 OOF feature를 추가하는 것이 필요하다.

## EXP-48. ResNet18 5-fold OOF

### 목적

ConvNeXt OOF만으로는 모델 다양성이 부족하므로, 기존에 강했던 ResNet18 layer4+fc 구조를 5-fold OOF로 확장했다.

### 설정

- 모델: ImageNet pretrained ResNet18
- 학습 범위: `layer4 + fc`
- fold: `oof/folds_seed42.csv`의 5-fold stratified split
- epoch: fold별 6
- optimizer: `AdamW(lr=1e-4, weight_decay=1e-4)`
- augmentation: resize 224, horizontal flip, rotation 7도
- loss: weighted CrossEntropyLoss

### 결과

| metric | value |
|---|---:|
| OOF accuracy | 0.98524 |
| precision | 0.99044 |
| recall | 0.98968 |
| f1 | 0.99006 |
| confusion matrix | `[[1304, 37], [40, 3835]]` |
| test label 0 | 184 |
| test label 1 | 440 |
| test_0244 | 1 |
| test_0025 | 1 |
| test_0539 | 1 |

### 저장 파일

```text
scripts/run_exp48_resnet18_oof.py
oof/preds/exp48_resnet18_oof_probs.csv
oof/preds/exp48_resnet18_test_probs_mean.csv
oof/preds/exp48_resnet18_combined_result.json
oof/submissions/exp48_resnet18_mean_submit.csv
```

### 해석

ResNet18 OOF는 ConvNeXt OOF보다 accuracy와 recall이 높아 강한 OOF feature가 되었다. 하지만 test 평균 예측이 `test_0244`, `test_0025`, `test_0539`를 모두 1로 바꾼다. 이 세 샘플은 기존 public 결과 기준 0 유지가 더 안전했던 민감 샘플이므로, ResNet18 OOF 단독 제출은 보류한다. 핵심 가치는 단독 제출이 아니라 OOF stacking의 보완 feature다.

## EXP-49. ConvNeXt + ResNet OOF stacking

### 목적

EXP-46 ConvNeXt OOF와 EXP-48 ResNet18 OOF를 결합해 OOF 기반 Ridge/Logistic meta-model을 검증했다. meta-model도 fold별 CV로 평가해, OOF feature 전체에 fit한 training score만 보지 않도록 했다.

### 설정

- 입력 모델: ConvNeXt-Tiny OOF, ResNet18 OOF
- meta-model: RidgeClassifier, LogisticRegression
- feature: `prob_1`, rank, margin, uncertainty, 모델 간 통계 feature
- 평가: `fold` 기준 meta-CV
- 제출 기준: `test_0244`, `test_0025`, `test_0539` 유지 여부와 EXP-26 대비 변경 수 확인

### 주요 결과

| 후보 | meta-CV accuracy | precision | recall | f1 | test label 0/1 | EXP-26 대비 변경 | 민감 샘플 |
|---|---:|---:|---:|---:|---:|---:|---|
| `prob_only ridge alpha=1` | 0.98735 | 0.99301 | 0.98994 | 0.99147 | 189/435 | 35 | 0244=1, 0025=0, 0539=1 |
| `prob_margin ridge alpha=3` | 0.98716 | 0.99301 | 0.98968 | 0.99134 | 189/435 | 35 | 0244=1, 0025=0, 0539=1 |

### 저장 파일

```text
scripts/run_exp49_oof_convnext_resnet_stacking.py
oof/preds/exp49_convnext_resnet_oof_stacking_summary.csv
oof/submissions/exp49_ridge_prob_only_1_submit.csv
```

### 해석

ConvNeXt+ResNet 조합은 meta-CV accuracy가 높지만, 상위 후보들이 `test_0539`를 1로 유지하고 `test_0244`도 1로 바꾸는 경향이 있었다. 이전 public 결과에서 `test_0539`의 0->1 변경은 하락과 연결되었으므로, EXP-49 후보는 제출 우선순위가 낮다. 다음 단계에서는 `test_0539`를 0으로 강하게 잡는 EfficientNet-B0 OOF를 추가하는 것이 타당하다.

## EXP-50. EfficientNet-B0 5-fold OOF

### 목적

기존 EXP-06 EfficientNet-B0가 단독 leaderboard는 낮았지만 stacking에서 보완 정보를 제공했으므로, 이를 5-fold OOF로 확장했다.

### 설정

- 모델: ImageNet pretrained EfficientNet-B0
- 초기 시도: `features[7:] + classifier`만 학습
- 초기 결과: fold0 best accuracy 0.9301로 낮아 중단
- 최종 설정: `features[5:] + classifier` 학습
- optimizer: AdamW
- LR: feature `5e-5`, classifier `1e-4`
- epoch: fold별 6

### 결과

| metric | value |
|---|---:|
| OOF accuracy | 0.98505 |
| precision | 0.99248 |
| recall | 0.98735 |
| f1 | 0.98991 |
| confusion matrix | `[[1312, 29], [49, 3826]]` |
| test label 0 | 245 |
| test label 1 | 379 |
| test_0244 | 0 |
| test_0025 | 0 |
| test_0539 | 0 |

### 저장 파일

```text
scripts/run_exp50_efficientnet_b0_oof.py
oof/preds/exp50_efficientnet_b0_oof_probs.csv
oof/preds/exp50_efficientnet_b0_test_probs_mean.csv
oof/preds/exp50_efficientnet_b0_combined_result.json
oof/submissions/exp50_efficientnet_b0_mean_submit.csv
```

### 해석

EfficientNet-B0 OOF는 ResNet18보다 recall은 낮지만 precision이 높고, 민감 샘플 3개를 모두 0으로 유지했다. 이는 EXP-49에서 문제가 된 ResNet의 과한 label 1 경향을 보정할 수 있는 좋은 보완 정보다. 단독 제출보다는 ConvNeXt와 결합한 OOF stacking 후보로 보는 것이 맞다.

## EXP-51. ConvNeXt + ResNet + EfficientNet OOF stacking

### 목적

EXP-46, EXP-48, EXP-50의 OOF probability를 결합해 multi-model OOF stacking 후보를 만들었다. 전체 3모델뿐 아니라 ResNet 제외, EfficientNet 제외 등 subset도 함께 비교했다.

### 설정

- 입력 모델: ConvNeXt-Tiny OOF, ResNet18 OOF, EfficientNet-B0 OOF
- meta-model: RidgeClassifier, LogisticRegression
- feature: probability, rank, margin, uncertainty, direct prediction, subset별 통계
- 평가: fold 기준 meta-CV
- 후보 필터:
  - `test_0244=0`, `test_0025=0`, `test_0539=0`
  - test label 1 개수 380~420
  - EXP-26 대비 변경 수 1~24

### 주요 후보

| 후보 파일 | meta-CV accuracy | precision | recall | f1 | FP/FN | test label 0/1 | EXP-26 대비 변경 | 민감 샘플 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `oof/submissions/exp51_ridge_cnx_eff_prob_only_50_submit.csv` | 0.98677 | 0.99506 | 0.98710 | 0.99106 | 19/50 | 230/394 | 8 | 0244=0, 0025=0, 0539=0 |
| `oof/submissions/exp51_logistic_cnx_eff_prob_rank_0p3_submit.csv` | 0.98639 | 0.99198 | 0.98968 | 0.99083 | 31/40 | 226/398 | 4 | 0244=0, 0025=0, 0539=0 |
| `oof/submissions/exp51_ridge_cnx_eff_prob_only_300_submit.csv` | 0.98658 | 0.99326 | 0.98865 | 0.99095 | 26/44 | 228/396 | 6 | 0244=0, 0025=0, 0539=0 |

### 저장 파일

```text
scripts/run_exp51_oof_3model_stacking.py
oof/preds/exp51_oof_3model_stacking_summary.csv
oof/submissions/exp51_ridge_cnx_eff_prob_only_50_submit.csv
oof/submissions/exp51_logistic_cnx_eff_prob_rank_0p3_submit.csv
oof/submissions/exp51_ridge_cnx_eff_prob_only_300_submit.csv
```

### 해석

EXP-51에서 가장 안정적인 흐름은 ResNet을 제외한 ConvNeXt+EfficientNet 조합이었다. ResNet18은 OOF 성능은 강하지만 test 민감 샘플을 모두 1로 밀어 올리는 경향이 있었고, EfficientNet-B0는 민감 샘플을 0으로 유지했다. 따라서 EXP-51의 제출 후보는 단순히 meta-CV accuracy가 가장 높은 후보만 보지 말고, EXP-26 대비 변경 수와 민감 샘플 유지 여부를 함께 기준으로 삼아야 한다.

제출 우선순위는 보수적으로는 `exp51_logistic_cnx_eff_prob_rank_0p3_submit.csv`, 공격적으로는 `exp51_ridge_cnx_eff_prob_only_50_submit.csv`가 적합하다. 전자는 EXP-26 대비 변경이 4개로 적고 recall이 높으며, 후자는 meta-CV accuracy가 가장 높지만 EXP-26 대비 8개를 모두 1->0으로 바꾼다.

### 리더보드 결과

| 제출 파일 | Leaderboard Score | 기존 EXP-26 대비 | 해석 |
|---|---:|---:|---|
| `exp51_logistic_cnx_eff_prob_rank_0p3_submit.csv` | 0.9551282051 | +0.0032051282, +2샘플 | 최고점 갱신 |
| `exp51_ridge_cnx_eff_prob_only_50_submit.csv` | 0.9519230769 | 동점 | meta-CV는 더 높았지만 public 개선 없음 |

EXP-51 결과는 OOF 기반 stacking 방향이 타당했음을 보여준다. 다만 meta-CV accuracy가 가장 높았던 공격 후보보다, EXP-26 대비 변경 수가 4개로 적었던 Logistic 후보가 public에서 더 좋았다. 따라서 다음 실험에서도 OOF 점수 최고 후보를 바로 제출하기보다, 변경 수와 민감 샘플 유지 여부를 함께 보는 전략을 유지한다.

### 과제용 해석

5-fold OOF 기반 stacking은 각 모델의 train set 전체 예측을 더 공정하게 meta-model에 제공한다는 장점이 있다. 이번 실험에서는 ConvNeXt와 EfficientNet-B0 조합이 민감 샘플을 안정적으로 유지하면서 meta-CV accuracy 0.98677까지 도달했다. 다만 public leaderboard에서는 1~2개 샘플 차이로 순위가 바뀔 수 있으므로, OOF 점수만으로 최종 우위를 단정하지 않고 EXP-26 대비 변경 샘플을 함께 확인해야 한다.

## EXP-52. EXP-51 low-change search

### 목적

EXP-51 보수 후보가 leaderboard 최고점 `0.9551282051`을 기록했으므로, 같은 ConvNeXt+EfficientNet OOF 조합 주변에서 변경 수를 더 작게 제한한 후보를 탐색했다. 새 학습은 하지 않고 Ridge/Logistic regularization과 decision threshold만 조정했다.

### 설정

- 기준 최고 후보: `exp51_logistic_cnx_eff_prob_rank_0p3_submit.csv`
- 입력 feature:
  - `cnx_prob1 + eff_prob1`
  - `cnx_prob1 + eff_prob1 + cnx_rank + eff_rank`
  - `cnx_prob1 + eff_prob1 + margin`
- meta-model:
  - RidgeClassifier
  - LogisticRegression
- threshold: `-0.10 ~ 0.25`
- 필터:
  - `test_0244=0`, `test_0025=0`, `test_0539=0`
  - EXP-26 대비 변경 2~8개
  - EXP-51 최고 후보와 완전 중복 제외

### 주요 후보

| 후보 파일 | meta-CV accuracy | recall | FP/FN | test label 0/1 | EXP-26 대비 변경 | EXP-51 대비 변경 | 변경 샘플 |
|---|---:|---:|---:|---:|---:|---:|---|
| `oof/submissions/exp52_logistic_cnx_eff_prob_only_p0p25_t0p0_submit.csv` | 0.98658 | 0.99045 | 33/37 | 227/397 | 5 | 1 | `test_0490`: 1->0 |
| `oof/submissions/exp52_ridge_cnx_eff_prob_only_p500_t0p025_submit.csv` | 0.98677 | 0.98942 | 28/41 | 227/397 | 5 | 1 | `test_0490`: 1->0 |
| `oof/submissions/exp52_ridge_cnx_eff_prob_margin_p500_tm0p025_submit.csv` | 0.98677 | 0.98736 | 20/49 | 228/396 | 6 | 2 | `test_0201`, `test_0490`: 1->0 |
| `oof/submissions/exp52_ridge_cnx_eff_prob_margin_p300_t0p0_submit.csv` | 0.98677 | 0.98710 | 19/50 | 230/394 | 8 | 4 | `test_0141`, `test_0201`, `test_0222`, `test_0490`: 1->0 |

### 저장 파일

```text
scripts/run_exp52_low_change_search.py
oof/preds/exp52_low_change_summary.csv
oof/submissions/exp52_logistic_cnx_eff_prob_only_p0p25_t0p0_submit.csv
oof/submissions/exp52_ridge_cnx_eff_prob_only_p500_t0p025_submit.csv
oof/submissions/exp52_ridge_cnx_eff_prob_margin_p500_tm0p025_submit.csv
```

### 해석

EXP-52의 가장 보수적인 후보는 현재 최고 EXP-51에서 `test_0490` 하나만 1에서 0으로 바꾼다. 실제 제출 결과 leaderboard score가 `0.9567307692`로 상승했으며, EXP-51 대비 정확히 `1/624` 개선되었다. 따라서 public 기준에서는 `test_0490=0` 방향이 맞았을 가능성이 높다.

2순위 후보인 `exp52_ridge_cnx_eff_prob_margin_p500_tm0p025_submit.csv`는 `test_0201`과 `test_0490`을 함께 1에서 0으로 바꿨고 leaderboard score가 `0.9551282051`로 하락했다. 따라서 이후 low-change 탐색에서는 `test_0490=0`은 유지하되, `test_0201=1`을 보호 조건으로 둔다.

제출 우선순위는 `exp52_logistic_cnx_eff_prob_only_p0p25_t0p0_submit.csv`가 1순위다. 같은 예측을 내는 중복 파일이 있으므로 하나만 제출하면 된다.

## EXP-53. EXP-52 constraint low-change search

### 목적

EXP-52 1순위 후보가 새 최고점이 되었으므로, 이를 기준선으로 삼아 추가로 1~3개 샘플만 바꾸는 후보를 찾았다. 단, public 결과상 유리했던 방향과 불리했던 방향을 guardrail로 반영해 보호 샘플을 더 엄격하게 설정했다.

### 설정

- 기준 최고 후보: `oof/submissions/exp52_logistic_cnx_eff_prob_only_p0p25_t0p0_submit.csv`
- 기준 leaderboard score: `0.9567307692`
- 입력 feature:
  - `cnx_prob1 + eff_prob1`
  - `cnx_prob1 + eff_prob1 + cnx_rank + eff_rank`
  - `cnx_prob1 + eff_prob1 + cnx_margin + eff_margin`
- meta-model:
  - RidgeClassifier
  - LogisticRegression
- 보호 샘플:
  - `test_0025=0`
  - `test_0539=0`
  - `test_0244=0`
  - `test_0490=0`
  - `test_0201=1`
- 필터:
  - EXP-52 winner 대비 변경 1~3개
  - test label 1 개수 395~399
  - meta-CV accuracy가 EXP-52 winner 대비 0.001 이상 낮으면 제외

### 결과

| 항목 | 값 |
|---|---:|
| 전체 탐색 row 수 | 1012 |
| 보호 샘플 5개 유지 | 통과 후보 있음 |
| label band 395~399 | 통과 후보 있음 |
| EXP-52 winner 대비 1~3개 변경 | 통과 후보 없음 |
| 최종 candidate 수 | 0 |

### 저장 파일

```text
scripts/run_exp53_constraint_low_change_search.py
oof/preds/exp53_constraint_low_change_summary.csv
```

### 해석

EXP-53에서는 보호 샘플 5개를 유지하면서 EXP-52 winner와 다른 제출 후보가 나오지 않았다. 즉, 현재 ConvNeXt+EfficientNet OOF 조합 안에서는 추가 low-change 제출 후보를 억지로 만들기보다 새 정보원을 확보하는 편이 낫다. 다음 단계는 ConvNeXt 계열 seed OOF 추가 학습이다.

## EXP-54. ConvNeXt seed123 5-fold OOF

### 목적

EXP-53에서 현재 ConvNeXt+EfficientNet OOF 조합 안의 추가 low-change 후보가 없었기 때문에, 새 정보원 확보를 위해 ConvNeXt-Tiny seed123 5-fold OOF를 추가로 생성했다. 모델 구조와 학습 설정은 EXP-46과 동일하게 유지하고 seed만 바꿔 diversity를 확보하는 것이 목적이다.

### 설정

- 모델: ImageNet pretrained ConvNeXt-Tiny
- 학습 범위: `features[-1] + classifier`
- seed: 123
- folds: 5-fold, `oof/folds_seed42.csv` 사용
- epoch: 6
- batch size: 32
- optimizer: AdamW
- learning rate: 5e-5
- loss: Weighted CrossEntropyLoss

### 결과

| 항목 | 값 |
|---|---:|
| OOF accuracy | 0.98102 |
| OOF precision | 0.99064 |
| OOF recall | 0.98374 |
| OOF F1 | 0.98718 |
| OOF confusion matrix | `[[1305, 36], [63, 3812]]` |
| test label 0/1 | 221 / 403 |

보호 샘플 확인:

| sample | pred | prob_1 |
|---|---:|---:|
| `test_0244` | 0 | 0.47937 |
| `test_0025` | 0 | 0.44476 |
| `test_0539` | 1 | 0.91007 |
| `test_0490` | 1 | 0.59618 |
| `test_0201` | 1 | 0.67760 |

### 저장 파일

```text
scripts/run_exp54_convnext_seed123_oof.py
oof/preds/exp54_convnext_tiny_seed123_oof_probs.csv
oof/preds/exp54_convnext_tiny_seed123_test_probs_mean.csv
oof/submissions/exp54_convnext_tiny_seed123_mean_submit.csv
```

### 해석

EXP-54 단독 mean 제출은 `test_0539`와 `test_0490`을 1로 예측한다. 특히 `test_0490=0`은 EXP-52에서 public score 개선과 연결되었으므로, EXP-54 단독 제출은 우선순위가 낮다. 다만 OOF accuracy가 EXP-46보다 높고 seed가 다르므로 stacking feature로는 의미가 있다.

## EXP-55. ConvNeXt seed OOF stacking

### 목적

EXP-54 seed123 ConvNeXt OOF를 기존 ConvNeXt/ EfficientNet-B0 OOF feature에 추가했을 때, 현재 최고 EXP-52보다 더 나은 low-change 후보가 생기는지 확인했다.

### 설정

- 입력 feature:
  - 기존 ConvNeXt OOF
  - ConvNeXt seed123 OOF
  - EfficientNet-B0 OOF
- meta-model:
  - RidgeClassifier
  - LogisticRegression
- threshold 탐색: decision score 기준 `-0.10 ~ 0.20`
- 보호 샘플:
  - `test_0025=0`
  - `test_0539=0`
  - `test_0244=0`
  - `test_0490=0`
  - `test_0201=1`
- 후보 조건:
  - 현재 최고 EXP-52 대비 변경 1~8개
  - 우선 후보는 변경 1~3개
  - test label 1 개수 395~399
  - meta-CV accuracy floor 0.98558 이상

### 주요 후보

| 후보 파일 | meta-CV accuracy | recall | FP/FN | test label 0/1 | 현재 최고 대비 변경 | 변경 샘플 |
|---|---:|---:|---:|---:|---:|---|
| `oof/submissions/exp55_ridge_cnx123_eff_prob_rank_p50_tm0p1_submit.csv` | 0.98677 | 0.98994 | 30/39 | 228/396 | 1 | `test_0222`: 1->0 |
| `oof/submissions/exp55_ridge_cnx_eff_compact_p3000_t0p025_submit.csv` | 0.98658 | 0.98968 | 30/40 | 228/396 | 1 | `test_0504`: 1->0 |

### 저장 파일

```text
scripts/run_exp55_convnext_seed_stacking.py
oof/preds/exp55_convnext_seed_stacking_summary.csv
oof/submissions/exp55_ridge_cnx123_eff_prob_rank_p50_tm0p1_submit.csv
oof/submissions/exp55_ridge_cnx_eff_compact_p3000_t0p025_submit.csv
```

### 해석

EXP-55 1순위 `exp55_ridge_cnx123_eff_prob_rank_p50_tm0p1_submit.csv`는 새 seed123 ConvNeXt와 EfficientNet-B0의 probability/rank를 사용하며, 현재 최고 대비 `test_0222` 하나만 1에서 0으로 바꾼다. 보호 샘플 5개를 모두 유지하고 meta-CV accuracy도 EXP-52보다 약간 높았다.

실제 제출 결과 leaderboard score는 `0.9551282051`로 현재 최고 EXP-52의 `0.9567307692`보다 하락했다. 따라서 public 기준에서는 `test_0222=1`을 유지하는 편이 유리했을 가능성이 높다. 이후 low-change 탐색에서는 `test_0222=1`도 보호 조건에 추가한다.

2순위 `exp55_ridge_cnx_eff_compact_p3000_t0p025_submit.csv`는 새 seed123을 쓰지 않는 compact feature 후보이며, `test_0504` 하나만 1에서 0으로 바꾼다. 다만 EXP-55 1순위가 하락했으므로, 2순위도 바로 제출하기보다는 `test_0222=1` 보호 조건을 포함한 다음 constrained search 이후에 판단한다.

## EXP-56. 보호 샘플 기반 low-change search

### 목적

EXP-55 제출에서 `test_0222: 1->0` 변경이 public score 하락과 연결되었으므로, `test_0222=1`을 새 보호 조건으로 추가했다. 이후 기존 EXP-55 feature 공간에서 현재 최고 EXP-52 대비 1~3개만 바꾸는 후보를 다시 탐색했다.

### 설정

- 기준 최고 후보: `oof/submissions/exp52_logistic_cnx_eff_prob_only_p0p25_t0p0_submit.csv`
- 입력 feature:
  - 기존 ConvNeXt OOF
  - ConvNeXt seed123 OOF
  - EfficientNet-B0 OOF
- meta-model:
  - RidgeClassifier
  - LogisticRegression
- 보호 샘플:
  - `test_0025=0`
  - `test_0539=0`
  - `test_0244=0`
  - `test_0490=0`
  - `test_0201=1`
  - `test_0222=1`

### 결과

| 후보 파일 | meta-CV accuracy | recall | FP/FN | test label 0/1 | 현재 최고 대비 변경 | 변경 샘플 |
|---|---:|---:|---:|---:|---:|---|
| `oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv` | 0.98658 | 0.98968 | 30/40 | 228/396 | 1 | `test_0504`: 1->0 |

### 저장 파일

```text
scripts/run_exp56_protected_low_change_search.py
oof/preds/exp56_protected_low_change_summary.csv
oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv
```

### 해석

EXP-56은 현재 보호 조건을 모두 유지하면서 `test_0504` 하나만 1에서 0으로 바꾸는 후보를 만들었다. EXP-55에서 하락한 `test_0222`는 1로 유지되므로 EXP-55 1순위와는 다른 독립적인 1샘플 후보다.

실제 제출 결과 leaderboard score는 `0.9583333333`으로 상승했다. EXP-52의 `0.9567307692` 대비 정확히 `1/624` 개선되었으므로, public 기준에서는 `test_0504=0` 방향이 맞았을 가능성이 높다. 이후 탐색에서는 `test_0504=0`도 보호 조건에 추가한다.

## EXP-57. EXP-56 기준 보호 샘플 low-change search

### 목적

EXP-56이 새 최고점이 되었으므로, `test_0504=0`까지 보호 조건에 추가한 뒤 같은 OOF feature 공간에서 추가 1~2개 변경 후보가 있는지 확인했다. 목표는 public에서 확인된 방향을 되돌리지 않으면서, 현재 최고 후보보다 아주 적은 샘플만 바꾸는 후보를 찾는 것이었다.

### 설정

- 기준 최고 후보: `oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv`
- 기준 score 파일: `oof/preds/exp56_ridge_cnx_eff_compact_p3000_t0p025_test_scores.csv`
- 입력 feature:
  - 기존 ConvNeXt OOF
  - ConvNeXt seed123 OOF
  - EfficientNet-B0 OOF
- meta-model:
  - RidgeClassifier
  - LogisticRegression
- 보호 샘플:
  - `test_0025=0`
  - `test_0539=0`
  - `test_0244=0`
  - `test_0490=0`
  - `test_0201=1`
  - `test_0222=1`
  - `test_0504=0`

### 결과

| 항목 | 값 |
|---|---:|
| 탐색 row 수 | 7,956 |
| 최고 meta-CV accuracy | 0.98696 |
| 보호 샘플 7개 유지 후보 | 0 |
| 저장된 제출 파일 | 0 |

### 저장 파일

```text
scripts/run_exp57_protected_low_change_search.py
oof/preds/exp57_protected_low_change_summary.csv
```

### 해석

EXP-57에서는 meta-CV가 높은 후보들이 존재했지만, 상위 후보들이 보호 샘플 중 하나 이상을 깨서 제출 후보에서 제외되었다. 따라서 현재 ConvNeXt/ConvNeXt seed123/EfficientNet-B0 OOF feature 공간 안에서는 public 결과로 확인된 guardrail을 유지하면서 추가로 제출할 만한 low-change 후보가 없다. 다음 단계는 같은 공간의 threshold 미세 탐색보다 calibration을 짧게 확인하거나, 새 정보원 OOF를 추가하는 방향이 더 타당하다.

## EXP-58. Calibrated OOF stacking

### 목적

EXP-57에서 같은 feature 공간의 단순 low-change search 후보가 나오지 않았으므로, meta-model score를 calibration해서 decision boundary 주변 샘플을 다르게 정렬할 수 있는지 확인했다. 단, EXP-56 이후 확인된 보호 샘플 7개는 모두 유지하고, 현재 최고 대비 변경 수가 작은 후보만 저장했다.

### 설정

- 기준 최고 후보: `oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv`
- 입력 feature:
  - 기존 ConvNeXt OOF
  - ConvNeXt seed123 OOF
  - EfficientNet-B0 OOF
- meta-model:
  - RidgeClassifier + calibration
  - LogisticRegression + calibration
  - LinearSVC + calibration
- calibration:
  - sigmoid
  - isotonic
- 보호 샘플:
  - `test_0025=0`
  - `test_0539=0`
  - `test_0244=0`
  - `test_0490=0`
  - `test_0201=1`
  - `test_0222=1`
  - `test_0504=0`

### 결과

| 후보 파일 | meta-CV accuracy | recall | FP/FN | test label 0/1 | 현재 최고 대비 변경 | 변경 샘플 |
|---|---:|---:|---:|---:|---:|---|
| `oof/submissions/exp58_svc_all_margin_stats_isotonic_p0p2_t0p575_submit.csv` | 0.98639 | 0.99174 | 39/32 | 226/398 | 2 | `test_0188`: 0->1, `test_0568`: 0->1 |

### 저장 파일

```text
scripts/run_exp58_calibrated_stacking.py
oof/preds/exp58_calibrated_stacking_summary.csv
oof/preds/exp58_svc_all_margin_stats_isotonic_p0p2_t0p575_test_scores.csv
oof/preds/exp58_svc_all_margin_stats_isotonic_p0p2_t0p575_changed_vs_current_best.csv
oof/submissions/exp58_svc_all_margin_stats_isotonic_p0p2_t0p575_submit.csv
```

### 해석

EXP-58 후보는 보호 샘플 7개를 모두 유지하면서 현재 최고 EXP-56 대비 두 샘플만 `0->1`로 바꾼다. 두 샘플이 모두 맞으면 public 기준 최대 `+2/624` 개선 가능성이 있었지만, 실제 제출 결과 leaderboard score는 `0.9551282051`로 하락했다.

따라서 public 기준에서는 `test_0188` 또는 `test_0568` 중 최소 하나 이상은 `0`을 유지하는 편이 유리했을 가능성이 높다. EXP-58 결과는 calibration 자체가 무효라는 뜻이라기보다는, 현재 보호 샘플을 유지하면서 recall을 더 올리는 방향이 public에서는 과한 FP 증가로 이어질 수 있음을 보여준다. 현재 최고 후보는 계속 EXP-56이다.

## EXP-65. ConvNeXt V2 Tiny 5-fold OOF

### 목적

ConvNeXt V1, EfficientNet-B0, ResNet18과 다른 오류 방향을 가진 일반화 모델을 확보하기 위해 ConvNeXt V2 Tiny를 OOF 모델 pool에 추가했다. 목적은 단독 제출이 아니라 이후 stacking feature로 사용하는 것이다.

### 설정

- 모델: `convnextv2_tiny.fcmae_ft_in22k_in1k`
- 구현: `timm==1.0.27`
- 입력 크기: 224
- fine-tuning 범위: `stage3 + head`
- trainable params: 약 15.49M
- optimizer: AdamW
- LR: `5e-5`
- loss: Weighted CrossEntropyLoss
- augmentation:
  - Resize 224
  - RandomHorizontalFlip 0.5
  - RandomRotation 7
  - ImageNet mean/std normalize

### 결과

| 항목 | 값 |
|---|---:|
| OOF accuracy | 0.97680 |
| OOF precision | 0.98906 |
| OOF recall | 0.97961 |
| OOF F1 | 0.98431 |
| OOF confusion matrix | `[[1299, 42], [79, 3796]]` |
| test mean label 0/1 | 208/416 |
| 총 학습 시간 | 약 95.3분 |

Fold별 결과:

| fold | best epoch | valid accuracy | recall | confusion matrix | test label 0/1 |
|---:|---:|---:|---:|---|---:|
| 0 | 6 | 0.98180 | 0.98452 | `[[262, 7], [12, 763]]` | 226/398 |
| 1 | 5 | 0.97699 | 0.97290 | `[[265, 3], [21, 754]]` | 214/410 |
| 2 | 2 | 0.96836 | 0.97677 | `[[253, 15], [18, 757]]` | 185/439 |
| 3 | 5 | 0.97987 | 0.98581 | `[[258, 10], [11, 764]]` | 213/411 |
| 4 | 5 | 0.97699 | 0.97806 | `[[261, 7], [17, 758]]` | 206/418 |

보호/관찰 샘플 test mean:

| sample | pred | prob_1 | 해석 |
|---|---:|---:|---|
| `test_0025` | 1 | 0.6723 | 기존 보호 방향 0을 깨므로 단독 제출에 불리 |
| `test_0539` | 1 | 0.7343 | 기존 보호 방향 0을 깨므로 단독 제출에 불리 |
| `test_0244` | 1 | 0.9960 | 기존 보호 방향 0을 강하게 깨므로 단독 제출에 불리 |
| `test_0490` | 1 | 0.9784 | EXP-52에서 확인된 0 방향을 깨므로 단독 제출에 불리 |
| `test_0201` | 1 | 0.9881 | 보호 방향 유지 |
| `test_0222` | 1 | 0.7316 | 보호 방향 유지 |
| `test_0504` | 1 | 0.7145 | EXP-56에서 확인된 0 방향을 깨므로 단독 제출에 불리 |
| `test_0188` | 1 | 0.5489 | EXP-58 하락 후보와 같은 방향 |
| `test_0568` | 0 | 0.2983 | EXP-58 하락 후보와 반대 방향 |

### 저장 파일

```text
scripts/run_exp65_convnextv2_tiny_oof.py
oof/exp65_convnextv2_tiny_precheck.md
oof/preds/exp65_convnextv2_tiny_stage3_oof_probs.csv
oof/preds/exp65_convnextv2_tiny_stage3_test_probs_by_fold.csv
oof/preds/exp65_convnextv2_tiny_stage3_test_probs_mean.csv
oof/preds/exp65_convnextv2_tiny_stage3_combined_result.json
oof/preds/exp65_convnextv2_tiny_stage3_changed_vs_exp56.csv
oof/submissions/exp65_convnextv2_tiny_stage3_mean_submit.csv
```

### 해석

EXP-65 ConvNeXt V2 Tiny는 학습 자체는 안정적으로 완료되었지만, OOF accuracy가 기존 ConvNeXt V1 OOF보다 낮고 test mean이 label 1을 416개로 많이 예측한다. 특히 `test_0025`, `test_0244`, `test_0490`, `test_0504`, `test_0539`처럼 public 결과 기반 보호 방향이 있는 샘플을 다수 1로 바꾼다.

따라서 EXP-65 단독 mean 제출은 하지 않는다. 다만 ConvNeXt V2가 기존 모델과 다른 방향으로 판단하는 샘플이 많기 때문에, EXP-66에서는 stacking feature로 제한적으로 넣어 보고 보호 샘플 조건을 유지하는 후보만 검토한다.

## EXP-66. ConvNeXt V2 포함 OOF stacking

### 목적

EXP-65 ConvNeXt V2 Tiny 단독 mean은 보호 샘플을 많이 깨서 제출 후보로 부적절했다. 하지만 기존 ConvNeXt/EfficientNet 계열과 다른 오류 방향을 가질 수 있으므로, 단독 예측이 아니라 stacking feature로 제한적으로 넣어 추가 정보가 되는지 확인했다.

### 설정

- 기준 최고 후보: `oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv`
- 추가 feature:
  - 기존 ConvNeXt OOF probability
  - ConvNeXt seed123 OOF probability
  - EfficientNet-B0 OOF probability
  - ConvNeXt V2 Tiny OOF probability
  - core/all mean, std, ConvNeXt V2와 core mean 차이
- meta-model: RidgeClassifier / LogisticRegression 탐색
- 제출 후보 필터:
  - EXP-56 대비 변경 수 1~6개
  - 보호 샘플 7개 유지
  - `test_0188`, `test_0568`을 동시에 1로 바꾸는 후보 제외
  - test label 1 개수 394~400
  - meta-CV가 EXP-56 기준에서 크게 낮지 않을 것

### 결과

| 항목 | 값 |
|---|---:|
| candidate count | 6 |
| unique submission count | 1 |
| best meta-CV accuracy | 0.98696 |
| best meta-CV precision | 0.99377 |
| best meta-CV recall | 0.98865 |
| best meta-CV F1 | 0.99120 |
| best meta-CV confusion matrix | `[[1317, 24], [44, 3831]]` |
| test label 0/1 | 229/395 |
| EXP-56 대비 변경 수 | 1 |

1순위 후보:

```text
oof/submissions/exp66_ridge_all_prob_stats_with_cnxv2_p10_t0p15_submit.csv
```

EXP-56 대비 변경 샘플:

| sample | EXP-56 | EXP-66 | decision score |
|---|---:|---:|---:|
| `test_0141.png` | 1 | 0 | 0.08557 |

보호 샘플 확인:

| sample | EXP-66 label |
|---|---:|
| `test_0025.png` | 0 |
| `test_0539.png` | 0 |
| `test_0244.png` | 0 |
| `test_0490.png` | 0 |
| `test_0201.png` | 1 |
| `test_0222.png` | 1 |
| `test_0504.png` | 0 |
| `test_0188.png` | 0 |
| `test_0568.png` | 0 |

### 저장 파일

```text
scripts/run_exp66_convnextv2_stacking.py
oof/preds/exp66_convnextv2_stacking_summary.csv
oof/preds/exp66_ridge_all_prob_stats_with_cnxv2_p10_t0p15_test_scores.csv
oof/preds/exp66_ridge_all_prob_stats_with_cnxv2_p10_t0p15_changed_vs_current_best.csv
oof/submissions/exp66_ridge_all_prob_stats_with_cnxv2_p10_t0p15_submit.csv
```

### 해석

EXP-66은 ConvNeXt V2를 단독 후보로 쓰지 않고 보조 feature로만 사용했을 때 의미가 있는지 확인한 실험이다. 결과적으로 보호 샘플 7개를 모두 유지하면서 EXP-56 대비 `test_0141` 하나만 1에서 0으로 바꾸는 후보가 생성되었다.

meta-CV는 EXP-56보다 약간 높고 변경 수도 1개라 제출 후보로는 합리적이었다. 그러나 실제 leaderboard score는 `0.9567307692`로 EXP-56의 `0.9583333333`보다 낮았다. 따라서 public 기준에서는 `test_0141`을 1에서 0으로 바꾸는 방향이 이득이 아니었고, 최고 기준은 계속 EXP-56으로 유지한다.

## EXP-67. RegNetY-3.2GF 5-fold OOF

### 목적

RegNetY 계열은 단일 split에서 단독 성능은 낮았지만, 일부 민감 샘플을 0으로 보는 경향이 있었다. 기존 ConvNeXt/EfficientNet 계열과 다른 오류 방향을 가진 CNN family를 확보하기 위해 5-fold OOF 모델로 확장했다.

### 설정

- 모델: `torchvision.models.regnet_y_3_2gf`
- 입력 크기: 224
- fine-tuning 범위: `trunk_output.block4 + fc`
- trainable params: 약 4.81M
- optimizer: AdamW
- LR: block4 `1e-5`, fc `5e-5`
- loss: Weighted CrossEntropyLoss
- epoch: 6 x 5 folds

### 결과

| 항목 | 값 |
|---|---:|
| OOF accuracy | 0.96166 |
| OOF precision | 0.99039 |
| OOF recall | 0.95768 |
| OOF F1 | 0.97376 |
| OOF confusion matrix | `[[1305, 36], [164, 3711]]` |
| test mean label 0/1 | 273/351 |
| EXP-56 대비 test 변경 수 | 69 |

Fold별 결과:

| fold | best epoch | valid accuracy | recall | confusion matrix | test label 0/1 |
|---:|---:|---:|---:|---|---:|
| 0 | 6 | 0.97031 | 0.96774 | `[[263, 6], [25, 750]]` | 251/373 |
| 1 | 6 | 0.96644 | 0.96258 | `[[262, 6], [29, 746]]` | 261/363 |
| 2 | 6 | 0.95110 | 0.94323 | `[[261, 7], [44, 731]]` | 290/334 |
| 3 | 5 | 0.96261 | 0.96258 | `[[258, 10], [29, 746]]` | 292/332 |
| 4 | 5 | 0.95781 | 0.95226 | `[[261, 7], [37, 738]]` | 279/345 |

보호/관찰 샘플 test mean:

| sample | pred | prob_1 | 해석 |
|---|---:|---:|---|
| `test_0025` | 0 | 0.1709 | 보호 방향 유지 |
| `test_0539` | 0 | 0.0994 | 보호 방향 유지 |
| `test_0244` | 1 | 0.5703 | 보호 방향 0을 깸 |
| `test_0490` | 1 | 0.7261 | 보호 방향 0을 깸 |
| `test_0201` | 1 | 0.5242 | 보호 방향 유지 |
| `test_0222` | 0 | 0.3346 | 보호 방향 1을 깸 |
| `test_0504` | 0 | 0.4307 | 보호 방향 유지 |
| `test_0141` | 0 | 0.0345 | EXP-66 결과상 유지해야 할 1 방향을 깸 |
| `test_0188` | 0 | 0.4433 | EXP-58 하락 방향 회피 |
| `test_0568` | 0 | 0.1633 | EXP-58 하락 방향 회피 |

### 저장 파일

```text
scripts/run_exp67_regnet_y_3_2gf_oof.py
oof/preds/exp67_regnet_y_3_2gf_oof_probs.csv
oof/preds/exp67_regnet_y_3_2gf_test_probs_by_fold.csv
oof/preds/exp67_regnet_y_3_2gf_test_probs_mean.csv
oof/preds/exp67_regnet_y_3_2gf_combined_result.json
oof/preds/exp67_regnet_y_3_2gf_changed_vs_exp56.csv
oof/submissions/exp67_regnet_y_3_2gf_mean_submit.csv
```

### 해석

EXP-67 RegNetY OOF는 precision은 높지만 recall이 낮아 FN이 164개로 많다. test mean도 label 1을 351개만 예측해 현재 최고 후보보다 훨씬 보수적이다. 따라서 단독 제출은 제외한다.

다만 `test_0025`, `test_0539`, `test_0188`, `test_0568`을 0으로 강하게 보는 특성이 있어, EXP-68에서 stacking 보조 feature로만 사용했다.

## EXP-68. RegNetY 포함 OOF stacking

### 목적

EXP-67 RegNetY 단독 mean은 약하지만, 기존 ConvNeXt/EfficientNet 계열이 흔들리는 일부 샘플에서 보정 정보가 될 수 있는지 확인했다. EXP-66 결과를 반영해 `test_0141=1`까지 보호 조건에 추가했다.

### 설정

- 기준 최고 후보: `oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv`
- feature:
  - ConvNeXt OOF probability
  - ConvNeXt seed123 OOF probability
  - EfficientNet-B0 OOF probability
  - RegNetY-3.2GF OOF probability/rank/margin
  - core/all mean, std, vote_sum
- meta-model: RidgeClassifier / LogisticRegression
- 필터:
  - 보호 샘플 8개 유지
  - `test_0188`, `test_0568` 동시 1 제외
  - test label 1 개수 394~400
  - EXP-56 대비 변경 수 1~6개
  - meta-CV floor 유지

### 결과

| 항목 | 값 |
|---|---:|
| 탐색 row 수 | 3724 |
| 최고 meta-CV accuracy | 0.98735 |
| 보호 샘플 8개 유지 candidate | 0 |
| 저장된 제출 파일 | 0 |

상위 후보들은 meta-CV가 높았지만 보호 샘플을 깼다. 특히 후보 필터 범위 안에서 `test_0490`, `test_0504`, `test_0141`, `test_0539`가 자주 반대 방향으로 움직였다.

### 저장 파일

```text
scripts/run_exp68_regnet_stacking.py
oof/preds/exp68_regnet_stacking_summary.csv
```

### 해석

RegNetY는 새 CNN family로서 오류 방향 다양성은 제공했지만, 현재 public 결과 기반 보호 규칙 안에서는 제출 가능한 low-change 후보를 만들지 못했다. 따라서 RegNetY 계열은 당장 제출 후보로 쓰기보다, 이후 더 넓은 OOF pool에서 feature importance를 보고 선택적으로 사용하는 편이 낫다.

## EXP-69. DINOv2 feature extractor + Ridge/Logistic

### 목적

DINOv2는 ConvNeXt/EfficientNet/RegNet과 다른 ViT 기반 foundation feature family다. 새 모델을 끝까지 fine-tuning하기 전에 frozen feature extractor로 흉부 X-ray 분류에 보완 정보가 있는지 확인했다.

### 설정

- 모델: `timm` `vit_small_patch14_dinov2`
- 입력 크기: 518x518
- feature dimension: 384
- 학습 방식: DINOv2 backbone은 학습하지 않고 train/test image feature만 추출
- meta-model:
  - `StandardScaler + RidgeClassifier`
  - `StandardScaler + LogisticRegression`
- 검증 방식: 기존 5-fold split 기반 meta-CV
- 출력 파일:
  - `scripts/run_exp69_dinov2_features_stacking.py`
  - `oof/preds/exp69_dinov2_vits14_train_features.npz`
  - `oof/preds/exp69_dinov2_vits14_test_features.npz`
  - `oof/preds/exp69_dinov2_vits14_summary.csv`
  - `oof/preds/exp69_dinov2_vits14_result.json`

### 결과

- train feature rows: 5,216
- test feature rows: 624
- feature 추출 시간: 약 623.6초
- 최고 meta-CV accuracy: 0.97373
- 보호 샘플과 low-change 조건을 모두 만족한 candidate 수: 0
- 저장된 제출 파일 수: 0
- 상위 후보 예시:
  - Logistic C=0.3, threshold -0.300
    - meta-CV accuracy: 0.97373
    - test label 1 개수: 302
    - EXP-56 대비 변경 수: 104
  - Ridge alpha=100, threshold -0.050
    - meta-CV accuracy: 0.97316
    - test label 1 개수: 388
    - EXP-56 대비 변경 수: 42

### 판단

DINOv2 frozen feature는 새 모델 family 확보 관점에서는 의미가 있지만, 이번 설정에서는 현재 CNN OOF pool보다 약했다. 상위 후보들이 `test_0244`, `test_0490`, `test_0504`, `test_0141` 등 보호 방향을 자주 깨고, 변경 샘플 수도 30개 이상으로 커져 제출하지 않는다.

### 다음 해석

DINOv2를 완전히 버릴 필요는 없지만, raw frozen feature + 선형 meta-model만으로는 부족하다. 다시 시도한다면 DINOv2 feature를 단독 제출 후보로 보기보다, PCA/feature pruning, 작은 MLP/TabM meta-model, 또는 의료 이미지에 더 맞는 foundation feature와 비교하는 방향이 더 타당하다.

## EXP-70. EfficientNetV2-S 재설계 5-fold OOF

### 목적

EXP-42 EfficientNetV2-S는 `features[7] + classifier`만 학습해 validation accuracy가 0.87261로 낮았다. 이번에는 같은 모델 family를 버리기 전에 fine-tuning 범위를 `features[6:] + classifier`로 넓혀, EfficientNetV2-S가 OOF 정보원으로 쓸 수 있는지 재평가했다.

### 설정

- 모델: `torchvision.models.efficientnet_v2_s(weights=IMAGENET1K_V1)`
- 학습 범위: `features[6:] + classifier`
- 입력 크기: 224x224
- batch size: 24
- epoch: 6 x 5 folds
- optimizer: AdamW
- LR:
  - feature blocks: 2e-5
  - classifier: 8e-5
- loss: class-weighted CrossEntropyLoss
- 출력 파일:
  - `scripts/run_exp70_efficientnet_v2_s_oof.py`
  - `oof/preds/exp70_efficientnet_v2_s_wide_oof_probs.csv`
  - `oof/preds/exp70_efficientnet_v2_s_wide_test_probs_mean.csv`
  - `oof/preds/exp70_efficientnet_v2_s_wide_combined_result.json`
  - `oof/submissions/exp70_efficientnet_v2_s_wide_mean_submit.csv`

### 결과

| 항목 | 값 |
|---|---:|
| OOF accuracy | 0.97642 |
| OOF precision | 0.99110 |
| OOF recall | 0.97703 |
| OOF F1 | 0.98402 |
| OOF confusion matrix | `[[1307, 34], [89, 3786]]` |
| test label 0/1 | 208 / 416 |
| EXP-56 대비 변경 수 | 34 |

fold별 best accuracy:

| fold | best epoch | valid accuracy |
|---:|---:|---:|
| 0 | 5 | 0.9751 |
| 1 | 6 | 0.9799 |
| 2 | 6 | 0.9741 |
| 3 | 4 | 0.9741 |
| 4 | 6 | 0.9789 |

### 보호 샘플 확인

| sample | EXP-70 pred | prob_1 | 판단 |
|---|---:|---:|---|
| test_0025 | 1 | 0.6687 | 보호 방향 위반 |
| test_0539 | 1 | 0.5383 | 보호 방향 위반 |
| test_0244 | 1 | 0.9028 | 보호 방향 위반 |
| test_0490 | 1 | 0.9110 | 보호 방향 위반 |
| test_0201 | 1 | 0.8746 | 유지 |
| test_0222 | 0 | 0.4090 | 보호 방향 위반 |
| test_0504 | 1 | 0.5698 | 보호 방향 위반 |
| test_0141 | 1 | 0.8956 | 유지 |
| test_0188 | 1 | 0.6869 | 단독으로는 주의 |
| test_0568 | 0 | 0.2469 | 유지 |

### 판단

EfficientNetV2-S 재설계는 EXP-42 대비 명확히 개선되었다. 마지막 block만 학습한 것이 실패 원인이었을 가능성이 크다. 다만 OOF accuracy는 기존 ConvNeXt/EfficientNet-B0 OOF보다 낮고, 단독 mean 제출은 보호 샘플을 다수 깨므로 제출하지 않는다.

### 다음 해석

EXP-70은 단독 제출 후보가 아니라 stacking diversity 후보로 보류한다. 다음 단계에서는 EXP-70 probability를 낮은 가중치 또는 feature importance 기반으로 제한적으로 넣어, EXP-56 보호 방향을 유지하는 후보가 나오는지 확인하는 것이 타당하다.

## EXP-71. EfficientNetV2-S 포함 OOF stacking

### 목적

EXP-70 EfficientNetV2-S는 단독 제출에는 위험했지만, EXP-42 대비 크게 회복되어 OOF 정보원으로는 의미가 있었다. 따라서 기존 ConvNeXt/ConvNeXt seed123/EfficientNet-B0 OOF feature에 EfficientNetV2-S를 낮은 가중치로 추가해, 보호 샘플을 유지하는 low-change 후보가 생기는지 확인했다.

### 설정

- 기준 최고 후보: `oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv`
- 추가 정보원: EXP-70 EfficientNetV2-S OOF/test probability
- meta-model: RidgeClassifier / LogisticRegression
- feature 조합:
  - core probability + `effv2_prob1`
  - core probability + `effv2_rank`
  - core probability + `effv2_margin`
  - compact core feature + `effv2` light feature
  - all probability/margin stats
- 필터:
  - 보호 샘플 8개 유지
  - `test_0188`, `test_0568` 동시 1 제외
  - test label 1 개수 394~400
  - EXP-56 대비 변경 수 1~6개
  - meta-CV floor 유지

### 결과

| 항목 | 값 |
|---|---:|
| 탐색 row 수 | 6318 |
| 최고 meta-CV accuracy | 0.98888 |
| 보호 샘플 유지 row 수 | 5 |
| 보호 샘플 유지 비중복 candidate | 0 |
| 저장된 제출 파일 | 0 |

상위 후보는 meta-CV와 label 분포가 괜찮았지만 보호 샘플을 깨는 경우가 많았다. 특히 top 50 후보 기준으로 `test_0490`, `test_0504`는 모두 1로 예측했고, `test_0539`, `test_0244`도 대부분 1로 예측했다.

보호 샘플을 모두 유지한 후보 5개는 모두 EXP-56과 test 예측이 완전히 동일했다. 따라서 새 제출 후보로 저장하지 않았다.

### 저장 파일

```text
scripts/run_exp71_effv2_stacking.py
oof/preds/exp71_effv2_stacking_summary.csv
```

### 판단

EfficientNetV2-S는 EXP-70에서 단독 OOF 정보원으로 회복됐지만, 현재 최고 후보를 개선하는 방향의 안전한 test 변경은 만들지 못했다. meta-CV 최고값만 보면 좋아 보이지만, public에서 이미 불리했던 보호 샘플 방향을 자주 깨므로 제출하지 않는다.

## EXP-72. ConvNeXt-Tiny seed777 5-fold OOF

### 목적

현재 최고 후보는 ConvNeXt/EfficientNet-B0 OOF 기반 stacking에서 나왔다. 따라서 새로운 모델 family보다 우선순위가 높은 작업으로, ConvNeXt-Tiny의 추가 seed를 학습해 강한 family 내부의 seed diversity를 확보했다.

### 설정

- 모델: ConvNeXt-Tiny ImageNet pretrained
- 구조: EXP-46/54와 동일
- 학습 seed: 777
- fold split: 기존 `oof/folds_seed42.csv`
- epoch: 6 x 5 folds
- optimizer: AdamW
- LR: 5e-5
- loss: class-weighted CrossEntropyLoss
- 출력 파일:
  - `scripts/run_exp72_convnext_seed777_oof.py`
  - `oof/preds/exp72_convnext_tiny_seed777_oof_probs.csv`
  - `oof/preds/exp72_convnext_tiny_seed777_test_probs_mean.csv`
  - `oof/preds/exp72_convnext_tiny_seed777_combined_result.json`
  - `oof/submissions/exp72_convnext_tiny_seed777_mean_submit.csv`

### 결과

| 항목 | 값 |
|---|---:|
| OOF accuracy | 0.98006 |
| OOF precision | 0.98809 |
| OOF recall | 0.98503 |
| OOF F1 | 0.98656 |
| OOF confusion matrix | `[[1295, 46], [58, 3817]]` |
| test label 0/1 | 210 / 414 |
| EXP-56 대비 변경 수 | 20 |

fold별 best accuracy:

| fold | best epoch | valid accuracy |
|---:|---:|---:|
| 0 | 3 | 0.9875 |
| 1 | 3 | 0.9818 |
| 2 | 5 | 0.9684 |
| 3 | 4 | 0.9818 |
| 4 | 5 | 0.9808 |

### 보호 샘플 확인

| sample | EXP-72 pred | prob_1 | 판단 |
|---|---:|---:|---|
| test_0025 | 1 | 0.6264 | 보호 방향 위반 |
| test_0539 | 1 | 0.9347 | 보호 방향 위반 |
| test_0244 | 1 | 0.6845 | 보호 방향 위반 |
| test_0490 | 1 | 0.6290 | 보호 방향 위반 |
| test_0201 | 1 | 0.8188 | 유지 |
| test_0222 | 1 | 0.8681 | 유지 |
| test_0504 | 1 | 0.6950 | 보호 방향 위반 |
| test_0141 | 1 | 0.5440 | 유지 |
| test_0188 | 1 | 0.8683 | 단독으로는 주의 |
| test_0568 | 0 | 0.0195 | 유지 |

### 판단

seed777 ConvNeXt는 OOF accuracy 0.98006으로 강한 편이며, seed diversity 확보 목적은 달성했다. 다만 단독 mean은 EXP-56 대비 20개가 바뀌고 보호 샘플 다수를 1로 바꾸므로 제출하지 않는다. 다음 단계에서는 EXP-72를 기존 ConvNeXt seed123, EfficientNet-B0와 함께 stacking에 넣되, 보호 샘플을 유지하는 후보만 검토한다.

## EXP-73. ConvNeXt seed777 포함 OOF stacking

### 목적

EXP-72 seed777 ConvNeXt는 단독 mean 제출은 위험했지만 OOF accuracy가 0.98006으로 강했다. 따라서 기존 ConvNeXt seed42, ConvNeXt seed123, EfficientNet-B0 OOF pool에 seed777을 추가해, 보호 샘플을 유지하면서 새 low-change 후보가 나오는지 확인했다.

### 설정

- 기준 최고 후보: `oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv`
- 추가 정보원: EXP-72 ConvNeXt-Tiny seed777 OOF/test probability
- meta-model: RidgeClassifier / LogisticRegression
- feature 조합:
  - ConvNeXt seed 조합 probability/rank/margin
  - ConvNeXt seed + EfficientNet-B0 compact feature
  - all probability stats
- 필터:
  - 보호 샘플 8개 유지
  - `test_0188`, `test_0568` 동시 1 제외
  - test label 1 개수 394~400
  - EXP-56 대비 변경 수 1~8개
  - meta-CV floor 유지

### 결과

| 항목 | 값 |
|---|---:|
| 탐색 row 수 | 20592 |
| 최고 meta-CV accuracy | 0.98754 |
| 보호 샘플 유지 row 수 | 21 |
| 보호 샘플 유지 비중복 candidate | 0 |
| 저장된 제출 파일 | 0 |

상위 후보들은 `test_0504`, `test_0490`, `test_0539`를 자주 1로 바꿔 보호 조건을 통과하지 못했다. 보호 샘플을 모두 유지한 row는 21개였지만 모두 EXP-56과 test 예측이 동일했다.

### 저장 파일

```text
scripts/run_exp73_convnext_seed777_stacking.py
oof/preds/exp73_convnext_seed_stacking_summary.csv
```

### 판단

ConvNeXt seed777은 OOF 정보원으로는 강하지만, 현재 guardrail 안에서는 새 제출 후보를 만들지 못했다. seed 추가가 완전히 무의미한 것은 아니지만, 현재 보호 샘플 기반 low-change 영역은 상당히 소진된 것으로 보인다.

## EXP-77. ConvNeXt-Tiny Lion fold0 smoke

### 목적

ConvNeXt 계열은 현재 가장 강한 family지만, seed 추가만으로는 새 제출 후보가 나오지 않았다. 이번에는 optimizer를 AdamW에서 Lion으로 바꿔 학습 동역학과 오류 방향이 달라지는지 fold0 smoke test로 확인했다.

### 설정

- 모델: ConvNeXt-Tiny ImageNet pretrained
- fold: 0 only
- 학습 범위: 마지막 stage `features[-1] + classifier`
- seed: 42
- epoch: 6
- optimizer: Lion
- LR: 2e-5
- weight decay: 1e-4
- loss: class-weighted CrossEntropyLoss
- 비교 기준: EXP-45 AdamW fold0

### 결과

| 항목 | AdamW fold0 | Lion fold0 |
|---|---:|---:|
| best epoch | - | 5 |
| accuracy | 0.98467 | 0.98563 |
| precision | 0.99608 | 0.99479 |
| recall | 0.98323 | 0.98581 |
| F1 | 0.98961 | 0.99028 |
| confusion matrix | `[[266, 3], [13, 762]]` | `[[265, 4], [11, 764]]` |
| test label 0/1 | - | 234 / 390 |
| AdamW fold0 대비 변경 수 | - | 18 |
| EXP-56 대비 변경 수 | - | 26 |

### 보호 샘플 확인

| sample | Lion pred | prob_1 | 판단 |
|---|---:|---:|---|
| test_0025 | 0 | 0.2595 | 유지 |
| test_0539 | 1 | 0.6337 | 보호 방향 위반 |
| test_0244 | 0 | 0.1513 | 유지 |
| test_0490 | 1 | 0.6424 | 보호 방향 위반 |
| test_0201 | 1 | 0.5218 | 유지 |
| test_0222 | 1 | 0.7895 | 유지 |
| test_0504 | 0 | 0.1029 | 유지 |
| test_0141 | 0 | 0.1894 | 보호 방향 위반 |
| test_0188 | 0 | 0.4101 | 유지 |
| test_0568 | 0 | 0.0014 | 유지 |

### 저장 파일

```text
scripts/run_exp77_convnext_lion_fold0.py
oof/preds/exp77_convnext_tiny_lion_fold0_oof_probs.csv
oof/preds/exp77_convnext_tiny_lion_fold0_test_probs.csv
oof/preds/exp77_convnext_tiny_lion_fold0_result.json
oof/preds/exp77_convnext_tiny_lion_fold0_summary.csv
oof/preds/exp77_changed_vs_exp56.csv
oof/submissions/exp77_convnext_tiny_lion_fold0_submit.csv
```

### 판단

Lion fold0는 AdamW fold0보다 accuracy와 recall이 소폭 개선되었다. 특히 FN이 13개에서 11개로 줄어, optimizer 변경이 단순 노이즈만은 아닐 가능성이 있다. 다만 `test_0539`, `test_0490`, `test_0141` 보호 방향을 깨므로 fold0 단독 제출은 제외한다. 다음 단계에서는 Lion 5-fold OOF 확장을 검토할 수 있다.

## EXP-78. ConvNeXt-Tiny Lion 5-fold OOF

### 목적

EXP-77 fold0 smoke에서 Lion이 AdamW보다 FN을 소폭 줄였으므로, optimizer 변경이 5-fold 전체에서도 유효한지 확인했다. 목표는 단독 제출보다 ConvNeXt 계열의 optimizer diversity를 확보해 이후 stacking feature로 활용하는 것이다.

### 복구 메모

IDE/Docker 메모리 문제로 fold2 진행 중 세션이 중단되었다. 중단 전 fold0/fold1 산출물은 정상 저장되어 있었고, 스크립트를 fold2~4만 재실행한 뒤 fold0~4 전체를 병합하도록 수정해 복구했다.

### 설정

- 모델: ConvNeXt-Tiny ImageNet pretrained
- fold: Stratified 5-fold
- 학습 범위: 마지막 stage `features[-1] + classifier`
- seed: 42
- epoch: 6 per fold
- optimizer: Lion
- LR: 2e-5
- weight decay: 1e-4
- loss: class-weighted CrossEntropyLoss

### 결과

| 항목 | 값 |
|---|---:|
| OOF accuracy | 0.98140 |
| OOF precision | 0.98786 |
| OOF recall | 0.98710 |
| OOF F1 | 0.98748 |
| OOF confusion matrix | `[[1294, 47], [50, 3825]]` |
| test label 0/1 | 204 / 420 |
| EXP-56 대비 변경 수 | 24 |

### Fold별 결과

| fold | best epoch | accuracy | confusion matrix |
|---:|---:|---:|---|
| 0 | 5 | 0.98563 | `[[265, 4], [11, 764]]` |
| 1 | 6 | 0.98082 | `[[261, 7], [13, 762]]` |
| 2 | 5 | 0.97795 | `[[255, 13], [10, 765]]` |
| 3 | 5 | 0.98370 | `[[261, 7], [10, 765]]` |
| 4 | 6 | 0.97891 | `[[252, 16], [6, 769]]` |

### 보호 샘플 확인

| sample | EXP-78 pred | prob_1 | 판단 |
|---|---:|---:|---|
| test_0025 | 0 | 0.4047 | 유지 |
| test_0539 | 1 | 0.8589 | 보호 방향 위반 |
| test_0244 | 1 | 0.6251 | 보호 방향 위반 |
| test_0490 | 1 | 0.9093 | 보호 방향 위반 |
| test_0201 | 1 | 0.9014 | 유지 |
| test_0222 | 1 | 0.9365 | 유지 |
| test_0504 | 1 | 0.5993 | 보호 방향 위반 |
| test_0141 | 1 | 0.7345 | 유지 |
| test_0188 | 1 | 0.6572 | 보호 방향 위반 |
| test_0568 | 0 | 0.1666 | 유지 |

### 저장 파일

```text
scripts/run_exp78_convnext_lion_oof.py
oof/preds/exp78_convnext_tiny_lion_oof_probs.csv
oof/preds/exp78_convnext_tiny_lion_test_probs_by_fold.csv
oof/preds/exp78_convnext_tiny_lion_test_probs_mean.csv
oof/preds/exp78_convnext_tiny_lion_fold_summary.csv
oof/preds/exp78_convnext_tiny_lion_combined_result.json
oof/preds/exp78_changed_vs_exp56.csv
oof/submissions/exp78_convnext_tiny_lion_mean_submit.csv
```

### 판단

EXP-78은 OOF 기준으로 강한 단독 모델이지만, test mean이 EXP-56 대비 24개를 모두 `0 -> 1`로 바꾸고 `test_0244`, `test_0490`, `test_0504`, `test_0188` 등 보호 방향을 깨므로 단독 제출은 제외한다. 다만 FN을 줄이는 방향의 모델이므로 이후 Ridge/Logistic stacking에서 낮은 가중치의 보조 feature로 활용할 수 있다.

## EXP-79. Lion OOF 포함 stacking

### 목적

EXP-78 Lion OOF는 단독 제출로는 공격적이지만 FN을 줄이는 성향이 있었다. 이번 실험은 Lion probability를 기존 ConvNeXt/EfficientNet OOF pool에 낮은 비중의 feature로 추가했을 때, EXP-56보다 나은 low-change 후보가 생기는지 확인하는 것이다.

### 설정

- 기준 최고 후보: `oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv`
- 추가 정보원: EXP-78 ConvNeXt-Tiny Lion 5-fold OOF/test mean probability
- 기존 정보원: EXP-46 ConvNeXt, EXP-54 ConvNeXt seed123, EXP-72 ConvNeXt seed777, EXP-50 EfficientNet-B0
- meta-model: RidgeClassifier / LogisticRegression
- feature: probability, rank, margin, uncertainty, compact feature 조합
- 보호 조건:
  - `test_0025=0`
  - `test_0539=0`
  - `test_0244=0`
  - `test_0490=0`
  - `test_0201=1`
  - `test_0222=1`
  - `test_0504=0`
  - `test_0141=1`
  - `test_0188`와 `test_0568` 동시 1 방지

### 결과

| 항목 | 값 |
|---|---:|
| 탐색 row 수 | 40,248 |
| 전체 최고 meta-CV accuracy | 0.98811 |
| 보호 샘플 유지 row 수 | 32 |
| label band 통과 row 수 | 11,904 |
| low-change 통과 row 수 | 16,325 |
| 최종 candidate 수 | 0 |
| 저장된 제출 파일 | 0 |

전체 최고 meta-CV 후보는 `all_prob_stats` logistic 조합으로 0.98811까지 나왔지만, 보호 샘플 조건을 통과하지 못했다. 반대로 보호 샘플을 모두 유지하면서 EXP-56 대비 1개만 바꾸는 Lion 포함 후보는 존재했지만, meta-CV accuracy가 0.98447로 현재 기준 0.98658보다 약 0.00211 낮았다.

가장 가까운 보호 후보:

| 항목 | 값 |
|---|---|
| spec | `cnx_lion_eff_prob_only` |
| model | RidgeClassifier |
| alpha | 500 |
| threshold | 0.20 |
| meta-CV accuracy | 0.98447 |
| test label 0/1 | 227 / 397 |
| EXP-56 대비 변경 | `test_0049: 0 -> 1` |

### 저장 파일

```text
scripts/run_exp79_lion_stacking.py
oof/preds/exp79_lion_stacking_summary.csv
oof/preds/exp79_ridge_cnx_lion_eff_prob_only_alpha500_t0p2_test_scores.csv
oof/preds/exp79_ridge_cnx_lion_eff_prob_only_alpha500_t0p2_changed_vs_exp56.csv
oof/submissions/exp79_ridge_cnx_lion_eff_prob_only_alpha500_t0p2_submit.csv
```

### 판단

Lion OOF를 stacking에 넣으면 일부 meta-CV는 올라가지만, 좋은 후보들은 보호 샘플을 깨고, 보호 조건을 만족하는 후보는 meta-CV 하락폭이 크다. 따라서 EXP-79는 기본적으로 제출 비추천이다. 다만 사용자가 1개 후보를 제출해보고 싶다고 요청해, 가장 가까운 보호 후보인 `cnx_lion_eff_prob_only`, Ridge alpha 500, threshold 0.20 제출 파일을 별도로 생성했다. 이 후보는 EXP-56 대비 `test_0049` 하나만 `0 -> 1`로 바꾸며 보호 샘플은 모두 유지했다.

Leaderboard 결과는 `0.9567307692`로 EXP-56의 `0.9583333333`보다 낮았다. 따라서 `test_0049=1` 방향은 현재 public 기준에서는 이득이 아니었고, `test_0049=0`을 우선 유지하는 편이 낫다.

## 데일리 로그

### 2026-06-18

- baseline 노트북 구조 확인
- 데이터 클래스 분포 확인
- train/validation split 추가
- augmentation, class weight, BatchNorm, Dropout 적용
- 개선 SimpleCNN validation 결과 기록
- ResNet18 fine-tuning 섹션 추가
- 실험 기록용 markdown 파일 생성
- ResNet18 layer4+fc 학습 완료
- `submissions/resnet18_submit.csv`로 이동/정리
- ResNet18 epoch 10 추가 실험 완료
- `submissions/resnet18_epoch10_submit.csv`로 이동/정리
- ResNet18 layer3+layer4+fc fine-tuning 실험 완료
- `submissions/resnet18_layer34_submit.csv`로 이동/정리
- ResNet18 threshold search 완료
- threshold별 제출 파일을 `submissions/` 폴더로 이동/정리
- EfficientNet-B0 fine-tuning 실험 완료
- EfficientNet-B0 모델 가중치와 valid/test probability 저장, 제출 CSV는 `submissions/`에 정리
- EXP-02 ResNet18 최고 구조를 EXP-07로 재현
- ResNet18 모델 가중치와 valid/test probability 저장

### 2026-06-19

- ConvNeXt-Tiny 단독 모델 학습 및 제출 파일 생성
- ConvNeXt 중심 soft voting 후보 생성
- ConvNeXt-Tiny sqrt class weight 실험 완료
- ConvNeXt-Tiny no horizontal flip augmentation 실험 완료
- ConvNeXt-Tiny seed7 추가 학습 및 EXP-12와 seed ensemble 후보 생성
- ConvNeXt-Tiny EXP-12 probability 기반 threshold 상향 후보 생성
- ConvNeXt-Tiny full fine-tuning 실험 완료 및 leaderboard 동점 확인
- EXP-12와 EXP-18 soft voting 제출 후보 생성
- RidgeClassifier stacking 제출 후보 생성
- LogisticRegression stacking 제출 후보 생성
- RidgeClassifier feature v2 제출 후보 생성
- RidgeClassifier feature pruning 제출 후보 생성
- RidgeClassifier stacking feature importance 분석 완료
- 딥러닝 모델 내부 feature importance/Grad-CAM 분석 필요성 기록
- Ridge alpha 미세 조정 완료, test 예측은 EXP-20과 동일
- DenseNet 제거 Ridge stacking 제출 후보 생성
- DenseNet prob/rank 영향 제한 분석 완료, 새 제출 후보 없음
- EXP-12 ConvNeXt disagreement 샘플 LIME 분석 완료
- Grad-CAM/LIME 비교 패널 및 XAI 리포트 생성
- ResNet/DenseNet/EfficientNet/ConvNeXt 다중 모델 Grad-CAM 비교 완료

### 2026-06-20

- EXP-26 Ridge without DenseNet 제출 결과 기록, leaderboard score `0.9519230769`, 순위 19위 확인
- EXP-20/EXP-26 validation FP/FN 오분류 비교 분석 완료
- validation 오분류 XAI 후보 grid, summary CSV, counts JSON 생성
- baseline 노트북에 `13-11. Validation FP/FN 분석` 섹션 추가
- EXP-33 validation 판단 변경 샘플 Grad-CAM/LIME 분석 완료
- EXP-34 EXP-26 Ridge threshold search 완료, 제출 후보 CSV 생성
- baseline 노트북에 `13-12. Validation 판단 변경 샘플 XAI`, `13-13. EXP-26 Ridge Threshold Search` 섹션 추가
- `submissions/exp34_ridge_threshold_m0p02_submit.csv` 제출 결과 leaderboard score `0.9503205128` 확인, EXP-26보다 하락
- EXP-35 ConvNeXt 영향 제한 후보 비교 완료, 제출 추천 후보 없음
- baseline 노트북에 `13-14. EXP-35 ConvNeXt 영향 제한` 섹션 추가
- EXP-40 Swin-Tiny head-only probe 학습 완료, 단독 제출 비추천
- EXP-40 Swin-Tiny stacking 추가 후보 확인, `test_0244`를 1로 바꿔 제출 보류
- baseline 노트북에 `13-15. EXP-40 Swin-Tiny 추가 실험` 섹션 추가
- EXP-36 test_0244 고정 stacking 탐색 완료, `core_prob_only alpha100` 제출 후보 생성
- baseline 노트북에 `13-16. EXP-36 test_0244 고정 stacking 탐색` 섹션 추가
- `submissions/exp36_fixed0244_core_prob_only_alpha100_submit.csv` 제출 결과 leaderboard score `0.9503205128` 확인, EXP-26보다 하락
- EXP-37 original+hflip TTA probability 생성 완료
- EXP-38 TTA 기반 Ridge stacking 후보 생성, 제출 우선순위 낮음
- baseline 노트북에 `13-17. EXP-37/38 TTA probability와 stacking` 섹션 추가
- `submissions/exp38_tta_all_prob_only_alpha50_submit.csv` 제출 결과 leaderboard score `0.9487179487` 확인, EXP-26보다 하락
- EXP-39 ConvNeXt-Tiny seed123 추가 학습 완료, 단독 제출 비추천 및 stacking diversity 후보로 기록
- baseline 노트북에 `13-18. EXP-39 ConvNeXt seed123 추가 학습` 섹션 추가
- EXP-41 5-fold OOF 설계 완료, fold split과 OOF prediction 포맷 정의
- baseline 노트북에 `13-19. EXP-41 5-fold OOF 설계` 섹션 추가
- EXP-42 EfficientNetV2-S 단일 split probe 완료, FN 증가와 민감 샘플 변경으로 제출 보류
- Lion optimizer 비교는 calibration/selective TTA 계열 실험 전 우선순위에 추가
- EXP-43 RegNetY-3.2GF 단일 split probe 완료, 단독 제출은 보류하고 stacking diversity 후보로 기록
- EXP-44 RegNetY 포함 Ridge stacking 후보 생성, EXP-26 대비 5개 변경 및 민감 샘플 3개 유지
- `submissions/exp44_regnet_drop_resnet18_add_reg_alpha150_submit.csv` 제출 결과 leaderboard score `0.9503205128` 확인, EXP-26보다 하락
- EXP-45 ConvNeXt-Tiny OOF fold0 smoke test 완료, OOF/test probability 포맷 검증 통과
- EXP-46 ConvNeXt-Tiny 5-fold OOF 완성, OOF 5,216 rows와 test mean 624 rows 생성
- EXP-47 OOF Ridge baseline 완료, 최고 OOF 후보는 민감 샘플을 바꿔 제출 보류

### 2026-06-21

- EXP-43~47 산출물과 기록을 git commit `ae3fc1b`로 저장
- EXP-48 ResNet18 5-fold OOF 완료, OOF accuracy `0.98524`
- EXP-48은 OOF 성능은 강하지만 민감 샘플 `test_0244`, `test_0025`, `test_0539`를 모두 1로 예측해 단독 제출 보류
- EXP-49 ConvNeXt+ResNet OOF Ridge/Logistic stacking 완료
- EXP-49는 meta-CV 최고 `0.98735`까지 도달했지만 `test_0539`를 1로 유지해 제출 우선순위 낮음
- EXP-50 EfficientNet-B0 OOF 초기 설정 `features[7:]+classifier`는 fold0 `0.9301`로 낮아 중단
- EXP-50을 `features[5:]+classifier`, feature LR `5e-5`, classifier LR `1e-4`로 재설정 후 5-fold OOF 완료
- EXP-50 OOF accuracy `0.98505`, 민감 샘플 3개를 모두 0으로 유지
- EXP-51 ConvNeXt+ResNet+EfficientNet OOF stacking 완료
- EXP-51에서는 ResNet 제외 ConvNeXt+EfficientNet 후보가 가장 안정적이며, 제출 후보 10개 생성
- 보수적 제출 후보: `oof/submissions/exp51_logistic_cnx_eff_prob_rank_0p3_submit.csv`
- 공격적 제출 후보: `oof/submissions/exp51_ridge_cnx_eff_prob_only_50_submit.csv`
- `oof/submissions/exp51_logistic_cnx_eff_prob_rank_0p3_submit.csv` 제출 결과 leaderboard score `0.9551282051`로 최고점 갱신
- `oof/submissions/exp51_ridge_cnx_eff_prob_only_50_submit.csv` 제출 결과 leaderboard score `0.9519230769`로 EXP-26과 동점
- public 결과상 변경 수가 적은 보수 후보가 더 좋았으므로, 다음 제출도 low-change 후보를 우선한다
- EXP-52에서 EXP-51 최고 후보 주변 low-change 탐색 완료
- EXP-52 1순위 후보는 `oof/submissions/exp52_logistic_cnx_eff_prob_only_p0p25_t0p0_submit.csv`
- EXP-52 1순위 후보는 EXP-51 최고 대비 `test_0490` 하나만 1->0으로 변경
- `oof/submissions/exp52_logistic_cnx_eff_prob_only_p0p25_t0p0_submit.csv` 제출 결과 leaderboard score `0.9567307692`로 최고점 갱신
- 당시 최고 순위는 public leaderboard 기준 14위였음
- `oof/submissions/exp52_ridge_cnx_eff_prob_margin_p500_tm0p025_submit.csv` 제출 결과 leaderboard score `0.9551282051`로 하락
- EXP-52 결과를 바탕으로 `test_0490=0`, `test_0201=1`을 보호 조건에 추가
- EXP-53 constraint low-change search 완료, 보호 샘플 5개를 유지하는 비중복 후보가 없어 제출 파일 없음
- EXP-54 ConvNeXt seed123 5-fold OOF 완료, OOF accuracy `0.98102`
- EXP-54 단독 mean은 `test_0539=1`, `test_0490=1`로 제출 우선순위 낮음
- EXP-55 ConvNeXt seed123 포함 OOF stacking 완료
- EXP-55 1순위 후보: `oof/submissions/exp55_ridge_cnx123_eff_prob_rank_p50_tm0p1_submit.csv`
  - meta-CV accuracy `0.98677`
  - 현재 최고 EXP-52 대비 `test_0222` 하나만 1->0
  - 보호 샘플 5개 모두 유지
- `oof/submissions/exp55_ridge_cnx123_eff_prob_rank_p50_tm0p1_submit.csv` 제출 결과 leaderboard score `0.9551282051`로 하락
- EXP-55 결과를 바탕으로 `test_0222=1`을 보호 조건에 추가
- EXP-55 2순위 후보: `oof/submissions/exp55_ridge_cnx_eff_compact_p3000_t0p025_submit.csv`
  - 현재 최고 EXP-52 대비 `test_0504` 하나만 1->0
- EXP-56 보호 샘플 기반 low-change search 완료
- EXP-56 후보: `oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv`
  - meta-CV accuracy `0.98658`
  - 현재 최고 EXP-52 대비 `test_0504` 하나만 1->0
  - `test_0222=1` 포함 보호 샘플 6개 모두 유지
- `oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv` 제출 결과 leaderboard score `0.9583333333`으로 최고점 갱신
- 최고 순위는 최신 public leaderboard 기준 13위로 확인
- EXP-56 결과를 바탕으로 `test_0504=0`을 보호 조건에 추가
- EXP-57에서 EXP-56 winner 기준 보호 샘플 7개 유지 low-change search 완료
  - 최고 meta-CV accuracy `0.98696`
  - 보호 샘플 7개를 모두 유지하는 비중복 후보 0개
  - 제출 파일 없음
- EXP-58 calibrated OOF stacking 완료
  - 1순위 후보: `oof/submissions/exp58_svc_all_margin_stats_isotonic_p0p2_t0p575_submit.csv`
  - meta-CV accuracy `0.98639`
  - EXP-56 현재 최고 대비 `test_0188`, `test_0568` 두 개만 0->1 변경
  - 보호 샘플 7개 모두 유지
- `oof/submissions/exp58_svc_all_margin_stats_isotonic_p0p2_t0p575_submit.csv` 제출 결과 leaderboard score `0.9551282051`
  - EXP-56 최고점 `0.9583333333`보다 하락
  - `test_0188`, `test_0568`을 모두 1로 바꾸는 후보는 이후 우선순위 하락
- EXP-65 ConvNeXt V2 Tiny 사전 검토 완료
  - `timm==1.0.27` 추가
  - `convnextv2_tiny.fcmae_ft_in22k_in1k` pretrained 로딩 성공
  - 파라미터 약 27.87M, stage3+head trainable 약 15.49M
  - 224 입력, ImageNet mean/std 사용 가능
  - 예상 비용은 stage3+head 5-fold 기준 약 40~70분
  - 다음 실행 후보는 fold0 smoke 후 5-fold OOF
- EXP-65 ConvNeXt V2 Tiny stage3+head 5-fold OOF 완료
  - OOF accuracy `0.97680`, recall `0.97961`
  - confusion matrix `[[1299, 42], [79, 3796]]`
  - test mean label 0/1: `208/416`
  - EXP-56 대비 변경 샘플 30개
  - 보호 샘플 다수를 1로 바꿔 단독 mean 제출은 보류
  - 다음 단계는 EXP-66에서 protected stacking feature로 제한 사용
- EXP-66 ConvNeXt V2 포함 OOF stacking 완료
  - 1순위 후보: `oof/submissions/exp66_ridge_all_prob_stats_with_cnxv2_p10_t0p15_submit.csv`
  - meta-CV accuracy `0.98696`
  - EXP-56 대비 `test_0141` 하나만 1->0 변경
  - 보호 샘플 7개와 `test_0188/test_0568=0` 모두 유지
  - 제출 결과 leaderboard score `0.9567307692`
  - EXP-56 최고점보다 하락했으므로 `test_0141=1` 방향을 우선 유지
- EXP-67 RegNetY-3.2GF 5-fold OOF 완료
  - OOF accuracy `0.96166`, recall `0.95768`
  - confusion matrix `[[1305, 36], [164, 3711]]`
  - test mean label 0/1: `273/351`
  - EXP-56 대비 변경 샘플 69개
  - 단독 mean은 FN이 많고 보호 샘플 일부를 깨서 제출 제외
- EXP-68 RegNetY 포함 OOF stacking 완료
  - 최고 meta-CV accuracy `0.98735`
  - 보호 샘플 8개 유지 candidate 0개
  - 제출 파일 없음
  - RegNetY는 현재 guardrail 안에서는 제출 후보를 만들지 못함
- EXP-69 DINOv2 feature extractor + Ridge/Logistic 완료
  - `vit_small_patch14_dinov2` frozen feature 384-dim 추출
  - feature 추출 시간 약 623.6초
  - 최고 meta-CV accuracy `0.97373`
  - 보호 샘플 유지 candidate 0개
  - 제출 파일 없음
  - ViT/foundation feature family 확보 실험으로 기록하되, raw frozen feature + 선형 meta-model은 현재 제출 후보로 부적합
- EXP-70 EfficientNetV2-S 재설계 5-fold OOF 완료
  - `features[6:] + classifier` 학습으로 EXP-42의 `features[7]` 제한 설정보다 크게 개선
  - OOF accuracy `0.97642`, recall `0.97703`
  - confusion matrix `[[1307, 34], [89, 3786]]`
  - test mean label 0/1: `208/416`
  - EXP-56 대비 변경 수 34개
  - 보호 샘플 `test_0025`, `test_0539`, `test_0244`, `test_0490`, `test_0222`, `test_0504`를 깨므로 단독 제출 제외
  - 다음에는 EXP-70을 낮은 가중치의 stacking 보조 feature로만 검토
- EXP-71 EfficientNetV2-S 포함 OOF stacking 완료
  - 탐색 row 수 `6318`
  - 최고 meta-CV accuracy `0.98888`
  - 보호 샘플 유지 row 수 5개
  - 보호 샘플 유지 row는 모두 EXP-56과 동일 예측이라 비중복 candidate 0개
  - 제출 파일 없음
  - EfficientNetV2-S는 현재 guardrail 안에서 새 제출 후보를 만들지 못함
- EXP-72 ConvNeXt-Tiny seed777 5-fold OOF 완료
  - OOF accuracy `0.98006`, recall `0.98503`
  - confusion matrix `[[1295, 46], [58, 3817]]`
  - test mean label 0/1: `210/414`
  - EXP-56 대비 변경 수 20개
  - 보호 샘플 다수를 1로 바꾸므로 단독 제출 제외
  - 다음 단계는 EXP-72 포함 ConvNeXt seed stacking
- EXP-73 ConvNeXt seed777 포함 OOF stacking 완료
  - 탐색 row 수 `20592`
  - 최고 meta-CV accuracy `0.98754`
  - 보호 샘플 유지 row 수 21개
  - 보호 샘플 유지 row는 모두 EXP-56과 동일 예측이라 비중복 candidate 0개
  - 제출 파일 없음
  - ConvNeXt seed 추가만으로는 현재 guardrail 안에서 새 제출 후보를 만들지 못함
- EXP-77 ConvNeXt-Tiny Lion fold0 smoke 완료
  - Lion LR `2e-5`, weight decay `1e-4`
  - fold0 accuracy `0.98563`, recall `0.98581`
  - AdamW fold0 accuracy `0.98467`, recall `0.98323`보다 소폭 개선
  - confusion matrix `[[265, 4], [11, 764]]`
  - EXP-56 대비 변경 수 26개
  - `test_0025`, `test_0244`, `test_0504`는 유지했지만 `test_0539`, `test_0490`, `test_0141`을 깨므로 단독 제출 제외
  - Lion 5-fold OOF 확장 후보로 기록

## 리더보드 제출 기록

| 제출 순서 | 당시 저장 파일명 | 모델 | Validation Accuracy | Leaderboard Score | 순위 | 메모 |
|---:|---|---|---:|---:|---:|---|
| 1 | submissions/baseline_submit.csv | 개선 SimpleCNN | 0.97318 | 0.81891 | 34 | 최초 제출. 원본 baseline은 제출하지 않음 |
| 2 | submissions/resnet18_submit.csv | ResNet18 layer4+fc | 0.98659 | 0.9278846154 | 26 | 현재 최고 점수. SimpleCNN 대비 약 +0.10897 상승 |
| 3 | submissions/resnet18_epoch10_submit.csv | ResNet18 layer4+fc epoch10 | 0.98659 | 0.8782051282 | 26 | epoch 증가 후 leaderboard 하락 |
| 4 | submissions/resnet18_layer34_submit.csv | ResNet18 layer3+layer4+fc | 0.98563 | 0.8894230769 | 26 | fine-tuning 범위 확대 후 leaderboard 하락 |
| 5 | submissions/resnet18_threshold_0p7_submit.csv | ResNet18 threshold 0.70 | 0.9751 | 0.891025641 | 26 | threshold 조정 후 최고 점수 갱신 실패 |
| 6 | submissions/resnet18_threshold_0p65_submit.csv | ResNet18 threshold 0.65 | 0.9780 | 0.8798076923 | 26 | threshold 조정 후 최고 점수 갱신 실패 |
| 7 | submissions/efficientnet_b0_submit.csv | EfficientNet-B0 | 0.97605 | 0.8894230769 | 26 | 단독 성능은 낮지만 앙상블 후보 |
| 8 | submissions/exp07_resnet18_submit.csv | ResNet18 layer4+fc 재현 | 0.98659 | 0.9278846154 | 26 | EXP-02와 동일 점수 재현, 앙상블용 probability 저장 |
| 9 | submissions/exp08_softvote_r70_e30_submit.csv | ResNet18 0.7 + EfficientNet-B0 0.3 | 0.98563 | 0.9230769231 | 26 | soft voting 후 최고 점수 갱신 실패 |
| 10 | submissions/exp09_densenet121_submit.csv | DenseNet121 | 0.97797 | 0.9262820513 | 26 | ResNet 최고와 근접, 앙상블 유력 후보 |
| 11 | submissions/exp10_softvote_res50_den50_submit.csv | ResNet18 0.5 + DenseNet121 0.5 | 0.99042 | 0.9294871795 | 26 | soft voting으로 최고 점수 갱신 |
| 12 | submissions/exp10_softvote_res45_den55_submit.csv | ResNet18 0.45 + DenseNet121 0.55 | 0.98946 | 0.9342948718 | 26 | 현재 최고 점수 갱신 |
| 13 | submissions/exp10_softvote_res40_den60_submit.csv | ResNet18 0.4 + DenseNet121 0.6 | 0.98946 | 0.9358974359 | 26 | 현재 최고 점수 갱신 |
| 14 | submissions/exp11_softvote_res35_den65_submit.csv | ResNet18 0.35 + DenseNet121 0.65 | 0.98946 | 0.9342948718 | 26 | DenseNet 0.65는 최고점보다 소폭 하락 |
| 15 | submissions/exp12_convnext_tiny_submit.csv | ConvNeXt-Tiny | 0.98084 | 0.94711 | 23 | 현재 최고 점수 갱신 |
| 16 | submissions/exp13_softvote_cnx50_den50_submit.csv | ConvNeXt 0.5 + DenseNet121 0.5 | 0.98659 | 0.9455128205 | 23 | ConvNeXt 단독보다 하락 |
| 17 | submissions/exp14_convnext_tiny_sqrt_weight_submit.csv | ConvNeXt-Tiny sqrt class weight | 0.98563 | 0.9439102564 | 23 | validation은 개선됐지만 leaderboard 하락 |
| 18 | submissions/exp15_convnext_tiny_no_flip_submit.csv | ConvNeXt-Tiny no horizontal flip | 0.98084 | 0.9391025641 | 23 | validation은 EXP-12와 동일했지만 leaderboard 하락 |
| 19 | submissions/exp16_softvote_cnx12_70_seed7_30_submit.csv | ConvNeXt seed ensemble 0.70/0.30 | 0.98467 | 0.9455128205 | 23 | validation은 개선됐지만 leaderboard는 EXP-12보다 하락 |
| 20 | submissions/exp17_convnext_tiny_threshold_0p54_submit.csv | ConvNeXt threshold 0.54 | 0.97989 | 0.9455128205 | 23 | EXP-12 대비 test 예측 1개만 변경했지만 하락 |
| 21 | submissions/exp18_convnext_tiny_full_finetune_submit.csv | ConvNeXt-Tiny full fine-tuning | 0.97989 | 0.9471153846 | 23 | EXP-12와 leaderboard 동점, 앙상블 후보 |
| 22 | submissions/exp19_softvote_cnx12_70_full30_submit.csv | EXP-12 0.7 + EXP-18 0.3 soft voting | 0.98276 | 0.9439102564 | 23 | validation은 개선됐지만 leaderboard 하락 |
| 23 | submissions/exp20_ridge_stacking_alpha100_submit.csv | RidgeClassifier stacking alpha=100 | 0.98755 | 0.9503205128 | 19 | 현재 최고 점수 갱신 |
| 24 | submissions/exp21_logistic_stacking_C1_submit.csv | LogisticRegression stacking C=1.0 | 0.98755 | 0.9503205128 | 19 | Ridge와 leaderboard 동점 |
| 25 | submissions/exp22_ridge_feature_v2_alpha3000_submit.csv | RidgeClassifier feature v2 alpha=3000 | 0.98276 | 0.9471153846 | 19 | feature 추가 후 leaderboard 하락 |
| 26 | submissions/exp23_ridge_pruned_cnx_den_res_alpha100_submit.csv | Ridge feature pruning, CNX+DenseNet+ResNet | 0.98755 | 0.9487179487 | 19 | EXP-20보다 하락, EXP-12보다는 개선 |
| 27 | submissions/exp26_ridge_without_densenet_alpha30_submit.csv | Ridge without DenseNet | 0.98563 | 0.9519230769 | 19 | 현재 최고 점수 갱신, 순위는 유지 |
| 28 | submissions/exp34_ridge_threshold_m0p02_submit.csv | EXP-26 threshold -0.02 | 0.98563 | 0.9503205128 | 19 | `test_0539.png` 1개만 0->1 변경했지만 하락 |
| 29 | submissions/exp36_fixed0244_core_prob_only_alpha100_submit.csv | test_0244 고정 probability-only Ridge | 0.98946 | 0.9503205128 | 19 | `test_0025.png` 1개만 0->1 변경했지만 하락 |
| 30 | submissions/exp38_tta_all_prob_only_alpha50_submit.csv | TTA probability-only Ridge | 0.98851 | 0.9487179487 | 19 | `test_0244`는 유지했지만 `test_0025` 0->1 포함으로 하락 |
| 31 | submissions/exp44_regnet_drop_resnet18_add_reg_alpha150_submit.csv | RegNetY 포함 Ridge stacking | 0.98659 | 0.9503205128 | 19 | 민감 샘플 3개는 유지했지만 EXP-26보다 하락 |
| 32 | oof/submissions/exp51_logistic_cnx_eff_prob_rank_0p3_submit.csv | OOF Logistic, ConvNeXt+EfficientNet prob/rank | 0.98639 meta-CV | 0.9551282051 | 19 | 최고점 갱신, EXP-26 대비 +2샘플 |
| 33 | oof/submissions/exp51_ridge_cnx_eff_prob_only_50_submit.csv | OOF Ridge, ConvNeXt+EfficientNet prob only | 0.98677 meta-CV | 0.9519230769 | 19 | EXP-26과 동점, meta-CV 최고가 public 최고는 아니었음 |
| 34 | oof/submissions/exp52_logistic_cnx_eff_prob_only_p0p25_t0p0_submit.csv | OOF Logistic, ConvNeXt+EfficientNet prob only low-change | 0.98658 meta-CV | 0.9567307692 | 14 | 최고점 갱신, EXP-51 대비 `test_0490` 1개만 1->0 |
| 35 | oof/submissions/exp52_ridge_cnx_eff_prob_margin_p500_tm0p025_submit.csv | OOF Ridge, ConvNeXt+EfficientNet prob/margin low-change | 0.98677 meta-CV | 0.9551282051 | 19 | `test_0201`까지 1->0으로 바꿔 하락 |
| 36 | oof/submissions/exp55_ridge_cnx123_eff_prob_rank_p50_tm0p1_submit.csv | OOF Ridge, ConvNeXt seed123+EfficientNet prob/rank | 0.98677 meta-CV | 0.9551282051 | 19 | `test_0222` 1개만 1->0으로 바꿨지만 하락 |
| 37 | oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv | OOF Ridge, ConvNeXt+EfficientNet compact | 0.98658 meta-CV | 0.9583333333 | 13 | 최고점 갱신, `test_0504` 1개만 1->0 |
| 38 | oof/submissions/exp58_svc_all_margin_stats_isotonic_p0p2_t0p575_submit.csv | Calibrated SVC, OOF margin stats | 0.98639 meta-CV | 0.9551282051 | 19 | `test_0188`, `test_0568` 두 개를 0->1로 바꿨지만 하락 |
| 39 | oof/submissions/exp66_ridge_all_prob_stats_with_cnxv2_p10_t0p15_submit.csv | ConvNeXt V2 포함 OOF Ridge stacking | 0.98696 meta-CV | 0.9567307692 | 13 | `test_0141` 1개만 1->0으로 바꿨지만 EXP-56보다 하락 |
| 40 | oof/submissions/exp79_ridge_cnx_lion_eff_prob_only_alpha500_t0p2_submit.csv | Lion OOF 포함 Ridge stacking | 0.98447 meta-CV | 0.9567307692 | - | `test_0049` 1개만 0->1로 바꿨지만 EXP-56보다 하락 |
