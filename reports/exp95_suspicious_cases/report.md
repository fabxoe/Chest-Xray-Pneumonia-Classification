# EXP-95 Suspicious Test Case Review Pack

목적: 제출 라벨을 leaderboard probing 방식으로 뒤집기보다, 모델 간 불일치와 불확실성이 큰 test 이미지를 리뷰 대상으로 묶는다.

- 기준 제출: `oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv`
- 사용 모델 수: `10`
- 사용 모델: `cnx, resnet, eff, cnx123, cnx777, cnx_lion, eff123, cnxv2, regnet, effv2`
- action counts: `{'keep': 553, 'review': 60, 'avoid_change_protected': 9, 'review_high_priority': 2}`

## 해석 원칙

- `suspicious_score`가 높다는 것은 바로 라벨을 바꾸라는 뜻이 아니다.
- `review_high_priority`는 모델 간 반대가 크고 EXP-56 decision score가 경계에 가까운 샘플이다.
- `avoid_change_protected`는 이전 public 결과에서 방향성이 확인된 guardrail 샘플이므로 변경하지 않는 쪽이 우선이다.
- 의료 이미지에서는 이미지 패널과 XAI 반응을 보더라도 실제 병변이라고 단정하지 않고, 모델 반응 위치와 예측 불확실성으로만 해석한다.

## Top 15

| file_name | exp56_label | suspicious_score | prob_std | vote_1_count | model_disagrees_with_exp56 | exp56_decision_score | recommended_action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test_0049.png | 0 | 0.9957 | 0.3981 | 6 | 6 | -0.0038 | avoid_change_protected |
| test_0539.png | 0 | 0.9879 | 0.3865 | 7 | 7 | 0.0114 | avoid_change_protected |
| test_0188.png | 0 | 0.9855 | 0.3183 | 7 | 7 | -0.0109 | review_high_priority |
| test_0620.png | 0 | 0.9830 | 0.3294 | 5 | 5 | -0.1307 | review |
| test_0315.png | 1 | 0.9799 | 0.3902 | 7 | 3 | 0.2386 | review |
| test_0377.png | 0 | 0.9759 | 0.2958 | 6 | 6 | -0.1389 | review_high_priority |
| test_0342.png | 0 | 0.9726 | 0.3057 | 3 | 3 | -0.2298 | review |
| test_0222.png | 1 | 0.9701 | 0.3169 | 6 | 4 | 0.1335 | avoid_change_protected |
| test_0040.png | 1 | 0.9691 | 0.3772 | 7 | 3 | 0.3094 | review |
| test_0478.png | 1 | 0.9683 | 0.3392 | 7 | 3 | 0.2698 | review |
| test_0060.png | 0 | 0.9677 | 0.3005 | 4 | 4 | -0.2410 | review |
| test_0600.png | 1 | 0.9676 | 0.3207 | 7 | 3 | 0.1530 | review |
| test_0547.png | 1 | 0.9669 | 0.3475 | 7 | 3 | 0.3453 | review |
| test_0137.png | 0 | 0.9659 | 0.3341 | 5 | 5 | -0.3621 | review |
| test_0244.png | 0 | 0.9658 | 0.3021 | 6 | 6 | -0.2610 | avoid_change_protected |

## Outputs

- `reports/exp95_suspicious_cases/exp95_suspicious_cases.csv`
- `reports/exp95_suspicious_cases/exp95_suspicious_top50.csv`
- `reports/exp95_suspicious_cases/exp95_suspicious_panel_top50.png`