# 위험 후보 선택 정책 비교 실험

## 1. 목적

YOLO 단독 탐지와 공간관계 분석 결과의 결합 정책을 비교하고,
위험 코드 재현율과 판단 보류 문제를 분석한다.

## 2. 정책 정의

| 논문 표기 | 코드 표기 | 설명 |
|---|---|---|
| A | A | YOLO 단독 |
| B | D | 교집합 기반 |
| C | E | YOLO 우선·공간관계 보완 |
| D | F | 합집합 기반 |

## 3. 데이터셋

- 전체: 1,256장 (위험 575장, 정상 681장)
- 유사 이미지 제외: 1,203장 (위험 545장, 정상 658장)

## 4. 실행 방법

프로젝트 루트에서 다음 명령어를 실행한다.

    python evaluation/evaluate_recall_priority.py

필요 입력 파일:

- evaluation/results/team_ablation/ablation_details.csv
- evaluation/results/ab_deduplicated/dedup_details.csv

출력 파일:

- evaluation/results/recall_priority/comparison.csv
- evaluation/results/recall_priority/prediction_details.csv

## 5. 주요 결과

전체 1,256장 기준:

| 정책 | 위험 코드 재현율 | Coverage |
|---|---:|---:|
| A | 80.87% | 100.00% |
| B | 56.00% | 79.22% |
| C | 90.78% | 100.00% |
| D | 90.96% | 100.00% |

C 정책은 YOLO 단독 A 대비 위험 코드 재현율이
9.91%p 향상되었다.

유사 이미지 제외 1,203장 평가에서 C 정책의
위험 코드 재현율은 90.28%였다.

## 6. 해석 시 주의사항

- C 정책은 위험 후보 선택 단계에 적용된다.
- 최종 법적 위반 여부 및 위험등급 확정과는 구분한다.
- 보류가 발생하는 정책은 Coverage를 함께 고려한다.
- 대표 정답 위험 코드와 다른 추가 탐지를 모두 실제 오탐으로 단정할 수 없다.
- A 대비 오경보율은 2.35%에서 3.67%로 증가하였다.
