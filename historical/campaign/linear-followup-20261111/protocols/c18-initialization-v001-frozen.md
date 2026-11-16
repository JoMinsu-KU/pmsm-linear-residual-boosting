# C18 초기 상태 개입 — root 실행 설계 고정

2026-11-11: 아래 제안을 C18 후속 탐색 설계로 채택한다. 네 dataset/arm 조합을 각 stage 최대 1200초, 출력 1GB로 제한한다. 각 stage에서 모델/예측 고정 후 저장 정답 평가를 별도 단계로 수행한다. 신규 확인이나 유의성 검정은 없다. 기존 자원·실패·확인 이력과 연장 기한을 유지한다. 학습 및 결과 계산의 독립 검증 후 기여를 판정한다.

---

# C18 초기 상태 개입 제안 — 실행 전 고정 대상

작성: 2026-11-11 KST, combination_ml. 이 문서는 root가 완료한 diagnostic-v002 결과와 기존 특징 생성 소스를 읽고 작성한 제안이다. 새 적합·예측·가중치 선택은 수행하지 않았다. root의 프로토콜 고정·실행 등록 전에는 실행하지 않는다.

## 관측된 문제와 남은 대안 설명

IPMSM id10에서 D9는 RMSE7.000295°C/편향−6.547779°C, 선형+잔차 전문가는 RMSE8.690756°C/편향−8.008112°C였다. 고정 반반의 RMSE 추가 손실 +0.828862°C 중 편향 제곱 항은 +0.680746°C, 분산 항은 +0.148117°C다. 전문가 품질 손실 +0.894339°C를 오차 차이의 상쇄 −0.065476°C가 충분히 보상하지 못한다. 잔차 보정 자체는 선형 사전의 RMSE9.211999→8.690756°C로 도움을 준다. “잔차 모델이 사전을 망가뜨렸다”는 설명은 이 주기에 맞지 않는다.

평가 시작300행 이후 첫40%가 반반 추가 손실 중 +0.670330°C를 차지한다. 초기 상태 가정은 검토할 가치가 있지만 아직 원인으로 입증되지 않았다. 전류·속도·냉각수·정답의 개별 FIT 범위 이탈은0%지만 **전체33 좌표 중 하나 이상이 범위를 벗어나는 비율은22.5728%**다. 특히 필터링한 `I*|speed|` 좌표와 현재 `I*|speed|`의 이탈이 크다. 주변 변수 범위가 같다는 사실만으로 결합 부하/이력 분포 차이가 없다고 할 수 없다.

## 하나의 개입 질문과 두 독립 요인 arm

질문은 “모델 종류·학습 자료·용량을 유지하면서 관측 이력 필터의 기록 시작 상태만 바꾸면, 원 IPMSM의 초반 음의 편향과 C18의 상대적 악화가 완화되는가?”다. 대상은 기존 동일33특징 구현을 쓰는 IPMSM과 Paderborn 두 자료다. RWTH는 별도 시간 간격 필터와 특징 설계라 이번 개입에 섞지 않는다.

기존 `code/combination_c12_fit_rc_v001.py::raw_features`는 span별 `[ambient,water]` 두 경계 필터를 **둘 다 ambient[0]**으로 시작한다. heating 세 입력 `[I²,|speed|²,I*|speed|]`는0으로 시작한다. 현재 코드는 잘못된 수치 구현이 아니라 명시된 가정이며 과거 결과는 그대로 남긴다.

기준 arm `legacy`는 이미 감사된 원 특징·모델·예측을 그대로 재사용한다. 새 학습은 다음 두 arm만 한다.

1. **own_boundary:** 경계 필터의 t=0 값을 `[ambient[0],water[0]]`로 둔다. heating 초기값은0 유지. 각 관측 채널을 자기 첫 관측값으로 시작하는 EMA 관례를 시험한다. rotor의 초기 정답은 사용하지 않는다.
2. **first_load:** 경계 필터는 원래처럼 두 채널 모두 ambient[0] 유지. 각 heating 필터의 t=0 값을 해당 첫 구동 입력 `[I²(0),|speed(0)|²,I(0)*|speed(0)|]`로 둔다. 기록 이전에 초기 부하가 유지됐다는 대안 가정을 시험한다. 이를 실제 정상상태 발열이나 초기 rotor 온도라고 해석하지 않는다.

두 변경을 함께 적용하는 joint arm은 추가하지 않는다. 두 arm을 보고 유리한 초기값·span·기간을 다시 탐색하지 않는다.

각 span s에서 a=1−2/(s+1), 필터는 z0=지정 초기값, `zt=a*z(t−1)+(1−a)*ut`로 원 구현과 같게 계산한다. 초기값만 Δz0 바꾼 필터 차이는 `a^t*Δz0`다. 이 식과 원 recurrence 재생성을 synthetic fixture 및 저장 특징으로 검증한다. 느린 span에서는 warmup300 이후에도 초기 상태 영향이 남을 수 있지만, 현재 코드의 span이 물리 시간상수를 식별했다는 뜻은 아니다.

## 공정한 재적합 범위

- 원 FIT8/IPMSM, FIT41/Paderborn과 모든 기존 비FIT 역할을 유지한다. 모든 프로파일 전체 행에서 필터를 계산한 뒤 원 학습300::5/평가300:를 그대로 쓴다. 초기 행 제거·추가 warmup·프로파일 제외를 하지 않는다.
- 각 arm에서 **D9와 선형 사전+잔차 모델이 같은 변경 특징**을 받는다. D9에만 원 특징을 유지한 채 다른 모델을 개선하는 비교는 하지 않는다.
- D9 용량/기타 설정은 원 FIT-CV가 선택한 설정을 그대로 고정한다. IPMSM7leaves×500iterations, Paderborn15leaves×1000iterations. λ는 각 자료 원 linear33 최종 모델의 FIT-CV 선택값을 그대로 사용하며 다시 고르지 않는다.
- FIT-only qscale·feature mean/std·target median/IQR와 동일 profile-balanced 행 가중을 원 절차대로 재계산한다. 동일33 입력의 선형 ridge(비벌점 절편 포함)를 재적합하고, 전체 FIT in-sample 잔차를 원15leaves×200iterations HGB로 적합한다. optimizer/seed/minleaf/learning rate/정규화 설정은 원 소스와 동일하다.
- 따라서 **두 arm×두 자료×(D9+잔차 HGB)=8개 booster 적합**, 선형 prior4개 적합이다. 기존 baseline에 대한 추가 재학습은 필요 없다. 단일 dataset/arm stage로 나누면 각 stage는 booster2개+linear1개다. 실제 cap과 비용은 root가 등록한다.
- 모든 변경 모델과 전체 비FIT 예측을 먼저 잠그고 그 뒤 원 정답을 읽는다. 이제 모든 역할은 노출된 탐색 자료임을 명시한다. 원 모델·예측·공개 패키지는 덮어쓰지 않는다.

최소 출력 패널은 arm별 D9,linear33 prior,linear33+잔차,`0.5*(D9+linear33+잔차)`,`0.5*(D9+linear33 prior)`다. baseline의 동일 패널과 기존 direct200/free25/용량 대조는 별도 보존하며, 변경 arm에서 적합하지 않은 모델을 은근히 다른 특징 모델로 대체하지 않는다.

## 어떤 결과가 무엇을 구별하는가

모든 역할의 profile-macroRMSE·과소/과대 지표·모든 주기별 차이를 보고한다. id10은 사전에 지정한 원인 진단이며 전체 성과를 대신하지 않는다. 동일5개의 평가행 진행 구간과 기존 MSE/ΔRMSE 분해를 재사용한다.

핵심 대조는 arm 안의 `H−D9`와 이 상대 차이가 baseline에서 얼마나 달라졌는지다. 단순히 변경 H의 절대 RMSE가 내려간 사실만으로 결합의 기여가 늘었다고 결론 내리지 않는다. D9도 같은 만큼 좋아졌다면 초기 상태 처리는 공통 표현의 효과일 수 있다. 선형 prior의 음의 편향·초반 손실이 줄고 반반의 상대 손실도 완화된다면 초기 상태 가정이 한 기여 요인이라는 **이 절차 안의 개입 근거**가 된다. 반대로 효과가 작거나 다른 기록에서 손해가 크면 결합 부하 이력의 분포 차이, 누락된 실제 초기 열 상태 등 대안 설명이 남는다.

이 개입은 모델을 재적합하므로 단일 물리적 원인을 고립한 실험은 아니다. 온도 정답을 사용한 초기화, 새로운 CAL 가중치, 유리한 구간만의 주 판정, 새 p-value 또는 확인 성공 선언은 없다. 기준보다 나아진 arm을 발견해도 이번 기존 자료는 새 확인으로 바뀌지 않는다.

## 출처

- `runs/diagnostic-v002/ipmsm__confirmation/profile_10.json`: whole.decomposition.linear_residual, whole.residual_effect, progress_blocks, outside_fit_ranges, feature_range_diagnostic.
- 원 특징 소스: campaign `code/combination_c12_fit_rc_v001.py`, raw_features/filtered/scaling/boost_features.
- 원 선형33 적합·예측: campaign `code/c15_paper_core_v001.py`, fit_linear/predict_linear/training.
- 같은33 특징과 선택 D9 설정의 현 C18 경로: `code/c18_capacity_v001.py`.

실행용 고정 프로토콜과 최종 소스·입력 hash는 root 승인 후 별도 기록한다. 이 제안은 canonical 상태 또는 확인 소비 이력을 수정하지 않는다.
