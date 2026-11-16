# RWTH 신규 모터: 고정 상수 결합의 확인 실험 동결본 v002

상태: **ROOT FROZEN, 2026-11-10T15:36Z**. 원 초안 SHA256: b2415153c33295e2db823362f333319267187c9521a60f89769d28dc7b26d46c. 본문에서 제안/초안은 이 동결본에 의해 채택된 조건을 뜻한다. 2026-11-11 KST. 실행 코드는 작성하거나 실행하지 않았다. 읽은 새 자료는 이미 허용된 FIT 구조 보고서뿐이며 CAL/CONF 수치에는 접근하지 않았다. 과거 v001 구조 HOLD와 분할·확인 이력은 그대로 보존한다.

## 1. 주 질문과 선택 근거

주 후보는 직접 HGB `D9`와 양수 25차원 초기 응답+잔차 HGB `C15`의 **CAL profile-macroRMSE 최적 상수 혼합**이다. 새로운 입력 게이트나 비대칭 손실은 이번 새 자료의 후보가 아니다. 이 선택은 새 CAL/CONF 성능이 아니라 기존 노출 자료의 감사된 C17 v002 결과에 근거한다.

Paderborn의 기존 평가에서 해당 혼합은 3.1625767°C였다. 동일 규칙으로 적합한 다른 혼합은 direct200 3.5732454, linear33-residual 3.4664997, free25-residual 3.3188540, positive-prior 3.2231946°C였다. 기존 C16 raw MSE MLP 세 시드 평균은 3.2772696°C다. 원본 IPMSM에서는 주 혼합이 D9 단독으로 돌아갔다. 양 자료의 v002 독립 감사는 PASS이며 각각 16,155/45,292개 검사에서 실패 0건이다. 따라서 새 모터에서 결합 효용과 구성요소 특이성을 확인할 이유는 있지만, 이미 일반화나 물리 제약의 우월성이 확인된 것은 아니다.

주 형식 판정은 **고정된 주 혼합 대 D9** 한 건이다. 다른 혼합과 MLP는 실질 기여를 평가하는 필수 대조이며, CONF에서 유리한 쌍으로 주 후보를 바꾸지 않는다.

## 2. 분할과 구조 변경의 근거

`configs/rwth-partition-v001.json` SHA256 `5d5117312eee65bd9811f0ede7326d01d8a0a1aa3d0b0e3fb6ef60ae47959be1`의 ID 전용 20/8/10 분할을 그대로 사용한다.

- FIT: `[38,27,30,23,21,19,29,20,26,37,6,8,35,34,4,15,7,13,18,1]`.
- CAL: `[10,11,22,9,16,31,28,17]`.
- CONF: `[5,32,33,36,24,2,25,14,3,12]`.

자료 버전 DOI는 `10.5281/zenodo.15599966`, ZIP SHA256은 `a6b219a17acbbc2a9899fa7d750e1b0531756eeefcf16efd56009c43a0c4fce6`이다. 원 구조 보고 `runs/rwth-fit-structure-v001/summary.json`은 `HOLD_STRUCTURE`이며, CAL/CONF 내용 미개봉·특징 미생성·모델 미적합·확인 미소비를 기록한다. FIT 20개의 합계는 76,273행, 범위는 70~11,520행이다. 제공된 FIT 값은 유한하고 시간은 엄격히 증가하지만 간격이 일정하지 않으며, ID 23은 70행이어서 기존 300행 워밍업을 적용할 수 없다.

변경은 이 두 **구조 조건**에 한정한다.

1. 모든 프로파일에 워밍업 **0**, 학습 표본 `0::5`, 모든 검증/평가 시점 `0:`를 적용한다. 짧은 주기를 제외하거나 역할을 바꾸지 않는다.
2. 고정 샘플 필터를 아래의 시간 간격 반영 필터로 바꾼다. 보간·재표본화·온도 기반 초기화는 하지 않는다.

이 변경은 FIT 구조를 본 뒤, 새 자료의 어떤 모델 성능이나 CAL/CONF 값도 보기 전에 고정한다. 온도 성능에 따라 워밍업·필터·분할을 더 바꾸지 않는다. 짧은 주기도 profile-macro 평가에서 다른 주기와 같은 가중치를 갖는다는 해석상의 조건을 명시한다.

## 3. 관측 입력과 공통 33차원 인과 특징

입력은 `i,motor_speed,torque,coolant` 네 채널이다. `rotor`는 학습/평가 정답이고 `stator`는 주 분석에서 숫자로 파싱하거나 입력으로 쓰지 않는다. 주변 온도 또는 d/q축 전류·전압을 합성하지 않는다. 측정 토크를 이용할 수 있는 조건부 비교이며, 토크 센서 없는 배치 환경까지 적용성을 주장하지 않는다.

회전수 n의 공개 단위 rpm을 이용해 `omega=2*pi*n/60`, `p=torque*omega`를 계산한다. 부호 있는 원 관측을 그대로 유지하고 음의 속도·토크를 오류처럼 잘라내지 않는다. 순간 특징 8개를 다음 순서로 고정한다.

`[i, omega, torque, coolant, i², |omega|, omega², |p|]`.

필터 구동량은 `g=[coolant, i², |omega|, omega², |p|]`, spans는 `[16,64,256,1024,4096]`이다. `dt_ref`는 **전체 FIT20의 모든 연속 시각 차이를 합친 표본의 중앙값** 한 개로 고정한다. 이 timestamp-only 구조 상수는 CV 전체에서도 공통으로 사용하므로 CV fold의 시각 정보까지 이용한다는 사실을 기록한다. CAL/CONF 간격은 dt_ref 추정에 쓰지 않는다. 다른 입력·정답 스케일은 각 fold의 학습 19개에서만 적합한다.

프로파일 첫 시점은 냉각수 상태를 첫 coolant 값으로, 네 heating 상태를 0으로 두며 **첫 시점에는 필터를 진전시키지 않는다**. t>0에서

`alpha_(t,s) = 1 - (1 - 2/(s+1))^((time_t-time_(t-1))/dt_ref)`,

`z_(t,s) = (1-alpha_(t,s))*z_(t-1,s) + alpha_(t,s)*g_t`.

구현은 작은 간격의 소실을 피하도록 `-expm1(log1p(-2/(s+1))*dt/dt_ref)`를 사용한다. 시간과 dt_ref의 동일 단위가 상쇄되므로 독립 검증되지 않은 시간 단위를 초라고 선언하지 않는다. 초기 rotor 값은 어떤 상태에도 쓰지 않는다. dt_ref>0, dt>0, 유한값을 요구하며 시각 재정렬·클리핑·관측 제외를 하지 않는다.

25개 상태는 **coolant의 5 span, 이어서 heating 4채널 각각의 5 span** 순서로 저장한다. 공통 X33은 순간 8개와 상태 25개를 연결한다. 모든 모델·대조·시드는 같은 관측과 동일한 상태를 받는다. 현재 행까지의 값만 사용하며 profile ID·미래값·전체 프로파일 평균을 예측 입력으로 쓰지 않는다. 기존 C15와 33차원 수는 같지만 물리적 열 의미는 다른 자료 전용 표현이다.

## 4. 공통 적합·검증 가중과 스케일

학습하는 프로파일 집합을 F, 그 개수를 P, `0::5` 표본 수를 n_k라 한다. 통계적 손실은 `(1/P) sum_k (1/n_k) sum_t loss_(k,t)`이다. 라이브러리에는 평균이 1인 행 가중치 `N/(P*n_k)`를 전달한다. 최종 짧은 minibatch도 원래의 가중 평균 손실을 사용한다. CV는 각 프로파일을 한 번씩 제외한 20회 LOPO이며, 제외 프로파일의 **모든 행** RMSE를 계산해 20개를 산술평균한다. 최종 CAL/CONF도 모든 행의 profile-macroRMSE다.

각 적합에서 pooled 학습 행(`0::5`)만으로 X33 열 평균과 표준편차를 계산하고, 표준편차 하한은 1e-8이다. 입력 정규화는 HGB와 MLP에 동일하게 적용한다. 정답의 중앙값 `y_mid`와 `s_y=max(IQR(y),1°C)`도 같은 학습 행만으로 추정한다. 모든 표본 통계·열 순서·행 인덱스·원 CSV 행 번호를 저장한다. 목표 통계에 CAL/CONF 정답을 섞지 않는다. 모델별로 다른 프로파일을 제외하거나 입력을 추가하지 않는다.

## 5. 직접 HGB와 FIT-only D9 선택

`HistGradientBoostingRegressor`의 공통 설정은 squared_error, learning_rate .05, min_samples_leaf 20, max_bins 255, early_stopping false, random_state 2026111201이다. 아래 순서를 그대로 유지한다.

| grid ID | max_leaf_nodes | max_iter | l2_regularization |
|---:|---:|---:|---:|
| 0 | 15 | 200 | .1 |
| 1 | 15 | 500 | .1 |
| 2 | 15 | 1000 | .1 |
| 3 | 31 | 200 | .1 |
| 4 | 31 | 500 | .1 |
| 5 | 31 | 1000 | .1 |
| 6 | 7 | 200 | .1 |
| 7 | 7 | 500 | .1 |
| 8 | 15 | 100 | 1 |

9×20=180개 CV 적합 후 평균 held-profile RMSE가 최소인 grid를 선택한다. 정확한 목적함수 동률은 낮은 grid ID로 푼다. CAL/CONF로 grid를 바꾸지 않는다. 선택한 D9와 고정 grid0 `direct200`을 FIT20 전체에 각각 적합한다. D9=grid0이면 완전히 같은 적합을 하나만 수행하고 두 논리 방법의 alias를 명시한다. 최종 target은 °C 단위 rotor다. CV와 최종 모델의 tree nodes/bin thresholds/weights/seed/학습 표본·예측을 감사 가능하게 저장한다.

## 6. 25차원 초기 응답과 대조: 정확한 정규화와 목적함수

각 적합의 pooled FIT `0::5`에서 네 heating 구동량의 RMS `q_j=max(sqrt(mean(g_j²)),1e-8)`를 구한다. 하나의 q_j를 해당 구동량의 모든 span에 공통 사용한다. 필터 상태 H를 q_j로 나누되 평균을 빼지 않는다. coolant 상태 B의 정규화는 `(B-y_mid)/s_y`, target은 `u=(rotor-y_mid)/s_y`다. 따라서 A25는 `[(B-y_mid)/s_y, H_j/q_j]` 순서다.

양수 초기 응답 계수 θ25는 다음 목적함수를 최소화한다.

`J(θ)= (1/P) sum_k mean_t((A_(k,t) θ-u_(k,t))²) + λ ||θ||²`,

제약은 `θ>=0`, `sum_(j=0..4)θ_j=1`이며 **별도의 절편은 없다**. 출력은 `y_mid+s_y*Aθ`다. 제약 아래에서 이 식은 `sum_s θ_s B_s + s_y*sum_(heating) θ_j H_j/q_j`가 되어, y_mid가 실제 경계 온도의 상수 이득 조건을 바꾸지 않는다. 이 조건은 안정한 필터 사전의 구조이지 실제 열저항/열용량 식별이나 완전한 열방정식의 증명은 아니다.

`free25`는 **동일 A25·target·λ penalty에서 두 제약을 모두 제거**한다. 추가 독립 절편을 두지 않는다. 따라서 positive25/free25는 제약만 다른 대조다. 자유 계수에서 y_mid 복원은 계수합과 연결된 상수항을 유발하지만 별도 자유 절편은 아니다. 과거 C15의 legacy free25+독립 절편 대조와 구별한다.

`linear33`은 정규화 X33과 target u를 이용하며 비벌점 독립 절편 하나를 둔다. 목적함수는 같은 profile-balanced MSE+λ times 33개 slope 제곱합이다. 단위 복원은 `y_mid+s_y*(β0+Z33β)`다. 이것은 더 많은 관측 특징을 활용한 일반 선형 사전 대조이며 제약만의 ablation은 아니다.

세 family 각각 λ `[.0001,.01,1]`을 FIT20 LOPO로 선택한다(3×20×3=180개 작은 선형/QP 적합). 모든 fold의 A/q/target/feature scaler는 그 fold의 학습 19개만 사용한다. 정확 동률은 작은 λ를 택한다. 선택된 λ로 각 prior를 FIT20 전체에 적합한다. 이후 각 prior의 **전체 FIT in-sample 잔차**를 같은 X33의 고정 grid0 HGB(15 leaves,200 iterations)로 예측한다. 양수 prior+잔차가 C15, free25+잔차와 linear33+잔차가 강한 대조다. OOF 잔차 booster를 새로 추가하지 않는다.

### 수치 적격성은 결과 이전에 고정

정규화 행 가중 W의 합은 1이다. `Q=2(A'WA+λI)`, `r=2A'Wu`, `c=u'Wu`; prior QP는 `.5θ'Qθ-r'θ+c`다. Q의 Cholesky 성공과 유한값을 요구한다. `e=[1,1,1,1,1,0,...,0]`이고, 양수 QP는 feasible active-set/KKT 해를 사용한다. 최대 1000 active-set outer iterations, 최소 계수≥−1e-10, `|e'θ−1|≤1e-10`을 고정한다. 수치 실패 시 다른 과학 방법으로 조용히 fallback하지 않는다.

인증에 equality multiplier ν와 nonnegative multiplier μ를 저장한다. `b=r−νe+μ`, `z=Q^−1 b`, `dual=c−ν−.5*b'z`이다. `μ=max(Qθ−r+νe,0)`로 dual feasibility를 확보하고 stationarity/complementarity를 기록한다. 안정한 gap은 `.5*(θ−z)'Q*(θ−z)+μ'θ+ν*(1−e'θ)`이다. `gap≥−1e-10`, `gap≤1e-8*(1+|primal|)`, `|primal−dual−gap|`도 같은 gap 허용값 이하를 요구한다. 계수·Q/r/c·ν/μ·직접/안정 gap을 독립 감사한다. 미세 허용범위를 벗어난 음의 계수 clipping이나 성능 기반 허용오차 확대는 하지 않는다. free25/linear33은 같은 이차 목적함수의 직접 선형 해와 잔차·목적함수 인증을 저장한다.

## 7. 신경망 강한 대조

별도 조합이나 위험 손실을 넣지 않는 raw MSE MLP 세 개를 적합한다. 구조33→64→32→1, ReLU, 4,289개 파라미터, dropout 없음. 시드는 `[2026111701,2026111702,2026111703]`. AdamW(lr .001, betas .9/.999, eps1e-8, weight_decay .0001), batch4096, 고정200epochs, early stopping/CAL checkpoint 선택 없음. 각 epoch의 전체 행 순서는 NumPy PCG64(seed+epoch−1)의 명시적 int64 permutation이다. FP64 특징/스케일, FP32 네트워크·행 가중, FP64 단위 복원을 사용한다.

목표는 FIT20에서 정한 u, loss는 가중 squared error다. 초기/최종 weight arrays, optimizer 설정, 정확 표본/가중치, epoch/permutation 기록을 저장한다. 세 시드를 모두 보고하고 **시드별 지표의 산술평균**을 부가 요약한다. 최선 시드 선택이나 세 예측의 앙상블로 바꾸지 않는다. 새로운 MLP epoch/grid 탐색이나 새 데이터의 CAL로 구조를 고르는 절차는 없다.

## 8. 동일 CAL 결합과 정확한 출력 패널

왼쪽은 모든 쌍에서 D9다. 오른쪽은 이 고정 순서다.

1. C15 (양수25+잔차): **주 후보**.
2. direct200.
3. linear33+잔차.
4. free25+잔차.
5. positive25 prior 단독.

각 쌍에 대해 CAL8의 모든 행에서 `mean_profile RMSE(d+w*(h−d),y)`를 최소화하는 w∈[0,1] 한 개만 적합한다. 기존 C17과 동일한 bounded scalar solver(xatol1e-12,maxiter500)와 명시적 w=0,1 끝점을 비교하고, 목적함수 정확 동률은 낮은 w로 푼다. subgradient interval·convex gap·모든 endpoint 목적함수를 저장하고 독립 scalar oracle로 확인한다. 최적 CAL 가중치가 CONF 최적 가중치라는 의미는 아니다.

각 쌍의 w=0,1,.5를 비선택 진단으로 함께 보존하므로 **5쌍×4규칙=20개 논리 출력 + raw MLP3개**다. primary는 첫 쌍의 CAL-macroRMSE 상수 혼합 하나다. 별도 CAL-MSE 선택기·입력 gate·shift·risk loss를 새 확인에 추가하지 않는다. CAL8 LOPO로 같은 스칼라 규칙의 안정성을 진단할 수 있으나 최종 규칙/정규화를 바꾸는 근거로 사용하지 않는다. 원 전문가 예측과 모든 진단 결과를 함께 보존한다.

## 9. 데이터 접근·단계·기한

최초 캐시는 FIT와 CAL까지만 만든다. FIT에는 rotor 정답을 포함한다. CAL은 base training/grid/λ/epoch 선택에 쓰지 않고 **base 모델 고정 이후** 결합 가중치와 사전 지정 진단에만 쓴다. CONF는 root의 마지막 확인 admission 전에는 멤버 내용을 열거나 특징 캐시를 만들지 않는다. 단일 archive가 이미 로컬에 있다는 사실은 CONF 수치 접근 허가를 뜻하지 않는다.

제안된 bounded 단계:

1. 고정 FIT 구조 변경·시각/특징 recurrence 수치 fixture와 FIT 캐시 생성. 원 HOLD 보존, 새 stage 사용.
2. FIT 두 프로파일 비용 측정: largest HGB 설정 한 개와 MLP2epochs만 수행. 과학적 성능 선택에 사용하지 않으며 출력/시간만으로 잔여 기한 내 전체 적합 가능성을 판단한다.
3. FIT-only 선형 λ LOPO와 D9 LOPO. HGB는 한 stage당 한 grid의20fold, cap1200초; 선형 전체 cap600초. 상태를 완성된 개별 job 단위로 잠그고 parent가 잔여 기한에 맞춰 등록한다. 결과가 불리하다고 나머지 grid를 생략하지 않는다.
4. D9/direct200, prior3개 및 residual3개, MLP3seed 최종 적합. MLP는 seed별 stage, 실측 근거를 갖춘 cap≤1800초. 동시 GPU1개. 모델 전체를 잠근다.
5. CAL 예측, 동일5쌍 scalar 적합, CAL LOPO 진단, 독립 감사. 코드·scale·grid·λ·seed·모델·CAL 선택 전부 동결한다. 각 CPU 후처리 stage≤600초.
6. root가 남은 확인1회를 기존 이력에 소비한 뒤 CONF 입력을 읽는다. 모든11개 base 모델(aliases 제외 실제 수는 다를 수 있음)과20+3개 논리 예측을 **전부 저장·잠근 다음** rotor 정답을 별도 단계에서 읽는다. CONF stator는 계속 미사용이다.
7. 완전한 독립 감사와 고정 판정. 각 audit stage≤1200초로 분할 가능하되 검사 누락을 완전 감사라고 부르지 않는다.

전체FIT 76,273행이므로 학습 stride5는 약15.3k행 규모지만, 실제 학습 시간을 측정 전 확정하지 않는다. CPU1thread, 추가 캐시/모델/출력 상한1GB를 제안한다. 기존 총자원 한도와 **새 학습 cutoff 2026-11-10T20:23:59Z, 전체 deadline2026-11-11T00:23:59Z**를 유지한다. 기한 내 전체 적합·감사·확인 잠금이 불가능하면 CONF를 열지 않고 미완료로 보고한다. 중간 실패를 새 확인으로 초기화하지 않는다.

## 10. 확인 판정과 해석

CONF10의 주 paired 차이는 `Δ_k=RMSE_primary,k−RMSE_D9,k`다. 주 형식 성공은 **mean(Δ)<0 및 개선 방향 일측 exact sign p≤.025** 두 조건을 모두 만족하는 경우다. 정확한 0 차이는 sign test에서 제외하고 유효 n 및 tie 수를 보고한다. 무승부 없는10개에서는9승(p=.0107421875) 이상이 필요하다. 새 p-value 여러 개 중 하나를 고르거나 효과 크기 최소3%를 사후 도입하지 않는다. 남은 확인 시도는1회이며 이전 확인의 소비 이력은 그대로 유지한다.

필수 보조 비교는 네 다른 CAL 혼합 및 MLP 세 시드/시드평균과의 평균·프로파일별 RMSE다. 주 형식 판정이 통과해도 단순 direct/linear/free 혼합이나 MLP가 같거나 더 좋다면 그 사실을 기여 평가에서 숨기지 않는다. 이 보조 비교에 새로운 유의성 성공 선언을 붙이지 않는다. 이득이 일반 앙상블 효과인지 positive prior 특이 효과인지 분리하여 설명한다.

under-RMS·over-MAE·FIT95 이상 온도 구간 지표를 기술 통계로 보고할 수 있지만 주 기준을 바꾸지 않는다. 시간 단위가 검증되지 않았으므로 지속시간은 timestamp units 또는 dt_ref-normalized units로만 표기하며 초 단위 안전 주장이나 conformal 보증을 하지 않는다. 초기 행을 모두 포함하므로 잠재적인 cold-start 오차도 주 결과에 남는다. 기간/온도 구간 중 유리한 부분만으로 주 성패를 재판정하지 않는다.

최종 가능한 결론은 성공/실패/수치 또는 실행 미완료이며, 작은 실용적 개선도 숨기지 않고 보고한다. 단일 추가 모터의 10개 운전 주기라는 범위, 공개 자료 사용, 토크 입력 가정, 불완전한 열 경계, 동일 모터 주기 간 의존성을 명시한다. 확인 성공이 자동으로 독창적 이론·안전성·SCIE 게재 가능성을 입증하는 것은 아니다.


## 11. Root 최종 판정·실행 명세 보충 (모든 새 성능 열람 전 고정)

형식 검정에서 유효 n=0이면 p=1이다. 그 외 wins=w, n_eff=n에서 p=sum(comb(n,j),j=w..n)/2**n. mean delta에는 무승부를 포함한 CONF10 전체가 남는다. 운전 주기 간 독립성 가정이 완전 검증되지 않았음을 명시한다.

**주제 선정의 실질 기여 조건**은 위 형식 성공과 더불어, 주 혼합의 CONF macroRMSE가 (a) 네 개 다른 CAL 혼합, (b) C15와의 고정 50:50 혼합, (c) raw MSE MLP 3seed의 metric 평균 각각보다 모두 엄격하게 낮은 것이다. 크기 최소 3% 조건은 없다. 이 조건은 새로운 유의성 주장이 아닌 사전 고정된 평균 성능 우월 조건이다. 같으면 별도 기여가 확인되지 않은 것으로 판정한다. 제약 자체에 대한 인과적/이론적 일반성은 이 비교만으로 주장하지 않는다. 한 모터 또는 몇 주기에 이득이 집중되면 사실대로 한정한다. 주 판정 실패를 다른 보조 지표나 다른 혼합의 성공으로 대체하지 않는다.

진단용 각 프로파일 제외 평균과 전체 주기별 차이를 모두 공개한다. 일부 제외 결과로 주 판정을 뒤집지 않는다. 공통 목표는 rotor이고 측정 토크 사용은 명시한다.

데이터 준비 단계는 FIT/CAL source arrays와 X33을 별도 역할로 저장할 수 있다. CAL rotor는 base fitting이 불가능한 분리 파일에 보관하며 fitter는 FIT cache만 읽도록 구현한다. CAL 결합 단계는 모든 base model lock 이후에만 CAL labels를 읽는다. CONF cache 생성·rotor 열람은 root 확인 admission 이후 별도 predict/evaluate 단계로 분리한다. 모든 단계에 읽은 파일 SHA256과 역할을 남긴다. cache 최소 표준은 profile_id, source_row(0-based data row), time, X33, 각 프로파일의 원 raw inputs이며 FIT label만 FIT cache에 포함한다. worker 간 실제 key/schema는 구현 전에 문서화한다.

추가 모델/캐시/출력 합계 상한 1GB, CPU single thread, 동시 GPU1, 기존 전체 상한·기한을 유지한다. 실패한 수치 solve는 원 기록과 고정 목적함수를 유지하는 구현 보정만 최대 두 번 허용한다. 새 모델 가족·탐색 범위·성능 기반 전처리 변경은 본 확인에 추가하지 않는다.
