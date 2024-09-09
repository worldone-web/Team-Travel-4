# STAN
## 특징
- 기존의 Next Location Recommendation 모델은, POI 추천에 지리적 인접성만 고려한다.
  - STAN은 시공간을 고려해, 인접하지 않거나, 연속적이지 않은 POI가 Next Location에 고려되게끔 한다.
  - 인전하지 않거나/연속적이지 않은 POI의 고려는 다음 이유로 필요하다.
    - 사용자는 며칠 전 방문한 POI가 방금 방문한 POI보다 더 의존할 수 있다.
    - 사용자가 관련성 있는 먼 장소를 방문하는 것은 드믈지 않다.

- STAN은 기존 모델과 달리 PIF를 반영한다.
  - 따라서 반복된 방문의 중요성을 고려할 수 있다.



## 구성
- Multimodal Embegging Module
  - Historical Trajectory (u, l, t)와 Location(l, lon, lat)를 인코딩

- Attention Aggregation Layer
  - 관련있는 POI를 집계
  - 각 방문의 representation과 가중치 갱신

- Attention Matching Layer
  - 갱신된 Matrix로 부터 후보 POI의 가능성을 예측

- Balanced Sampler
  - 크로스 앤트로피 계산


## 인터페이스
  - 사용자 u의 1~(m-1)번째 체크인 => [STAN] => m번째 방문 장소 예측
  - 사용자 u의 m번째 방문 POI가 라벨로 사용됨

## 평가
  - Recall@K

## 하이퍼파라미터
실험에서는 다음과 같이 사용
  - latent = 50  	# 잠재요인 수
  - max-length-of-trajectory-sequence=100  # Next POI 예측에 사용할 이전 check-in의 수
  - optimizer=Adam, 
  - learning-rate=0.003, 
  - dropout-rate=0.2
  - epoch=50
