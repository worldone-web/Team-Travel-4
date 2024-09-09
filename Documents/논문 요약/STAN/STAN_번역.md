# STAN: 다음 장소 추천을 위한 시공간 주의집중 네트워크 (Spatio-Tempral Attention Network for Next Location Recommendation)

## ABSTRACT
다음 장소 추천은 다양한 장소 기반 어플리케이션의 핵심이다.   
(The next location recommendation is at the core of various location based application.)

현재 최첨단 모델은 공간적 희소성을 계층적 그리드로 해결하고 명시적 시간 간격으로 시간적 관계를 모델화하려 시도하나, 해결되지 않은 문제가 남아있다.   
(Current state-of-the-art models have attempted to solve spatial sparsity with hierachical gridding and model temporal relation with explicit time intervals while some vital questions remain unsolved.)

인접하지 않은 장소와 연속적이지 않은 방문은 사용자의 행동을 이해하는 데 사소하지 않은 상관관계를 제공하나 거의 고려되지 않았다.   
(Non-adjacent locations and non-consecutive visits provide non-trivial correlations for understanding a user's behavior but were rarely considered.)

사용자 궤적으로 부터 모든 관련된 방문을 집계하고 가중화된 표현에서 가장 그럴듯한 후보를 이끌어내기 위해, 장소 추천을 위한 STAN을 제안한다.   
(To aggregate all relevant visits from user trajectory and recall the most plausible condidates from weighted representations, here we propse a Spatio-Temporal Attention Network(STAN) for location recommation.)

STAN은 궤적을 따라 self-attention 레이어를 사용해 모든 체크인의 상대적인 시공간 정보를 명시적으로 활용한다.   
(STAN explicitly exploits relative spatiotemporal information of all the check-ins with self attention layers along the trajectory.)

이러한 개선으로 명시적인 시공간 효과를 통해 인접하지 않은 장소와 비연속적인 체크인 사이의 지점간 상호작용을 가능해졌다.   
(This improvement allows a point to point iteraction between non-adjacent locations and non-consecutive check-ins with explicit spatio-temporal effect.)

STAN은 먼저 사용자 궤적에서 시공간 상관관계를 집계한 다음 PIF(개인화 항목 빈도)를 고려한 대상을 이끌어 내는 2중 레이어 주의 아키텍처를 사용한다.   
(STAN uses a bi-layer attention architecture that firstly aggregates spatiotemporal correlation within user trajectory and then recalls the target with consideration of personalized item frequency(PIF).)

시각화를 통해, STAN이 위의 직관과 일치함을 보여준다.   
(By visualization, we show that STAN is in line with the above intuition.)

실험적 결과는 우리의 모델이 기존의 최첨단 모델보다 9-17% 더 나은 성능을 보인다는 것을 명백히 보여준다.
(Experimental results unequivocally show that our model outperforms the existing state-of-the-art methods by 9-17%)

## INTRODUCTION
다음 POI 추천은 Yelp, Foursquare 그리고 Uber와 같은 장소 기반 서비스의 성장으로 최근 몇년간 집중적 연구를 불러일으켰다.
(Next Point-of-Interest(POI) recommendation raises intensive studies in recent years owing to the growth of location-based services such as Yelp, Foursquare, and Uber)

방대한 과거 체크인 데이터는 서비스 제공자에게 다음 움직임에 대한 사용자의 선호도를 이해하기 위한 귀중한 정보를 제공한다. 과거의 궤적은 모든 의사 결정에서 사용자 행동 패턴을 드러내기 때문이다.   
(The large volume of historical check-in data gives service providers invaluable information to understand user perferences on next movements, as the historical trajectories reveal the user's behavioral pattern in making every decision.)

한편, 이러한 시스템은 사용자에게 현재 상태뿐만 아니라 이전 방문을 기반으로 어디를 갈 지와 어떻게 하루를 계획할지 편리함을 제공할 수 있다.   
(Meanwhile, such as system can also provide users with the convenience to decide where to go and how to plan the day, based on previous visits as well as current status.)

이전의 접근법은 다양한 관점을 광범위하게 연구했고 개인화 추천을 위한 많은 모델을 제안했다.  
(Previous approaches have extensively studied various aspects and propsed many models to make a personalized recommendation.)

초기 모델은 주로 마르코프 연쇄와 같은 순차적 전이에 집중했다.   
(Early models mainly focus on sequential transitions, such as Markov chains.)

나중에, 메모리 메커니즘을 갖춘 RNN은 추천 정확도를 향상시켰고, 사용자 궤적의 긴 주기와 짧은 순차적 특징을 더 잘 추출하기 위한 RNN 변형을 제안하려는 다음 작업에 영감을 주었다.   
(Later on, recurrent neural networks(RNNs) with memory mechanism improved recomendation precision, inspiring following works to proposed RNN variants to better extract the long periodic and short sequential features of user trajectories.)

순차적 규칙성 외에도, 연구자들은 시간적 공간적 관계를 순차적 추천을 보조하기 위해 활용했다.   
(Besides sequential regularities, researchers have exploited temporal and spatial relation to assist sequential recommendation.)

최근 최첨단 모델은 두 연속적인 방문 사이의 시간 간격 및/또는 공간적 거리를 제공해 명시적으로 각 이동 사이의 시공간 간격의 영향을 표현했다.   
(The recent state-of-the art models fed time intervals and/or spatial distance between two consecutive visits to explicitly represent the effect of the spatiotemporal gap between each movement.)

선행 연구 또한 시공간 정보의 희소성 문제를 이산적으로 시간을 표기하고 공간을 계층적 그리드로 분할하여 다뤘다.   
(Prior works have also addressed the sparsity problem of spatiotemporal information by discretely denoting time in hours and partitioning spatial areas by hierachical grids.)

게다가, 그들은 뉴럴 아키텍처를 변경하거나 추가적인 모듈을 쌓아 추가적인 정보를 통합했다.   
(Beside, they modified neural architectures or stacked extra modules to integrate these additional information.)


지속적으로 출시되는 새로운 모델이 이동성 예측의 이해를 향상시키면서, 몇가지 핵심 문제가 해결되지 않고 남아있다.   
(With the continuously upcoming novel models pushing forward our understanding of mobility prediction, several key problems remain unsolved)

첫째, 인접하지 않은 장소와 비연속적인 방문 사이의 상관관계가 효과적으로 학습되지 않았다.   
(First, the correlations between non-adjacent locations and non-contiguous visits have not been learned effectively.)

사용자의 이동성은 며칠 전 방문한 관련 장소를 방금 방문한 관련없는 장소보다 더 의존할 수 있다.   
(The mobility of user may depend more on relevant locations visited a few days ago rather than irrelevant locations visited just now.)

더욱이, 사용자가 기능적으로 관련있는/유사한 먼 장소를 방문하는 것은 드믈지 않다.   
(Moreover, it is not rare for a user to visit distanced locations that are functionally relevant/similar.)

그림 1에서 보여주는 특별한 예시에서, 사용자는 항상 금요일 저녁에 직장 근처의 특정 레스토랑에서 식사를 하고, 토요일 아침에 몇몇 쇼핑몰을 간 다음 저녁에는 그곳 근처의 임의의 레스토랑에서 식사를 한다.   
(In a special example shown in Figure 1, a user always dines at a certaion restaurant near the workplace on Friday evening, go to some shopping mall on Saturday morning, and dine at a random restaurant near a mall on Saturday evening.)

이 경우, 사용자는 사실상 인접하지 않은 레스토랑을 비연속적으로 두번 방문하였다. 집과 쇼핑몰 사이의 공간적 거리와 식사 사이의 명시적 시간 간격은 토요일 저녁의 정확한 장소 예측을 위한 사소하지 않은 정보를 제공한다.   
(In this case, the user have de facto made two non-consecutive visits to non-adjacent restaurants, where the explicit spatial distances between home and shopping malls and the explicit temporal interval between meals provide non-trivial information for predicting the exact location for Saturday dinner)

그러나, 대부분의 현재 모델은 현재와 미래 단계간 공간적 및/또는 시간적 차이에 집중하고 궤적의 시공간적 상관관계를 무시한다.   
(However, most current models focused on spatial and/or temporal differences between current and future steps while ignoring spatiotemporal correlation with the trajecctory.)

둘째, 공간 이산화를 위해 이전에 실행된 계층적 그리드는 공간적 거리에 둔감하다.   
(Second, the previously practiced hierarchical gridding for spatial discretization is insensitive to spatial distance.)

그리드 기반 주의 집중 네트워크는 이웃 장소를 집계하나, 공간적 거리를 인지할 수 없다.   
(The gridding-based attention network aggregate neighboring locations but cannot perceive spatial distance.)

서로 가까운 그리드는 그렇지 않은 차이를 나타내지 않아 많은 공간적 정보를 전달한다.   
(Grids that are close to each other reflect no difference to those that are not, tossing a lot of spatial information.)

셋째, 이전 모델은 광범위하게 개인화 항목 빈도를 간과했다.   
(Third, previous models extensively overlooked personalized item frequency(PIF).)

같은 장소에 대한 반복된 방문은 반복된 장소의 중요성과 사용자의 재방문 가능성을 강조하는 빈도를 나타낸다.   
(Repeated visits to the same plcae reflect the frequency, which emphasizes the importance of the repeated locations and the possibility of users revisiting.)

이전 RNN 기반 모델과 self-attention 모델은 메모리 매커니즘과 정규화 연산으로 인해 PIF를 반영하지 못했다.   
(Previous RNN-based models and self-attention models can hardly reflect PIF due to the memory mechanism and normalization operation respectively.)

이를 위해 우리는 다음 장소 추천을 위한 STAN을 제안했다.   
(To this end, we proposed STAN, a Spatio-Temporal Self-Attention Network for the next location recommendation.)

STAN에서 과거 궤적 내 중요한 장소를 집계하기 위한 self-attention 레이어와 가장 그럴듯한 후보지를 recall하기 위한 self-attention 레이어를 설계한다. 둘 모두 지점간 명시적 시공간 영향을 고려한다.   
(In STAN, we design a self-attention layer for aggregating important locations within the historical trajectory and another self-attention layer for recalling the most plausible candidates, both with the consideration of a point to point explicit spatiotemporal effect.)

Self-attention layers는 다른 가중치를 궤적 내 각 방문에 할당할 수 있다. 이것은 흔히 순환 레이어에 사용되는 장기 의존성 문제를 극복한다.    
(Self-attention layers can assign different weights to each visit within the trajectory, which overcomes the long-term dependency problem of the commonly used recurrent layers.)

이중 계증 시스템은 PIF를 고려한 효과적인 집게를 가능하게 한다.
(The bi-layer system allows effective aggregation that considers PIF.)

우리는 희소성 문제를 해결하고 시공간전이 행렬의 임베딩을 위한 선형 보간을 사용한다. 이것은 GPS 그리드와 달리 공간적 거리에 민감하다.   
(We employ linear interpolation for the embedding of spatiotemporal transition matrix to address the sparsity problem, which is sensitive to spatial distance unlike GPS gridding.)

STAN은 모델에 제공되는 모든 체크인의 시공간 영향으로 인접하지 않은 장소와 연속적이지 않은 방문 사이의 상관관계를 학습할 수 있다.   
(STAN can learn correlations between non-adjacent locations and non-contiguous visits owing to the spatiotemporal effect of all check-ins fed into the model.)

요약하자면, 우리의 기여는 다음과 같다.   
(To summarize, out contributions are listed as follows:)

- 우리는 관련 장소를 집계하기 위한 시공간 영향을 충분히 고려하기 위해 STAN을 제안한다. STAN은 인접하지 않은 장소와 비연속적인 방문 사이의 규칙성을 학습하기 위해 시공간적 상관관계를 명시적으로 통합하는 POI 추천의 첫번째 모델이다.   
- (We propose STAN, a spatiotemporal bi-attention model, to fully consider the spatiotemporal effect for aggregating relevant locations. To out best recollection, STAN is the first model in POI recommendation that explicitly incorporates spatiotemporal correlation to learn the regularities between non-adjacent locations and non-contiguous visits.)

- 우리는 GPS 그리드를 공간적 이산화를 위한 단순한 선형 보간법 기술로 대체한다. 이것은 단지 이웃을 집계하는 대신, 공간적 거리를 복구하고 사용자의 공간적 선호도를 나타낸다. 정확한 표현을 위해 우리는 이 방법을 STAN에 통합한다.
- (We replace the GPS gridding with a simple linear interpolation technique for spatial discretization, which can recover spatial distances and reflect user spatial preference, instead of merely aggregating neighbors. We integrate this method into STAN for more accurate representation.)

- 우리는 특히 PIF를 위한 이중 집중 주의 아키텍처를 제한한다. 첫번째 레이어는 업데이트된 표현을 위해 궤적 내의 관련된 장소를 집계한다. 이로써 두번째 레이어는 반복을 포함하여 모든 체크인에 대상을 일치시킬 수 있다.
- (We specifically propose a bi-attention architecture for PIF. The first layer aggregates relevant locations within the trajectory for updated representation, so that the second layer can match the target to all check-ins, including repetition.)

- 제안된 방법의 성능을 평가하기 위해 실제 세계에 대한 4개의 데이터세트에 실험을 수행한다. 결과는 제안된 STAN이 최신 모델의 정확도를 10%이상 능가한다는 것을 보여준다
- (Experiments on four real-world datasets are conducted to evaluate the performances of the proposed method. The result shows that the proposed STAN outperforms that accuracy of state-of-the-art models by more that 10%.)

## RELATED WORKS
이 섹션에서, 우리는 순차적 추천과 다음 POI 추천에 관한 일부 연구를 간략하게 검토한다.   
(In this section, we briefly review some works on sequential recommendation and the next POI recommendation.)

다음 POI 추천은 공간 정보를 이용한 순차적 추천의 하위 태스크로 볼 수 있다.   
(The next POI recommendation can be viewd as a special sub-task of sequential recommendation with spatial information.)

### 2.1 Sequential Recommendation
순차 추천은 주로 두 연구 모델 (Markov 기반 모델과 딥러닝 기반 모델)에 의해 모델링 되었다.   
(The sequential recommendation was mainly modeled by two schools of models: Markov-based models and deep learning-based models.)

Markov 기반 모델은 전이 행렬을 통해 다음 행동의 가능성을 예측한다.   
(Markov-based models predict the probaility of the next behavior via a transition matrix.)

순차적 데이터의 희소성 때문에, Markov 모델은 간헐적인 방문의 전이를 전혀 포착할 수 없다.  
(Duo to the sparsity of sequential data, the Markov model can hardly capture the transition of intermittent visits.)

MF 모델은 이 문제에 대한 접근법을 제안했다. 추가적인 확장을 통해 명시적인 시공간적 정보가 추천 성능에 많은 도움이 된다는 것이 발견되었다.  
(Matrix factorization models are proposed to approach this problem, with further extensions find that explicit spatial and temporal information help a lot with recommendation performance.)

일반적으로, Markov 기반 모델은 주로 두 연속적인 방문간 전이 가능성에 집중한다.  
(In genral, Markov-based models mainly focus on the transition probability between two consecutive visits.)

Markov 모델의 결점에 의해, 딥러닝 기반 모델이 대체이 이를 대체한다.   
(Challenged by the flaws of Markov models, deep learning-based models thrive to replace them.)

그 중, RNN 기반 모델들이 강력한 기준선으로 대표적이며 빠르게 발전하고 있다.   
(Among them, models based on RNN are representative and quickly develop as strong baselines.)

그들은 다양한 작업 (세션 기반 추천, 다음 장바구니 추천, 다음 항목 추천 등)으로 만족스러운 성능을 성취했다.   
(They have achieved satisfactory performances on variety of tasks, such as session-based recommendation, next basket recommendation and next item recommendation.)

한편, 사용자의 이력의 동적 특성을 더 잘 보존하기 위해, 인접한 행동간 시간 간격은 RNN 기반 모델에 통합되었다.   
(Meanwhile, time intervals between adjacent behaviors are incorporated in the RNN-based recommendation models, for better preserving the dynamic characteristics of user history.)

RNN 외에도 다른 딥러닝 방법 또한 고려되었다.   
(Besides RNN, other deep learning methods are also considerd.)

예를 들어, metric embedding algorithms, CNN, 강화 학습 그리고 graph network는 순차적 추천을 위해 하나씩 제안되었다.  
(For example, metric embedding algorithms, convolutional neural networks, reinforcement learing algorithms, and graph network are proposed one by one for sequential recommendation.)

최근, 연구자들은 순차적 추천을 위한 self-attention을 광범위하게 사용한다. 이것은 SASRec 모델이 제안했다.   
(Recently, researchers extensively use self-attention for sequential recommendation, where a model named SASRec is proposed.)

SASRec에 기반해, 사용자의 순서 내 시간 간격이 고려되었다.   
(Based on SASRec, time intervals within user sequence are considerd.)

또한, 논의한 바와 같이, PIF는 순차적 추천에 매우 중요하다.   
(Moreover, as discussed in, Personalized Item Frequency(PIF) is very important for sequential recommendation.)


RNN 기반 순차적 추천 시스템은 PIF를 효과적으로 포착할 수 없음이 증명되었다.   
(RNN-based sequential recommenders have been proven to be unable for effectively capturing PIF.)

Self-attention 기반 모델에서, PIF는 주의집중 모듈의 정규화로인해 포착하기 어렵다.   
(In Models based on self-attention, PIF is also hard to capture due to the normalization in attention modules.)

정규화 후, 이전 이력의 표현은 임베딩 차원의 단일 벡터로 축소된다.   
(After normalization, the representation of previous histories is reduced to a single vector of embedding dimension.)

이 표현으로 각 후보를 일치시키는 것은 것은 PIF 정보를 거의 반영하지 못한다.   
(Matching each candidate with this representation can hardly reflect PIF information.)

### 2.2 Next POI Recommendation
대부분의 기존 다음 POI 추천 모델은 RNN에 기반한다.   
(Most existing next POI recommendation models are based on RNN.)

STRNN은 모델의 성능을 향상시키기 위해 모든 두 연속적인 방문사이의 시공간 간격을 명시적 정보로 사용한다. 이것은 또한 공공 보안 평가에 적용되었다.   
(STRNN uses temporal and spatial intervals between every two consecutive visits as explicit information to improve model performance, which has also been applied in public security evaluation.)

SERM은 사용자의 선호도를 반영하는 시간적 의미적 문맥을 공동으로 학습한다.   
(SERM jointly learns temporal and semantic contexts that reflect user preference.)

DeepMove는 장기간 주기적으로 학습하기 위해 주의집중 레이어와 단기간 순차 규칙성을 학습하기 위한 순환 레이어를 결합한다. 그리고 상관관계가 높은 궤적으로부터 학습한다.   
(DeepMove combines an attention layer for learning long-term periodicity with a recurrent layer for learning short-term sequential regularity and learned from highly correlated trajectories.)

다음 위치 추천을 위한 시공간 정보 사용과 관련해, 많은 이전 연구는 순환 레이어의 두 연속적 방문간 명시적 시공간 간격만 사용했다.   
(Regarding the use of spatiotemporal information in the next location recommendation, many previous works only used explicit spatiotemporal intervals between two successive visits in a recurrent layer.)

STRNN은 RNN의 연속적인 방문 사이의 시공간 간격을 직접 사용한다.   
(STRNN directly uses spatiotemporal intervals between successive visits in a recurrent neural network.)

그 다음 Time-LSTM은 시공간의 영향을 더 잘 적용하기 위해 LSTM 구조에 시간 게이트를 추가할 것을 제안한다.   
(Then, Time-LSTM proposes to add time gates to the LSTM structure to better adapt the spatiotemporal effect.)

STGN은 LSTM 구조를 시공간 게이트를 추가함으로써 더욱 강화했다.   
(STGN further enhances the LSTM structure by adding spatiotemporal gates.)

ATST-LSTM은 주의집중 메커니즘을 사용해 LSTM이 각 체크인에 서로 다른 가중치를 할당하도록 지원한다. 이것은 주의집중을 사용하기 시작하지만 여전히 연속적인 방문만 고려한다.   
(ATST-LSTM uses an attention mechanism to assist LSTM in assigning different weights to each check-in, which starts to use attention but still only considered successive visits.)

LSTPM은 최근 방문 장소를 집계하는 지리적 확장 RNN을 제한하지만, 단기 선호도만 해당된다.   
(LSTPM proposes a geo-dilated RNN that aggregates locations visited recently, but only for short-term preference.)

순차적 항목 추천에 영감을 받은 GeoSAN은 궤적 내 지점 간 상호작용을 허용하는 다음 장소 추천에 self-attention 모델을 사용한다.   
(Inspired by sequential item recommendation, GeoSAN uses self-attention model in next location recommendation that allows point-to-point interaction within the trajectory.)

그러나 GeoSAN은 시간 간격과 공간적 거리의 명시적 모델을 무시한다. GeoSAN에서 사용되는 공간적 이산화를 위한 그리드 방법은 정확한 거리를 잘 포착하지 못하기 때문이다.   
(However, GeoSAN ignores the explicit modeling of time intervals and spatial distances, as the gridding method for spatial discretization used in GeoSAN can not well capture the exact distances.)

다시 말해, 모든 이전 방법은 인접하지 않은 위치와 연속적이지 않은 방문 사이의 사소하지 않은 상관관계를 효과적으로 고려하지 못했다.   
(In other words, all previous methods have not effectively considered non-trivial correlations between non-adjacent locations and non-contiguous visits.)

게다가 이러한 모델들은 또한 PIF 정보를 모델링하는 데 문제가 있다.   
(Moreover, these model also have problem in modeling PIF information)

## PREELIMINARIES
이 섹션에서, 우리는 문제 공식화와 용어 정의를 할 것이다.   
(In this section, we give problem formulations and term definitions.)

### Historical Trajectory
사용자 u-i의 궤적은 일시적으로 체크인으로 지시된다.   
(The trajactory of user u-i is temporally ordered check-ins.)

사용자 u-i의 궤적 내 각 체크인 rk는 (u-i, l-k, t-k) 튜플이다. 여기서 lk는장소, tk는 타임스탬프이다.   
(Each check-in rk within the trajectory of user ui is a tuple (u-i, l-k, t-k), in which l-k is the location and t-k is the timestamp.)

각 사용자는 가변 길이 궤적 tra(u-i) = { r1, r2, ..., r-mi }를 가질 수 있다.   
(Each user may have variable-length trajectory tra(u-i) = { r1, r2, ..., r-mi }.)

우리는 각 궤적은 고정 길이 시퀸스 seq(u-i) = {r1, r2, ... r-n}로 변환한다. 여기서 n은 우리가 고려하는 최대 길이다.   
(We transform each trajectory into a fixed-length sequence seq(u-i) = {r1, r2, ... r-n}, with n as the maximum length we consider.)

만약 n < m-i 이면, 가장 최근 n개의 체크인만 고려한다.   
(If n < m-i, we only consider the most recent n checkins.)

만약 n > m-i 이면, 시퀸스의 길이가 n이 될때까지 오른쪽을 0으로 채워넣고 계산하는 동안 그것을 마스킹한다.   
(If n > m-i, we pad zeros to the right until the sequence length is n and mask off the padding items during calucation.)

### Trajectory Spatio-Temporal Relation Matrix
우리는 시간 간격과 지리적 거리를 두 방문 장소 사이의 명시적 시공간 관계로 모델링한다.   
(We model time intervals and geographical distances as the explicit spatio-temporal relation between two visited locations.)

우리는 i번째와 j번째 방문 사이의 시간 간격을 dt-(i, j) = |t-i - t-j|로 나타내고, i번째 방문과 j번째 방문의 GPS 위치 사이의 공간적 거리를 ds-(i, j) = Haversine(GPS-i, GPS-j)로 나타낸다.   
(We denote temporal interval between i-th and j-th visits as dt-(i, j) = |t-i - t-j|, and denote spatial distance between the GPS location of i-th visit and the GPS location of j-th visit as ds-(i, j) = Haversine(GPS-i, GPS-j).)

구체적으로 궤적의 공간적 관계 행렬과 궤적의 시간적 관계 행렬은 각각 다음과 같이 표현된다.   
(Specifically, the trajectory spatial relation matrix ds ∈ R^n*n and the trajectory temporal relation matrix dt ∈ R^n*n are separately represented as)

### Candidate Spatio-Temporal Relation Matrix
내부적으로 명시적인 관계외에, 우리는 논문에서 다음 시공간 행렬도 고려한다.   
(Besides the internal explicit relation, we also consider a next spatiotemporal matrix in the paper.)

이것은 각 장소 후보 i와 체크인 j사이의 거리를 Havorsine으로 계산하고 t-(m+1)과 { t1, t2, ..., t-j}간 시간 간격을 N^t-(i, j) = |t-(m+1) - t-j|로 표현한다.   
(It calculates the distance between each location candidate i ∈ [1, L] and each location of the check-ins j ∈ [1, n] as N^s-(i,j) = Haversine(GPS-i, GPS-j), and represents the time intervals between t-(m+1) and { t1, t2, ..., t-j} that are repeated L times to expand into 2D as N^t-(i, j) = |t-(m+1) - t-j|.)

후보 공간적 관계 행렬과 후보 시간적 관계 행렬은 각각 다음과 같이 표현된다.   
(The candidate spatial relation matrix N^s ∈ R^L*n and the candidate temporal relation matrix N^t ∈ R^L*n are separately represented as)

### Mobility Prediction
주어진 사용자 궤적 (r1, r2, ..., r-m), 장소 후보 L, 시공간 관계 행렬 d(t, x) 그리고 다음 시공간 관계 행렬 N^(t,s)에 대해, 우리의 목표는 이상적인 출력 l ∈ r-(m+1)을 찾는 것이다.   
(Given the user trajectory (r1, r2, ...,r-m), the location candidate L = { l1, l2, ...l-L }, the spatio-temporal relation matrix d(t,x), and the next spatio-temporal matrix N^(d,s), out goal is to find the desired output l ∈ r-(m+1).)


## THE PROPOSED FRAMEWORK
우리가 제안하는 STAN은 다음으로 구성된다.   
(Our proposed Spatio-Temporal Attention Network(STAN) consists of:)

**Multimodal Embedding Module**, 사용자 장소, 시간, 그리고 시공간적 영향의 표현의 밀집된 표현을 학습.   
(a multimodal embedding module that learns the dense representation of user, location, time, and spatiotemporal effect)
**Self-Attention Aggregation Layer**, 각 체크인 표현을 업데이트 하기 위한 사용자 궤적 내 중요한 관련 장소를 수집.    
(a self-attention aggregation layer that aggregates important relevant locations within the user trajectory to update the representation of each check-in)

**Attention Matching Layer**, 다음 장소를 위한 각 장소 후보의 확률 계산을 위해 가중치가 적용된 체크인 표현으로부터 소프트 맥스 확률을 계산   
(an attention matching layer that calculates softmax probability from weighted check-in representations to compute the probability of each location candidate for next location.)

**Balanced Sampler**, 크로스 앤트로피 손실을 계산하기 위해 양수 샘플과 일부 음수 샘플을 사용.   
(a balanced sampler that use a positive sample and several negative samples to compute the cross-entropy loss.)

제안된 STAN 뉴럴 아키텍처는 그림 2와 같다.   
(The neural architecture of the proposed STAN is show in Figure 2.)

### Multimodal Embedding Module
Multimodal Embedding Module은 두 부분(궤적 임베딩 레이어, 시공간 임베딩 레이어)으로 구성돼있다.    
(The multi-modal embedding module consists of two parts, namely a trajectory embedding layer and a spatio-temporal embedding layer.)
#### User Trajectory Embedding Layer
Multimodal Embedding Module는 사용자, 장소, 시간을 잠재적 표현으로 인코딩하는데 사용된다.   
(A multi-modal embedding layer is used to encode user, location and time into latent representations.)

사용자, 장소, 시간을 위해 우리는 임베딩 표현으로 각각 e^u ∈ R^d, e^l ∈ R^d, e^t ∈ R^d 으로 나타낸다.   
(For user, location and time, we denote their embedding representations as e^u ∈ R^d, e^l ∈ R^d, e^t ∈ R^d, respectively.)

임베딩 모듈은 계산을 줄이고 표현을 향상시키기 위해 스칼라를 밀집 벡터로 변환하기 위한 다른 모듈과 통합된다.    
(The embedding module is incorporated into the other modules to transform the scalars into dense vectors to reduce computation and improve representation.)

지속적인 타임 스탬프는 168시간으로 나눠져 정확히 1주일을 표현한다. 이것은 168차원으로 사상한다.   
(Here, the continuous timestamp is divided by 7x27 = 168 hours that represents the exact hour in a week, which maps the original time into 168 dimensions.)

이러한 시간적 이산화는 주기성을 반영하는 하루나 일주일의 정확한 시간을 나타낼 수 있다.   
(This temporal discretization can indicate the exact time in a day or a week, reflecting periodicity.)

그러므로 임베딩 e^u, e^l, e^t의 입력 차원은 각각 U, L, 168 이다.   
(Therefore, the input dimensions of the embeddings e^u, e^l and e^t are U, L and 168, respectively.)

각 체크인 r에 대한 사용자 궤적 임베딩 출력은 e^r = e^u + e^l + e^t ∈ R^d 이다.   
(The output of user trajectory embedding layer for each check-in r is the sum e^r = e^u + e^l + e^t ∈ R^d.)

각 사용자 시퀸스 seq(u-i) = { r1, r2, ..., r-n }의 임베딩을 위해 우리는 E(u-i) = { e^r1, e^r2, ..., e^r-n} ∈ R^n*d 으로 나타낸다.   
(For the embedding of each user sequence seq(u-i) = { r1, r2, ..., r-n }, we denote as E(u-i) = { e^r1, e^r2, ..., e^r-n} ∈ R^n*d)

#### Spatio-Temporal Embedding Layer
임베딩 레이어의 유닛은 한 시간과 100미터를 기본 단위로 하는 시공간 차이의 조밀하게 표현하는데 사용된다.   
(A unit embedding layer is used for the dense representation of spatial and temporal differences with an hour and hundred meters as basic units, respectively.)

만약 우리가 최대 공간과 시간 간격을 임베딩의 수로 간주하고 모든 간격을 이산화 하면 그것은 쉽게 희소 관계 인코딩으로 이어질 수 있음을 기억하라.    
(Recall that if we regard the maximum space or time intervals as the number of embeddings and discretize all the intervals, it can easily lead to a sparse relation encoding.)

이 레이어는 공간과 시간 간격에 각각 유닛 임베딩 벡터 eds와 edt를 곱한다.   
(This layer multiplies the space and time intervals each with a unit embedding vector eds and edt, respectively.)

유닛 임베딩 벡터는 기본 유닛과의 연속적인 시공간 맥락을 반영하고 밀집된 희소성 인코딩을 방지한다.   
(The unit embedding vectors reflect the continuous spatiotemporal context with the basic unit and avoid sparsity encoding with the dense dimensions.)

특히, 우리는 공간적 거리에 민감한 이 기법을 사용해 인접한 장소만 집계하고 공간적 거리를 표현할 수 없는 계층적 그리드 방식을 대체할 수 있다.    
(Especially, we can use this technique that is sensitive to spatial distance to replace hierarchical gridding method, which only aggregates adjacent locations and is not capable to represent spatial distance.)

수학적으로 시공간 차 임베딩은 ed-(imj) ∈ R^d 이다.   
(In mathematics, the spatiotemporal difference embedding is ed-(imj) ∈ R^d)

우리는 대안적인 보간 입베딩 레이어도 고려할 수 있다. 이것은 상한 유닛 임베딩 벡터와 하한 유닛 임베딩 벡터로 구성되고 선형 보간법으로 명시적인 간격을 표현한다. 이것은 유닛 임베딩 벡터를 근사한다.   
(We may also consider an alternative interplation embedding layer that sets a upper-bound unit embedding vector and a lower-bound unit embedding vector and represents the explicit intervals as a linear interplation, which is an approximation to the unit embedding layer.)

실험적으로, 두 방법은 유사한 효율성을 갖는다.   
(In experiments, the two methods have similar efficiency.)

보간 임베딩은 다음과 같이 계산된다.   
(The interpolation embedding is calcuated as:)

이 레이어는 두 행렬(궤적 시공간 관계 행렬, 후보 시공간 관계 행렬)을 처리한다.   
(This layer processes two matrices: the trajectory spatio-temporal relation matrix and the candidate spatio-temporal relation matrix, as described in preliminaries.)

이러한 임베딩은 E(dt) ∈ R^n*n*d, E(ds) ∈ R^L*n*d, and E(N^s) ∈ R^L*n*d 이다.
(Their embeddings are E(dt) ∈ R^n*n*d, E(ds) ∈ R^L*n*d, and E(N^s) ∈ R^L*n*d.)

우리는 가중치가 적용된 마지막 차원의 합을 사용할 수 있고 공간적, 시간적 임베딩을 추가할 수 있다.   
(We can use a weighted sum of the last dimension and add spatial and temporal embeddings together to create)

### Self-Attention Aggregation Layer
Self-attention 매커니즘에 영감을 받아, 우리는 궤적 내 두 방문간 공간적 차이와 시간 간격을 고려하는 확장 모듈을 제안한다.   
(Inspired by self-attention mechanisms, we propose an extensional module to consider the different spatial distances and time intervals between two visits in a trajectory.)

이 모듈은 관련 방문 장소를 집계하고 각 방문의 표현을 갱신하는데 목표를 두고있다.   
(This module aims at aggregating relevant visited locations and updating the representation of each visits.)

Self-attention 레이어는 장기 의존성을 포착하고 궤적 내 각 방문에 다른 가중치를 할당할 수 있다.   
(Self-attention layer can capture long-term dependency and assign differnt weights to each visit within the trajectory.)

궤적 내 지점간 상호작용은 레이어가 관련 방문에 가중치를 더 많은 할당할 수 있도록 한다.   
(This point-to-point interaction within the trajectory allows the layer to assign more weights to relevant visits.)

더욱이, 우리는 명시적 시공간 간격을 상호작용으로 쉽게 통합할 수 있다.   
(Moreover, we can easily incorporate the explicit spatio-temporal intervals into the interaction.)

길이 m'로 패딩하지 않은 사용자 임베디드 궤적 행렬 E(u)과 시공간 관계 행렬 E(d)에 기반해, 레이어는 우선 마스크 행렬 M ∈ R^n*n (좌상단 R^m*m은 1, 나머지는 0 으로 구성)을 구성한다.   
(Given the user embedded trajectory matrix E(u) with non-padding length m' and the spatio-temporal relation metrices E(d), this layer firstly construct a mask matrix M ∈ R^n*n with uper left elements R^m'*m' being ones and other elements being zeros.)

그 다음 레이어는 별개의 파라미터 행렬 W-Q, W-K, W-V ∈ R^d*d 로 변환한 다음, 새 시퀸스 S를 계산한다.   
(Then the layer computes a new sequences S after converting them through distinct parameter matrices W-Q, W-K, W-V ∈ R^d*d as)

여기서, 마스크와 소프트 맥스 attention만 요소별로 곱해지고 다른 것들은 행렬 곱셈을 한다.   
(Here, only the mask and softmax attention are multiplied element by element, while others use matrix multiplication.)

(m'+1)번째 방문을 예측할 때, 우리가 궤적 내 처음 m'개의 방문만 모델에 이용해 인과관계를 고려하는 것은 매우 중요하다.    
(It is very important for us to consider cauality that only the first m' visits in the trajectory are fed into the model while predicting the (m'+1)-st location.)

그러므로 훈련을 할 때, m' ∈ [1, m] 을 사용해 입력 시퀸스를 마스킹하고 선텍한 레이블에 따른다.   
(Therefore, during training, we use all the m' ∈ [1, m] to mask the input sequence and accordingly to the selected label.)

우리는 사용자 궤적의 갱신된 표현에서 S(u) ∈ R^n*d 을 얻을 수 있다.   
(We can get S(u) ∈ R^n*d as the updated representation of the user trajectory.)

또 다른 대안적인 구현은 TiSARec처럼 명시적인 시공간 간격을 E(u)W-K and E(u)W-V에게 주는 것이다.   
(Another alternative implementation is to feed explicit spatio-temporal intervals into both E(u)W-K and E(u)W-V, as TiSARec did.)

그러나 실험적으로 우리는 유사한 성능을 갖는 두가지 방법을 찾았다.   
(However, in experiments, we found out the two methods have similar performances.)

우리의 구현은 요소별 계산 대신 행렬 곱셈만 사용하는 더 간결한 형식이다.   
(Our implementation is in a more concise form using only matrix multiplication instead of element-wise calcuation.)

### Attention Matching Layer
이 모듈은 사용자 궤적의 갱신된 표현에서 매칭함으로써 가장 그럴듯한 후보를 모든 L 장소로 부터 recall하는것을 목표로 삼는다.   
(This module aims at recalling the most plausible candidates from all the L locations by matching with the updated representation of the user trajectory.)

갱신된 궤적 표현 S(u) ∈ R^n*d, 임베디드된 장소 후보 시공간 관계 행렬 E(N) ∈ R^L*n에 대해, 이 레이어는 각 장소 후보의 다음 장소가 되기 위한 가능성을 계산한다.   
(Given the updated trajectory representation S(u) ∈ R^n*d, the embedded location candidate spatio-temporal relation matrix E(N) ∈ R^L*n, this layer computes the probability of each location candidate to be the next location as)

여기서, 합 연산은 마지막 차원의 가중치가 적용된 합으로 A(u)의 차원을 R^L로 변환한다.   
(Here, the Sum operation is a weighted sum of the last dimension, converting the dimension of A(u) to be R^L.)

공식 8에서, 다른 self-attention 모델이 PIF 정보를 축소하는 것과 달리, 우리는 각 후보 장소의 일치시키는 것에 모두 참여하는 체크인의 갱신된 표현을 보여준다.   
(In Eq.(8), we show that the updated representation of check-ins all participate in the matching of each candidate location, unlike other self-attention models that reduce the PIF information.)

이것은 우선 관련 장소를 집계하고 그다음 PIF의 고려한 표한으로부터 recall하는 이중 레이어 시스템 덕분이다.   
(This is due to the design of a bi-layer system that firstly aggregates relevant location and then recalls from representations with consideration of PIF.)

### Balanced Sampler
A(u)에서 양수 샘플과 음수 샘플의 불균형한 규모로 인해, 크로스 엔트로피 손실로 최적화하는 것은 더이상 효율적이지 않다. 손실 가중치가 정확한 예측에 거의 영향을 미치지 않기 때문이다.   
(Due to the unbalanced scale of positive and negative samples in A(u), optimizing the cross-entropy loss is no longer efficient as the loss weights little on the momentum to push forward the correct prediction.)

손실이 줄어들고, recall 비율도 감소하는 것이 관찰되는 것은 정상이다.   
(It would be normal to observe that as the loss goes down, the recall rate also goes down.)

사용자 i의 시퀸스 seq(u-i), 각 후보 장소의 일치 확률 a-j ∈ A(u-i) 그리고 장소 집합 L 내의 지시의 수 k에 대한 라벨 l-k에 대해 일반적인 크로스 엔트로피 손실은 다음과 같다.   
(Given the user i's sequence seq(u-i), the matching probaility of each candidate location a-j ∈ A(u-i) for j ∈ [1, L], and the label l-k with number of order k in the location set L, the ordinary cross-entropy loss is written as)

이 형식에서, 모든 양수 샘플 a-k에 대해, 동시에 우리는 L-1 개의 음수 샘플들 계산한다.   
(In this form, for every positive sample a-k, we need to compute L-1 negative samples in the meantime.)

달느 구현은 광범위하게 이진 크로스 엔트로피 손실을 사용하는 것이다. 이것은 양수 샘플을 하나의 음수 샘플과만 계산한다.   
(Ohter implementations also extensively used binary cross-entrophy loss that computes only one negative sample algong with a positive sample.)

그러나 이것은 아마 또한 전체 훈련에서 사용되지 않는 라벨이 없는 샘플을 남길 것이다.   
(However, this may also leave many non-label samples unused throughout the entire training.)

여기서 우리는 단순히 크로스 엔트로피 손실에 사용되는 음수 샘플의 수를 하이퍼 파라미터 s로 사용할 수 있다.   
(Here, we can simply set the number of negative samples used in cross-entropy loss as a hyperparmeter s.)

우리는 각 훈련 단계마다 임의로 음수 샘플을 샘플링하는 Balanced Sampler를 제안한다.   
(Here we propose a balanced sampler for randomly sampling negative samples at each step of training.)

결론적으로 우리는 각 훈련 단계가 끝난 후에 음수 샘플의 난수 시드를 갱신한다.   
(Consequently, we update the random seed of the negative sampler after each trainging step.)

손실은 다음과 같이 계산된다.   
(The loss is calcuated as)

## EXPERIMENTS
이 섹션에서, 우리는 다른 모델과 양적으로 공평한 비교를 위해 경험적 결과를 보여준다.   
(In this section, we show our empirical results to make a fair comparision with other models quantitatively.)

우리 데이터 셋의 테이블과, topk recall 비율 수치에 기반한 성능 평가 테이블 그리고 STAN 집계에서 attention 가중치의 시각화를 보여준다.   
(We show a table of datasets, a table of recommendation performance under the evaludation of topk recall rates figures of model stability, and the visualization of attention weights in STAN aggregation.)

### Datasets
우리는 제안한 STAN 모델을 네가지 실세계 데이터 세트 (Gowalla, SINm TKY, NYC)로 평가한다.   
(We evaluate our proposed STAN model on four real-wrold datasets: Gowalla, SIN, TKY and NYC.)

각 데이터셋의 사용자, 장소, 체크인 수는 Table.1과 같다.   
(The numbers of users, locations, and check-ins in each dataset are shown in Table.1)

이 실험에서, 우리는 각 장소의 GPS와 사용자 체크인 기록만 포함한 데이테셋 원형을 사용하고, 다음 절차에 따라 전처리한다.   
(In experiments, we use the original raw datasets that only contain the GPS of each location and user check-in records, and pre-process them following each work's protocol.)

데이터셋의 전처리와 관련해, 많은 이전 연구는 고정 길이 윈도우나 최대 시간 간격으로 궤적을 슬라이스하였다.   
(In regard to the pre-processing technique of datasets, many previous works used sliced trajectory with a fixed length window or maximum time interval.)

비록 이것이 모델의 장기 의존성 학습을 방해할 수 있지만, 우리는 각 연구 설정을 따른다.   
(We follow each work's setup, although this could prevent this model from learning long-time dependency.)

m 개의 체크인을 가진 각 사용자에 대해, 우리는 데이터셋을 훈련, 검증, 테스트 데이터셋으로 분할한다.   
(For each user that has m check-ins, we divide a dataset into training, validation and test datasets.)

훈련셋의 수는 m - 3 으로 처음 m' ∈ [1, m - 3] 체크인은 입력 시퀸스이고 [2, m - 2]번째 방문 장소는 라벨이다.   
(The number of training set is m - 3, with the first m' ∈ [1, m - 3] check-ins as input sequence and the [2, m - 2]-nd visited location as label.)

검증셋은 처음 m - 2 개의 체크인을 입력 시퀸스로 사용하고  (m - 1)번째 방문 장소는 라벨이다.   
(The validation set uses the first m - 2 check-ins as input sequence an dthe (m - 1)-st visited location as label.)

테스트셋은 처음 m - 1 개의 체크인을 입력 시퀸스로 사용하고 m번째 방문 장소가 라벨이다.   
(The test set uses the first m - 1 check-ins as input sequence and the m-th visited location as label.)

데이터셋의 분할은 미래의 데이터는 미래의 데이터 예측에 사용하지 않는 인과율를 따른다.   
(The split of datasets follows the causality that no future data is used in the prediction of future data.)

### Baseline Models
우리는 STAN을 다음 기존 모델과 비교한다.   
(We compare out STAN with the following baselines:)

- STRNN: 연속적인 방문 사이의 시공간 특성을 통합하는 RNN 모델의 변형
- (STRNN: an invariant RNN model that incorporates spatio-temporal features between consecutive vists.)

- DeepMove: 순환과 attention 레이어로 주기성을 포착하는 최첨단 모델
- (DeepMove: a state-of-the-art model with recurrent and attention layers to capture periodicity.)

- STGN: 시간과 거리 게이트 간격을 LSTM에 추가한 최첨단 모델   
- (STGN: a state-of-the art model that add time and distance interval gates to LSTM.)

- ARNN: 의미적이고 공간적인 정보를 지식 그래프를 구성하는데 사용하고 순차적 LSTM 모델의 성능을 향상시키는 최첨단 모델   
- (ARNN: a state-of-the-art model that uses semantic and spatial information to construct knowledge graph and improve the performance of sequential LSTM model.)

- LSTPM: 장기 와 단기 순차 추천 모델을 결합한 최첨단 모델
- (LSTPM: a-state-of-the-art model that combines long-term and short-term sequential models for recommendation.)

- TiSARec: self-attention 레이어를 순차 추천을 위한 명시적 시간 간격과 사용하는 최첨단 모델, 단 공간적 정보는 사용하지 않음   
- (TiSASRec: a-state-of-the-art model that uses self-attention layers with explicit time intervals for sequential recommentation, but it uses no spatial information.)

- GeoSAN: 공간적 이산화를 위한 GPS 장소의 계층적 그리드와 명시적 시공간 간격 없는 매칭을 위한 사용자 self-attention 레이어 사용하는 최첨단 모델   
- (GeoSAN: a state-of-the-art model that uses hierarchical gridding of GPS locations for spatial discretization and users self-attention layers for matching, without use of exlicit spatio-temporal interval.)

### Evaluation Matrices
우리는 추천 성능 평가하는데, Recall@5와 Recall@10을 채택한다.   
(We adopt the topk recall rates, Recall@5 and Recall@10, to evaluate recommendation performance.)

Recall@k는 모든 P 샘플에서 TP의 비율을 센다. 우리의 경우, topk의 가능성을 나타내는 샘플에서 라벨의 비율을 의미한다.   
(Recall@k counts the rate of true positive samples in all positive samples, which in our case means the rate of the label in the topk probability samples.)

평가를 위해, 우리는 Balanced Sampler 모듈을 제거하고, attention matching layer의 출력, A로부터 직접 대상을 recall한다.   
(For evaluattion, we drop the balanced sampler module and directly recall the target from A,
the output of the attention matching layer.)

Recaller@k가 커질 수록, 성능이 더 좋아진다.   
(The larger the Recall@k, the better the performance.)

### Settings
두 개의 하이퍼 파라미터가 있다.   
(There are two kinds of hyperparameters)

하나는 모든 모델이 공유하는 공유 하이퍼 파라미터다.   
((i) common hyperparameters that are shared by all models)

또 다른 하나는 각 모델의 프레임워크에 의존하는 고유 하이퍼 파라미터다.   
((ii) unique hyperparamters that depend on each models'framework.)

우리는 공통 파라미터를 단순한 RNN으로 학습한 다음, 모든 모델에 적용한다. 이것은 훈련의 부담을 줄이는데 도움된다.    
(We train the common hyperparameters on a simple recurrent neural network and then apply them to all models, which helps reduce the training burden.)

임베딩 차원 d는 TKY, SIN, NYC는 50이고, gowalla는 10이다.   
(The embedding dimension d to 50 for TKY, SIN and NYC datasets and 10 for gowalla datasets.)

우리는 optimizer=Adam, learning-rate=0.003, dropout-rate=0.2, epoch=50, max-length-of-trajectory-sequence=100 을 사용한다.   
(We use the Adam optimizer with default betas, the learning rate of 0.003, the dropout rate of 0.2, the training epoch of 50, and the maximum length for trajectory sequence of 100.)

공통 하이퍼 파라미터를 고정하고, 각 모델의 고유 파라미터를 미세 조정한다.   
(Fixing these common hyperparameters, we find-tune the unique hyperparameters for each model.)

우리의 모델에서, Balanced Sampler의 N 샘플의 수는 10이 최적이다.   
(In our model, the number of negative samples in the balanced sampler is optimal at 10.)

### Recommendation Performance
Table.2는 4개의 데이터셋에 대한 우리 모델과 기존 모델의 추천 성능을 보여준다.   
(Table.2 shows the recommendation performance of our model and baselines on the four datasets.)

다른 방법간 모든 차이는 통계적으로 의미있다.   
(All the differences between different methods are statistically significant.)

우리는 STAN의 성능 향상을 평가하기 위해 p-value가 0.01인 T-test를 사용한다.   
(We use a T-test with a p-value of 0.01 to evaluate the performance improvement provided by STAN.)

여기서 우리는 10회 시행에 대한 평균 성능을 사용했고 H0 가설을 기각했다.   
(Here, we use the averaged performance run by 10 times and reject the H0 hypothesis.)

그러므로 우리는 STAN의 향상이 통계적으로 의미있음을 알 수있다.   
(Therefore, we know the improvement of STAN is statistically sigificant.)

우리는 우리의 모델이 비교한 모든 모델보다 recall 비율이 분명히 9%-17% 향상됨을 볼 수 있다.   
(We can see that our model unequivocally outperforms all compared models with 9%-17% improvement in recall rates.)

Firgure.3과 4에서 하이퍼 파라미터 조정에서 모델이 안정적임을 볼 수 있다.   
(We show in Figures.3 and 4 that the model is stable under hyperparameter tuning.)

기존 모델 중, self-attention 모델(TiSRARec, GeoSAN)이 RNN 기반 모델보다 명백히 더 성능이 좋다.   
(Among baseline models, self-attention models such as TiSARRec and GeoSAN clearly have better performance over RNN-based models.)


이것은 놀랍지 않다. 이전 RNN 모델은 종종 슬라이싱한 짧은 궤적은 긴 궤적 대신 사용했기 때문이다. 이것은 장기 주기성을 버리고 각 방문에서 다음 이동의 정확한 영향력을 거의 포착할 수 없다.   
(It is not a surprise since previous RNN-based models often use sliced short trajectories instead of long trajectory, which tossed long-term periodicity and can hardly capture the exact influence of each visits towrds the next movement.)

ARNN에서 메타 경로를 수행하는데 지식 그래프를 구축하는데 우리가 어떤 의미적 정보도 사용하지 않는 데 유의해야한다. 비교에서 다른 모델에 의미적 분석이 사용되지 않았기 때문이다.   
(It should be noted that we do not use any semantic information to construct knowledge graph to perform meta-path in ARNN, as semantic analysis was not performed by other baselines in the comparision.)

RNN 기반 모델 중 LSTPM과 DeepMove는 상대적으로 더 좋은 성능을 보였다. 이들은 주기성을 고려하기 때문이다.   
(Among RNN-based models, LSTPM and DeepMove have relatively better performances, due to their consideration of periodicity.)

Self-attention 모델 중, TiSASRec은 시간적 간격을 사용했고 GeoSAN은 지리적 분할을 고려했다.   
(Among self-attention models, TiSASRec used temporal intervals and GeoSAN considered geographical partitions.)

오직 STAN만 비연속적인 방문과 인접하지 않은 장소의 모델링에 대한 시퀸스 내 시공간 간격을 완전히 고려한다. 그리고 변환 구조를 직접 상속하는 대신, attention 아키텍처를 PIF 정보를 적용할 수 있도록 변경한다.   
(Only STAN fully considers the spatio-temporal intervals within the sequences for modeling non-consecutive visits and non-adjacent locations, and modifies attention architecture to adapt PIF information instead of inheriting the transformer structure directly.)

또한, STRANN과 TiSARect은 모두 시간적 간격을 사용하기 때문에 우리는 이들의 성능을 self-attention 모듈 vs recurrent 레이어의 성능 향상으로 비교할 수있다.   
(In addition, because STRNN and TiSASRec both use temporal intervals, we can compare their performances to evaluate the improvement provided by self-attention modules versus recurrent layers.)

우리는 Table.3에서 시공간 간격과 Balanced Sampler가 없는 STAN 모델의 변형을 표현하는 -ALL 모델도 참조했다.   
(We can also refer to Table.3 where the -ALL model represents a variant STAN model without spatio-temporal intervals and the balanced sampler.)

-ALL 모델은 기존의 self-attention 모델과 PIF 정보를 고려하는 이중 계층 시스템의 사용만 다르다.   
(-ALL model is different from ordinary self-attention models only on the bi-layer system, which considers PIF information.)

-ALL은 4개의 데이터셋의 recall에 대해 GeoSAN에 비해 간소하게 성능이 나쁘다. 하지만 간소하게 TiSARect보단 성능이 좋고 RNN보단 훨씬 좋다.   
(-ALL has a slightly worse performance then GeoSAN on the recall rates of the four datasets, but is slightly better than TiSASRec and much better than RNN-based models.)

이것은 우리에게 PIF를 고려하는 이중 계층 시스템이 시간 간격을 attention 시스템에 통합하는것 만큼 중요하다는 것을 말한다.   
(This tells us that the bi-layer system which considers PIF is approximately as important as time intervals incorporated into the attention systems.)

### Ablation Study
우리 모델을 다른 모듈에 대해 분석하는데, 이 섹션에서 절제 연구를 한다.   
(To analyze different modules in our model, we conduct an ablation study in this section.)

우리는 시공간 간격과 Balanced Sampler를 사용하는 STAN을 기반 모델로 나타낸다.   
(We denote the based model as STAN, with spatio-temporal intervals and a balanced sampler.)

다른 구성요소는 변형을 형성하기 위해 제거한다.   
(We drop different components to form variants.)

구성요소는 다음과 같다.   
(The components are listed as:)

- SIM: 궤적 내의 명시적 공간 간격을 행렬로 나타낸다.   
- (SIM(Spatial Intervals in Matrix): This denotes the explicit spatial intervals we use within the trajectory as a matrix.)

- EWSI: 요소별 공간 간격을 TiSASRec의 구조에 따라 나타낸다.   
- (EWSI (Element-Wise Spatial Intervals): This denotes the element-wise spatial intervals following the structure of TiSASRec)

- TIM: 궤적 내 명시적 시간 간격을 행렬로 나타낸다.   
- TIM (Temporal Intervals in Matrix): This denotes the explicit temporal intervals we use within the trajectory as a matrix.

-EWTI: 요소별 시간 간격을 TiSASRec의 구조에 따라 나타낸다.   
-  EWTI (Element-Wise Temporal Intervals): This denotes the element-wise temporal intervals following the structure of TiSASRec

- BS: 손실 계산을 위한 Balanced Sampler.   
- (BS(Balanced Sampler): Balanced sampler for calculating loss.)

Table.3는 절제 연구의 결과를 보여준다.   
(Table.3 shows the result of the abalation study.)

Balanced Sampler는 추천 성능의 향상에 중요함을 알 수 있다. 이것은 recall 비율의 5-12% 향상시킨다.   
(We find that a balanced sampler is crucial for improving the recommendation performance, which provides a nearly 5-12% increase in recall rates.)

시공간 간격은 명시적으로 비연속적인 방문과 인전하지 않은 장소 사이의 상관관계를 표현한다.   
(Spatial and temporal intervals can explicity express the correlation between non-consecutive visits and non-adjacent locations.)

공간적 거리와 시간적 간격을 추가하는 것은 recall 비율을 4-8% 향상시킨다.   
(Adding spatial distances and temporal intervals all provide nearly 4-8% increase in recall rates.)

시공간 상관관계를 소개하기 위한 우리의 방법은 TiSASRec에서 사용된 방법과 같다는 것도 알아냈다. 하지만 우리의 방법은 구현하기 더 쉽고, 행렬 형식 덕에 계산하기 더 편하다.   
(We also find that our method to introduce spatio-temporal correlations is equivalent to the method used in TiSASRec, while our method is easier to implement and can be computationally convenient due to its matrix form.)

최악의 조건은 시공간 간격이 없거나 Balanced Sampler가 없는 것으로 Recall@5와 Recall@10을 크게 감소시킨다.   
(The worst condition is that none of the spatio-temporal intervals nor balanced sampler isused, in which the Recall@5 and Recall@10 decrease drastically.)

그럼에도 불구하고, -ALL 절제 모델은 여전히 이전에 보고된 RNN 기반 모델(DeepMove, STRNN, STGN)보다 뛰어난 성능을 보인다.   
(Even so, this -ALL ablated model still outperforms previously reported RNN-based models such as DeepMove, STRNN, and STGN.)

이중 계층 시스템을 포함한 -ALL 모델은 PIF 정보를 고려할 수 있다.   
(-ALL model with the bi-layer system can consider PIF imformation.)

이것은 -ALL이 여전히 TuSARec과 RNN 기반 모델보다 성능이 더 좋은 이유를 설명한다.   
(This explains why -ALL still has a better performance over TiSASRec and RNN-based models.)

이것은 우리에게 PIF를 고려한 이중 계층 시스템이 시간 간격을 self-attention 시스템에 통합하는 것만큼 중요하다는 것을 말한다.   
(This tells us that the bi-layer system which consider PIF is as important as time intervals incorporated into self-attention systems.)

### Stability Study
#### Embedding dimension

우리는 다양한 임베딩 모듈 내 임베딩 차원 d를 10에서 60으로 10단계에 나눠 변화시킨다.   
(We vary the dimension of embedding d in the multimodal embedding module from 10 to 60 with step 10.)

Figure.3는 d=50이 궤적과 시공간 임베딩을 위한 최고의 차원임을 보여준다.   
(Figure.3 shows that d = 50 is the best dimension for trajetory and spatio-temporal embedding.)

일반적으로 우리의 추천 성능은 하이퍼 파라미터 d에 민감하지 않은데,  Gowalla는 6% 미만으로 변화하고 나머지 데이터셋은 2% 미만이다.   
(In general, the recommendation performance of our model is insensitive to the hyperparameter d, with less than 6% change rate for the Gowalla dataset and less than 2% change rate for other datasets.)

d가 30이상인 한, 추천 성능의 변화는 0.5%로 무시될 수 있다.   
(As long as d is large than 30, the change in recommendation performance will be less than 0.5% which can be ignored.)

#### Number of negative samples
우리는 Balanced Sampler 내 일련의 N 샘플의 수 s = [1, 10, 20, 30, 40, 50]로 실험한다.   
(We experiment a series of number of negative samples s = [1, 10, 20, 30, 40, 50] in the balanced sampler.)

Figure.4는 20 미만인 N 샘플의 수가 모든 데이터셋을 위한 추천 안정성을 제공함을 보여준다.   
(Figure.4 shows that the number of negative samples less than 20 can all produce stable recommendations for all datasets.)

STAN은 121944개의 장소를 포함한 Gowalla 데이터셋에서 N 샘플의 수에 특히 민감하지 않았다.   
(STAN is specifically insensitive to the number of negative samples for the Gowalla dataset, which has as many as 121944 locations.)

이것은 데이터셋이 클수록 최적의 N 샘플의 수가 더 많아짐을 나타낸다.   
(This indicates that the larger the dataset, the larger the optimal number of negative samplers.)

N 샘플의 수가 증가하기 때문에, 균형 손실은 일반적인 크로스 엔트로피 손실을 사용하는 경향이 있다.   
(As the number of netavie samples increases, the balanced loss will tend to the ordinary cross-entropy loss.)

Table.3 에서 Balanced Sampler는 추천 성능 향상에 중요함을 알 수 있다.   
(In Table.3, we found that the balanced sampler is crucial for improving recommendation performance.)

N 샘플의 수가 임계값을 너으면 recall 비율은 극적으로 떨어진다.   
(If the number of negative samples is above the threshold, the recall rate will drop drastically.)

### Interpretablity Study
STAN의 매커니즘을 이해하기 위해 self-attention 집계 레이어를 사용해 비연속적인 방문과 인접하지 않은 장소를 집계하는 것 핵심이다.   
(To understand the mechanism of STAN, the aggregation of non-consecutive visits and non-adjacent locations performed by the self-attention aggregation layer is at the core)

우리는 attention 가중치의 상관관계 행렬 Cor을 Figure.5로 시각화한다.   
(We visualize the correlation matrix Cor of the attention weights in Figure.5.)

행렬의 각 요소 Cor-(i,j)는 i번째 방문한 장소에서 j번째로 방문한 장소로의 가중화된 영향력을 나타낸다.    
(Each element Cor-(i,j) of the matrix represents the weighted influence of j-th visited location on i-th visited location.)

상관관계 행렬은 self-attention 집계 레이어의 쿼리와 키를 곱한 소프트 맥스 함수로 계산된다.   
(The correlation matrix is calculated as the softmax of the multiplication of query and key in the self-attention aggregation layer.)

소프트 맥스 함수의 결과로 상관관계 행렬 내 각 요소의 값은 1 또는 0인 경향이 있다.   
(The value of each element in this correlation matrix is either tending to 1 or 0 as a result of softmax operation.)

원래의 체크인 임베딩에 곱하기 위해 상관관계 행렬을 사용하여, 궤적을 갱신할 수 있다.   
(Using the correlation matrix to times the original check-in embeddings, we can update the representions of the trajectory.)

Figure.5는 Figure.1의 소개에서 논의된 실제 사용자의 궤적 예제의 슬라이스에 기반한다.   
(Figure.5 is based on a slice of real user trajectory example that is discussed in Introduction Section and Figure.1)

서로 다른 장소는 0에서 6으로 명명되어 분류된다.   
(Here, diffent locations are classified and named by numbers from 0 to 6.)

정확한 GPS의 질의에 의해, 장소 0, 1, 2는 각각 집, 직장, 쇼핑몰을 나타낸다.   
(By query of the exact GPS, we find that locations 0, 1, 2 are home, workplace and shopping mall, respectively.)

장소 3, 4, 5, 6은 레스토랑이다.   
(Location 3, 4, 5 and 6 are restaurants.)

Figure.5(a)는 Figure.5(b)로 얻은 방문한 장소의 공간적 상관관계를 보여준다. 이때 노란색으로 칠해진 장소와 검은 원 내 범위는 함께 집계되었다.   
(Figure.5(a) shows the spatial correlation of visited locations that is attained by Figure.5(b), where locations with the yellow-colord marks and locations within the range of the same dark circles are aggregated togeter.)

이것은 인접한 장소뿐만 아니라 인접하지 않은 장소의 상관관계도 보여준다.    
(This shows that not only adjacent locations but also non-adjacent locations are correlated.)

장소 3, 4, 5, 6은 모두 레스트랑이고 식사를 위해 종종 정확한 시간에 방문된다.   
(Locations 3, 4, 5 and 6 are all restaurants and are often visited at the exact time for meals.)

공간적으로 거리가 있음에도 불구하고, 우리는 상관관계 행렬로부터 관련성을 말할 수 있다.   
(We can tell from the correlation matrix that they are relevant, despite that they are spatially distanced.)

이 궤적 예제에서 시간적 순서는 Figure.1의 타임라인을 보여준다.   
(The temporal order of this trectory example is shown in the timeline of Figure.1.)

이것은 레스토랑의 상관관계에 집중하기 위해 관련없는 방문을 편집하기 위한 슬라이스된 희소 궤적이다.   
(This is a sliced sparse trajectory as we edit off the irrelevant visits to focus on the correlation of restaurants.)

방문한 레스토랑의 시간과 순서는 연속적이지 않지만 여전히 함께 집계된다.   
(The time and order of these restaurants being visited are not consecutive but are still aggregated together.)

공간과 시간의 증거 모두 우리의 동기를 보여준다.   
(Both shreds of evidence in space and time demonstrate our motivation.)

## CONCLUSION
본 연구에서 우리는 STAN을 제안한다.   
(In this work, we propose a spatio-temporal attention network, abreviated as STAN.)

우리는 실제 궤적 예제를 사용해 인접하지 않은 장소와 연속적이지 않은 방문 사이의 기능적 관련을 설명한다. 그리고 이중 attention 시스템을 사용하는 궤적 내 시공간 상관관계를 명시적으로 학습할 것을 제안한다.   
(We use a real trajectory example to illustrate the functional relevance between non-adjacent locations and non-consecutive visits, and propose to learn the explict spatio-temporal correlations within the trajectory using a bi-attention system.)

이 아키텍처는 우선 궤적 내 시공간 간격을 집계한 다음 대상을 recall한다.   
(This architecture firstly aggregates spatio-temporal intervals within the trajectory and then recalls the target.)

궤적의 모든 표현에 가중치가 적용되므로, 대상을 recall하는 것은 PIF의 영향을 완전히 고려한다.   
(Because all the representations of the trajectory are weighted, the recall of the target fully considers the effect of personalized item frequency (PIF).)

우리는 크로스 엔트로피 손실 계산을 일치시키기 위한 Balanced Sampler를 제안한다. 이것은 흔히 이용되된 이진 및 일반적인 크로스 엔트로피 손실을 능가한다.   
(We propose a balanced sampler for matching calculating cross-entropy loss, which outperform the commonly practiced binary and/or ordinary cross-entropy loss.)

우리는 실험 섹션에서 종합적인 절제 연구, 안정성 연구, 해석성 연구를 수행한다.   
(We perform comprehensive ablation study, stability study, and interpretability study in the experimental section)

우리는 제안된 구성요소에 의한 의한 recall 비율의 향상을 증명하고 하이퍼 파라미터의 변형에 대한 강력한 안정성을 변경한다.   
(We prove an improvement of recall rates by the proposed components and vary robust stability against hyperparameter's variation.)

우리는 또한 공간적 이산화를 위한 계층적 그리드 방식을 단순 선형 보간 기술로 교체할 것을 제안한다. 이것은 밀집된 표현을 제공하면서 연속적인 공간적 거리를 나타낼 수 있다.   
(We also propose to replace the hierachical gridding method for spatial discretization with a simple linear interpolation technique, which can reflect the continuous spatial distance while providing dense representation.)

기존 모델과의 실험적 비교는 우리 모델의 우수성을 명백히 보여준다. STAN은 최첨단 모델을 9-17% 능가하는 recall 비율을 향상시킨다.   
(Experimental comparision with baseline models unequivocally demonstrates the superiority of our model, as STAN improves recall rates to new records that surpass the state-of-the-art-models by 9-17%)
