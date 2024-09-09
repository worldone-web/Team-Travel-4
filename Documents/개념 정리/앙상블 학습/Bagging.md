# 앙상블 학습법(Ensemble Learning)
## 앙상블 학습의 개념
앙상블 학습이란, 하나의 학습 알고리즘을 사용하는 것 보다 더 좋은 예측 성능을 얻기 위해, 복수의 학습 알고리즘을 사용하는 것을 의미한다. (wikipedia, 2024)

## 용어
- 앙상블(Ensemble): 둘 이상의 모델을 결합하여 더 나은 결과를 얻는 기법
- 학습 알고리즘(learning algorithm): 기계 학습에 사용하는 학습 방법 (로지스틱 회귀, 결정 트리, 인공신경망 등)


## 앙상블 학습: 배깅(Bagging)
![배깅](https://blog.kakaocdn.net/dn/brvKWi/btrav6KEpto/jLG48GsfPaUotdvoIsXsSk/img.png)

배깅은 Bootstrap aggregation을 의미한다.  
배깅은 훈련셋의 원형으로 서브셋을 만들고, 동일한 학습 알고리즘으로 서브셋마다 모델을 만들어 최종적으로 앙상블하는 기법이다.   
배깅의 대표적인 알고리즘은 랜덤 포레스트(random forest)이다.   
배깅은 Bootstrapping과 Aggregation 두 과정을 거친다.   
### Bootstrapping
통계학에서 Bootstrpping은 복원추출(random sampling with replacement)을 사용해 표본을 재표집(resampling)하는 기법을 의미한다. (wikipedia, 2024)   
복원 추출이란 샘플을 추출해 값을 기록하고 제자리에 돌려 놓는 것을 의미한다. (ayi4067, 2024)

![복원추출](https://velog.velcdn.com/images%2Fayi4067%2Fpost%2F8c2342ee-4c33-44ae-8cf2-7f08077b9983%2Fimage.png)

기계 학습에서 Bootstrapping은 훈련 셋을 재표집한 bootstrapped set을 생성하는 것이다.   
훈련 셋에서 등장한 데이터는 Bootstrapped set에서 0번 이상 등장할 수 있다.   

Bootstrpped set을 생성한 후, 각 데이터 셋을 훈련 셋으로 삼아 모델을 학습시킨다.    
이때, 모든 모델은 동일한 학습 알고리즘을 사용한다.   

### Aggregation
Aggregation은 각 모델의 결과를 모아 결과를 얻는 앙상블 과정이다.   
이것은 보팅(Voting)을 사용해 수행된다.   
회귀 문제일 경우, 기본 모델의 결과의 평균값을 예측하고   
분류 문제일 경우, 다수결로 클래스를 예측한다.   


# 참고
[앙상블 학습] wikipedia, Ensemble learning, 2024, https://en.wikipedia.org/wiki/Ensemble_learning   
[Bootstrapping] wikipedia, Bootstrapping (statistics), 2024, https://en.wikipedia.org/wiki/Bootstrapping_(statistics)   
[복원 추출] ayi4067, DAY27, velog, 2024, https://velog.io/@ayi4067/DAY27   

