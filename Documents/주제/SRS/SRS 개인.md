## 1. 소개(목적)
- 소개: 2019년에는 코로나 팬데믹으로 인한 여행산업의 수요가 줄어들어 관광시장의 규모가 축소되었습니다.
  그러나 최근 "2023년 부산관광기업 모니터링 조사" 결과에 따르면, 관광업계의 규모와 수익성은
  BSI 지수를 통해 평가되었을 때 모두 100 이상의 수치를 기록했습니다.
  또한, 계절적인 요인을 배제한 내국인 관광객 수도 BSI 수치가 100 이상을 유지하며 부산 내국인 관광객 수가 확대될 것으로 전망했습니다. 

  팬더믹이후 배우자, 자녀, 친구등 관광참여는 점차감소하는추세이나, 1인여행과 가족 소규모 여행이 정차 증가함.
  또한 환경 및 기술의 변화와 재택 및 원격 근무의 증가로 일과 생활의 경계가 허물어져  요일에 관계없이 개인화 취향의 여행수요가 생겨났습니다.

- 목적: 종래의 관광지 추천 서비스는 관광지를 추천만 할 뿐, 주변에 경험할 수 있는 먹거리/즐길거리와 같은 시설이나 행사에 대한 
  정보   제공이 부족하다.
  이러한 배경으로 관광객을 대신해 관광지 주변의 즐길거리를 탐색하고, 관광객의 취향에 맞는 장소를 제안해 관광객의 의사결정을 도울 수 있는 시스템이 필요하다.

- 용어 정의
  - BSI: 전체 응답 중 증가와 호전 등 긍정적인 응답비중과 감소와 악화 등 부정적인 응답비중의 차이로 산출된다. 100 이상이면 경기가 좋다는 의미로, 100 이하면 경기가 좋지 않다는 의미로 해석된다.

## 2. 기능적 요구사항
- 평점 기반 추천 (★★★★★)
  - 추천 시스템의 최소 조건

- 위치 기반 추천 (★★★★★)
  - POI 데이터(위도/경도) 활용
  - 거리에 따른 가중치 부과
  - 혹은, 교통편에 따른 실제 이동거리에 따라 가중치 부과 (★)

- 추천 장소 시각화 (★★★★)
  - 추천 장소를 지도 위에 표기
  - 사용자가 추천 장소를 선택하면 경로 제시 (★)
  - 추천 장소와 관련된 정보(공식 홈페이지 등)과 연결 (★★)

- 사용자의 SNS 기반 추천 (★★)
  - 인스타그램 등에서 사용자의 과거 여행 경험을 수집
  - 게시물의 위치 정보, 사진, 해시태그 등으로 여행 경험을 분류
  - 사용자의 해시태그를 이용
  - 사용자의 리뷰를 이용

- 웹 또는 앱 서비스 (★★★)
  - 시스템을 웹/앱 형태로 제작 및 배포
  
## 3. 비기능적 요구사항
- 정확성 
  - 대표적인 관광앱과 비교하여 추천 관광지의 95%가 일치해야합니다.
- 최적화
  - 방문할만 관광지에 대한 최단 경로를 만들어 줍니다.
- 보안성 
  - 사용자의 개인정보를 관광지 추천외에는 사용하지 않습니다.

## 4. 참고문서
- [대한민국 구석구석](https://korean.visitkorea.or.kr/main/main.do)
- [tripadviser](https://www.tripadvisor.co.kr )
- https://korean.visitkorea.or.kr/main/main.do
- https://kr.trip.com/travel-guide/
- https://travel4u.naver.com/
- [2023년 2분기 부산관광기업 모니터링 조사](https://www.cleaneye.go.kr/user/empStudyReport.do?entId=2012000016&itemId=studyReport&fixedYear=2019&pastYear=2015&beyondYear=2023&openedDate=2020.07.01&entName=%EB%B6%80%EC%82%B0%EA%B4%80%EA%B4%91%EA%B3%B5%EC%82%AC&entKind=006003&budgetSumYear=2020&openedDateHalf=2019&fixedHalfYear=2019&pastHalfYear=2015&dtFlagHalf=500220&openedDateQuarter=2020&fixedQuarterYear=2020&pastQuarterYear=2016&dtFlagQuarter=500110)
- [2023년 3분기 부산관광기업 모니터링 조사](https://www.cleaneye.go.kr/user/empStudyReport.do?entId=2012000016&itemId=studyReport&fixedYear=2019&pastYear=2015&beyondYear=2023&openedDate=2020.07.01&entName=%EB%B6%80%EC%82%B0%EA%B4%80%EA%B4%91%EA%B3%B5%EC%82%AC&entKind=006003&budgetSumYear=2020&openedDateHalf=2019&fixedHalfYear=2019&pastHalfYear=2015&dtFlagHalf=500220&openedDateQuarter=2020&fixedQuarterYear=2020&pastQuarterYear=2016&dtFlagQuarter=500110)
- [2023년 4분기 부산관광기업 모니터링 조사](https://www.cleaneye.go.kr/user/empStudyReport.do?entId=2012000016&itemId=studyReport&fixedYear=2019&pastYear=2015&beyondYear=2023&openedDate=2020.07.01&entName=%EB%B6%80%EC%82%B0%EA%B4%80%EA%B4%91%EA%B3%B5%EC%82%AC&entKind=006003&budgetSumYear=2020&openedDateHalf=2019&fixedHalfYear=2019&pastHalfYear=2015&dtFlagHalf=500220&openedDateQuarter=2020&fixedQuarterYear=2020&pastQuarterYear=2016&dtFlagQuarter=500110)
- [2023 부산 방문 관광객 실태조사](https://www.cleaneye.go.kr/user/empStudyReport.do?entId=2012000016&itemId=studyReport&fixedYear=2019&pastYear=2015&beyondYear=2023&openedDate=2020.07.01&entName=%EB%B6%80%EC%82%B0%EA%B4%80%EA%B4%91%EA%B3%B5%EC%82%AC&entKind=006003&budgetSumYear=2020&openedDateHalf=2019&fixedHalfYear=2019&pastHalfYear=2015&dtFlagHalf=500220&openedDateQuarter=2020&fixedQuarterYear=2020&pastQuarterYear=2016&dtFlagQuarter=500110)
