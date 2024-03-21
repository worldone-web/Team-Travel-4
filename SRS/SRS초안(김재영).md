## 1. 소개(목적)
- 1.1목적
  - 이 문서의 목적은 2019년 코로나 팬데믹으로 인해 여행산업의 하락을 겪은 후, 최근 관광시장의 회복에 대한 정보를 토대로
    부산 지역의 관광 산업 활성화를 위해 개인화된 관광지 추천 시스템을 개발하기 위한 요구사항을 명세하는 것입니다.

OR

2019년에는 코로나 팬데믹으로 인한 여행산업의 수요가 줄어들어 관광시장의 규모가 축소되었습니다. 그러나 최근 "2023년 부산관광기업 모니터링 조사" 결과에 따르면, 
관광업계의 규모와 수익성은 BSI 지수를 통해 평가되었을 때 모두 100 이상의 수치를 기록했습니다. 
또한, 계절적인 요인을 배제한 내국인 관광객 수도 BSI 수치가 100 이상을 유지하며 부산 내국인 관광객 수가 확대될 것으로 전망했습니다.

이와 같이 부산 관광산업의 전망이 좋아짐에도 불구하고, "2023년 부산 방문 관광객 실태조사" 결과에 따르면, 
방문한 내국인 중 37.2%가 부산 시내교통이 복잡해서 재방문 의향이 없다는 결과가 나왔습니다. 
이를 보완하고자 부산 지역의 관광 산업을 더욱 활성화시키고 재방문률을 높이기 위해 개인화된 관광지 추천 시스템을 개발하는 것이 필요합니다.







## 2. 전체 시스템 설명(제품의 범위)
- 2.1 제품 위치
  - 시스템은 부산 지역의 관광 산업을 활성화하고 사용자의 여행 경험을 향상시키기 위해 개발됩니다. 이는 웹 서비스로 제공됩니다.
    
- 2.2 제품 기능
  - 사용자는 시스템을 통해 해시태그를 선택하여 자신의 관심사를 나타낼 수 있습니다.
  - 시스템은 선택된 해시태그를 기반으로 사용자의 개인화된 관심사를 파악하고, 이를 활용하여 관광지와 먹거리 정보를 지도상에 표시합니다.
  - 사용자가 추천된 관광지를 선택하면 해당 관광지 주변의 먹거리 및 다른 관광지의 추천 이동 경로를 제시하여 편의성을 제공합니다.
    
- 2.3 사용자 클래스와 특성
  - 이 시스템은 주로 관광객 및 지역 주민을 대상으로 합니다. 관광객은 부산을 방문하여 관광 명소와 먹거리를 탐색할 수 있습니다.

OR

이 시스템은 사용자에게 해시태그를 제공하고 사용자가 원하는 해시태그를 선택하여 개인화된 관심사를 파악합니다. 이 정보를 기반으로 관광지와 먹거리 정보를 지도상에 표시합니다. 
또한, 사용자가 개인화를 이용하여 추천된 관광지 중 하나를 선택하면 해당 관광지를 중심으로 주변의 먹거리와 다른 관광지의 추천 이동 경로를 표시하여 편의성을 제공합니다.

## 3. 참조 문서
- [2023년 2분기 부산관광기업 모니터링 조사](https://www.cleaneye.go.kr/user/empStudyReport.do?entId=2012000016&itemId=studyReport&fixedYear=2019&pastYear=2015&beyondYear=2023&openedDate=2020.07.01&entName=%EB%B6%80%EC%82%B0%EA%B4%80%EA%B4%91%EA%B3%B5%EC%82%AC&entKind=006003&budgetSumYear=2020&openedDateHalf=2019&fixedHalfYear=2019&pastHalfYear=2015&dtFlagHalf=500220&openedDateQuarter=2020&fixedQuarterYear=2020&pastQuarterYear=2016&dtFlagQuarter=500110)
- [2023년 3분기 부산관광기업 모니터링 조사](https://www.cleaneye.go.kr/user/empStudyReport.do?entId=2012000016&itemId=studyReport&fixedYear=2019&pastYear=2015&beyondYear=2023&openedDate=2020.07.01&entName=%EB%B6%80%EC%82%B0%EA%B4%80%EA%B4%91%EA%B3%B5%EC%82%AC&entKind=006003&budgetSumYear=2020&openedDateHalf=2019&fixedHalfYear=2019&pastHalfYear=2015&dtFlagHalf=500220&openedDateQuarter=2020&fixedQuarterYear=2020&pastQuarterYear=2016&dtFlagQuarter=500110)
- [2023년 4분기 부산관광기업 모니터링 조사](https://www.cleaneye.go.kr/user/empStudyReport.do?entId=2012000016&itemId=studyReport&fixedYear=2019&pastYear=2015&beyondYear=2023&openedDate=2020.07.01&entName=%EB%B6%80%EC%82%B0%EA%B4%80%EA%B4%91%EA%B3%B5%EC%82%AC&entKind=006003&budgetSumYear=2020&openedDateHalf=2019&fixedHalfYear=2019&pastHalfYear=2015&dtFlagHalf=500220&openedDateQuarter=2020&fixedQuarterYear=2020&pastQuarterYear=2016&dtFlagQuarter=500110)
- [2023 부산 방문 관광객 실태조사](https://www.cleaneye.go.kr/user/empStudyReport.do?entId=2012000016&itemId=studyReport&fixedYear=2019&pastYear=2015&beyondYear=2023&openedDate=2020.07.01&entName=%EB%B6%80%EC%82%B0%EA%B4%80%EA%B4%91%EA%B3%B5%EC%82%AC&entKind=006003&budgetSumYear=2020&openedDateHalf=2019&fixedHalfYear=2019&pastHalfYear=2015&dtFlagHalf=500220&openedDateQuarter=2020&fixedQuarterYear=2020&pastQuarterYear=2016&dtFlagQuarter=500110)
  
## 4. 결론 
코로나 팬데믹으로 인한 여행산업의 하락에도 불구하고, 2023년 부산관광기업의 모니터링 조사 결과는 부산 지역의 관광 산업이 꾸준히 성장하고 있다는 긍정적인 메시지를 전달합니다. 
관광업계의 매출과 수익성은 매분기마다 지속적으로 상승하고 있으며, 내국인 관광객의 수도 늘어나고 있는 것으로 나타났습니다. 
계절적인 요인을 배제한 내국인 관광객의 수치는 매 분기마다 100 이상의 BSI 수치를 보여주며, 전년도에 비해 확대되고 있는 것으로 전망됩니다.

이러한 경향을 고려하여 부산 지역의 관광 산업을 더욱 활성화하기 위해 개인화된 관광지 추천 시스템이 필요합니다. 
이 시스템은 사용자가 선택한 해시태그를 기반으로 개인화된 관심사를 파악하고, 
관광지와 먹거리 정보를 지도상에 표시하여 사용자에게 최적화된 여행 경험을 제공합니다. 
또한, 사용자가 선택한 추천 관광지 중 하나를 선택하면 해당 관광지를 중심으로 주변의 먹거리와 다른 관광지로의 추천 이동 경로를 제시하여 편의성을 높여줍니다.

이러한 시스템은 부산 지역의 관광 산업을 더욱 활성화시키고, 사용자들에게 더 나은 관광 경험을 제공하여 지속적인 성장을 도모할 것으로 기대됩니다.

## 4. 비고
- BSI는 전체 응답 중 증가와 호전 등 긍정적인 응답비중과 감소와 악화 등 부정적인 응답비중의 차이로 산출된다. 100 이상이면 경기가 좋다는 의미로, 100 미만면 경기가 좋지 않다는 의미로 해석된다.

#버전 1.0: 초기 버전
