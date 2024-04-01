import pandas as pd

df = pd.read_csv("Philadelphia_reviews.csv")

# 각 사업체의 리뷰 수 계산
business_review_counts = df['business_id'].value_counts()
print(business_review_counts)

# 리뷰 수 상위 n개 사업체 선택
top_businesses = business_review_counts.head(500).index

# 각 사업체에서 최대 n개의 리뷰만 선택
selected_data = pd.DataFrame(columns=df.columns)
for business in top_businesses:
    business_reviews = df[df['business_id'] == business].head(200)
    selected_data = pd.concat([selected_data, business_reviews])

# 새로운 데이터셋으로 저장
selected_data.to_csv("500x200_philadelphia_reviews.csv", index=False)
