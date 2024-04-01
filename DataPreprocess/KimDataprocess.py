import pandas as pd
import json


def load_data(file):
    with open(file, encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    return data

def preprocess_reviews(data):
    reviews = pd.DataFrame(data)
    reviews = reviews[['user_id', 'business_id', 'stars']]
    return reviews

review_file = 'DataSet/yelp_academic_dataset_review.json'
reviews_data = load_data(review_file)
reviews = preprocess_reviews(reviews_data)


reviews.to_csv('reviews.csv', index=False)