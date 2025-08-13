#/bin/sh
python prep.py yelp_review_polarity/train.csv train.pt 56000
python prep.py yelp_review_polarity/test.csv test.py 38000
