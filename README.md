# MyNaiveClassificationModel

LOGIC:
It first converts all columns to categorical columns then simply equate features from test data to train data (think of it as 
creating buckets of features -> entire n-D space divided in buckets and then choose the right bucket given our test data)
and then calculating percent and count for that bucket.

CAUTION:
Do LABEL ENCODING instead of one hot encoding
