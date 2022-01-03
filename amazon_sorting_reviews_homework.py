# Business Problem:
# To calculate product ratings more accurately work and product reviews more accurately to sort.

# Dataset Story:
# This dataset, which includes Amazon product data, includes product categories and various metadata.
# The product with the most reviews in the electronics category has user ratings and reviews.

# Variables

# reviewerID : User ID
# asin : Product ID
# reviewerName : Username
# helpful : Degree of Useful Evaluation (Sample : 2/3)
# reviewText : Review (User-written review text)
# overall : Product Rating
# summary : Evaluation Summary
# unixReviewTime : Evaluation Time
# reviewTime : Review Time
# day_diff : Number of Days Since Evaluation
# helpful_yes : The Number of Times the Review was Found Helpful
# total_vote – Number of Votes Given to the Review


import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv("/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_5/amazon_review.csv")
df.head()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

df.info()

df["overall"].mean()

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22, w5 =20):
    return dataframe.loc[dataframe["day_diff"] <= 30, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 30) & (dataframe["day_diff"] <= 90), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 90) & (dataframe["day_diff"] <= 180), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 180) & (dataframe["day_diff"] <= 270), "overall"].mean() * w4/ 100 + \
           dataframe.loc[(dataframe["day_diff"] > 270), "overall"].mean() * w5 / 100

time_based_weighted_average(df)

rate =  (( time_based_weighted_average(df) - df["overall"].mean() ) / df["overall"].mean()  ) * 100

df["priority_review"] = df["helpful_yes"] / df["total_vote"]
df["priority_review"].fillna(0, inplace = True)
df["priority_review"] = df["priority_review"].astype(int)
df.head(20)

df["helpful_yes"].value_counts()

df["helpful_no"] =  df["total_vote"] - df["helpful_yes"]
df.head(20)

comments = pd.DataFrame({"up": df["helpful_yes"], "down": df["helpful_no"]})

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"],x["down"]),axis=1)

comments.sort_values("wilson_lower_bound", ascending=False).head(20)

#         up  down  wilson_lower_bound
# 2031  1952    68                0.96
# 3449  1428    77                0.94
# 4212  1568   126                0.91
# 317    422    73                0.82
# 4672    45     4                0.81
# 1835    60     8                0.78
# 3981   112    27                0.73
# 3807    22     3                0.70
# 4306    51    14                0.67
# 4596    82    27                0.66
# 315     38    10                0.66
# 1465     7     0                0.65
# 1609     7     0                0.65
# 4302    14     2                0.64
# 4072     6     0                0.61
# 1072     5     0                0.57
# 2583     5     0                0.57
# 121      5     0                0.57
# 1142     5     0                0.57
# 1753     5     0                0.57

