# read the data from the pkl file
import pandas as pd
df = pd.read_pickle("data/origin_line_leg_scores copy.pkl")

# print the first 5 rows
print(df.head(500))

# print the columns
print(df.columns)


# set show all columns in pandas debugger
pd.set_option('display.max_columns', None)

# print the first 5 rows
print(df.head(5))

print(df.describe())
