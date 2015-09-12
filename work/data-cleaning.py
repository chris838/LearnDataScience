import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# import the data
df = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

# remove the '%' suffix from each row
df['Interest.Rate'] = [float(x.strip('%')) for x in df['Interest.Rate']]

# remove the ' months' suffix from each row
df['Loan.Length'] = [int(x.strip('months')) for x in df['Loan.Length']]

# remove the outlier rows
df = df[np.abs(df['Monthly.Income']-df['Monthly.Income'].mean())<=(3*df['Monthly.Income'].std())]

# remove rows with NA
def isBadData(x) :
  return re.match("(N/A|NA|na|n/a)", str(x))

df = df.applymap(lambda x: np.nan if isBadData(x) else x)
df = df.dropna()



