'''
Convert Date time to features
Source: https://github.com/createai/Footfall-regression/blob/master/notebook/Ultimate_Student_Hunt.ipynb
'''

# Converting 'Date' column into datetime datatype
df["Date"] = pd.to_datetime(df["Date"], format='%d-%m-%Y')

# Splitting Date into Day, Month and Year 
df['Day'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

df.describe()