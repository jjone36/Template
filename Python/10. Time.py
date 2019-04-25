import pandas as pd
from datetime import datetime

####################### Autocorrelation, Partial autocorrelation
from statsmodels.graphics import tsaplots

tsaplots.plot_acf(df['col'], lags = 40)    # plot_pacf()
plt.show()

# Times series decomposition
import statsmodels.api as sm
decomposition = sm.tsa.seasonal_decompose(df)
print(dir(decompostition))

trend = decomposition.trend
trend.plot()

####################### Manipulating time data
# pd.Timestamp
time_stamp = pd.Timestamp(datetime(2018, 1, 1))
timestamp('2018-01-01 00:00:00', freq = 'M')

# pd.Period()
period = pd.Period('2018-01')
period.asfreq('D')   # Convert to daily frequency
period.to_timestamp().to_period('M')   # Conver to pd.Timestamp() and back
period + 2

# date_range()
index = pd.date_range(start = '2017-01-01', periods = 7, freq = 'M')

# Upsampling
df.asfreq('D').info()

# .shift(), .div(), sub(), nul(), .diff(), .pct_change()
df.price.shift(periods = -1)
df.price.sub(100).mul(2)
df.price.diff()
df.price.pct_change()
