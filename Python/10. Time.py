# -*- coding: utf-8 -*-

import pandas as pd

####################### Manipulating time data
# Create the range of dates here
seven_days = pd.date_range(start = '2017-01-01', periods = 7)

# Iterate over the dates and print the number and name of the weekday
for day in seven_days:
    print(day.dayofweek, day.weekday_name)
    
    