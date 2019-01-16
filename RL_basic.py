#####################################################
############# Upper Confident Bound
import random

N = 10000   # number of users
n_ads = 10  # number of ads

ads_selected = []
total_reward = 0
for n in range(N):
    ad = random.randrange(n_ads)
    ads_selected.append(ad)
    reward = df.values[n, ad]
    total_reward = total_reward + reward
