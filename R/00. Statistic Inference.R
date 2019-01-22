
'https://www.datacamp.com/courses/foundations-of-inference'
####################### Inference for Categorical Data #####################
'https://www.datacamp.com/courses/inference-for-categorical-data'
###### Chapter 1 ######
library(infer)
gss

gss2016 = gss %>%
  filter(year == 2016)

gss2016 %>%
  ggplot(aes(x = cappun)) + geom_bar()

str(gss2016$cappun)

p_hat = gss2016 %>%
  summarise(mean(cappun == 'FAVOR', na.rm = T)) %>%
  pull()
p_hat

## Bootstrapping (sampling with replacement. same size of the original data)
boot = gss2016 %>%
  specify(response = cappun, success = 'FAVOR') %>%
  generate(reps = 500, type = 'bootstrap') %>%
  calculate(stat = 'prop')

## Null Hypothesis - Point Estimation
null = gss2016 %>%
  specify(response = cappun, success = 'FAVOR') %>%
  hypothesize(null = 'point', p = .5) %>%
  generate(reps = 500, type = 'simulate') %>%
  calculate(stat = 'prop')

mean(boot$stat)
p_hat
mean(null$stat)

null %>%
  mutate(boot_stat = boot$stat) %>%
  ggplot() +
  geom_density(aes(x = stat, col = 'blue')) +
  geom_density(aes(x = boot_stat)) +
  geom_vline(aes(xintercept = p_hat, col = 'green'))


gss2016$postlife %>% table()
ggplot(gss2016, aes(x = postlife)) + geom_bar()

p_hat = gss2016 %>%
  summarize(mean(postlife == 'YES', na.rm = T)) %>%
  pull()
p_hat

sim1 = gss2016 %>%
  specify(response = postlife, success = 'YES') %>%
  hypothesize(null = 'point', p = .79) %>%
  generate(reps = 500, type = 'simulate')

ggplot(sim1, aes(x = postlife)) + geom_bar()

null = sim1 %>%
  calculate(stat = 'prop')

ggplot(null, aes(x = stat)) +
  geom_density() +
  geom_vline(aes(xintercept = p_hat, col = 'red'))

p_value = null %>%
  summarise(mean(stat > p_hat)) %>%
  pull()


## Null Hypothesis - Test of Independence
ggplot(gss2016, aes(x = cappun, fill = sex)) + geom_bar() + theme_bw()

p_hat = gss2016 %>%
  group_by(sex) %>%
  summarise(prop = mean(cappun == 'FAVOR', na.rm = T))
d_hat = diff(p_hat$prop)
d_hat

null2 = gss2016 %>%
  specify(cappun ~ sex, success = 'FAVOR') %>%
  hypothesize(null = 'independence') %>%
  generate(reps = 500, type = 'permute') %>%
  calculate(stat = 'diff in props', order = c('FEMALE', 'MALE'))

ggplot(null2, aes(x = stat)) + geom_density() +
  geom_vline(xintercept = d_hat, col = 'red')

## p-value
null2 %>%
  summarize(mean(stat >= d_hat)) %>%
  pull()*2

## Standard Error (= standard deviation of the sample)
SE = gss2016 %>%
  specify(cappun ~ sex, success = 'FAVOR') %>%
  generate(reps = 500, type = 'bootstrap') %>%
  calculate(stat = 'diff in props', order = c('FEMALE', 'MALE')) %>%
  summarise(sd(stat)) %>%
  pull()
SE

ci = c(d_hat - 2*SE, d_hat + 2*SE)

## Rejection Region
alpha = .05
upper_tail = null2 %>%
  summarise(quantile(stat, probs = (1 - alpha/2))) %>%
  pull()

lower_tail = quantile(null2$stat, probs = alpha/2)

ci2 = c(lower_tail, upper_tail)

ggplot(null2, aes(x = stat)) +
  geom_density() +
  geom_vline(xintercept = c(lower_tail, upper_tail), color = 'red') +
  geom_vline(xintercept = d_hat, lty = 2, color = 'blue')

d_hat %>% between(lower_tail, upper_tail)


## Approximation shortcut (independent observations & large number)
p_hat = gss2016 %>%
  summarise(mean(cappun == 'FAVOR', na.rm = T)) %>%
  pull()
n = nrow(gss2016)
c(n*p_hat, n*(1-p_hat))

SE_approx = sqrt(p_hat*(1-p_hat)/n)
SE

###### Chapter 2 ######
## Contingency table
gss2016 = gss %>%
  filter(year == 2016)

table(gss2016$partyid)

new_party = gss2016 %>%
  filter(partyid != 'OTHER PARTY') %>%
  pull(partyid) %>%
  as.character()     # factor -> char variable!!

DEM = grepl(pattern = 'DEM', x = new_party)
new_party[DEM] <- 'D'

new_party[grepl(pattern = 'REP', x = new_party)] = 'R'
new_party[new_party == 'INDEPENDENT'] = 'I'

table(new_party)
new_partyid = as.factor(new_party)
str(new_party)

gss_party = gss2016 %>%
  filter(partyid != 'OTHER PARTY') %>%
  mutate(party = new_party)

table(gss_party$party)

ggplot(gss_party, aes(x = party, fill = natspac)) + geom_bar()
ggplot(gss_party, aes(x = party, fill = natspac)) + geom_bar(position = 'fill')
gss_party %>%
  filter(!is.na(natspac)) %>%
  ggplot(aes(x = party, fill = natspac)) + geom_bar(position = 'fill')  # -> No tendency

gss_party %>%
  filter(!is.na(natarms)) %>%
  ggplot(aes(x = party, fill = natarms)) +
  geom_bar(position = 'fill')

## table <-> tidy
tab = gss_party %>%
  select(natspac, party) %>%
  table()

tab %>%
  tidy() %>%
  uncount(Freq)

## Chi-squared test : natarms ~ party
perm_1 = gss_party %>%
  specify(natarms ~ party) %>%
  hypothesize(null = 'independence') %>%
  generate(reps = 1, type = 'permute')

perm_1 %>%
  ggplot(aes(x = party, fill = natarms)) + geom_bar(position = 'fill')

tab = perm_1 %>%
  ungroup() %>%
  select(natarms, party) %>%
  table()

chisq.test(tab)$statistic    # repeat permutation!


null_arms = gss_party %>%
  specify(natarms ~ party) %>%
  hypothesize(null = 'independence') %>%
  generate(reps = 100, type = 'permute') %>%
  calculate(stat = 'Chisq')

chi_obs_arms = chisq.test(gss_party$natarms, gss_party$party)$statistic

ggplot(null_arms, aes(x = stat)) +
  geom_density() +
  geom_vline(xintercept = chi_obs_arms, color = 'red')     # -> Chi-squared Distribution


## Chi-squared test : natspec ~ party
null_spac = gss_party %>%
  specify(natspac ~ party) %>%
  hypothesize(null = 'independence') %>%
  generate(reps = 100, type = 'permute') %>%
  calculate(stat = 'Chisq')

chi_obs_spac = chisq.test(gss_party$natspac, gss_party$party)$statistic

ggplot(null_spac, aes(x = stat)) +
  geom_density() +
  geom_vline(xintercept = chi_obs_spac, color = 'red') +
  stat_function(fun = dchisq, args = list(df = 4), color = 'blue')

############################################################################
####################### Inference for Numerical Data #######################
'https://www.datacamp.com/courses/inference-for-numerical-data'
###### Chapter 1 ######
library(openintro)
data("stem.cell")
head(stem.cell)
str(stem.cell)

diff_avg = stem.cell %>%
  mutate(change = after - before) %>%
  group_by(trmt) %>%
  summarise(mean(change)) %>%
  pull() %>%
  diff()

## Ha : difference between the mean of the change in the control & treatment groups
null = stem.cell %>%
  specify(change ~ trmt) %>%
  hypothesize(null = 'independence') %>%
  generate(reps = 1000, type = 'permute') %>%
  calculate(stat = 'diff in means', order = c('esc', 'ctrl'))

ggplot(null, aes(stat)) + geom_density() +
  geom_vline(xintercept = diff_avg)

null %>%
  summarise(mean(stat > diff_mean))


## Ha : weights of babies are differenct between smoker & non-smoker mothers
ncbirths %>% head()
summary(ncbirths$habit)

ncbirths_habit = ncbirths %>%
  filter(!is.na(habit))

ggplot(ncbirths_habit, aes(x = weight, fill = habit)) + geom_histogram()

weight_avg = ncbirths_habit %>%
  group_by(habit) %>%
  summarise(avg = mean(weight)) %$%
  diff(avg)

null = ncbirths_habit %>%
  specify(weight ~ habit) %>%
  hypothesize(null = 'independence') %>%
  generate(reps = 1000, type = 'permute') %>%
  calculate(stat = 'diff in means', order = c('nonsmoker', 'smoker'))

(p = ggplot(null, aes(x = stat)) +
    geom_density() +
    geom_vline(xintercept = c(weight_avg, -weight_avg), col = 'red', lty = 2))

p_value = null %>%
  summarise(mean(stat < weight_avg)) %>%
  pull()   # reject!

boot = ncbirths_habit %>%
  specify(weight ~ habit) %>%
  generate(reps = 1500, type = 'bootstrap') %>%
  calculate(stat = 'diff in means', order = c('nonsmoker', 'smoker'))

lower_tail = quantile(boot$stat, .025)
upper_tail = quantile(boot$stat, .975)
c(lower_tail, upper_tail)

p + geom_vline(xintercept = c(lower_tail, upper_tail), col = 'blue')

se = boot %>%
  summarise(sd(stat)) %>%
  pull()
c(weight_avg - 2*se, weight_avg + 2*se)


###### Chapter 2 ######
url = 'https://assets.datacamp.com/production/course_5103/datasets/manhattan.csv'
manhattan = read_csv(url)
manhattan

rent_med_obs = median(manhattan$rent)

boot_rent = manhattan %>%
  specify(response = rent) %>%
  generate(reps = 15000, type = 'bootstrap') %>%
  calculate(stat = 'median')

## t Distribution -> SE method
t_value = qt(p = .975, df = nrow(manhattan)-1)

rent_ci_med = boot_rent %>%
  summarise(SE = sd(stat)) %>%
  summarise(lower = rent_med_obs - t_value * SE,
            upper = rent_med_obs + t_value * SE)

## percentage method
lower_tail = quantile(boot_rent$stat, probs = .025)
upper_tail = quantile(boot_rent$stat, probs = .975)
c(lower_tail, upper_tail)


ggplot(boot_rent, aes(x = stat)) +
  geom_histogram(fill = 'darkred') +
  geom_vline(xintercept = rent_med_obs, col = 'yellow') +
  geom_vline(xintercept = c(lower_tail, upper_tail), col = 'blue', size = 2) +
  geom_vline(xintercept = c(rent_ci_med$lower, rent_ci_med$upper), col = 'orange', size = 2) +
  theme_bw()


## P(t < T) = p : pt(q, df) = p | qt(p, df) = T
pt(q = 3, df = 10)    # -> probability
qt(.975, df = 10)    # -> cutoff value


###### Chapter 3 ######
url = 'https://assets.datacamp.com/production/course_5103/datasets/gss_wordsum_class.csv'
vocab = read_csv(url)
vocab

levels(vocab$class)

vocab$class = factor(vocab$class, levels = c('UPPER', 'MIDDLE', 'WORKING', 'LOWER'))

ggplot(vocab, aes(x = wordsum, fill = class)) + geom_histogram(show.legend = F) +
  facet_grid(class ~ .) + theme_bw()

vocab %>%
  group_by(class) %>%
  summarise(var = sd(wordsum))

## ANOVA
(m = aov(wordsum ~ class, data = vocab))
tidy(m)

oneway.test(wordsum ~ class, data = vocab)

## multiple comparison
pairwise.t.test(x = vocab$wordsum, g = vocab$class, p.adjust.method = 'none')
pairwise.t.test(x = vocab$wordsum, g = vocab$class, p.adjust.method = 'BH')

TukeyHSD(m)


############################################################################
####################### Bayesian Inference #################################
'https://www.datacamp.com/courses/fundamentals-of-bayesian-data-analysis-in-r'
###### Chapter 1 ######
n_samples = 10000
n_ads = 100
prop_cliked = runif(n = n_samples, min = 0, max = .2)    # -> prior
hist(prop_cliked)

n_clicked = rbinom(n = n_samples, size = n_ads, prob = prop_cliked)    # -> likelihood
n_clicked2 = rbinom(n = n_samples, size = n_ads, prob = .1)
hist(n_clicked)
hist(n_clicked2)

prior = data.frame(prop_cliked, n_clicked)
head(prior)

prior_const = data.frame(prop_cliked, n_clicked2)
head(prior_const)

posterior = prior[prior$n_clicked == 4, ]    # -> data
head(posterior)

prior2 = posterior
prior2$n_clicked = rbinom(n = nrow(posterior), size = n_ads, prob = posterior$prop_cliked)    # -> update
head(prior2)
hist(prior2$n_clicked)
hist(prior$n_clicked)

##Comparison
(p = ggplot(prior, aes(x = n_clicked, y = prop_cliked)) +
    geom_jitter(col = '#377EB8') +
    labs(x = 'Number of Clicked', y = 'Probabilities') +
    theme_bw())

mean(prior$n_clicked == 13)
mean(prior$n_clicked == 6)
hist(prior$n_clicked)

posterior_video = prior[prior$n_clicked == 13, ]
posterior_text = prior[prior$n_clicked == 6, ]

p + geom_vline(xintercept = c(13, 6), col = 'dark red', lty = 4, size = 1.5) +
  geom_text(aes(x = 7.6, y = 0.005, label = format('Text Ads')), fontface = 'bold') +
  geom_text(aes(x = 15, y = 0.005, label = format('Video Ads')), fontface = 'bold')
hist(posterior_video$prop_cliked, xlim = c(0, .25), col = '#377EB8', xlab = 'Probabilities', main = 'Probability D. of Video Ads')
hist(posterior_text$prop_cliked, xlim = c(0, .25), col = '#377EB8', xlab = 'Probabilites', main = 'Probability D. of Test Ads')

dim(posterior_text)
dim(posterior_video)
posterior = data.frame(video_prop = posterior_video$prop_cliked[1:470],
                       text_prop = posterior_text$prop_cliked[1:470])
head(posterior)

posterior$diff_prop = posterior$video_prop - posterior$text_prop
mean(posterior$diff_prop > 0)

visior_spend = 2.53
video_cost = .25
text_cost = .05

posterior$video_profit = posterior$video_prop*visior_spend - video_cost
posterior$text_profit = posterior$text_prop*visior_spend - text_cost
posterior$diff_profit = posterior$video_profit - posterior$text_profit
head(posterior)

mean(posterior$diff_profit > 0)

###### Chapter 2 ######
n_trials = 10000
n_coins = 100
n_head = rbinom(n = 10000, size = n_coins, prob = .5)

##P(n_head==13\prob=.5) : fair�� ������ ���Ͽ� 100�� ���� �� �ո��� 13�� ���� Ȯ��
dbinom(x = 13, size = n_coins, prob = .5)

##P(n_head\prob=.5) : fair�� ������ ���Ͽ� 100�� ���� �� ���� �� �ִ� �ո��� ����
B = dbinom(x = 1:100, size = n_coins, prob = .5)
plot(x = 1:100, y = B, type = 'h', xlab = 'n_head', ylab = 'Probabilities', main = 'Probaility Distribution of All n_head (given that p = 0.1)')

##P(n_head==13\prob) : �پ��� ������ ���Ͽ� 100�� ���� �� �ո��� 13�� ���� Ȯ��
C = dbinom(x = 13, size = n_coins, prob = seq(0, 1, .01))
prob_head = seq(0, 1, .01)
plot(x = prob_head, y = C, type = 'h', xlab = 'Various Biased Coins', ylab = 'Probabilities', main = 'Probaility Distribution of n_head = 13')

##Probabilities of All Possible Cases
pars = expand.grid(prop_head = seq(0, 1, .01), n_head = 1:100)
pars[c(8, 441, 1382, 1765, 829, 73), ]

pars$prior = dunif(x = pars$prop_head, min = .3, max = .7)
pars$likelihood = dbinom(x = pars$n_head, size = n_coins, prob = pars$prop_head)

pars$posterior = pars$prior*pars$likelihood
sum(pars$posterior)
pars$posterior = pars$posterior / sum(pars$posterior)


## The IQ test of a bunch of zombies
# Q. What is the median value of IQ
iq <- c(55, 44, 34, 18, 51, 40, 40, 49, 48, 46)
dnorm(x = 55, mean = 40, sd = 2)

(a = dnorm(x = iq, mean = 40, sd = 2))
prod(a)

pars = expand.grid(mu = seq(0, 150, length.out = 100),
                   sigma = seq(0.1, 50, length.out = 100))
pars[c(8, 441, 829, 73), ]

pars$mu_prior = dnorm(x = pars$mu, mean = 100, sd = 10)
pars$sigma_prior = dunif(x = pars$sigma, min = .1, max = 50)
pars$prior = pars$mu_prior*pars$sigma_prior

for(i in 1:nrow(pars)){
  likelihood = dnorm(x = iq, mean = pars$mu[i], sd = pars$sigma[i])
  pars$likelihood[i] = prod(likelihood)
}

pars$posterior = pars$prior * pars$likelihood
pars$posterior = pars$posterior / sum(pars$posterior)
sum(pars$posterior)
pars[c(8, 441, 829, 73), ]

sample_index = sample(x = 1:nrow(pars), size = nrow(pars), replace = T, prob = pars$posterior)
sample_index[1:20]

pars_sample = pars[sample_index, c('mu', 'sigma')]
head(pars_sample)
hist(pars_sample$mu)
median(pars_sample$mu)   # -> median value of IQ
quantile(x = pars_sample$mu, probs = c(.025, .5, .975))

# Q. What is the posibilities the next zombie's IQ is over 60?
pred_iq = rnorm(n = nrow(pars_sample), mean = pars_sample$mu, sd = pars_sample$sigma)
hist(pred_iq)
mean(pred_iq)
sd(pred_iq)

mean(pred_iq >= 60)

# Markov Chain
library(rjags)
zombie_model = 'model{
   # Likelihood
   for(i in 1:length(Y)){
   Y[i] ~ dnorm(m, s^(-2))
   }
   # Prior for m and s
   m ~ dnorm(100, 100)
   s ~ dunif(.1, 50)
}'
zombie_jags = jags.model(file = textConnection(zombie_model),
                         data = list(Y = iq),
                         inits = list(.RNG.name = 'base::Wichmann-Hill', .RNG.seed = 1989))
zombie_sim = coda.samples(model = zombie_jags, variable.names = c('m', 's'), n.iter = 10000)

head(zombie_sim)
summary(zombie_sim)

zombie_chains = data.frame(zombie_sim[[1]], iter = 1:10000)
head(zombie_chains)
head(pars_sample)

par(mfrow = c(1, 2))
hist(zombie_chains$m)
hist(pars_sample$mu)
hist(pars$prior_m)     # ???

###### Chapter 3 : Markov Chain ######
url = 'https://assets.datacamp.com/production/repositories/2096/datasets/62737a3d23519405d7bfe3eceb85be0f97a07862/sleep_study.csv'
sleep_data = read_csv(url)
sleep_data

sleep_data$diff = sleep_data$day_3 - sleep_data$day_0
hist(sleep_data$diff)
mean(sleep_data$diff)
sd(sleep_data$diff)

library(rjags)
# DEFINE the Normal - Normal model
sleep_model = 'model{
    # Likelihood
   for(i in 1:length(Y)){
   Y[i] ~ dnorm(m, s^(-2))
   }

   # Prior for m and s
   m ~ dnorm(50, 25^(-2))
   s ~ dunif(0, 200)
}'

# COMPILE the model
sleep_jags = jags.model(file = textConnection(sleep_model),
                        data = list(Y = sleep_data$diff),
                        inits = list(.RNG.name = 'base::Wichmann-Hill', .RNG.seed = 1989))

# SIMULATE the posterior
sleep_sim = coda.samples(model = sleep_jags, variable.names = c('m', 's'), n.iter = 10000)
plot(sleep_sim)
head(sleep_sim)
summary(sleep_sim)    # check Naive standard eroor

sleep_chains = data.frame(sleep_sim[[1]], iter = 1:10000)
head(sleep_chains)

sleep_chains[1:100, ] %>%
  ggplot(aes(x = iter, y = m)) + geom_line()

ggplot(sleep_chains, aes(x = iter, y = m)) + geom_line()
ggplot(sleep_chains, aes(x = m)) + geom_density()
plot(sleep_sim, trace = F)


############################################################################
####################### Bayesian Regression ################################
'https://www.datacamp.com/courses/bayesian-modeling-with-rjags'
###### Chapter 1 ######
a = rnorm(n = 10000, mean = 0, sd = 200)
b = rnorm(n = 10000, mean = 1, sd = .5)
s = runif(n = 10000, min = 0, max = 20)

samples = data.frame(set = 1:10000, a, b, s)
head(samples)

prior_rep = bind_rows(replicate(n = 50, expr = samples[1:12, ], simplify = F))
head(prior_rep)

prior_sim = prior_rep %>%
  mutate(height = rnorm(n = 600, mean = 170, sd = 10)) %>%
  mutate(weight = rnorm(n = 600, mean = (a + b*height), sd = s))
head(prior_sim)
ggplot(prior_sim, aes(x = height, y = weight)) +
  geom_point() +
  geom_smooth(method = 'lm', se = F, size = .75) +
  facet_wrap( ~ set)

library(rjags)
library(openintro)
data("bdims")
head(bdims)

weight_model = 'model{
   # Likelihood
   for(i in 1:length(Y)){
   Y[i] ~ dnorm(m[i], s^(-2))
   m[i] = a + b * X[i]
   }
   # Prior for a, b, s
   a ~ dnorm(0, 200^(-2))
   b ~ dnorm(1, 0.5^(-2))
   s ~ dunif(0, 20)
}'
weight_jags = jags.model(file = textConnection(weight_model),
                         data = list(X = bdims$hgt, Y = bdims$wgt),
                         inits = list(.RNG.name = 'base::Wichmann-Hill', .RNG.seed = 1989))
weight_sim = coda.samples(model = weight_jags, variable.names = c('a', 'b', 's'), n.iter = 10000)
plot(weight_sim)
summary(weight_sim)

weight_chains = data.frame(weight_sim[[1]], iter = 1:10000)    # -> posterior a, b and s
head(weight_chains)
mean(weight_chains$b)    # -> slope
mean(weight_chains$a)    # -> intercept

ggplot(bdims, aes(x = hgt, y = wgt)) +
  geom_point() +
  geom_abline(intercept = mean(weight_chains$a), slope = mean(weight_chains$b), color = 'red')

ggplot(bdims, aes(x = hgt, y = wgt)) +
  geom_point() +
  geom_abline(intercept = weight_chains$a[1:20], slope = weight_chains$b[1:20], color = 'grey')

(ci_95 = quantile(x = weight_chains$b, probs = c(.025, .975)))     # -> credible interval

# Q. What's the posterior probability that b > 1.1
mean(weight_chains$b > 1.1)
table(weight_chains$b > 1.1)

ggplot(weight_chains, aes(x = b)) +
  geom_density() +
  geom_vline(xintercept = 1.1, color = 'red')


# Calculating m[i] & Y predition value
weight_chains = weight_chains %>%
  mutate(m_180 = a + b * 180)     # -> m[i] value when X = 180
head(weight_chains)

ggplot(weight_chains, aes(x = m_180)) +
  geom_density() +
  geom_vline(xintercept = c(quantile(weight_chains$m_180, probs = c(.025, .975))), color = 'red')


weight_chains = weight_chains %>%
  mutate(Y_180 = rnorm(n = 10000, mean = m_180, sd = s))     # -> prediction Y when ~ N(m_180, s)
head(weight_chains)

ci_y_180 = quantile(weight_chains$Y_180, probs = c(.025, .975))

ggplot(bdims, aes(x = hgt, y = wgt)) +
  geom_point() +
  geom_abline(intercept = mean(weight_chains$a), slope = mean(weight_chains$b), color = 'blue', size = 1.5) +
  geom_segment(x = 180, xend = 180, y = ci_y_180[1], yend = ci_y_180[2], color = 'blue', size = 1.5)
