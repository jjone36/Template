###### Association Rule Learning
############# Apriori
dataset = pd.read_csv('Basket.csv', header = None)

# transforming each transaction into lists
transactions = []
for i in range(0, len(dataset)+1):
    transactions.append([str(dataset.values[i, :])])

from apyroi import apriori
rules = apriori(transactions, min_support, min_confidence, min_lift, min_length = 5)
results = list(rules)


###########################################################
###### Association Rule Learning in R
## Sparse Matrix
library(arules)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep=',', rm.duplicates = T)

summary(dataset)
size(head(dataset))
size(dataset) %>% quantile(prob = seq(0.1, 0.5, 0.1))

itemFrequencyPlot(dataset, topN = 10)

## 1. Apriori
rules = apriori(data = dataset, parameter = list(support = 0.003, confidence = 0.2))

rules_conf = sort(rules, by = 'confidence', decreasing = T)
rules_lift = sort(rules, by='lift', decreasing = T)
inspect(rules_conf[1:10])

rules2 = apriori(dataset, parameter = list(support = .001, confidence = 0.2, minlen = 2),
                 appearance = list(default = 'rhs', lhs = 'cereals'), control = list(verbose = F))

library(arulesViz)
sub_rules = subset(rules, confidence > 0.9)
sub_rules
plot(sub_rules, method = 'matrix')
sub_rules2 = subset(rules, lift > 6)
plot(sub_rules2, method = 'graph')


## 2. Eclat
rules = eclat(data = dataset, parameter = list(support = 0.003, minlen = 2))
rules = eclat(dataset, parameter = list(support = 0.06, maxlen = 5))

inspect(head(rules))
A = sort(rules, by='support')
inspect(A[1:10])

itemsets = sort(rules)
as(items(itemsets), 'list')
