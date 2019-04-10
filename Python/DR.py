#################################################
###### Dimensionality Reduction in python
# Pricipal Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
pca.components_
pca.n_components_
pca.mean_

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

# Kernel PCA (Non-liear data case)
from sklearn.decomposition import kernelPCA
kpca = kernelPCA(n_components = 2, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

# trucated SVD
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD()
Z = svd.fit_transform(X)

# Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)    # LDA: supervised model
X_test = lda.transform(X_test)

# t-SNE => 2D
from sklearn.manifold import TSNE
model = TSNE(learning_rate = 200)
tsne_features = model.fit_transform(samples)
x = tsne_features[:, 0]
y = tsne_features[:, 1]
plt.scatter(x, y, c = label)
plt.show()




# Non-Negative Factorization
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()
# Create an NMF model: nmf
nmf = NMF(n_components = 20)
# Create a Normalizer: normalizer
normalizer = Normalizer()
# Create a pipeline: pipeline
pipeline = make_pipeline(scaler, nmf, normalizer)
# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(items)
# Create a DataFrame: df
df = pd.DataFrame(norm_features, index = item_names)
# Select row of 'Bruce Springsteen': artist
artist = df.loc['Water Bomb']
# Compute cosine similarities of the item between the itmes
similarities = df.dot(itmes)
# Display those with highest cosine similarity
print(similarities.nlargest())

#################################################
###### Dimensionality Reduction in R ######
############# PCA
library(e1071)
pca = prcomp(x = training[, -ncol(training)], center = T, scale. = T)

library(caret)
pca2 = preProcess(x = training[, -ncol(training)], method = 'pca', pcaComp = 2)
summary(pca)

training_pca = predict(pca, training)
testing_pca = predict(pca, testing)

plot(pca, type = 'l')
pr.var = pca$sdev^2
pve = pr.var / sum(pr.var)

plot(pve, ylim = c(0, 1), type = 'b')   #scree plot
plot(cumsum(pve), ylim = c(0, 1), type = 'b')

biplot(pca)
plot(pca$x[, c(1,2)], col = diagonosis + 1)

df_pca = pca$x %>% as.data.frame()
ggplot(df_pca, aes(x = PC1, y = PC2, col = dataset$diagnosis)) +
  geom_point() +
  theme_classic()

library(ggord)
ggord(pca, grp_in = dataset$diagnosis)

############# Kernel PCA
library(kernlab)
k_pca = kpca(~., data = training[, 3:4], kernel = 'rbfdot', features = 2)

training_k_pca = predict(k_pca, training) %>% as.data.frame()
training_k_pca$Purchased = training$Purchased
testing_k_pca = predict(k_pca, testing) %>% as.data.frame()
testing_k_pca$Purchased = testing$Purchased

############# FactorMinorR



############# LDA
library(MASS)
lda = lda(Customer_Segment ~., data = training)

training_lda = predict(lda, training) %>% as.data.frame()
testing_lda = predict(lda, testing) %>% as.data.frame()
