####################### Visualization #######################
import matplotlib.pyplot as plt

iris.plot(kind = 'scatter', x = 'Sepal_Length', y = 'Sepal_Width')
iris['Speices'].plot(subplots = True, title = 'Iris Species')

plt.xlabel('Iris Sepal Length')
plt.ylabel('Iris Sepal Width')
plt.title('Iris Plot')

plt.xticks(rotation = 60)
plt.twinx()
plt.axis('off')
plt.grid('off')

plt.xlim(20, 40)
plt.ylim(100, 400)
plt.axis((20, 40, 100, 400))
plt.axes([x_lo, y_lo, width, height], 'equal')

plt.show()
plt.clf()

# facet_grid()
cols = ['weight', 'mpg']
df[cols].plot(kind = 'box', subplots = True)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.hist(df['weight'], normed = True, bins = 30, range = (0, .3))
ax2.scatter(df['weight'], df['mpg'])
plt.tight_layout()
plt.show()

'https://campus.datacamp.com/courses/python-for-r-users/plotting-4?ex=7'
facet = sns.FacetGrid(df, col, row)
facet.map(plt.scatter, col_x, col_y)
plt.show()

# position : 1. 9. 2 / 6. 10. 7 / 3. 8. 4 / 0. 5
plt.subplots(2, 2, 2)
plt.plot(year, computer_science, color = 'red', legend = 'Computer Science')
plt.subplots(2, 2, 1)
plt.plot(year, physical_sciences, color = 'blue', legend = 'Physical Sciences')
plt.legend(loc = 'lower center')

# plt.annotate
cs_max = computer_science.max()
year_max = year[computer_science.argmax()]

plt.annotate('Maximum', xy = (year_max, cs_max), xytext = (year_max+5, cs_max+5),
             arrowprops = dict(fontcolor = 'k'))

# plot style
print(plt.style.available)
plt.style.use('ggplot')

plt.savefig('plot1.png')

#####################################################################
####################### boxplot #######################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# geom_point()
df.plot(x, y, kind = 'scatter')
df.scatter(x, y)

plt.scatter(df['x'], df['y'])

sns.lmplot(x, y, data, fit_reg = False)
sns.regplot(x, y, data, scatter = None, label = 'Order 1')

# geom_baxplot()
df.boxplot(column, by)
sns.boxplot(x, y, data)

# geom_hist()
plt.hist(df['x'])
sns.distplot(x)

# geom_bar()
sns.countplot(x, data)
sns.barplot(x, y, data)

plt.show()

####################### Seaborn #######################
# set default Seaborn style
sns.set()

# geom_point()
sns.lmplot(x = 'weight', y = 'mpg', data = auto, col = 'cyl')
plt.scatter(auto['weight'], auto['mpg'], color = 'red', label = 'data')
sns.regplot(x = 'weight', y = 'mpg', data = auto, color = 'blue', scatter = None, label = 'Order 1')
# geom_smooth(method = 'lm', se = F)
sns.lmplot(x = 'weight', y = 'mpg', data = auto, col = 'cyl')
sns.lmplot(x = 'weight', y = 'mpg', data = auto, hue = 'origin', palette = 'Set1')
sns.lmplot(x = 'weight', y = 'mpg', data = auto, row = 'origin')      # row-wise grouping
# sns.regplot(order = 2) -> polynomial regreesion
sns.regplot(x = 'weight', y = 'mpg', data = auto, color = 'green', scatter = None, order = 2, label = 'Order 2')
plt.legend(loc = 'upper right')
plt.show()

# residaul plot
sns.residplot(x = 'weight', y = 'mpg', data = auto, color = 'indianred')
# strip plot
sns.stripplot(x = 'cyl', y = 'hp', data = auto, size = 3, jitter = True)
# swarm plot
sns.swarmplot(x = 'hp', y = 'cyl', data = auto, hue = 'origin', orient = 'h')
sns.boxplot()

# violin plot / + geom_boxplot(varwidth = T)
sns.violinplot(x = 'cyl', y = 'hp', data = auto)
sns.violinplot(x = 'cyl', y = 'hp', data = auto, inner = None, color = 'lightgray')

# joint distributions
sns.jointplot(x = 'hp', y = 'mpg', data = auto, kind = 'scatter')     # 'scatter', 'reg', 'resid', 'kde', 'hex'
# ggpairs()
sns.pairplot(auto)
sns.pairplot(auto, hue = 'origin', kind = 'reg')
# covariance matrix plot (heatmap)
sns.heatmap(cov_matrix)

####################### Empirical cumulative distributions #######################
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    return x, y

plt.plot(x, y, data, marker = '.', linestyle = 'none')

a = np.array([2.5, 25, 50, 75, 97.5])
iqr = np.percentile(height, q = a)
plt.plot(iqr, a/100, marker = 'D', color = 'red')
plt.show()


####################### 2D Array Plot #######################
import numpy as np

a = np.linspace(-2, 2, 41)
b = np.linspace(-1, 1, 21)
X,Y = np.meshgrid(a, b)
Z = np.sin(np.sqrt(X**2 + Y**2))

plt.pcolor(Z, cmap = 'Blues')     # cmap = 'autumn', 'gray'
plt.colorbar()

plt.pcolor(X, Y, Z)
plt.show()

# contour plot
plt.contour(X, Y, Z, 30)
plt.contourf(X, Y, Z, 30)
plt.show()

# hist2D
plt.hist2d(hp, mpg, bins = (20, 20), range = ((40, 235), (8, 48)))
plt.hexbin(hp, mpg, gridsize = (15, 12), extent = (40, 235, 8, 48))

plt.colorbar()
plt.show()

####################### image #######################
# Load the image into an array: img
img = plt.imread('480px-Astronaut-EVA.jpg')
# Print the shape of the image
print(img)
print(img.shape)
# Display the image
plt.imshow(img)
# Hide the axes
plt.axis('off')
plt.show()

plt.imshow(img, extent = (-1,1,-1,1), aspect = 2)
