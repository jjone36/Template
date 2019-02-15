http://www.colors.commutercreative.com/grid/
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

# subplots()
cols = ['weight', 'mpg']
df[cols].plot(kind = 'box', subplots = True)

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharey = True)
ax1.hist(df['weight'], normed = True, bins = 30, range = (0, .3))
ax.axvline(x= np.median(df['weight']), color='m', label='Median', linestyle='--', linewidth=2)

ax2.scatter(df['weight'], df['mpg'])
ax2.set(xlabel = 'weight of auto', xlim = (10, 100), title = 'Scatter Plot')

plt.tight_layout()
plt.subplots_adjust(wspace = .5, hspace = .5)
plt.show()

## position : 1. 9. 2 / 6. 10. 7 / 3. 8. 4 / 0. 5
plt.subplots(2, 2, 2); plt.plot(year, computer_science, color = 'red', legend = 'Computer Science')
plt.subplots(2, 2, 1); plt.plot(year, physical_sciences, color = 'blue', legend = 'Physical Sciences')
plt.legend(loc = 'lower center')

# FacetGrid
f = sns.FacetGrid(df, row = 'smoker', col = 'diagnose', hue = 'gender')
f.map(plt.scatter, 'weight', 'age')
plt.show()

f = sns.FacetGrid(iris, row = 'species', row_order = ['Setosa', 'versicolor', 'Virginica'])
f.map(plt.scatter, 'Sepal Length')
plt.show()

sns.factorplot(kind = 'box', data = iris, x = 'Sepal Length', row = 'Degree_Type')

# PairGrid
p = sns.PairGrid(iris, vars = ['Sepal Length', 'Petal Length'])
p.map(plt.scatter)
p.map_diag()
p.map_offdiag()

# PairPlot: ggpairs()
sns.pairplot(df, vars = [], palette = 'husl', plot_kws = {'alpha': .5})
sns.pairplot(df, x_vars, y_vars, hue = 'origin', kind = 'reg', diag_kind = 'kde')   # diag_kws

# JointGrid: joint distributions
sns.jointplot(x = 'hp', y = 'mpg', data = auto, kind = 'scatter')     # 'scatter', 'reg', 'resid', 'kde', 'hex'

j = sns.JointGrid(x = 'hp', y = 'mpg', data = auto, xlim = (.1, 1))
j.plot(sns.regplot, sns.distplot)

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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# geom_point()
df.plot(x, y, kind = 'scatter')
df.scatter(x, y)

plt.scatter(df['x'], df['y'])
sns.regplot(x, y, data, scatter = None, label = 'Order 1')
sns.regplot(x, y, data, fit_reg = False, x_bins = 10)
sns.lmplot(x, y, data, fit_reg = False)

# geom_baxplot()
df.boxplot(column, by)
sns.boxplot(x, y, data)

# geom_hist()
plt.hist(df['x'])
df['x'].plot.hist()
sns.distplot(df['x'], kde = False, bins = 30)

# geom_bar()
df.plot(x, y, kind = 'barh')
sns.countplot(data = df, y = 'weight')
sns.barplot(x, y, data)

plt.show()
plt.clf()

####################### Seaborn special #######################
# set default Seaborn style
sns.set()
sns.set_style('whitegird')   # dark, white
sns.despine()    # removing the outside lines
sns.set_palette('colorblind')   # bright

sns.palplot(sns.color_palette("Purples", 8))   # husl

# geom_point()
plt.scatter(auto['weight'], auto['mpg'], color = 'red', label = 'data')
sns.regplot(x = 'weight', y = 'mpg', data = auto, color = 'blue', scatter = None, label = 'Order 1')
sns.lmplot(x = 'weight', y = 'mpg', data = auto, row = 'cyl')

# geom_smooth(method = 'lm', se = F)
sns.lmplot(x = 'weight', y = 'mpg', data = auto, col = 'cyl')
sns.lmplot(x = 'weight', y = 'mpg', data = auto, hue = 'origin', palette = 'Set1')
sns.lmplot(x = 'weight', y = 'mpg', data = auto, row = 'origin')      # row-wise grouping

# sns.regplot(order = 2) -> polynomial regreesion
sns.regplot(x = 'weight', y = 'mpg', data = auto, color = 'green', order = 2, label = 'Order 2')
plt.legend(loc = 'upper right')
plt.show()

sns.regplot(x = 'weight', y = 'mpg', data = auto, order = 3, x_estimator = np.mean)

# residaul plot
sns.residplot(x = 'weight', y = 'mpg', data = auto, color = 'indianred')
# strip plot
sns.stripplot(x = 'cyl', y = 'hp', data = auto, size = 3, jitter = True)
# swarm plot
sns.swarmplot(x = 'hp', y = 'cyl', data = auto, hue = 'origin', orient = 'h')

# violin plot / + geom_boxplot(varwidth = T)
sns.violinplot(x = 'cyl', y = 'hp', data = auto, color = 'lightgray')
sns.violinplot(x = 'cyl', y = 'hp', data = auto, inner = None,  palette = 'husl')

# covariance matrix plot (heatmap)
sns.heatmap(df.corr())

df_crosstab = pd.corsstab(df['weight'], df['height'], values = df['hp'], aggfunc = 'mean'))
sns.heatmap(df_crosstab, annot = True, fmt = 'd', cmap = 'YlGnBu', cbar = False,
            linewidths = .3, center = df_crosstab.loc = [6, 9])

#####################################################################
import matplotlib.patches as mpatches

x = np.array([100, 105, 110, 124, 136, 155, 166, 177, 182, 196, 208, 230, 260, 294, 312])
y = np.array([54, 56, 60, 60, 60, 72, 62, 64, 66, 80, 82, 72, 67, 84, 74])
z = (x*y) / 60

for index, val in enumerate(z):
    if index < 10:
        color = 'g'
    else:
        color = 'r'
    plt.scatter(x[index], y[index], s=z[index]*5, alpha=0.5, c=color)

red_patch = mpatches.Patch(color='red', label='Male')
green_patch = mpatches.Patch(color='green', label='Female')
plt.legend(handles=[green_patch, red_patch])

plt.title("French fries eaten vs height and weight")
plt.xlabel("Weight (pounds)")
plt.ylabel("Height (inches)")
plt.show()

#####################################################################
http://bokeh.github.io
####################### Bokeh #######################
from bokeh.plotting import figure
from bokeh.io import output_file, show

p = figure(x_axis_label = 'weight', y_axis_label = 'mpg',
           plot_height = 400, plot_width = 700, x_range, y_range)
p.circle(auto['weight'], auto['mpg'], color = 'blue', size = 10, alpha = .8)
p.circle(auto['height'], auto['mpg'], color = 'red', size = 10, alpha = .8)
output_file('auto.html')
show(p)

from bokeh.plotting import ColumnDataSource
source = ColumnDataSource(df)

# selection
p = figure(x_axis_type = 'datetime', tools = 'box_select, lasso_select')
p.circle(x = 'Year', y = 'Time', source = source, selection_color = 'blue', nonselection_alpha = .1)
output_file('auto_1.html')
show(p)

# HavorTool
from bokeh.models import HoverTool
hover = HoverTool(tooltips = None, mode = 'hline' # 'vline')

p = figure()
p.circle(x, y, fill_color = 'grey', line_color = None,
         hover_fill_color = 'firebrick', hover_line_color = 'grey', hover_alpha = .5)
p.add_tools(hover)

hover2 = HoverTool(tooltips = [('species name', '@species'),
                               ('petal length', '@petal_length')])
p = figure(tools = [hover, 'pan', 'wheel_zoom'])


# Colormapping
from bokeh.models import CategoricalColorMapper
mapper = CategoricalColorMapper(factors = ['setosa', 'virginica', 'versicolor'],
                                pallete = ['red', 'green', 'blue'])

p = figure(x_axis_label = 'petal_length', y_axis_label = 'sepal_length')
p.circle(x = 'petal_length', y = 'sepal_length', source = source,
         color = {'field': 'species', 'transform': mapper, legend = 'species'}, legend = 'species')

plot.legend.location = 'top_left'
p.legend.background_fill_color = 'lightgray'


# layout
from bokeh.layouts import row, column
row_2 = column([p1, p2])
layout = row([p3, row_2], sizing_mode = 'scale_width')
output_file('layout.html')
show(layout)

p1.x_range = p2.x_range
p1.y_range = p2.y_range


# gridplot
from bokeh.layouts import gridplot
row_1 = [p1, p2]
row_2 = [p3, p4]
layout = gridplot([row_1, row_2])
show(layout)


# tabs
from bokeh.models.widgets import Panel
tab_1 = Panel(child = p1, title = 'Setosa')
tab_2 = Panel(child = p2, title = 'Virginica')

from bokeh.models.widgets import Tabs
layout = Tabs(tabs = [tab1, tab2])
show(layout)


############# bokeh app
from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider

# slider
slider = Slider(title ='my slider', start = 0, end = 10, step = 0.1, value = 2)
layout = column(p, widgetbox(slider))

curdoc().add_root(layout)

bokeh serve current.py

# callback
from bokeh.models import ColumnDataSource, Select

source = ColumnDataSource(data = {'x': iris['Sepal_Length'], 'y': iris['Sepal_Width']})
p = figure()
p.circle('x', 'y', source = source)

def update_plot(attr, old, new):
    if new == 'Petal_Length':
        source.data = {'x': 'Sepal_Length', 'y': 'Petal_Length'}
    else:
        source.data = {'x': 'Sepal_Length', 'y': 'Sepal_Length'}

select = Select(title = 'part', options = ['sepal length', 'petal length'], value = 'Sepal_Length')
select.on_change('value', update)

layout = row(select, plot)
curdoc().add_root(layout)


# button
from bokeh.models import Button
button = Button(label = 'Hit me!')
def update():

button.on_click(update)

from bokeh.models import Toggle, CheckboxGroup, RadioGroup
toggle = Toggle(button_type = 'success', label = 'Toggle button')
checkbox = CheckboxGroup(labels = ['Option 1', 'Option 2', 'Option 3'])
radio = RadioGroup(labels = ['Option 1', 'Option 2', 'Option 3'])
curdoc().add_root(widgetbox(toggle, checkbox, radio))


# paragraph
from bokeh.models.widgets import Paragraph

p = Paragraph(text = '', width = 200, height = 400)
show(widgetbox(p))

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
