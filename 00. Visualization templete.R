head(iris)
iris.tidy = gather(iris, part, value, -Species) %>%
  separate(col = part, into = c('Part', 'Measure'), '\\.')
head(iris.tidy)
str(iris.tidy)

iris.wide = iris %>%
  mutate(Flower = 1:nrow(iris)) %>%
  gather(part, value, -Species, -Flower) %>%
  separate(col = part, into = c('Part', 'Measure'), sep = '\\.') %>%
  spread(Measure, value)
head(iris.wide)

myBlue = '#377EB8'
myPink = '#FEE0D2'
myRed = '#99000D'
#################################### Scatter Plot #### 
## Overplotting! jitter. stat_sum() ####
ggplot(diamonds, aes(x = clarity, y = carat, color = price)) + geom_point(alpha = .5)
ggplot(diamonds, aes(x = clarity, y = carat, color = price)) + geom_point(position = 'jitter', alpha = .5)

ggplot(mtcars, aes(x = cyl, y = wt)) + geom_point()
ggplot(mtcars, aes(x = cyl, y = wt)) + geom_jitter()
ggplot(mtcars, aes(x = cyl, y = wt)) + geom_jitter(width = .2, shape = 2)
ggplot(mtcars, aes(x = cyl, y = wt)) + geom_point(position = position_jitter(width = .2))

library(carData)
head(Vocab)
p_vocab = ggplot(Vocab, aes(x = education, y = vocabulary)) 

p_vocab + geom_point(alpha = .2)
p_vocab + geom_jitter(alpha = .2)
ggplot(Vocab, aes(x = education, y = vocabulary, col = factor(year))) + geom_jitter(alpha = .2)
p_vocab + geom_jitter(alpha = .2) + stat_smooth(method = 'lm', se = F)

p_vocab + stat_sum()      #calculate the total number of overlapping observations
p_vocab + stat_sum() + scale_size(range = c(1, 10))
p_vocab + stat_sum() + scale_size(range = c(1, 15))


## geom_smooth(), stat_smooth() ####
dia = ggplot(diamonds, aes(x = carat, y = price))
dia + geom_point()
dia + geom_point(alpha = .2)

ggplot(diamonds, aes(x = carat, y = price, color = clarity)) + geom_point() + geom_smooth(se = F)
dia + geom_point(aes(color = clarity)) + geom_smooth(se = F)
dia + geom_point(alpha = .2) + geom_smooth(aes(color = clarity), se = F)

p_mtc = ggplot(mtcars, aes(x = wt, y = mpg)) 
p_mtc + geom_point()
p_mtc + geom_smooth()      #stat_smooth()
p_mtc + geom_smooth(method = 'lm', se = F)

ggplot(mtcars, aes(x = wt, y = mpg, col = cyl)) + geom_point()       #size, shape
ggplot(mtcars, aes(x = wt, y = mpg, col = cyl)) + geom_smooth()
ggplot(mtcars, aes(x = wt, y = mpg, col = factor(cyl))) + geom_point()
ggplot(mtcars, aes(x = wt, y = mpg, col = factor(cyl))) + geom_smooth()

p_mtc + geom_point(aes(color = factor(cyl))) + geom_smooth(se = F)
ggplot(mtcars, aes(x = wt, y = mpg, col = factor(cyl))) + geom_point() + stat_smooth(aes(group = 1), se = F)
p_mtc + geom_point() + geom_smooth(aes(color = factor(cyl)), se = F)

ggplot(mtcars, aes(x = wt, y = mpg, col = factor(cyl))) + geom_point() + stat_smooth(method = 'lm', se = F) 
ggplot(mtcars, aes(x = wt, y = mpg, col = cyl)) +     #compare with 'col = factor(cyl)'
  geom_point() +  
  geom_smooth(method = 'lm', se = F) + 
  geom_smooth(aes(group = cyl), method = 'lm', se = F, lty = 2)


ggplot(mtcars, aes(x = wt, y = mpg)) + geom_point() + geom_smooth(se = F, span = .7)
ggplot(mtcars, aes(x = wt, y = mpg)) + geom_point() + geom_smooth(se = F)
ggplot(mtcars, aes(x = wt, y = mpg, col = factor(cyl))) +
  geom_point() +
  stat_smooth(method = 'lm', se = F) +
  stat_smooth(aes(group = 1, col = 'All'), se = F, span = .7)


## stat_summary(), stat_quantile() #### 
str(mtcars)
mtcars$cyl = factor(mtcars$cyl)
mtcars$am = factor(mtcars$am)

posn_d = position_dodge(width = .2)
posn_j = position_jitter(width = .1)
posn_jd = position_jitterdodge(jitter.width = .1, dodge.width = .2)

p2_mtc = ggplot(mtcars, aes(x = cyl, y = wt, col = am, fill = am, group = am))
p2_mtc + geom_point(alpha = .5)
p2_mtc + geom_point(alpha = .5, position = posn_d)
p2_mtc + geom_point(alpha = .5, position = posn_j)
p2_mtc + geom_point(alpha = .5, position = posn_jd)

# stat_summary(fun.data = , fun.args = ) 
p2_mtc + 
  geom_point(alpha = .5) +
  stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1))
p2_mtc + 
  geom_point(alpha = .5) +
  stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1), position = posn_d)
p2_mtc + 
  geom_point(alpha = .5) +
  stat_summary(fun.data = mean_cl_normal, position = posn_d)

p2_mtc +
  geom_point() +
  stat_summary(fun.data = mean_cl_normal, geom = 'crossbar', width = .2, alpha = .2)

# stat_quantile
ggplot(Vocab, aes(x = education, y = vocabulary, col = year, group = factor(year))) + 
  stat_quantile(alpha = .6) 
ggplot(Vocab, aes(x = education, y = vocabulary, col = year, group = factor(year))) + 
  stat_quantile(quantile = .1, alpha = .6) 

#################################### Box Plot ####
str(diamonds)
p = ggplot(diamonds, aes(x = carat, y = price)) 

p + geom_boxplot() # -> Continous variable 
p + geom_boxplot(aes(group = cut_number(carat, n = 10)))
p + geom_boxplot(aes(group = cut_interval(carat, n = 10)))
p + geom_boxplot(aes(group = cut_width(carat, width = .25)))

ggplot(diamonds, aes(x = cut, y = price)) + geom_boxplot()
ggplot(diamonds, aes(x = cut, y = price)) + geom_boxplot(varwidth = T)

#################################### Bar Plots #### 
## position 
p_mtc = ggplot(mtcars, aes(x = cyl, fill = am))
p_mtc + geom_bar()
p_mtc + geom_bar(position = 'stack')
p_mtc + geom_bar(position = 'fill')
p_mtc + geom_bar(position = 'dodge')

p_mtc + geom_bar(position = position_dodge(width = .2))
posn_d = position_dodge(width = .5)
p_mtc + geom_bar(position = posn_d, alpha = .6)


p_mtc + geom_bar(position = 'dodge') +
  scale_x_discrete(aes(labels = 'Cylinders')) +
  scale_y_continuous(aes(labels = 'Number')) +       #labs(x = 'Cylinders', y = 'Number')
  scale_fill_manual('Transmission', 
                    values = c('#E41A1C', '#377EB8'), 
                    labels = c('Manual', 'Automatic'))

#################################### histogram #### 
ggplot(mtcars, aes(x = mpg)) + geom_histogram()

ggplot(mtcars, aes(x = mpg)) + geom_histogram(binwidth = .5)
ggplot(mtcars, aes(x = mpg, y = ..density..)) + geom_histogram(binwidth = .5, fill = 'steelblue1')

ggplot(mtcars, aes(x = mpg, fill = factor(am))) + geom_histogram()
ggplot(mtcars, aes(x = mpg, fill = factor(am))) + geom_histogram(position = 'fill')
ggplot(mtcars, aes(x = mpg, fill = factor(am))) + geom_histogram(position = 'dodge')
ggplot(mtcars, aes(x = mpg, fill = factor(am))) + geom_histogram(position = 'identity')
ggplot(mtcars, aes(x = mpg, fill = factor(am))) + geom_histogram(position = 'identity', alpha = .4)

ggplot(mtcars, aes(x = mpg, color = factor(am))) + geom_freqpoly()

#################################### density #### 
ggplot(diamonds, aes(x = carat, fill = color)) + 
  geom_density(col = NA, alpha = .3) + 
  scale_x_continuous(limits = c(0, 3))

ggplot(diamonds, aes(x = carat, fill = color)) + 
  geom_density(col = NA, alpha = .3, trim = T) +   # trim
  scale_x_continuous(limits = c(0, 3))

diamonds2 = diamonds %>%
  group_by(color) %>%
  mutate(prop = n() / nrow(diamonds))

ggplot(diamonds2, aes(x = carat, fill = color)) + 
  geom_density(aes(weight = prop), col = NA, alpha = .3, trim = T) +   # weight
  scale_x_continuous(limits = c(0, 3))

ggplot(diamonds, aes(x = color, y = carat, fill = color)) + geom_boxplot()
ggplot(diamonds, aes(x = color, y = carat, fill = color)) + geom_violin()  
ggplot(diamonds2, aes(x = color, y = carat, fill = color)) + geom_violin(aes(weight = prop))  

## geom_density_2d
str(faithful)

ggplot(faithful, aes(x = waiting, y = eruptions)) + geom_density_2d()
ggplot(faithful, aes(x = waiting, y = eruptions)) + stat_density_2d(aes(col = ..level..), h = c(5, .5))

#################################### Line Plots #### 
head(economics)
recess

ggplot(economics, aes(x = date, y = unemploy)) + geom_line()
ggplot(economics, aes(x = date, y = unemploy/pop)) + geom_line()

ggplot(economics, aes(x = date, y = unemploy/pop)) + 
  geom_rect(data = recess, aes(xmin = begin, xmax = end, ymin = -Inf, ymax = +Inf), inherit.aes = F,
            fill = 'red', alpha = .2) +
  geom_line()

head(fish.tidy)
ggplot(fish.tidy, aes(x = Year, y = Capture, color = Species)) + geom_line()

ggplot(fish.tidy, aes(x = Year, y = Capture, color = Species)) + geom_area()
ggplot(fish.tidy, aes(x = Year, y = Capture, fill = Species)) + geom_area()
ggplot(fish.tidy, aes(x = Year, y = Capture, fill = Species)) + geom_area(position = 'fill')


#################################### coordinate #### 
scale_x_continuous(labels = scales::comma) 

## scale_cartesian() ####
(p_mtc = ggplot(mtcars, aes(x = wt, y = hp, col = am)) + geom_point() + geom_smooth())

p_mtc + scale_x_continuous(limits = c(3, 6))
p_mtc + scale_x_continuous(limits = c(3, 6), expand = c(0, 0))
p_mtc + coord_cartesian(xlim = c(3, 6))


## coord_trans(), scale_x_log10() coord_equal() ####
data("diamonds")
ggplot(diamonds, aes(x = carat, y = price, color = color)) +
  geom_point(alpha = .5, size = .5, shape = 16) +
  coord_trans(x = 'log10') +
  theme_classic()

ggplot(diamonds, aes(x = carat, y = price, color = color)) +
  geom_point(alpha = .5, size = .5, shape = 16) +
  scale_x_log10(limits = c(.1, 10)) +
  theme_classic()

ggplot(diamonds, aes(x = carat, y = price, color = color)) +
  geom_point(alpha = .5, size = .5, shape = 16) +
  scale_x_log10(limits = c(.1, 10)) +
  scale_y_log10(limits = c(100, 100000)) +
  theme_classic()

ggplot(diamonds, aes(x = carat, y = price, color = color)) +
  geom_point(alpha = .5, size = .5, shape = 16) +
  scale_x_log10(limits = c(.1, 10), expression(log[10](Carat))) +
  scale_y_log10(limits = c(100, 100000), expression(log[10](Price))) +
  coord_equal() +
  theme_classic()

(p_iris = ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, col = Species)) +
    geom_jitter() + geom_smooth(method = 'lm', se = F))
p_iris + coord_equal()

## coord_polar() ####
ggplot(mtcars, aes(x = 1, fill = cyl)) + geom_bar() + coord_polar()
ggplot(mtcars, aes(x = 1, fill = cyl)) + geom_bar() + coord_polar(theta = 'y')
ggplot(mtcars, aes(x = 1, fill = cyl)) + geom_bar(binwidth = .1) + coord_polar(theta = 'y') 

ggplot(mtcars, aes(x = cyl, fill = factor(am))) + geom_bar(position = 'fill')
ggplot(mtcars, aes(x = 1, fill = factor(am))) + 
  geom_bar(position = 'fill') +
  facet_grid(. ~ cyl) +
  coord_polar()
ggplot(mtcars, aes(x = factor(1), fill = factor(am))) + 
  geom_bar(position = 'fill') +
  facet_grid(. ~ cyl) +
  coord_polar(theta = 'y') +
  theme_void()

#################################### colour #### 
## scale_fill_brewer() ####
ggplot(mtcars, aes(x = cyl, fill = factor(am))) +
  geom_bar(position = 'fill') + 
  scale_fill_brewer()

ggplot(mtcars, aes(x = cyl, fill = factor(am))) +
  geom_bar() + 
  scale_fill_brewer(palette = 'Set2')

## scale_fill_manual() ####
myColor = brewer.pal(8, 'PuBuGn')    #up to 9 colors
color_range = colorRampPalette(myColor)    #extend color range
ggplot(diamonds, aes(x = carat, fill = clarity)) +
  geom_bar(position = 'fill') +
  scale_fill_manual(values = color_range(10))

myColor = c(brewer.pal(3, 'Accent'), 'black')
ggplot(mtcars, aes(x = wt, y = mpg, col = factor(cyl))) +
  geom_point() +
  stat_smooth(method = 'lm', se = F, span = .7) +
  stat_smooth(aes(group = 1, col = 'All'), se = F, span = .7) +
  scale_color_manual('Cylinders', values = myColor)

mtcars$cyl_am = paste(mtcars$cyl, mtcars$am, sep = '_')
myColors = rbind(brewer.pal(9, 'Blues')[c(3, 6, 8)],
                 brewer.pal(9, 'Reds')[c(3, 6, 8)])
ggplot(mtcars, aes(x = wt, y = mpg, col = cyl_am)) + 
  geom_point() +
  scale_color_manual(values = myColors)
 
## scale_color_gradientn() ####
ggplot(Vocab, aes(x = education, y = vocabulary, col = factor(year))) + 
  geom_smooth() + scale_color_brewer()
ggplot(Vocab, aes(x = education, y = vocabulary, col = year, group = factor(year))) + 
  geom_smooth(method = 'lm', se = F, alpha = .6, size = 2) +
  scale_color_gradientn(colors = brewer.pal(9, 'YlOrRd'))

#################################### themes #### 
(p = ggplot(mtcars, aes(x = wt, y = mpg, col = factor(cyl))) + 
  geom_point() +
  geom_smooth(method = 'lm', se = F) +
  scale_x_continuous(expand = c(0.2, 0.2)) +
  facet_wrap(~ cyl))

p + theme(plot.background = element_rect(fill = myPink))
p + theme(plot.background = element_rect(fill = myPink, color = 'black', size = 3))

no_panels = theme(rect = element_blank())
p + no_panels
p + no_panels + theme(plot.background = element_rect(fill = myPink, color = 'black', size = 3))

p + theme(legend.position = c(.85, .85))
p + theme(legend.direction = 'horizontal')
p + theme(legend.position = 'bottom')
p + theme(legend.position = 'none')


#################################### stat_function() ####
test_data = ch1_test_data
d = density(test_data$norm)
d$bw
mode = d$x[which.max(d$y)]

ggplot(test_data, aes(x = norm)) +
  geom_density() +
  geom_rug() + 
  geom_vline(xintercept = mode, col = 'red')

ggplot(test_data, aes(x = norm)) +
  geom_histogram(aes(y = ..density..), fill = 'grey80') +
  geom_density(col = 'red') +
  stat_function(fun = dnorm, args = list(mean = mean(test_data$norm), sd = sd(test_data$norm)), col = 'blue') +
  theme_classic()

#################################### ggplot2 part3 ####
pairs(iris[1:4])

library(GGally)
ggpairs(iris[1:4])


