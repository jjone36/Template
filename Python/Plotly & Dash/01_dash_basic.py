# https://plot.ly/python/
# https://dash.plot.ly/dash-core-components
# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html

import numpy as np
import pandas as pd
import plotly.graph_objs as go

# Launch the application
app = dash.Dash()

# Import the dataset
cosm = pd.read_csv('data.csv')

option_1 = ['Moisturizer', 'Cleanser', 'Treatment', 'Face Mask', 'Eye cream', 'Sun protect']
option_2 = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']
df = cosm[(cosm['Category'] == option_1[0]) & (cosm['Skin'] == option_2[0])]

# Create a plotly figure
trace_1 = go.Scatter(x = df.X, y = df.Y, mode = 'markers',
                   marker = dict(size = 10,
                                color = 'rgba(51, 204, 153, .7)',
                                symbol = 'pentagon',
                                line = {'width':1}))

layout = go.Layout(title = 'Item Plot',
                   xaxis = dict(title = 'X-values',
                                range = [-100, 100],
                                titlefont=dict(family='Arial, sans-serif',
                                                size=18,
                                                color='lightgrey')),
                   yaxis = dict(title = 'Y-values',
                                range = [-100, 100],
                                titlefont=dict(family='Arial, sans-serif',
                                                size=18,
                                                color='lightgrey')),
                   hovermode = 'closest')

fig = go.Figure(data = [trace_1], layout = layout)

# Create a Dash layout
app.layout = html.Div([
                dcc.Graph(id = 'plot_id', figure = fig)
                      ])

# Add the server clause
if __name__ == '__main__':
    app.run_server()
