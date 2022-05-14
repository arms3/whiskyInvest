import re
import plotly.graph_objs as go

# Formatting helpers
def format_whisky_type(whisky_type):
    m = {'BBF': 'First fill bourbon', 'BBR': 'Refill bourbon',
         'HHR': 'Refill hogshead', 'SBR': 'Refill sherry butt',
         'BRF':'Refill butt', 'SHR': 'Refill sherry hogshead'}
    processed = whisky_type.split('_')
    processed[0] = ' '.join([x[0].upper() + x[1:] for x in processed[0].split('-')])
    processed[-1] = m[processed[-1]]
    return ' '.join(processed)

def format_distill(distill):
    return ' '.join([x[0].upper() + x[1:] for x in distill.split('-')])

def format_markdown(text):
    return re.sub(r'\n\s+','\n',text)

# Create pitch chart
def create_pitch(dff, strategy=None):
    """Creates the whisky returns by pitch chart"""
    data = []
    if strategy == 2:
        xcol = 'annual_return'
        xtitle = 'Annual % return'
        ycol = 'APR_5%'
        ytitle = 'Compound Annual Return, %'
    else:
        xcol = 'days_to_close_spread'
        xtitle = 'Number of days to close bid ask spread'
        ycol = 'annual_return'
        ytitle = 'Annual Return, %'

    for _, grp in dff.groupby('distillery'):
        data.append(
            dict(x=grp[xcol], y=grp[ycol], text=grp.whisky_type.map(format_whisky_type),
                 mode='markers', name=grp.formattedDistillery.iloc[0], customdata=grp.index,
                 marker=dict(size=17, opacity=0.7, line={'color': 'rgb(255, 255, 255)', 'width': 1}))
        )

    figure = {'layout':
              dict(title=None,
                   autosize=True,
                   xaxis={'title': xtitle, 'showline': False, 'zeroline': False, },
                   yaxis={'title': ytitle, 'showline': False, 'zeroline': True, },
                   hovermode='closest', font={'family': 'inherit'},
                   modebar={'orientation': 'h'}, legend={'orientation': 'v'},
                   hoverlabel=dict(bordercolor='rgba(255, 255, 255, 0)', font={'color': '#ffffff'}),
                   margin={'l': 50, 'b': 50, 'r': 10, 't': 10}),
              'data': data}
    return figure

# Create bar chart for best returns
def best_returns_bar(table):
    data = [
        go.Bar(
            x=table['Whisky'],
            y=table['Annual Return, %'] * ~table['Own?'],
            text=table['Annual Return, %'],
            textposition="inside",
            textangle=0,
            # texttemplate = "%{text:.1f}%",
            hoverinfo="x+name",
            name='Not Owned', ),
        go.Bar(
            x=table['Whisky'],
            y=table['Annual Return, %'] * table['Own?'],
            text=table['Annual Return, %'],
            textposition="inside",
            textangle=0,
            # texttemplate = "%{text:.1f}%",
            hoverinfo="x+name",
            name='Owned',),
    ]

    if sum(table['Own?']) == 0:
        data = [data[0]]

    layout = dict(
        colorway=['#053061', '#ff610b', '#a3156d'],
        barmode='stack',
        hovermode='closest',
        yaxis={'showticklabels': False, 'showgrid': False},
        xaxis={'automargin': True},
        legend={"orientation":"v","xanchor":"auto"},
    )
    return {'layout': layout, 'data': data}

# Create timeseries chart
def create_time_series(dff, axis_type, pitches):
    dff = dff.merge(pitches,how='inner',on='pitchId')
    best_sell = dff.best_sell.values[0]
    best_buy = dff.best_buy.values[0]
    min_time = min(dff.time)
    max_time = max(dff.time)
    return dict(data=[
        go.Scatter(x=dff['time'],
                   y=dff['min_sell'],
                   mode='lines',
                   name='Bid £', ),
        go.Scatter(x=dff['time'],
                   y=dff['max_buy'],
                   mode='lines',
                   name='Ask £', ),
        go.Scatter(x=dff['time'],
                   y=dff['predict'],
                   mode='lines',
                   name='Model',
                   line={'width':1}),
    ], layout=dict(title=None, font={'family': 'inherit'}, hovermode='compare',
                   autosize=True,
                   legend=dict(orientation='h', xanchor='left', x=0.1, y=1.08, yanchor='top'),
                   margin={'l': 80, 'b': 20, 'r': 20, 't': 20},
                   yaxis={'type': 'linear' if axis_type == 'Linear' else 'log', 'title': 'Price, £'},
                   xaxis={'showgrid': False, 'rangeslider': {'visible':True, 'thickness':0.07},
                          'rangeselector': {'buttons': [{'step': 'month', 'count': 3}, {'step': 'year', 'count': 1}]}},
                   colorway=['#ff610b', '#053061', '#a3156d'],
                   annotations=[dict(showarrow=True, arrowcolor='rgba(255, 255, 255, 0.01)', ax=15, ay=0, opacity=0.5,
                                     xanchor='left', yanchor='bottom', x=min_time,
                                     y=best_sell, text='Current ask £{:.2f}'.format(best_sell)),
                                dict(showarrow=True, arrowcolor='rgba(255, 255, 255, 0.01)', ax=15, ay=0, opacity=0.5,
                                     xanchor='left', yanchor='bottom', x=min_time,
                                     y=best_buy, text='Current bid £{:.2f}'.format(best_buy)), ],
                   shapes=[dict(type='line', line={'dash':'dot'}, opacity=0.3, x0=min_time, x1=max_time, y0=best_sell,
                                y1=best_sell),
                           dict(type='line', line={'dash':'dot'}, opacity=0.3, x0=min_time, x1=max_time, y0=best_buy,
                                y1=best_buy), ],
                   ))

