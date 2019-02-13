import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import os
from random import randint
import plotly.graph_objs as go
import re
import pandas as pd
import numpy as np

# Load up and analyse data
from fetch import get_from_s3, calc_returns
pitches, all_whisky = get_from_s3()

# TODO: use flask-cache to save calls to whisky site
# https://pythonhosted.org/Flask-Cache/

# Main app definition
server = flask.Flask(__name__)
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootswatch/4.2.1/lux/bootstrap.min.css',]
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
app = dash.Dash(__name__, server=server,
                external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True

# Default template to load. Can customize favicon
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Whisky Invest</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


# Update the index
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return summary_table_layout
    elif pathname == '/about':
        return about_page_layout
    elif pathname == '/detail':
        return page_1_layout
    else:
        return page_1_layout
    # You could also return a 404 "URL not found" page here


# Formatting helpers
def format_whisky_type(whisky_type):
    m = {'BBF': 'First fill bourbon', 'BBR': 'Refill bourbon', 'HHR': 'Refill hogshead', 'SBR': 'Refill sherry butt'}
    processed = whisky_type.split('_')
    processed[0] = ' '.join([x[0].upper() + x[1:] for x in processed[0].split('-')])
    processed[-1] = m[processed[-1]]
    return ' '.join(processed)


def format_distill(distill):
    return ' '.join([x[0].upper() + x[1:] for x in distill.split('-')])


def format_markdown(text):
    return re.sub(r'\n\s+','\n',text)


# Formatting of pitches
pitches.formattedDistillery = pitches.formattedDistillery.map(format_distill)
pitches['formatted_whisky'] = pitches.whisky_type.map(format_whisky_type)

# Format summary table
table = pitches[['formatted_whisky','days_to_close_spread','annual_return','r_value']]\
                .query('r_value > 0.99').sort_values('annual_return',ascending=False)[:10].copy()
table['owned'] = False
table.columns = ['Whisky', 'Days to Close Bid Ask Spread', 'Annual Return, %', 'Confidence (R Value)', 'Own?']


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

    for name, grp in dff.groupby('distillery'):
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


def best_returns_bar():
    data = [
        go.Bar(
            x=table['Whisky'],
            y=table['Annual Return, %'] * ~table['Own?'],
            text=table['Annual Return, %'],
            textposition="inside",
            hoverinfo="x+name",
            name='Not Owned', ),
        go.Bar(
            x=table['Whisky'],
            y=table['Annual Return, %'] * table['Own?'],
            text=table['Annual Return, %'],
            textposition="inside",
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


# Common Parts
Nav = html.Nav(children=[
    html.Div([
        html.Img(src='/assets/whiskey.svg',height=40),
        html.A('Whisky Investor', className='navbar-brand', href='/'),
        html.Div([
            html.Ul([
                html.Li([html.A('Home', className='nav-link', href='/')], className='nav-item'),
                html.Li([html.A('Detail', className='nav-link', href='/detail')], className='nav-item'),
                html.Li([html.A('About', className='nav-link', href='/about')], className='nav-item'),
            ], className='navbar-nav'),
            html.Ul([],className='nav navbar-nav ml-auto'),
        ], id='navbarResponsive',className='collapse navbar-collapse'),
    ], className='container')
], className="navbar navbar-expand-lg navbar-dark bg-dark")

Footer = html.Footer([
    html.Div([
        html.Div([
            html.Ul([
                html.Li(html.A('Back to top',href='#top'),className='float-lg-right'),
            ],className='list-unstyled'),
            dcc.Markdown('''Created by [Angus Sinclair](https://github.com/arms3)  
Icons made by [Freepik](https://www.freepik.com/) from [www.flaticon.com](https://www.flaticon.com/) is licensed by
[CC 3.0 BY](http://creativecommons.org/licenses/by/3.0/)'''),
        ],className='col-lg-12'),
    ],className='row'),
],id='footer')

contact_card = dbc.Card([
    dbc.CardHeader(html.H3("Contact Details",style={'margin':0})),
    dbc.Row([
        dbc.Col(dbc.CardImg(src=app.get_asset_url('mug.jpg'), style={'max-width': '120px'}), width=3),
        dbc.Col([
            dbc.Row(['email: ', dbc.CardLink('angus.sinclair@mg.thedataincubator.com', href='mailto:angus.sinclair@mg.thedataincubator.com')]),
            dbc.Row(['git: ', dbc.CardLink('github.com/arms3/whiskyInvest', href='https://github.com/arms3/whiskyInvest')]),
            dbc.Row(['app: ', dbc.CardLink('whisky-invest.herokuapp.com', href='https://whisky-invest.herokuapp.com')]),
            dbc.Row(['about me: ', html.P('Drinker of good and bad whisky', className='text-primary')]),
        ], width=9, align='center'),
    ]),
])


about_page_layout = html.Div([
    # Navbar
    Nav,

    # Main container
    html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    html.P(dcc.Markdown(format_markdown('''
                    ## About this site
                    ##### Technologies
                    - Webscraper for pricing deployed on [AWS Lightsail](https://aws.amazon.com/lightsail/)
                    - Daily analysis (batch forecasting and data aggregation) deployed on [AWS Data pipeline](https://aws.amazon.com/datapipeline/)
                    - Webapp deployed on [Heroku](heroku.com), using [Flask](http://flask.pocoo.org/) and [Dash by Plot.ly](https://plot.ly/products/dash/)
                    ##### Purpose
                    - Fetches pricing and provides recommended whisky investments for the Scotch whisky trading platform [whiskyinvestdirect.com](https://www.whiskyinvestdirect.com/)
                    - Evaluates projected performance of whiskies adjusted for exchange fees and holding costs
                    - Hosts an interactive [dashboard](https://whisky-invest.herokuapp.com/) to display best performing whiskies in real time
                    '''))),
                ]),
            ],width=6),
            dbc.Col(contact_card, width=5),
        ],style={'margin-top':30})
    ], className='container', style={'margin-left':30}),
])

summary_table_layout = html.Div([
    Nav,
    dbc.Container([
        dbc.Row([
            dbc.Row([
                dbc.Col([
                    dbc.Jumbotron([
                        html.H2('What is this website telling me?'),
                        html.P(dcc.Markdown('This website helps you to pick the top performing whiskies on [whiskyinvestdirect.com](https://www.whiskyinvestdirect.com/).'),),
                        html.Hr(className="my-2"),
                        html.P("The performance of your investments depends on the strategy you use to buy and sell "
                                "whiskies. The annual returns calculated here are based on a buy and hold strategy for "
                                "one year. Included in this calculation are the buying and selling fees (1.75% for each "
                                "transaction), the bid ask spread, and the holding fees of 15p per month."),
                        html.P(dbc.Button("Got it thanks", color="primary"), className="btn-sm", id="show-hide-button"),
                    ], id='intro-text'),
                ],width=8, align='center'),
            ]),
            dbc.Col([
                html.H2("Top performing whiskies"),
                html.Div([dbc.Table.from_dataframe(table, dark=False, responsive='md', hover=True,
                                                   float_format='.2f', size='sm')],id='summary-table'),
            ]),
            dbc.Col([
                dbc.Row([
                    dcc.Graph(figure=best_returns_bar(),id='returns-bar', style={'margin-bottom':30}),
                ]),
                dbc.Row(html.H5("Input whiskies you own")),
                dbc.Row(html.P("To see the relative return of holding whiskies you currently own. This compares "
                               "the current sale value to the projected sales value in 1 year",
                               className="text-primary")),
                dbc.Row([
                    dcc.Dropdown(
                        id='whisky-dropdown',
                        options=pitches[['formatted_whisky']].reset_index().drop_duplicates() \
                            .rename({'formatted_whisky': 'label', 'pitchId': 'value'}, axis=1).to_dict(
                            orient='rows'),
                        multi=True,
                        style={'width':'100%', 'margin-bottom':50},
                    ),
                ]),
            ])
        ],),
        Footer,
        ],  className="mt-4",),
    html.Div(id='intermediate-value', style={'display': 'none'}),
])


@app.callback(Output('intro-text','style'),
              [Input('show-hide-button','n_clicks')])
def show_hide_intro(n):
    if not n:
        # Button has never been clicked
        return {}
    return {'display': 'none'}


@app.callback(
    Output('intermediate-value', 'children'),
    [Input(component_id='whisky-dropdown', component_property='value')]
)
def fetch_intermediate(value):
    # single fetch call for both table and chart
    global pitches
    pitches['owned'] = False
    table = pitches.query('r_value > 0.99').sort_values('annual_return',ascending=False)[:10]
    if value == '' or value is None:
        return table.to_json(orient='split')

    # Do pitches and recalculate returns based on owned.
    owned = pd.DataFrame({'pitchId': value, 'owned': np.ones(len(value), dtype=bool)}).set_index('pitchId')
    pitches.drop('owned', axis=1, inplace=True)
    pitches = pitches.join(owned, how='outer')
    pitches.owned = ~pitches.owned.isna()
    owned = calc_returns(pitches[pitches.owned])
    table = pd.concat([table, owned],axis=0).sort_values('annual_return', ascending=False)[:20].drop_duplicates('securityId')
    return table.to_json(orient='split')

@app.callback(
    Output('summary-table', 'children'),
    [Input('intermediate-value', 'children')]
)
def update_table(json):
    if json is None:
        return dbc.Table.from_dataframe(table, dark=False, responsive='md', hover=True, float_format='.2f', size='sm')
    df = pd.read_json(json, orient='split')
    df = df[['formatted_whisky', 'days_to_close_spread', 'annual_return', 'r_value', 'owned']]
    df['annual_return'] = df['annual_return'].map('{:.1f}%'.format)
    df = df.replace({'owned':{True:'✓',False:'✗'}})
    df.columns = ['Whisky', 'Days to Close Bid Ask Spread', 'Annual Return, %', 'Confidence (R Value)', 'Own?']
    return dbc.Table.from_dataframe(df, dark=False, responsive='md', hover=True, float_format='.2f', size='sm')


@app.callback(
    Output('returns-bar', 'figure'),
    [Input('intermediate-value', 'children')]
)
def update_bar_chart(json):
    if json is None:
        return best_returns_bar()
    global table
    table = pd.read_json(json, orient='split')
    table = table[['formatted_whisky', 'days_to_close_spread', 'annual_return', 'r_value', 'owned']]
    table['annual_return'] = table['annual_return'].map('{:.1f}%'.format)
    # table = table.replace({'owned': {True: '✓', False: '✗'}})
    table.columns = ['Whisky', 'Days to Close Bid Ask Spread', 'Annual Return, %', 'Confidence (R Value)', 'Own?']
    return best_returns_bar()


# Main chart page
page_1_layout = html.Div([
    Nav,
    # Main container
    html.Div([
        # Header
        html.Div([
            html.Div([
                html.Div([
                ], className='col-lg-12', style={'margin-top':20})
            ], className='row')
        ], className='page-header'),
        # Row containing charts
        dbc.Fade([
            # Left Side
            html.Div([
                html.Div([
                    # Left chart
                    html.Div([html.H3('Predicted Returns', style={'display':'inline-block','margin-bottom':'0px'})],
                             className='card-header'),
                    html.Div([

                        # Distillery picker
                        html.Div([
                            html.Div([
                                dcc.Dropdown(
                                    id='distillery-dropdown',
                                    options=pitches[['formattedDistillery', 'distillery']].drop_duplicates() \
                                        .rename({'formattedDistillery': 'label', 'distillery': 'value'}, axis=1).to_dict(
                                        orient='rows'),
                                    multi=True,
                                ),
                            ], className='col-lg-9', style={'padding-left':'0px'}),

                            # Radio items
                            html.Div([
                                dcc.RadioItems(
                                    id='radio-high-correlation',
                                    options=[{'label':'All data','value':1},{'label':'High R Value','value':2}],
                                    value=2,
                                    labelStyle={'display': 'inline-block', 'padding':'2px'},
                                ),
                            ], className='float-lg-right'),
                        ], className='row', style={'margin':'5px'}),

                        # Malt picker
                        html.Div([
                            dcc.Checklist(
                                id='grain-malt-chooser',
                                options=[{'label':i,'value':i} for i in ['Single Malt','Grain']],
                                values=['Single Malt','Grain'],
                                labelStyle={'display': 'inline-block', 'margin': '5px'},),
                        ], className='row', style={'margin':'5px'}),

                        # Chart row
                        html.Div([
                            # html.Div([
                            dcc.Graph(
                                id='whisky-return-graph',
                                hoverData={'points': [{'text': 'auchroisk_2012_Q4_HHR', 'customdata':1}]},
                                # style={'width':800},
                                # figure=create_pitch(pitches),
                                style={'height':'100%', 'width':'100%'}
                            ),
                        ], className='row', style={'margin':'5px','height':450})

                    ],className='section'),
                ], className='card border-secondary mb-3', style={'height':650}),
            ],className='col-lg-6'),

            # Right Side
            html.Div([
                html.Div([
                    html.Div([html.H3('Price History',style={'display':'inline-block','margin-bottom':'0px'}),
                              html.P('Filler', id='single-whisky-title', className='float-right',
                                     style={'margin-bottom': '0px'})],
                             className='card-header'),

                    dcc.Graph(id='single-whisky-chart',
                              style={'height':'100%'}),
                ],className='card border-secondary mb-3', style={'height':650}),
            ],className='col-lg-6'),

        ],className='row row-eq-height', id="chart-content", appear=False, is_in=True),

        # Footer
        Footer,

    ], className="container"),
])


@app.callback(
    Output('single-whisky-chart', 'figure'),
    [Input('whisky-return-graph', 'hoverData')])
def update_single_whisky(hoverData):
    whisky_name = hoverData['points'][0]['text']
    pitchId = hoverData['points'][0]['customdata']
    dff = all_whisky[all_whisky['pitchId'] == pitchId]
    return create_time_series(dff, 'Linear', whisky_name)


@app.callback(
    Output('single-whisky-title', 'children'), [Input('whisky-return-graph', 'hoverData')])
def update_title(hoverData):
    return hoverData['points'][0]['text']

@app.callback(
    Output('whisky-return-graph', 'figure'),
    [Input('distillery-dropdown', 'value'), Input('radio-high-correlation', 'value'),
     Input('grain-malt-chooser', 'values'),] #Input('radio-strategy', 'value')]
)
def update_pitches(distilleries, radio, malt_grain,): #strategy):
    # Check we have some distilleries selected otherwise show all
    if distilleries == None:
        dff = pitches
    elif len(distilleries) == 0:
        dff = pitches
    else:
        dff = pitches[pitches['distillery'].isin(distilleries)]

    if radio == 2:
        dff = dff[dff.r_value > 0.95]
    else:
       pass

    malt_grain_format = {'Grain':'GRAIN','Single Malt':'SINGLE_MALT'}
    if len(malt_grain) == 0:
        pass
    else:
        malt_grain = [malt_grain_format[x] for x in malt_grain]
        dff = dff[dff.categoryName.isin(malt_grain)]

    # if strategy == 2:
    #     return create_pitch(dff, 2)
    # else:
    return create_pitch(dff, 1)


def create_time_series(dff, axis_type, title):
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


if __name__ == '__main__':
    app.server.run(debug=True, threaded=True)
