import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash_html_template import Template
import os
from random import randint
import plotly.graph_objs as go
import re

# Load up and analyse data
from fetch import get_from_s3
pitches, all_whisky = get_from_s3()

# TODO: use flask-cache to save calls to whisky site
# https://pythonhosted.org/Flask-Cache/

server = flask.Flask(__name__)
external_stylesheets = [
    # 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css',
    'https://stackpath.bootstrapcdn.com/bootswatch/4.2.1/lux/bootstrap.min.css',
    # 'https://fonts.googleapis.com/css?family=Montserrat:400,700',
    ]

server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
app = dash.Dash(__name__, server=server,
                external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True

def create_pitch(dff):
    """Creates the whisky returns by pitch chart"""
    data = []
    for name, grp in dff.groupby('distillery'):
        data.append(
            dict(x=grp.days_to_close_spread, y=grp.annual_return, text=grp.whisky_type, mode='markers', name=name,
                 customdata=grp.index, marker=dict(size=17, opacity=0.6, line={'color': 'rgb(255, 255, 255)', 'width': 1}))
        )

    figure = {'layout':
                  dict(title=None,
                       xaxis={'title': 'Number of days to close bid ask spread', 'rangemode':'nonnegative', 'showline': False, 'zeroline': False, },
                       yaxis={'title': 'Annual % return', 'rangemode': 'nonnegative', 'showline': False, 'zeroline': False, },
                       hovermode='closest', font={'family': 'inherit'}, autosize=True,
                       modebar={'orientation': 'h'}, legend={'orientation': 'v'},
                       hoverlabel=dict(bordercolor='rgba(255, 255, 255, 0)', font={'color': '#ffffff'}),
                       margin={'l': 50, 'b': 50, 'r': 10, 't': 10}),
              'data': data}
    return figure

def format_markdown(text):
    return re.sub(r'\n\s+','\n',text)

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


def Nav():
    nav = html.Nav(children=[
        html.Div([
            html.Img(src='/assets/whiskey.svg',height=40),
            html.A('Whisky Investor', className='navbar-brand', href='/'),
            html.Div([
                html.Ul([
                    html.Li([html.A('About', className='nav-link', href='/about')], className='nav-item'),
                    # html.Li([html.A('Github', className='nav-link', href='/')], className='nav-item'),
                    html.Li([html.A('Github', className='nav-link', href='https://github.com/arms3')], className='nav-item'),
                ], className='navbar-nav'),
                html.Ul([],className='nav navbar-nav ml-auto'),
            ], id='navbarResponsive',className='collapse navbar-collapse'),
        ], className='container')
    ], className="navbar navbar-expand-lg navbar-dark bg-dark")
    return nav


about_page_layout = html.Div([
    # Navbar
    Nav(),

    # Main container
    # Header
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H1('What\'s this all about?', style={'margin-top':30}),
                    dcc.Markdown(format_markdown('''
                    - Fetches daily pricing from [whiskyinvestdirect.com](https://www.whiskyinvestdirect.com/)
                    - Evaluates performance via linear regression
                    - Creates a [dashboard](https://whisky-invest.herokuapp.com/) to display best performance whiskies
                    - Adjusts for exchange fees and holding costs
                    ''')),
                ], className='col-lg-12')
            ], className='row')
        ], className='page-header'),
    ], className='container')
])


# Main chart page
# TODO: Update this to use dash-bootsrap components
# https://dash-bootstrap-components.opensource.asidatascience.com/
page_1_layout = html.Div([
    Nav(),
    # Main container
    html.Div([
        # Header
        html.Div([
            html.Div([
                html.Div([
                    # html.H1('Whisky Price Explorer', style={'margin-top':30}),
                    html.P(dcc.Markdown('Top performing whiskies from [whiskyinvestdirect.com](https://www.whiskyinvestdirect.com/)')),
                ], className='col-lg-12', style={'margin-top':20})
            ], className='row')
        ], className='page-header'),
        # Row containing charts
        html.Div([
            # Left Side
            html.Div([
                html.Div([
                    # Left chart
                    html.Div([html.H3('Predicted Returns', style={'display':'inline-block','margin-bottom':'0px'})], className='card-header'), #className='card-header'
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
                            html.Div([
                                dcc.Graph(
                                    id='whisky-return-graph',
                                    hoverData={'points': [{'text': 'auchroisk_2012_Q4_HHR', 'customdata':1}]},
                                    # style={'width':800},
                                    figure=create_pitch(pitches),
                                    style={'height':'100%'}
                                ),
                            ],className='container',style={'overflow':'hidden'}),
                        ], className='row', style={'margin':'5px'})
                    ],className='section'),
                ], className='card border-secondary mb-3', style={'height':650}), #col-lg-6 style={'height':600} #className='card border-secondary mb-3'
            ],className='col-lg-6'),
            # Right Side
            html.Div([
                html.Div([
                    html.Div([html.H3('Price History',style={'display':'inline-block','margin-bottom':'0px'}),
                              html.P('Filler', id='single-whisky-title', className='float-right', style={'margin-bottom': '0px'})],
                             className='card-header'), #className='card-header'
                    # html.Div([html.P('Hello',className='lead')], className='row', style={'margin':'5px'}),
                    dcc.Graph(id='single-whisky-chart',
                              # style={'width': 850,'height':550},
                              style={'height':'100%'}),
                ],className='card border-secondary mb-3', style={'height':650}), #'col-lg-6' style={'height':600} #className='card border-secondary mb-3',
            ],className='col-lg-6'),
        ],className='row row-eq-height', ), #style={'display':'flex'}

        # Footer
        html.Footer([html.Div([html.Div([
            html.Ul([
                    html.Li(html.A('Back to top',href='#top'),className='float-lg-right'),
                    # html.Li(html.A('Home',href='/')),
                    # html.Li(html.A('About',href='/about')),
                    # html.Li(html.A('GitHub',href='https://github.com/arms3')),
            ],className='list-unstyled'),
            dcc.Markdown('''Created by [Angus Sinclair](https://github.com/arms3)'''),
            Template.from_string(
                '''<P>Icons made by <a href="https://www.freepik.com/" title="Freepik">Freepik</a> from <a href="https://www.flaticon.com/" 			    title="Flaticon">www.flaticon.com</a> is licensed by <a href="http://creativecommons.org/licenses/by/3.0/" 			    title="Creative Commons BY 3.0" target="_blank">CC 3.0 BY</a></P>'''),
            ],className='col-lg-12'),],className='row'),
        ],id='footer'),
    ], className="container"),
])


# Update the index
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return page_1_layout
    elif pathname == '/about':
        return about_page_layout
    else:
        return page_1_layout
    # You could also return a 404 "URL not found" page here


@app.callback(
    Output('single-whisky-chart', 'figure'),
    [Input('whisky-return-graph', 'hoverData')])
def update_single_whisky(hoverData):
    whisky_name = hoverData['points'][0]['text']
    pitchId = hoverData['points'][0]['customdata']
    dff = all_whisky[all_whisky['pitchId'] == pitchId]
    return create_time_series(dff, 'Linear', whisky_name)

@app.callback(
    Output('single-whisky-title', 'children'),[Input('whisky-return-graph', 'hoverData')])
def update_title(hoverData):
    m = {'BBF':'First fill bourbon', 'BBR': 'Refill bourbon', 'HHR': 'Refill hogshead', 'SBR': 'Refill sherry butt'}
    raw = hoverData['points'][0]['text']
    processed = raw.split('_')
    processed[0] = ' '.join([x[0].upper()+x[1:] for x in processed[0].split('-')])
    processed[-1] = m[processed[-1]]
    processed = ' '.join(processed)
    return processed

@app.callback(
    Output('whisky-return-graph', 'figure'),
    [Input('distillery-dropdown', 'value'), Input('radio-high-correlation','value'),
     Input('grain-malt-chooser', 'values'),]
)
def update_pitches(distilleries, radio, malt_grain):
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

    return create_pitch(dff)


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
                   name='Model', ),
    ], layout=dict(title=None, font={'family': 'inherit'}, hovermode='closest',
                   legend=dict(orientation='h', xanchor='left', x=0.1, y=1.11, yanchor='top'),
                   margin={'l': 80, 'b': 20, 'r': 20, 't': 20},
                   yaxis={'type': 'linear' if axis_type == 'Linear' else 'log', 'title': 'Price, £'},
                   xaxis={'showgrid': False, 'rangeslider': {'visible':True, 'thickness':0.07},
                          'rangeselector': {'buttons': [{'step': 'month', 'count': 3}, {'step': 'year', 'count': 1}]}},
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
    app.server.run(debug=False, threaded=True)
