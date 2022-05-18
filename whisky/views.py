from whisky import app
from whisky.utils import best_returns_bar, format_markdown
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

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
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Nav layout
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

# Footer layout
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

# Contact card layout
contact_card = dbc.Card([
    dbc.CardHeader(html.H3("Contact Details",style={'margin':0})),
    dbc.Row([
        dbc.Col(dbc.CardImg(src=app.get_asset_url('mug.png'), style={'max-width': '120px'}), width=3),
        dbc.Col([
            dbc.Row(['email: ', dbc.CardLink('sinclair.angus@gmail.com', href='mailto:sinclair.angus@gmail.com')]),
            dbc.Row(['git: ', dbc.CardLink('github.com/arms3/whiskyInvest', href='https://github.com/arms3/whiskyInvest')]),
            dbc.Row(['app: ', dbc.CardLink('whisky-invest.herokuapp.com', href='https://whisky-invest.herokuapp.com')]),
            dbc.Row(['about me: ', html.P('Drinker of good and bad whisky', className='text-primary')]),
        ], width=9, align='center'),
    ]),
])

# Summary layout
def summary_table_layout(table, pitches):
    return html.Div([
        Nav,
        dbc.Container([
            dbc.Row([
                dbc.Row([
                    dbc.Col([
                        dbc.Jumbotron([
                            html.H2('What is this website for?'),
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
                    html.Div([dbc.Table.from_dataframe(table.replace({'Own?':{True:'✓',False:'✗'}}), dark=False, responsive='md', hover=True,
                                                    float_format='.2f', size='sm')],id='summary-table'),
                ]),
                dbc.Col([
                    dbc.Row([
                        dcc.Graph(figure=best_returns_bar(table), id='returns-bar', style={'margin-bottom':30}),
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
                                orient='records'),
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

# About page layout
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
                    - Webscraper for pricing deployed on AWS Lambda
                    - Daily analysis (batch forecasting and data aggregation) deployed on AWS Lambda
                    - Webapp deployed on [Heroku](https://www.heroku.com/), using [Flask](http://flask.pocoo.org/) and [Dash by Plot.ly](https://plot.ly/products/dash/)
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

# Main chart page
def page_1_layout(pitches):
    return html.Div([
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
                            dbc.Row([
                                dbc.Col([
                                    dcc.Dropdown(
                                        id='distillery-dropdown',
                                        options=pitches[['formattedDistillery', 'distillery']].drop_duplicates() \
                                            .rename({'formattedDistillery': 'label', 'distillery': 'value'}, axis=1).to_dict(
                                            orient='records'),
                                        multi=True,
                                    ),
                                ], width=8, style={'padding-left':'0px'}),

                                # Radio items
                                dbc.Col([
                                    dcc.RadioItems(
                                        id='radio-high-correlation',
                                        options=[{'label':'All data','value':1},{'label':'High R Value','value':2}],
                                        value=2,
                                        labelStyle={'display': 'inline-block', 'padding':'2px'},
                                    ),
                                ], width=4),
                            ], style={'margin':'5px'}),

                            # Malt picker
                            html.Div([
                                dcc.Checklist(
                                    id='grain-malt-chooser',
                                    options=[{'label':i,'value':i} for i in ['Single Malt','Grain']],
                                    value  =['Single Malt','Grain'],
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
