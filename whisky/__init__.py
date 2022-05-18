import flask
import dash
from dash.dependencies import Input, Output
import pandas as pd
from random import randint
import dash_bootstrap_components as dbc
import numpy as np
import os

print("in init")

############################
# Import Utility Functions #
############################
from whisky.utils import format_distill, format_whisky_type, best_returns_bar, create_time_series, create_pitch
R_VALUE = 0.98


#######################
# Main app definition #
#######################
server = flask.Flask(__name__)
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootswatch/4.2.1/lux/bootstrap.min.css',]
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
app = dash.Dash(__name__, server=server,
                external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True


############################
# Load up and analyse data #
############################
from whisky.fetch import get_from_s3, calc_returns
pitches, all_whisky = get_from_s3()

# Formatting of pitches
pitches.formattedDistillery = pitches.formattedDistillery.map(format_distill)
pitches['formatted_whisky'] = pitches.whisky_type.map(format_whisky_type)

# Format summary table
### Adjust R_VALUE down dynamically to handle low confidence models
table_len=0
while table_len < 10:
    table = pitches[['formatted_whisky','days_to_close_spread','annual_return','r_value']]\
                    .query('r_value > '+ str(R_VALUE)).sort_values('annual_return',ascending=False)[:10].copy()
    table_len=len(table)
    if table_len < 10:
        R_VALUE -= 0.01
    if R_VALUE == 0:
        break
table['owned'] = False
# table = table.replace({'owned':{True:'✓',False:'✗'}})
table.columns = ['Whisky', 'Days to Close Bid Ask Spread', 'Annual Return, %', 'Confidence (R Value)', 'Own?']


##############
# Load views #
##############
import whisky.views


#############################
# Callbacks / Interactivity #
#############################

# Define main index
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    global table
    global pitches
    if pathname == '/':
        return whisky.views.summary_table_layout(table, pitches)
    elif pathname == '/about':
        return whisky.views.about_page_layout
    elif pathname == '/detail':
        return whisky.views.page_1_layout(pitches)
    else:
        return whisky.views.page_1_layout(pitches)
    # Could also return a 404 "URL not found" page here

# Show/hide the intro
@app.callback(Output('intro-text','style'),
              [Input('show-hide-button','n_clicks')])
def show_hide_intro(n):
    if not n:
        # Button has never been clicked
        return {}
    return {'display': 'none'}

# Fetch filtered values
@app.callback(
    Output('intermediate-value', 'children'),
    [Input(component_id='whisky-dropdown', component_property='value')]
)
def fetch_intermediate(value):
    # single fetch call for both table and chart
    global pitches
    pitches['owned'] = False
    table = pitches.query('r_value > ' + str(R_VALUE)).sort_values('annual_return',ascending=False)[:10]
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

# Update summary table
@app.callback(
    Output('summary-table', 'children'),
    [Input('intermediate-value', 'children')]
)
def update_table(json):
    if json is None:
        return dbc.Table.from_dataframe(table.replace({'Own?':{True:'✓',False:'✗'}}), dark=False, responsive='md', hover=True, float_format='.2f', size='sm')
    df = pd.read_json(json, orient='split')
    df = df[['formatted_whisky', 'days_to_close_spread', 'annual_return', 'r_value', 'owned']]
    df['annual_return'] = df['annual_return'].map('{:.1f}%'.format)
    df = df.replace({'owned':{True:'✓',False:'✗'}})
    df.columns = ['Whisky', 'Days to Close Bid Ask Spread', 'Annual Return, %', 'Confidence (R Value)', 'Own?']
    return dbc.Table.from_dataframe(df, dark=False, responsive='md', hover=True, float_format='.2f', size='sm')

# Update bar chart
@app.callback(
    Output('returns-bar', 'figure'),
    [Input('intermediate-value', 'children')]
)
def update_bar_chart(json):
    global table
    if json is None:
        return best_returns_bar(table)
    table = pd.read_json(json, orient='split')
    table = table[['formatted_whisky', 'days_to_close_spread', 'annual_return', 'r_value', 'owned']]
    table['annual_return'] = table['annual_return'].map('{:.1f}%'.format)
    # table = table.replace({'owned': {True: '✓', False: '✗'}})
    table.columns = ['Whisky', 'Days to Close Bid Ask Spread', 'Annual Return, %', 'Confidence (R Value)', 'Own?']
    return best_returns_bar(table)

# Update single whisky time series chart
@app.callback(
    Output('single-whisky-chart', 'figure'),
    [Input('whisky-return-graph', 'hoverData')])
def update_single_whisky(hoverData):
    whisky_name = hoverData['points'][0]['text']
    pitchId = hoverData['points'][0]['customdata']
    dff = all_whisky[all_whisky.pitchId == pitchId]
    return create_time_series(dff, 'Linear', pitches)

# Update single whisky timeseries title
@app.callback(
    Output('single-whisky-title', 'children'), [Input('whisky-return-graph', 'hoverData')])
def update_title(hoverData):
    return hoverData['points'][0]['text']

# Update displayed pitches
@app.callback(
    Output('whisky-return-graph', 'figure'),
    [Input('distillery-dropdown', 'value'), Input('radio-high-correlation', 'value'),
     Input('grain-malt-chooser', 'value'),]
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
        dff = dff[dff.r_value > R_VALUE]

    malt_grain_format = {'Grain':'GRAIN','Single Malt':'SINGLE_MALT'}
    if len(malt_grain) != 0:
        malt_grain = [malt_grain_format[x] for x in malt_grain]
        dff = dff[dff.categoryName.isin(malt_grain)]

    # if strategy == 2:
    #     return create_pitch(dff, 2)
    # else:
    return create_pitch(dff, 1)

def main():
    app.server.run(debug=False, threaded=True)