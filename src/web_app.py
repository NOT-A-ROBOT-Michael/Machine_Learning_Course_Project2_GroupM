"begin"
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import base64
import io
import plotly.graph_objs as go
import seaborn as sns
from prepare_data import prepare_dataset
from preprocess_data import process_data
from train_models import create_finalmodel
import dash_bootstrap_components as dbc

"palettes"
sns.color_palette("rocket", as_cmap=True)
"outliers, no_outliers"
df = pd.DataFrame()
"code to train models"
def prepare_outlier_data(df):
    df_temp = prepare_dataset(df)
    return df_temp

def prepare_non_outlier_data(df):
    df_temp = process_data(df)
    return df_temp

def train_data(df):
    df_temp = create_finalmodel(df)
    return df_temp

app = dash.Dash(external_stylesheets=[dbc.themes.LUX])
server = app.server

app.layout = html.Div([
    html.H1("Customer Segmentation Dashboard"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select CSV File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.P("Filter based on amount"),
    html.Div([
        dcc.RangeSlider(
            id='amount-slider',
            min=0,
            max=10000,
            value=[0, 10000],
            marks={i: str(i) for i in range(0, 10001, 2000)}
        )
    ], style={'marginBottom': '30px'}),
    html.P("Filter based on frequency"),
    html.Div([
        dcc.RangeSlider(
            id='frequency-slider',
            min=0,
            max=200,
            value=[0, 200],
            marks={i: str(i) for i in range(0, 201, 50)}
        )
    ], style={'marginBottom': '30px'}),
    html.Div([
        dcc.Graph(id='customer-segments-scatter'),
        dcc.Graph(id='customer-segments-scatter-2')
    ], style={'display': 'flex'}),
])

@app.callback(
    [Output('customer-segments-scatter', 'figure'),
     Output('customer-segments-scatter-2', 'figure')],
    [Input('upload-data', 'contents'),
     Input('amount-slider', 'value'),
     Input('frequency-slider', 'value')]
)
def update_output(contents, amount_range, frequency_range):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        temp_Out = prepare_outlier_data(df)
        temp_n_Out = prepare_non_outlier_data(temp_Out)

        temp_Out.to_csv('Out.csv', sep=',', index=False, header=True)
        temp_n_Out.to_csv('No_Out.csv', sep=',', index=False, header=True)

        Out = train_data(temp_Out)
        n_Out = train_data(temp_n_Out)
        Out.to_csv('Out.csv', sep=',', index=False, header=True)
        n_Out.to_csv('No_Out.csv', sep=',', index=False, header=True)

        # Filter data based on amount and frequency ranges
        Out = Out[(Out['Amount'] >= amount_range[0]) & (Out['Amount'] <= amount_range[1]) &
                  (Out['Frequency'] >= frequency_range[0]) & (Out['Frequency'] <= frequency_range[1])]
        n_Out = n_Out[(n_Out['Amount'] >= amount_range[0]) & (n_Out['Amount'] <= amount_range[1]) &
                      (n_Out['Frequency'] >= frequency_range[0]) & (n_Out['Frequency'] <= frequency_range[1])]

        # Plot 2D scatter plot of customer segments for both graphs
        fig = go.Figure(data=[go.Scatter(
            x=Out['Frequency'],
            y=Out['Amount'],
            mode='markers',
            marker=dict(
                size=5,
                color=Out['Cluster'],  # Color by cluster label
                opacity=0.8,
                colorscale='Viridis'
            )
        )])

        fig2 = go.Figure(data=[go.Scatter(
            x=n_Out['Frequency'],
            y=n_Out['Amount'],
            mode='markers',
            marker=dict(
                size=5,
                color=n_Out['Cluster'],  # Color by cluster label
                opacity=0.8,
                colorscale='Viridis'
            )
        )])

        fig.update_layout(
            title='2D Plot of Frequency and Amount - With outliers',
            xaxis_title='Frequency',
            yaxis_title='Amount',
            width=1000,
            height=700
        )

        fig2.update_layout(
            title='2D Plot of Frequency and Amount - Without outliers',
            xaxis_title='Frequency',
            yaxis_title='Amount',
            width=1000,
            height=700
        )

        return fig, fig2
    else:
        return {}, {}

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
"end"