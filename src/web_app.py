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

sns.color_palette("rocket", as_cmap=True)
df = pd.DataFrame()

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
    html.H1("Cluster Visualisation", style={'textAlign': 'center'}),
    html.P(style={'marginBottom': '30px'}),
    html.P("Filter based on amount (USD)"),
    html.Div([
        dcc.RangeSlider(
            id='amount-slider',
            min=0,
            max=80000,
            value=[0, 80000],
            marks={i: str(i) for i in range(0, 80001, 20000)}
        )
    ], style={'marginBottom': '30px'}),
    html.P("Filter based on frequency (days)"),
    html.Div([
        dcc.RangeSlider(
            id='frequency-slider',
            min=0,
            max=200,
            value=[0, 200],
            marks={i: str(i) for i in range(0, 201, 50)}
        )
    ], style={'marginBottom': '30px'}),
    dbc.Row([
        dbc.Col(dcc.Graph(id='customer-segments-scatter'), width=6),
        dbc.Col(dcc.Graph(id='customer-segments-scatter-2'), width=6)
    ]),
    html.Div([
        html.Button("Download Filtered Data", id="btn-download", n_clicks=0,
                    style={'display': 'block', 'margin': '0 auto', 'marginBottom': '30px'}),
        dcc.Download(id="download-dataframe-csv")
    ], style={'textAlign': 'center'}),
])

@app.callback(
    [Output('customer-segments-scatter', 'figure'),
     Output('customer-segments-scatter-2', 'figure'),
     Output('download-dataframe-csv', 'data')],
    [Input('upload-data', 'contents'),
     Input('amount-slider', 'value'),
     Input('frequency-slider', 'value'),
     Input('btn-download', 'n_clicks')]
)
def update_output(contents, amount_range, frequency_range, n_clicks):
    download_data = None
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        temp_Out = prepare_outlier_data(df)
        temp_n_Out = prepare_non_outlier_data(temp_Out)

        Out = train_data(temp_Out)
        n_Out = train_data(temp_n_Out)

        # Filter data based on amount and frequency ranges
        Out_filtered = Out[(Out['Amount'] >= amount_range[0]) & (Out['Amount'] <= amount_range[1]) &
                           (Out['Frequency'] >= frequency_range[0]) & (Out['Frequency'] <= frequency_range[1])]
        n_Out_filtered = n_Out[(n_Out['Amount'] >= amount_range[0]) & (n_Out['Amount'] <= amount_range[1]) &
                               (n_Out['Frequency'] >= frequency_range[0]) & (n_Out['Frequency'] <= frequency_range[1])]

        # Plot 2D scatter plot of customer segments for both graphs
        fig = go.Figure(data=[go.Scatter(
            x=Out_filtered['Frequency'],
            y=Out_filtered['Amount'],
            mode='markers',
            marker=dict(
                size=5,
                color=Out_filtered['Cluster'],  # Color by cluster label
                opacity=0.8,
                colorscale='Jet'
            )
        )])

        fig2 = go.Figure(data=[go.Scatter(
            x=n_Out_filtered['Frequency'],
            y=n_Out_filtered['Amount'],
            mode='markers',
            marker=dict(
                size=5,
                color=n_Out_filtered['Cluster'],  # Color by cluster label
                opacity=0.8,
                colorscale='Jet'
            )
        )])

        fig.update_layout(
            title='2D Plot of Frequency and Amount - With outliers',
            xaxis_title='Frequency',
            yaxis_title='Amount',
            width=700,
            height=700
        )

        fig2.update_layout(
            title='2D Plot of Frequency and Amount - Without outliers',
            xaxis_title='Frequency',
            yaxis_title='Amount',
            width=700,
            height=700
        )

        # Prepare data for download
        if n_clicks > 0:
            filtered_data = pd.concat([Out_filtered, n_Out_filtered])
            download_data = dcc.send_data_frame(filtered_data.to_csv, "filtered_data.csv")

        return fig, fig2, download_data
    else:
        return {}, {}, None

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
