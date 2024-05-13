import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'Amount': [100, 200, 150, 300, 250],
    'Frequency': [5, 3, 4, 6, 2],
    'Country': ['USA', 'UK', 'Germany', 'France', 'Canada'],
    'Item': ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
})

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    dcc.Graph(id='scatter-plot'),
    html.Div([
        html.Label('Select range for Amount:'),
        dcc.RangeSlider(
            id='amount-slider',
            min=0,
            max=500,
            step=10,
            marks={i: str(i) for i in range(0, 501, 50)},
            value=[0, 500]
        ),
        html.Label('Select range for Frequency:'),
        dcc.RangeSlider(
            id='frequency-slider',
            min=0,
            max=10,
            step=1,
            marks={i: str(i) for i in range(0, 11)},
            value=[0, 10]
        )
    ])
])

# Define callback to update scatter plot
@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [dash.dependencies.Input('amount-slider', 'value'),
     dash.dependencies.Input('frequency-slider', 'value')]
)
def update_scatter_plot(amount_range, frequency_range):
    filtered_df = df[(df['Amount'] >= amount_range[0]) & (df['Amount'] <= amount_range[1]) &
                     (df['Frequency'] >= frequency_range[0]) & (df['Frequency'] <= frequency_range[1])]
    fig = px.scatter(filtered_df, x='Amount', y='Frequency', color='Country', hover_data=['Item'])
    return fig

# Run the app
if __name__ == '_main_':
    app.run_server(debug=True)