from dash import Dash, dcc, html, Input, Output, callback

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Slider(0, 85000, 5000,
               value=60000,
               id='my-slider'
    ),
    html.Div(id='slider-output-container')
])

@callback(
    Output('slider-output-container', 'children'),
    Input('my-slider', 'value'))
def update_output(value):
    return 'You have selected the customer total amount of "{}"'.format(value)

if __name__ == '__main__':
    app.run(debug=True)