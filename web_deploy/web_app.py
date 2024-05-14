"begin"
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import base64
import io
import plotly.graph_objs as go
import seaborn as sns
import dash_bootstrap_components as dbc
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score

df = pd.DataFrame()
"""function to load the data"""
def prepare_dataset(df):
    "Data Cleaning"
    # Drop 'Country' and 'InvoiceNo' columns
    processed_df = df.drop(['Country','Description'], axis=1)
    # Remove rows with quantity less than or equal to zero
    processed_df = processed_df[processed_df['Quantity'] >= 0]
    # Remove rows with missing CustomerID
    processed_df = processed_df.dropna(subset=['CustomerID'])
    # Reset the index after removing rows
    processed_df.reset_index(drop=True, inplace=True)

    "Data Processing"
    processed_df['Quantity'] = processed_df['Quantity'].astype(int)
    processed_df['CustomerID'] = processed_df['CustomerID'].astype(str)
    processed_df['Amount'] = processed_df['Quantity']*processed_df['UnitPrice']
    # amount
    rfm_ds_n = processed_df.groupby('CustomerID')['Amount'].sum()
    rfm_ds_n.reset_index()
    rfm_ds_n.columns = ['CustomerID', 'Amount']
    # frequency
    rfm_ds_f = processed_df.groupby('CustomerID')['InvoiceNo'].count()
    rfm_ds_f = rfm_ds_f.reset_index()
    rfm_ds_f.columns = ['CustomerID','Frequency']
    # recency
    'date_diff'
    processed_df['InvoiceDate'] = pd.to_datetime(processed_df['InvoiceDate'],format='%m/%d/%Y %H:%M')
    max_date = max(processed_df['InvoiceDate'])
    processed_df['Diff'] = max_date - processed_df['InvoiceDate']
    rfm_ds_p = processed_df.groupby('CustomerID')['Diff'].min()
    rfm_ds_p = rfm_ds_p.reset_index()
    rfm_ds_p.columns = ['CustomerID', 'Diff']
    rfm_ds_p['Diff'] = rfm_ds_p['Diff'].dt.days
    # merge
    rfm_ds_final = pd.merge(rfm_ds_n, rfm_ds_f, on='CustomerID',how='inner')
    rfm_ds_final = pd.merge(rfm_ds_final, rfm_ds_p, on='CustomerID', how='inner')
    rfm_ds_final.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']
    return rfm_ds_final

def process_data(df):
        #Removing outliers
        Q1 = df['Amount'].quantile(0.25)
        Q3 = df['Amount'].quantile(0.75)
        IQR = Q3-Q1
        rfm_ds_final = df[(df['Amount'] > Q1 - 1.5*IQR) & (df['Amount'] < Q3 + 1.5*IQR)]

        Q1 = df['Recency'].quantile(0.25)
        Q3 = df['Recency'].quantile(0.75)
        IQR = Q3-Q1
        rfm_ds_final = df[(df['Recency'] > Q1 - 1.5*IQR) & (df['Recency'] < Q3 + 1.5*IQR)]

        Q1 = df['Frequency'].quantile(0.25)
        Q3 = df['Frequency'].quantile(0.75)
        IQR = Q3-Q1
        rfm_ds_final = df[(df['Frequency'] > Q1 - 1.5*IQR) & (df['Frequency'] < Q3 + 1.5*IQR)]
        
        #Dont need Min-max scaling
        X = rfm_ds_final
        return X

def getClusters(df):

    #model creation
    kmeans = KMeans(n_clusters= 3,max_iter= 50)
    kmeans.fit(df)
    lbs = kmeans.labels_

    "elbow-method"
    #appendin inertia
    #wss
    wss =[]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters= num_clusters, max_iter= 50)
        kmeans.fit(df)
        wss.append(kmeans.inertia_)
    #silhouette score
    n_cluster=0
    silhouette_no=0
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters= num_clusters, max_iter= 50)
        kmeans.fit(df)
        cluster_labels = kmeans.labels_
        silhouette_avg = silhouette_score(df, cluster_labels)
        print('For n_clusters{0}, the silhouette score is {1}'.format(num_clusters, silhouette_avg))
        if silhouette_avg>silhouette_no:
            silhouette_no=silhouette_avg
            n_cluster=num_clusters
    return n_cluster

"returns the final data-model"
def create_finalmodel(df):
    # base model
    df_scaled = df
    # final_model labels
    final_model = make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=3, random_state=42)
    )
    # Predict class labels
    cluster = final_model.fit_predict(df_scaled)

    df_scaled['Cluster'] = cluster
    return df_scaled


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
    html.P("Filter based on amount (USD)"),
    html.Div([
        dcc.RangeSlider(
            id='amount-slider',
            min=0,
            max=10000,
            value=[0, 10000],
            marks={i: str(i) for i in range(0, 10001, 2000)}
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
