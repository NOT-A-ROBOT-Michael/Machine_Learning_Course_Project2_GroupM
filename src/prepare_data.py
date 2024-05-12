"begin"
import pandas as pd
import datetime as dt

# local DataFrame
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
"end"