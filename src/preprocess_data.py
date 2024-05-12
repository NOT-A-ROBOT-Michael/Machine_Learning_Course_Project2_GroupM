"begin"
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# local DataFrame
df = pd.DataFrame()
"""function to process the data"""

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
        
        #scaling
        X = rfm_ds_final[['Amount', 'Frequency', 'Recency']]
        scaler = MinMaxScaler()
        rfm_ds_scaled = scaler.fit_transform(X)
        rfm_ds_scaled = pd.DataFrame(rfm_ds_scaled)
        rfm_ds_scaled.columns = ['Amount', 'Frequency','Recency']
        return rfm_ds_scaled

"end"