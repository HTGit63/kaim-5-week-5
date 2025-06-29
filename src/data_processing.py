import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer

def create_rfm_labels(raw_csv_path: str, output_csv_path: str):
    """
    Reads raw transaction data, computes RFM features per customer,
    clusters customers into 3 segments, labels the highest-risk segment,
    and writes the full transaction-level data with an 'is_high_risk' column.
    """
    # 1. Load data & parse dates
    df = pd.read_csv(raw_csv_path, parse_dates=['TransactionStartTime'])

    # 2. Compute snapshot date (day after last transaction)
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

    # 3. Aggregate RFM per customer
    rfm = (
        df.groupby('CustomerId')
          .agg(
              Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
              Frequency=('TransactionId', 'count'),
              Monetary=('Value', 'sum')
          )
    )

    # 4. Scale RFM features for clustering
    scaler = StandardScaler()
    rfm_scaled = pd.DataFrame(
        scaler.fit_transform(rfm),
        index=rfm.index,
        columns=rfm.columns
    )

    # 5. Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # 6. Identify the high-risk cluster
    cluster_summary = (
        rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']]
           .mean()
    )
    # Sort: highest Recency, lowest Frequency, lowest Monetary
    high_risk_cluster = (
        cluster_summary
        .sort_values(
            by=['Recency','Frequency','Monetary'],
            ascending=[False, True, True]
        )
        .index[0]
    )

    # 7. Create binary label
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)

    # 8. Merge label back to transaction-level data
    df = df.merge(
        rfm['is_high_risk'].rename('is_high_risk'),
        how='left',
        on='CustomerId'
    )

    # 9. Save the result
    df.to_csv(output_csv_path, index=False)
    print(f"Processed data with labels saved to: {output_csv_path}")

if __name__ == "__main__":
    create_rfm_labels(
        raw_csv_path="data/raw/data.csv",
        output_csv_path="data/processed/with_labels.csv"
    )

def create_rfm_labels(df: pd.DataFrame) -> pd.Series:
    snapshot = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby('CustomerId')
          .agg(
             Recency=('TransactionStartTime', lambda x: (snapshot - x.max()).days),
             Frequency=('TransactionId', 'count'),
             Monetary=('Value', 'sum')
          )
    )
    # Scale & cluster
    rfm_scaled = StandardScaler().fit_transform(rfm)
    clusters = KMeans(n_clusters=3, random_state=42).fit_predict(rfm_scaled)
    rfm['Cluster'] = clusters
    # Identify high-risk cluster
    prof = rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean()
    high = prof.sort_values(
        by=['Recency','Frequency','Monetary'],
        ascending=[False,True,True]
    ).index[0]
    return (rfm['Cluster'] == high).astype(int)

def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df['TransactionStartTime']
    return pd.DataFrame({
        'hour':  dt.dt.hour,
        'day':   dt.dt.day,
        'month': dt.dt.month,
        'year':  dt.dt.year
    })

def build_simple_pipeline(categorical_cols, numerical_cols):
    """
    Pipeline that imputes + ordinal-encodes categoricals,
    and imputes + scales numericals.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ord',     OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    return ColumnTransformer([
        ('cat', cat_pipe, categorical_cols),
        ('num', num_pipe, numerical_cols)
    ])

def main():
    # 1. Load raw data
    df = pd.read_csv("data/raw/data.csv", parse_dates=['TransactionStartTime'])

    # 2. Compute RFM-based risk label
    df['is_high_risk'] = create_rfm_labels(df)

    # 3. Extract datetime features separately
    dt_feats = extract_datetime_features(df)

    # 4. Define which columns to pipeline
    categorical_cols = [
        'BatchId', 'AccountId', 'SubscriptionId', 'CurrencyCode',
        'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId',
        'PricingStrategy'
    ]
    numerical_cols = ['Amount', 'Value']

    # 5. Build & apply the pipeline
    pipeline = build_simple_pipeline(categorical_cols, numerical_cols)
    X_catnum = pipeline.fit_transform(df)

    # 6. Assemble final feature DataFrame
    #    cat+num first, then datetime
    catnum_cols = categorical_cols + numerical_cols
    X_df = pd.DataFrame(X_catnum, columns=catnum_cols)
    final_df = pd.concat([X_df.reset_index(drop=True),
                          dt_feats.reset_index(drop=True)],
                         axis=1)

    # 7. Save features & labels
    final_df.to_csv("data/processed/features.csv", index=False)
    df[['is_high_risk']].to_csv("data/processed/labels.csv", index=False)

    print("✅ Features and labels saved to data/processed/")

if __name__ == "__main__":
    main()