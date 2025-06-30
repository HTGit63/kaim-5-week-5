import pandas as pd
import logging
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_rfm_labels(df: pd.DataFrame) -> pd.Series:
    """
    Given the full transaction DataFrame, compute RFM per CustomerId,
    cluster into 3 segments, and return a Series is_high_risk (0/1),
    indexed by CustomerId.
    """
    logging.info("Computing RFM metrics")
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby('CustomerId')
          .agg(
              Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
              Frequency=('TransactionId', 'count'),
              Monetary=('Value', 'sum')
          )
    )

    # Scale & cluster
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Identify high‑risk cluster
    cluster_summary = rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean()
    high_risk_cluster = (
        cluster_summary
        .sort_values(by=['Recency','Frequency','Monetary'], ascending=[False,True,True])
        .index[0]
    )

    # Create label
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)

    # Return only the label series, indexed by CustomerId
    return rfm['is_high_risk']

def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract hour, day, month, year from TransactionStartTime.
    """
    logging.info("Extracting datetime features")
    dt = df['TransactionStartTime']
    return pd.DataFrame({
        'hour':  dt.dt.hour,
        'day':   dt.dt.day,
        'month': dt.dt.month,
        'year':  dt.dt.year
    })

def build_simple_pipeline(categorical_cols, numerical_cols):
    """
    Build a ColumnTransformer that:
      - imputes + ordinal-encodes categoricals,
      - imputes + scales numericals.
    """
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
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 1. Load raw data
    logging.info("Loading raw data")
    df = pd.read_csv("data/raw/data.csv", parse_dates=['TransactionStartTime'])

    # 2. Compute RFM‑based risk label series
    logging.info("Computing is_high_risk label series")
    label_series = create_rfm_labels(df)  # Series indexed by CustomerId

    # 3. Merge that series onto the transaction-level df
    logging.info("Merging is_high_risk labels back to transactions")
    df = df.merge(
        label_series.rename('is_high_risk'),
        left_on='CustomerId',
        right_index=True,
        how='left'
    )

    # 4. Extract datetime features
    dt_feats = extract_datetime_features(df)

    # 5. Define feature columns
    categorical_cols = [
        'BatchId', 'AccountId', 'SubscriptionId', 'CurrencyCode',
        'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId',
        'PricingStrategy'
    ]
    numerical_cols = ['Amount', 'Value']

    # 6. Build & apply pipeline
    logging.info("Building feature pipeline")
    pipeline = build_simple_pipeline(categorical_cols, numerical_cols)
    logging.info("Transforming categorical + numerical features")
    X_catnum = pipeline.fit_transform(df)

    # 7. Assemble final feature DataFrame
    logging.info("Assembling final feature DataFrame")
    catnum_cols = categorical_cols + numerical_cols
    X_df = pd.DataFrame(X_catnum, columns=catnum_cols)
    final_df = pd.concat([X_df.reset_index(drop=True),
                          dt_feats.reset_index(drop=True)],
                         axis=1)

    # 8. Save outputs
    logging.info("Saving features.csv and labels.csv")
    final_df.to_csv("data/processed/features.csv", index=False)
    df[['is_high_risk']].to_csv("data/processed/labels.csv", index=False)

    logging.info("Data processing complete")

if __name__ == "__main__":
    main()
