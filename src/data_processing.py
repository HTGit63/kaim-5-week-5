import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
