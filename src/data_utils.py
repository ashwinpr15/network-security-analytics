import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def generate_network_data(n_samples=5000, include_categorical=True):
    """
    Generates synthetic network traffic data mimicking NSL-KDD features.
    Includes both continuous metrics and categorical metadata.
    """
    # 1. Continuous features
    data = {
        'duration': np.random.exponential(scale=10, size=n_samples),
        'src_bytes': np.random.lognormal(mean=5, sigma=1, size=n_samples),
        'dst_bytes': np.random.lognormal(mean=6, sigma=1, size=n_samples),
        'wrong_fragment': np.random.randint(0, 3, n_samples),
        'count': np.random.randint(1, 500, n_samples),
        'srv_count': np.random.randint(1, 500, n_samples),
        'serror_rate': np.random.rand(n_samples),
        'rerror_rate': np.random.rand(n_samples)
    }

    df = pd.DataFrame(data)

    # 2. Categorical features (Mimicking NSL-KDD)
    if include_categorical:
        protocols = ['tcp', 'udp', 'icmp']
        services = ['http', 'ftp', 'smtp', 'dns', 'other']
        flags = ['SF', 'S0', 'REJ', 'RSTR']
        
        df['protocol_type'] = np.random.choice(protocols, size=n_samples, p=[0.7, 0.25, 0.05])
        df['service'] = np.random.choice(services, size=n_samples)
        df['flag'] = np.random.choice(flags, size=n_samples)

    # 3. Target variable (Attack vs Normal)
    # Logic: High error rates + specific protocols = Higher Risk
    risk_factor = (df['serror_rate'] * 3) + (df['count'] / 200) + (df['src_bytes'] > 5000).astype(int)
    
    if include_categorical:
        risk_factor += (df['flag'] == 'S0').astype(int) * 2  # SYN flood pattern
    
    threshold = risk_factor.median()
    df['label'] = (risk_factor + np.random.normal(0, 0.5, n_samples) > threshold).astype(int)

    return df

def preprocess_data(df):
    """
    Scales features and One-Hot Encodes categoricals.
    Returns: X_reshaped (for CNN), y, preprocessor
    """
    y = df['label'].values
    X = df.drop(columns=['label'])

    # Identify columns
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Define Transformers
    transformers = [
        ('num', MinMaxScaler(), numeric_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
    ]

    preprocessor = ColumnTransformer(transformers)
    X_transformed = preprocessor.fit_transform(X)

    # Reshape for Conv1D: (Samples, Features, 1)
    X_reshaped = X_transformed.reshape(X_transformed.shape[0], X_transformed.shape[1], 1)

    return X_reshaped, y, preprocessor
