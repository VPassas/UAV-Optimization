"""Student A — Learned Channel Surrogate for UAV Placement.

Pipeline modules (filled in step by step):
    features   Step 3  -- extract_features(uav, user, city_geom) -> np.ndarray
    dataset    Step 4  -- (UAV, user) sampling + parallel rate generation
    model      Step 7  -- ChannelSurrogate (heteroscedastic Gaussian MLP)
    baselines  Step 6  -- mean / linear / RandomForest / KNN
"""
