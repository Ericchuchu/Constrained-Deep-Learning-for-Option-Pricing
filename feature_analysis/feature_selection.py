import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def calculate_vif(dataframe, features):
    # 選擇特徵
    X = dataframe[features].copy()

    # 檢查缺失值以及inf
    missing_count = X.isna().sum()
    inf_count = np.isinf(X).sum()

    if missing_count.sum() > 0 or inf_count.sum() > 0:
        print(f'存在缺失值或者inf')
        for col in features:
            miss = missing_count[col]
            inf = inf_count[col]
            if miss > 0 or inf > 0:
                print(f' - {col} : {miss} 缺失值 {inf} 無限值')

    # 計算 VIF
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    # 按 VIF 降序排序
    vif_data = vif_data.sort_values("VIF", ascending=False)
    
    return vif_data

def select_features_by_vif(dataframe, features, threshold=10.0, verbose=True):

    # 複製特徵列表，避免修改原始列表
    selected_features = features.copy()
    
    # 迭代移除 VIF 高的特徵
    max_vif = float('inf')
    iteration = 0
    
    while max_vif > threshold and len(selected_features) > 1:
        iteration += 1
        
        # 計算當前特徵的 VIF
        vif_data = calculate_vif(dataframe, selected_features)
        max_vif = vif_data["VIF"].max()
        
        if max_vif > threshold:
            # 移除 VIF 最高的特徵
            feature_to_drop = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
            selected_features.remove(feature_to_drop)
            
            if verbose:
                print(f"Iteration {iteration}: Dropped {feature_to_drop} with VIF = {max_vif:.2f}")
                print(f"Remaining features: {len(selected_features)}")
        
    # 計算最終的 VIF 值
    final_vif = calculate_vif(dataframe, selected_features)
    
    if verbose:
        print("\nFeature selection complete!")
        print(f"Initial features: {len(features)}")
        print(f"Selected features: {len(selected_features)}")
    
    return selected_features, final_vif

def plot_vif_results(initial_vif, final_vif):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 繪製初始 VIF
    sns.barplot(x="VIF", y="feature", data=initial_vif, palette="viridis", ax=ax1)
    ax1.set_title("Initial VIF Values", fontsize=14)
    ax1.axvline(x=10, color='red', linestyle='--', label='Threshold (VIF=10)')
    ax1.legend()
    
    # 繪製最終 VIF
    sns.barplot(x="VIF", y="feature", data=final_vif, palette="viridis", ax=ax2)
    ax2.set_title("Final VIF Values (After Selection)", fontsize=14)
    ax2.axvline(x=10, color='red', linestyle='--', label='Threshold (VIF=10)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 讀取數據 (請替換為您的實際數據路徑)
    df = pd.read_csv(r"/Users/chuchu/Desktop/option_pricing_thesis_project/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/prs_dataset_mpf.csv")
    
    # 選擇要考慮的數值特徵
    df = df[['volume','tau','impl_volatility','strike_price','delta','gamma','rho','theta','vega','pre_settle_price','settle_price_chg','S','theory_margin','theory_price','moneyness','option_price','time_value']]
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    initial_features = [f for f in numeric_features]
    
    print(f"Initial features for VIF analysis: {len(initial_features)}")
    
    # 計算初始 VIF
    initial_vif = calculate_vif(df, initial_features)
    print("\nInitial VIF values:")
    print(initial_vif)
    
    # 使用 VIF 選擇特徵 (閾值為 10)
    selected_features, final_vif = select_features_by_vif(df, initial_features, threshold=5.0)
    
    print("\nSelected features:")
    print(selected_features)
    
    print("\nFinal VIF values:")
    print(final_vif)
    
    # 繪製 VIF 結果
    plot_vif_results(initial_vif, final_vif)
    
    # 如果要保存選擇後的特徵數據
    selected_df = df[selected_features + ['option_price']]  # 加上目標變數
    print("\nSelected dataset shape:", selected_df.shape)
    
    # 可選：保存選擇後的特徵數據
    # selected_df.to_csv('selected_features_data.csv', index=False)

    # Selected features:
    # ['volume', 'tau', 'gamma', 'theta', 'pre_settle_price', 'settle_price_chg', 'theory_margin', 'theory_price', 'moneyness', 'time_value']