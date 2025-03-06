import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def df_to_image(df, output_file):

    fig, ax = plt.subplots(figsize=(20, 10))
    data = df.to_numpy()
    # 創建自定義的色彩映射
    cmap = mcolors.ListedColormap(['white', 'black'])
    bounds = [0, 0.5, 1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')


    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)

    ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)

# 文件路径
output_file_path = r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/2021_opt_01_adjusted.csv"

data = pd.read_csv(output_file_path)
data['date'] = pd.to_datetime(data['date'])
data['exdate'] = pd.to_datetime(data['exdate'])
data['opt_ID'] = data.groupby(['strike_price', 'exdate','cp_flag']).ngroup()
data = data.sort_values(by=['opt_ID']).reset_index(drop=True)

data = data[data['date'] >= '2021-01-01']
unique_dates = data['date'].unique()
unique_dates = np.sort(unique_dates)

opt_ids = data['opt_ID'].unique()
opt_timeline_df = pd.DataFrame(index=opt_ids, columns=unique_dates, dtype=float)

for opt_id in opt_ids:
    opt_data = data[data['opt_ID'] == opt_id]
    for date in unique_dates:
        if date in opt_data['date'].values:
            price = opt_data[opt_data['date'] == date]['option_price'].values[0]
            if pd.isna(price):
                opt_timeline_df.at[opt_id, date] = 0
            else:
                opt_timeline_df.at[opt_id, date] = 1
        else:
            opt_timeline_df.at[opt_id, date] = 0

opt_timeline_df = opt_timeline_df[:50]
print(opt_timeline_df)

# 調用函數,將 DataFrame 轉換為圖像
df_to_image(opt_timeline_df, 'option_price_timeline_raw_data.png')

