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
output_file_path = r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/core/filled_time_series_underlying_data.csv"

data = pd.read_csv(output_file_path)

data['opt_ID'] = data.groupby(['strike_price', 'exdate', 'PC']).ngroup()
data = data.sort_values(by=['opt_ID', 'date']).reset_index(drop=True)
data = data[data['date'] >= '2021-01-01']
unique_dates = data['date'].unique()
unique_dates = np.sort(unique_dates)

exdates = data['exdate'].unique()
underlying_timeline_df = pd.DataFrame(index=exdates, columns=unique_dates, dtype=float)

for exdate in exdates:
    opt_data = data[data['exdate'] == exdate]
    for date in unique_dates:
        if date in opt_data['date'].values:
            price = opt_data[opt_data['date'] == date]['S'].values[0]
            if pd.isna(price):
                underlying_timeline_df.at[exdate, date] = 0
            else:
                underlying_timeline_df.at[exdate, date] = 1
        else:
            underlying_timeline_df.at[exdate, date] = 0

print(underlying_timeline_df)

# 調用函數,將 DataFrame 轉換為圖像
df_to_image(underlying_timeline_df, 'option_underlting_timeline.png')

