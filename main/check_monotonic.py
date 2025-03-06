import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Feb08_001054_best_test_predicted_result.csv')
def process_date(date_str):
    parts = date_str.split('.')
    year = parts[0]
    month = parts[1].zfill(2)  # 確保是兩位數
    day = parts[2].zfill(2)    # 確保是兩位數
    return f"{year}-{month}-{day}"

# 處理日期
df['date'] = df['timestamp'].apply(lambda x: process_date(x))
df = df[df['date'] == '2021-05-03']

def plot_strike_price_vs_estimated(df):
    call_options = df[df['cp_flag'] == 0]
    
    plt.figure(figsize=(12, 8))
    
    for (date, ttm), group in call_options.groupby(['date', 'time_to_maturity']):
        if len(group) > 1:
            # 關鍵修改：確保數據點按照履約價格排序
            group_sorted = group.sort_values('strike_price', ascending=True)
            
            plt.plot(group_sorted['strike_price'], 
                    group_sorted['estimated_price'], 
                    marker='o', 
                    label=f'Date: {date}, TTM: {ttm:.3f}')
    
    plt.title('Strike Price vs Estimated Price for Call Options')
    plt.xlabel('Strike Price')
    plt.ylabel('Estimated Price')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.subplots_adjust(right=0.85)
    plt.savefig(f'check_monotic_test.png')
    plt.close()

# 執行視覺化
plot_strike_price_vs_estimated(df)