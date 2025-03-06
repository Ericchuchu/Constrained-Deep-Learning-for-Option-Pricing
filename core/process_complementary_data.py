import zipfile
import os
import pandas as pd

def get_expiry_date(date_str):
    date_str = str(date_str)
    # w2
    date_str = str(date_str)
    first_wednesday = 'w1' in date_str.lower()
    date_str = date_str.lower().replace('w1', '')
    second_wednesday = 'w2' in date_str.lower()
    date_str = date_str.lower().replace('w2', '')
    fourth_wednesday = 'w4' in date_str.lower()
    date_str = date_str.lower().replace('w4', '')

    year = int(date_str[:4])
    month = int(date_str[4:6])
    first_day = pd.Timestamp(year=year, month=month, day=1)
    
    # 找出該月所有的禮拜三
    wednesdays = []
    curr_day = first_day
    while curr_day.month == month:
        if curr_day.dayofweek == 2:  # 2 表示禮拜三
            wednesdays.append(curr_day)
        curr_day += pd.Timedelta(days=1)

    if first_wednesday:
        exdate = wednesdays[0]
    elif second_wednesday:
        exdate = wednesdays[1]
    elif fourth_wednesday:
        exdate = wednesdays[3]
    else:
        exdate = wednesdays[2]
    return exdate
        
def preprocess_csv(file_path, output_folder):
    data = pd.read_csv(file_path, encoding='big5')
    data = data[data['是否因訊息面暫停交易'] == "一般"]
    data = data.dropna(subset=['契約', '到期月份(週別)', '履約價', '成交量'])
    adjusted_data = pd.DataFrame(index=range(len(data)))
    adjusted_data['date'] = pd.to_datetime(data.index)
    adjusted_data['exdate'] = data['契約'].apply(get_expiry_date).reset_index(drop=True)
    adjusted_data['strike_price'] = data['到期月份(週別)'].reset_index(drop=True)
    adjusted_data['cp_flag'] = data['履約價'].map({'買權':'C','賣權':'P'}).reset_index(drop=True)
    adjusted_data['option_price'] = pd.to_numeric(data['成交量'], errors='coerce').reset_index(drop=True)
    adjusted_data = adjusted_data.dropna(subset=['option_price','exdate']).reset_index(drop=True)
    
    output_file = os.path.join(output_folder, os.path.basename(file_path))
    adjusted_data.to_csv(output_file, index=False)
    print(f"已儲存處理後檔案：{output_file}")

zip_file_path = r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/2021_opt.zip"
output_folder = r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/2021_opt_ajusted"
temp_folder = r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/temp_extracted"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(temp_folder, exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
    for file_name in zip_file.namelist():
        if file_name.endswith('.csv'):
            temp_file_path = zip_file.extract(file_name, path=temp_folder)     
            try:
                preprocess_csv(temp_file_path, output_folder)
            finally:
                os.remove(temp_file_path)
                
print("預處理完成")