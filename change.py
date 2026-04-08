import pandas as pd



# data = pd.read_csv("/home/zhanglongyang/Desktop/SWaTDec2015/Physical/SWaT_Dataset_Attack_v0.csv", index_col=False)

# Extract rows 227,831 through 228,231 (skiprows to skip first 227,830 rows, nrows to get 401 rows)
extracted_data = pd.read_csv("/home/zhanglongyang/Desktop/SWaTDec2015/Physical/SWaT_Dataset_Attack_v0.csv", skiprows=range(1, 227829), nrows=403)

# Save to new CSV file
extracted_data.to_csv("extracted_rows_227831_228231.csv", index=False, header=True)




# data = pd.read_excel("List_of_attacks_Final.xlsx")

# data.to_csv("List_of_attacks_Final.csv", index=False)

