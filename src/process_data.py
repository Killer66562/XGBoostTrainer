import pandas as pd

from sklearn.model_selection import train_test_split


def main():
    df_data = pd.read_csv("datasets/heart_2020_cleaned.csv")
    # 移除不需要的欄位
    df_data = df_data.drop(columns=['PhysicalHealth', 'MentalHealth', 'Race' , 'GenHealth'])


    # 定義映射
    HeartDisease_map = {'Yes': 1, 'No': 0}
    Smoking_map = {'Yes': 1, 'No': 0}
    AlcoholDrinking_map = {'Yes': 1, 'No': 0}
    Stroke_map = {'Yes': 1, 'No': 0}
    DiffWalking_map = {'Yes': 1, 'No': 0}
    Sex_map = {'Male': 0, 'Female': 1}
    AgeCategory_map = {
                        '0-4': 0,
                        '5-9': 1,
                        '10-14': 2,
                        '15-17': 3,
                        '18-24': 4,
                        '25-29': 5,
                        '30-34': 6,
                        '35-39': 7,
                        '40-44': 8,
                        '45-49': 9,
                        '50-54': 10,
                        '55-59': 11,
                        '60-64': 12,
                        '65-69': 13,
                        '70-74': 14,
                        '75-79': 15,
                        '80 or older': 16
                    }
    Diabetic_map = {'Yes (during pregnancy)':1 ,'Yes': 1, 'No': 0, 'No, borderline diabetes':0 }
    PhysicalActivity_map = {'Yes': 1, 'No': 0}
    Asthma_map = {'Yes': 1, 'No': 0}
    KidneyDisease_map = {'Yes': 1, 'No': 0}
    SkinCancer_map = {'Yes': 1, 'No': 0} 

    # 補齊資料
    df_data['HeartDisease'] = df_data['HeartDisease'].map(HeartDisease_map)
    df_data['Smoking'] = df_data['Smoking'].map(Smoking_map) 
    df_data['AlcoholDrinking'] = df_data['AlcoholDrinking'].map(AlcoholDrinking_map) 
    df_data['Stroke'] = df_data['Stroke'].map(Stroke_map) 
    df_data['DiffWalking'] = df_data['DiffWalking'].map(DiffWalking_map) 
    df_data['Sex'] = df_data['Sex'].map(Sex_map) 
    df_data['AgeCategory'] = df_data['AgeCategory'].map(AgeCategory_map) 
    df_data['Diabetic'] = df_data['Diabetic'].map(Diabetic_map) 
    df_data['PhysicalActivity'] = df_data['PhysicalActivity'].map(PhysicalActivity_map)
    df_data['Asthma'] = df_data['Asthma'].map(Asthma_map) 
    df_data['KidneyDisease'] = df_data['KidneyDisease'].map(KidneyDisease_map) 
    df_data['SkinCancer'] = df_data['SkinCancer'].map(SkinCancer_map) 

    # 將 'Sex' 和 'AgeCategory' 欄位分別移到 DataFrame 的第一和第二欄
    columns_order = ['Sex', 'AgeCategory'] + [col for col in df_data.columns if col not in ['Sex', 'AgeCategory']]
    df_data = df_data[columns_order]

    X = df_data.drop(labels=['HeartDisease'], axis=1)
    Y = df_data[['HeartDisease']]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train.to_csv("datasets/x_train.csv", index=False)
    X_test.to_csv("datasets/x_test.csv", index=False)
    Y_train.to_csv("datasets/y_train.csv", index=False)
    Y_test.to_csv("datasets/y_test.csv", index=False)
    
if __name__ == '__main__':
    main()