import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

Car_info = "./data/car.csv"
Cus_info = "./data/client.csv"
Con_info = "./data/contact.csv"


def hacarthon():
    buy = dict()            # 구객 구매 데이터
    check_result = dict()   # 최종 결과 확인용

    car_replace = dict()    # 차량 관점 대차
    car_add = dict()        # 차량 관점 추가구매

    cus_replace = dict()    # 고객 관점 대차
    cus_add = dict()        # 고객 관점 추가구매

    cus_data = dict()       # 고객 정보
    cus_contact = dict()    # 접촉 정보

    #차량 및 구매 정보 확인
    with open(Car_info, "r")as f:
        next(f)
        csv_reader = csv.reader(f)
        for line in csv_reader:
            if line[1] in buy:
                buy[line[1]].append([line[0], line[2], line[3], line[4], line[6], line[7]])
            else:
                buy[line[1]] = [[line[0], line[2], line[3], line[4], line[6], line[7]]]

    #대차 및 추가구매 분리
    for i in buy:
        if len(buy[i]) == 1:
            continue
        else:
            sorted_buy = sorted(buy[i], key=lambda x: x[1])
            buy_check = list()
            replace_data = [0, 0]   #대차 데이터 [대차 횟수, 대차 소요기간]
            add_data = [0, 0]       #추가구매 데이터 [추가구매 횟수, 추가구매 소요기간]
            for j in range(len(sorted_buy)):
                buy_check.append(False)
            buy_check[0] = True
            for j in range(len(sorted_buy)):
                if False in buy_check:
                    release_day = datetime.strptime(sorted_buy[j][1], "%Y%m%d")
                    end_day = sorted_buy[j][3]
                    if end_day == '': end_day = '99991231'
                    end_day = datetime.strptime(end_day, "%Y%m%d")
                    for k in range(j + 1, len(sorted_buy)):
                        period = np.nan
                        if buy_check[k]: continue
                        next_release_day = datetime.strptime(sorted_buy[k][1], "%Y%m%d")
                        if release_day <= next_release_day < end_day - timedelta(days=180): #추가구매
                            add_data[0] += 1
                            add_data[1] += (next_release_day - release_day).days
                            if sorted_buy[k][3] != '':
                                period = (datetime.strptime(sorted_buy[k][3], "%Y%m%d") -
                                          datetime.strptime(sorted_buy[k][1], "%Y%m%d")).days
                            car_add[sorted_buy[k][0]] = [period, sorted_buy[k][4], sorted_buy[k][5]]
                            buy_check[k] = True
                        else:
                            if end_day - timedelta(days=180) <= next_release_day <= end_day + timedelta(days=180): #대차
                                replace_data[0] += 1
                                replace_data[1] += (next_release_day - release_day).days
                                if sorted_buy[k][3] != '':
                                    period = (datetime.strptime(sorted_buy[k][3], "%Y%m%d") -
                                              datetime.strptime(sorted_buy[k][1], "%Y%m%d")).days
                                car_replace[sorted_buy[k][0]] = [period, sorted_buy[k][4], sorted_buy[k][5]]
                                buy_check[k] = True
                            break
            if add_data != [0, 0]:
                cus_add[i] = add_data
            if replace_data != [0, 0]:
                cus_replace[i] = replace_data
                check_result[i] = [datetime.strptime(str(sorted_buy[-1][1]), "%Y%m%d")]

    # 고객 정보 확인
    with open(Cus_info, "r") as f:
        next(f)
        csv_reader = csv.reader(f)
        for line in csv_reader:
            cus_info = list()
            if line[1] == 'Y':
                cus_info.append('YES')
            else:
                cus_info.append('NO')
            if line[2] == '남자':
                cus_info.append('MALE')
            else:
                cus_info.append('FEMALE')
            cus_info.append(datetime.today().year - int(line[3][:4]))
            cus_info.append(line[8])
            if cus_info[-1] == '':
                cus_info[-1] = np.nan
            cus_data[line[0]] = cus_info

    # 접촉 정보 확인
    with open(Con_info, "r") as f:
        next(f)
        csv_reader = csv.reader(f)
        contact_purpose = ['정비', '일반상담', '견적']
        for line in csv_reader:
            if line[4] in contact_purpose:
                if not line[1] in cus_contact:
                    cus_contact[line[1]] = [0, 0, 0]
                cus_contact[line[1]][contact_purpose.index(line[4])] += 1

    num_value = [1, 6, 7, 8]    # 숫자로 사용되는 정보 확인 후에 전처리
    for i in cus_add:           # 대차 데이터 정리
        for j in cus_data[i]:
            cus_add[i].append(j)
        if not i in cus_contact:
            cus_add[i] += [0, 0, 0]
        else:
            for j in cus_contact[i]:
                cus_add[i].append(j)
        if cus_add[i][0] != 1:
            for j in num_value:
                cus_add[i][j] /= cus_add[i][0]
        cus_add[i].append(i)

    id_list = list()            # 예측모델에서 아이디 확인을 위해 사용
    for i in cus_replace:       # 추가구매 데이터 정리
        id_list.append(i)
        for j in cus_data[i]:
            cus_replace[i].append(j)
        if not i in cus_contact:
            cus_replace[i] += [0, 0, 0]
        else:
            for j in cus_contact[i]:
                cus_replace[i].append(j)
        if cus_replace[i][0] != 1:
            for j in num_value:
                cus_replace[i][j] /= cus_replace[i][0]
        cus_replace[i].append(i)

    #데이터프레임(고객 관점)
    df_cus_add = pd.DataFrame(cus_add).transpose().rename(
        columns={0: 'count', 1: 'length', 2: 'private', 3: 'gender', 4: 'age', 5: 'money',
                 6: 'maintenance', 7: 'counseling', 8: 'estimate', 9: 'cus_id'})
    df_cus_add = df_cus_add.astype({'count': 'int', 'length': 'float', 'age': 'int', 'money': 'float',
                                    'maintenance': 'float', 'counseling': 'float', 'estimate': 'float'})
    df_cus_replace = pd.DataFrame(cus_replace).transpose().rename(
        columns={0: 'count', 1: 'length', 2: 'private', 3: 'gender', 4: 'age', 5: 'money',
                 6: 'maintenance', 7: 'counseling', 8: 'estimate', 9: 'cus_id'})
    df_cus_replace = df_cus_replace.astype({'count': 'int', 'length': 'float', 'age': 'int', 'money': 'float',
                                            'maintenance': 'float', 'counseling': 'float', 'estimate': 'float'})
    #데이터프레임(차량 관점)
    df_car_replace = pd.DataFrame(car_replace).transpose().rename(columns={0: 'length', 1: 'GCRD1', 2: 'GCRD2'})
    df_car_replace = df_car_replace.astype({'length': 'float'})
    df_car_add = pd.DataFrame(car_add).transpose().rename(columns={0: 'length', 1: 'GCRD1', 2: 'GCRD2'})
    df_car_add = df_car_add.astype({'length': 'float'})

    print(df_cus_replace.corr())
    print(df_cus_add.corr())

    plt.rc('font', family='Malgun Gothic')      #한글 깨짐 현상 해결
    plt.rc('axes', unicode_minus=False)

    sns.countplot(data=df_cus_replace, x='gender')
    plt.show()
    sns.countplot(data=df_cus_replace, x='private')
    plt.show()
    sns.countplot(data=df_cus_add, x='gender')
    plt.show()
    sns.countplot(data=df_cus_add, x='private')
    plt.show()

    sns.countplot(df_car_replace['GCRD1'])
    plt.show()
    sns.countplot(df_car_add['GCRD1'])
    plt.show()
    sns.countplot(df_car_replace['GCRD2'])
    plt.show()
    sns.countplot(df_car_add['GCRD2'])
    plt.show()

    house_money_avg = df_cus_replace['money'].mean()    #결측값 처리
    df_cus_replace = df_cus_replace.fillna(house_money_avg)

    train_df = df_cus_replace
    test_df = df_cus_replace

    x_train = train_df.drop(['length', 'cus_id'], axis=1)
    x_test = test_df.drop(['length', 'cus_id'], axis=1)
    y_train = train_df['length']
    y_test = test_df['length']
    check_id = df_cus_replace['cus_id']

    transformer = make_column_transformer((OneHotEncoder(), ['private', 'gender']), remainder='passthrough')
    transformer.fit(x_train)
    x_train = transformer.transform(x_train)
    x_test = transformer.transform(x_test)

    model = LinearRegression()  # 모델 생성
    model.fit(x_train, y_train)  # 모델 학습
    print("모델 검증 : ", model.score(x_test, y_test))  # 모델 검증

    ##########모델 예측
    x_test = test_df.drop(['length', 'cus_id'], axis=1)
    x_test = pd.DataFrame(x_test, columns=['count', 'private', 'gender', 'age', 'money', 'maintenance', 'counseling',
                                           'estimate'])
    x_test = transformer.transform(x_test)
    y_predict = model.predict(x_test)

    for i in range(len(y_predict)):
        check_result[id_list[i]].append(y_predict[i])

    predict_true, predict_false = 0, 0
    for i in check_result:
        predict_day = check_result[i][0] + timedelta(days=int(check_result[i][1]))
        if datetime.today() <= predict_day <= datetime.today() + timedelta(days=180):
            check_result[i].append(True)
            predict_true += 1
        else:
            check_result[i].append(False)
            predict_false += 1
    print("6개월내 대차 True : ", predict_true)
    print("6개월내 대차 False : ", predict_false)


if __name__ == '__main__':
    hacarthon()

