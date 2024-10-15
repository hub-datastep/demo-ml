import Levenshtein
import pandas as pd
import numpy as np
import re


def process_text(text):
    # 1. Заменяем все символы, кроме русских букв, на пробелы
    clean_text = re.sub(r'[^а-яА-ЯёЁa-zA-Z]', ' ', text)
    
    # 2. Разбиваем строку на слова
    words = clean_text.split()
    
    # 3. Сортируем слова в алфавитном порядке
    sorted_words = [i for i in sorted(words, key=lambda word: word.lower()) if len(i)>2]
    
    return ' '.join(sorted_words)

def levenshtein_compare(str1, str2):
    return Levenshtein.ratio(str1, str2) * 100

def get_nsi_data():
    sheets = pd.read_excel('c:\\Users\\Dmitry\\Downloads\\ynistroi.xlsx', sheet_name=None)

    arrays = {sheet_name: pd.DataFrame(df) for sheet_name, df in sheets.items()}

    first_sheet_name = list(arrays.keys())[0]
    training_data_df = arrays[first_sheet_name]

    selected_columns = ['name','group']
    training_data_df = training_data_df[selected_columns]
    # Пример данных
    data = np.array(training_data_df)
    # Разделение данных на номенклатуру и классы
    # X = data[:, 0]  # Номенклатура
    # y = data[:, 1]  # Классы
    return data

def get_test_data():
    sheets = pd.read_excel('c:\\Users\\Dmitry\\Downloads\\Classifier_ unistroy UTDs test-cases.xlsx', sheet_name=None)

    arrays = {sheet_name: pd.DataFrame(df) for sheet_name, df in sheets.items()}

    first_sheet_name = list(arrays.keys())[0]
    training_data_df = arrays[first_sheet_name]

    selected_columns = ['Номенклатура поставщика']
    training_data_df = training_data_df[selected_columns]
    # Пример данных
    data = np.array(training_data_df)
    # Разделение данных на номенклатуру и классы
    X = data[:, 0]  # Номенклатура
    # y = data[:, 1]  # Классы
    return list(X)

def save_to_excel(data, column_names=['номенклатура', "сопоставленная группа"], file_name='нечёткое_сравнение2.xlsx'):
    try:
        # Преобразуем массив в DataFrame
        df = pd.DataFrame(data)
        
        # Сохраняем DataFrame в Excel файл
        df = pd.DataFrame(data, columns=column_names)
        df.to_excel(file_name, index=False)
        
        return True
    except Exception as e:
        print(f"An error occurred: {e}")

data = get_test_data()
data = sorted(data, key=lambda word: word.lower())
data2 = list(set([process_text(i) for i in data]))
print(len(data2), len(data))


xlsx = get_nsi_data()
print(len(xlsx))
xlsx = list(i.split('||') for i in set([process_text(i[0]) + '||' + i[1] for i in xlsx]))
print(len(xlsx))

f = {}
for i in data:
    f[i] = {} # {"group1": [score1, score2], "group2": [score3, score4]}
    for x in xlsx:
        j = x[0]
        similarity_score = levenshtein_compare(j, process_text(i))
        if similarity_score > 0.1:
            try:
                f[i][x[1]].append(similarity_score)
            except:
                f[i][x[1]] = [similarity_score]
output = []
for i in f:
    val = list(f[i].keys())
    if len(val)==0:
        output.append([i, '-'])
    elif len(val)==1:
        output.append([i, val[0]])
    else:
        group = ''
        mx = 0
        for j in f[i]:
            m = sum(f[i][j])/len(f[i][j])
            if m > mx:
                group = j
                mx = m
        
print('---------')
# save_to_excel(output)
