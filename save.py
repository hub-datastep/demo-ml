import re
import numpy as np
import pandas as pd
import Levenshtein

def levenshtein_compare(str1, str2):
    return Levenshtein.ratio(str1, str2) * 100

class ModelPipeline:
    def __init__(self) -> None:
        self._project = 'Унистрой'
        self._dir      = 'week41'

        self._file_dataset_path = f"datasets/ynistroi.xlsx"
        self._file_dataset_sheet = "Sheet1"
        self._file_dataset_column = ["name", "group"]
        
        self._file_test_input_path = f"test-sets/Classifier_ unistroy UTDs test-cases (fix)_30.09.2024.xlsx"
        self._file_test_input_sheet = "test-cases"
        self._file_test_input_column = ["Номенклатура поставщика", "Ожидание группа"]

        self._file_test_output_path = f"Classifier_test-casse_unistqqroieed0fd.xlsx"
        self._file_test_output_column = ['name', 'group', 'vid']

        self._model_path = "model/model_unistroi.pkl"

        self._model = 0
        self._label_encoder = 0
        self._report=0
        self._conf_matrix=0
        self._execution_time=0
        self._accuracy=0

        self._X_train = 0
        self._X_test = 0
        self._y_train = 0
        self._y_test = 0
        self._texts = 0
        self._true_labels = 0

    
    def read_excel_columns(self, file_path, columns, sheet):
        try:
            df = pd.read_excel(file_path, usecols=columns, sheet_name=sheet).values.tolist()
            return df
        except FileNotFoundError:
            print(f"Ошибка: Файл '{file_path}' не найден.")
            return None
        except ValueError as e:
            print(f"Ошибка: Некорректные данные в файле или неверные колонки. {e}")
            return None
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            return None

        # try:
        #     result = df.values.tolist()
        #     data = np.array(result)
        #     # Разделение данных на номенклатуру и классы
        #     noms = data[:, 0]  # Номенклатура
        #     groups = data[:, 1]  # Классы
        # except Exception as e:
        #     print(f"Ошибка при преобразовании данных: {e}")
        #     return None

        # return noms, groups 

    def save_to_excel(self, data):
        try:
            # Преобразуем массив в DataFrame
            df = pd.DataFrame(data)
            
            # Сохраняем DataFrame в Excel файл
            df = pd.DataFrame(data, columns=self._file_test_output_column)
            df.to_excel(self._file_test_output_path, index=False)
            
            return True
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    

    def clean_text(self, text):
    # Удаляем пробелы в начале и конце строки, заменяем множественные пробелы на один
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        while cleaned_text[0]==' ':
            cleaned_text = cleaned_text[1:]
        return cleaned_text
    
    
    def run(self):
        c = 0
        dt = self.read_excel_columns('c:\\Users\\Lanutrix\\Downloads\\[ACTUAL] Материалы НСИ.xlsx', ["Материал","Группа", "Вид"], "Лист1")
        dt2 = self.read_excel_columns('c:\\Users\\Lanutrix\\Downloads\\[ACTUAL] LevelGroup_ training dataset (clear, with internal group, with view).xlsx', ["name","group", "view"], "nomenclatures")

        for i in range(len(dt)):
            mx = 0
            ans = ''
            for j in dt2:
                k = levenshtein_compare(dt[i][0],j[0])
                if k>mx:
                    mx = k
                    ans = j
            dt[i][1] = j[1]
            dt[i][2] = j[2]
        
        # dt = list(dt[0])
        # data = []
        # for i in range(len(dt)):
        #     dt[i] = self.clean_text(str(dt[i]))
        self.save_to_excel(dt)
p = '  djf '
print(p)

m = ModelPipeline()
print(m.clean_text(p))
f = m.run()