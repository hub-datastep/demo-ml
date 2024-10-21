import re
import joblib
import numpy as np
# Загрузка модели
m = joblib.load('model/model_unistroi_LogisticRegression+syntetics+clear_big_class+not_lestnitsa.pkl')

model = m['model']  # Это загруженная модель, например, GridSearchCV или другая обученная модель

label_encoder = m['label_encoder']
new_class = 'Лестничные марши и площадки железобетонные Материалы'
if new_class not in label_encoder.classes_:
    label_encoder.classes_ = np.append(label_encoder.classes_, new_class)

label_lestnitsa = label_encoder.transform([new_class])[0]

# Определение нового класса с переопределением только метода predict
class ModelAI:
    def __init__(self, model):
        self.model = model

    def predict(self, dt):
        output = []
        for t in dt:
            text = t.lower()
            ss = re.sub(r"шткм|[^ ,.\-авдклмнптуфш0-9]", "", text) == text
            if ss:
                output.append(label_lestnitsa)
            else:
                output.append(self.model.predict([t])[0])
        return output

    def __getattr__(self, attr):
        return getattr(self.model, attr)

dt = '''Кирпич силикатный пустотелый 2,0-НФ 250х120х103	Кирпич
Кирпич силикатный рядовой 0,5-НФ/М250/F150	Кирпич
Кирпич силикатный рядовой NF 240х120х71	Кирпич
Кирпич стеновой КСЛ-ПР 250х120х88/F50/M200	Кирпич
16ЛМ 28-12	Лестничные марши и площадки железобетонные Материалы
17ЛМ28-11п	Лестничные марши и площадки железобетонные Материалы
1ЛМ27-11-14-4	Лестничные марши и площадки железобетонные Материалы'''.split('\n')
q = []
answer = []
for i in [i.split('	') for i in dt]:
    q.append(i[0])
    answer.append(i[1])

# Создаем объект ModelAI с переданной моделью
ml = ModelAI(model)


answer_ai = label_encoder.inverse_transform(ml.predict(q))  # Пример использования
for i in range(len(answer_ai)):
    print(q[i], ';', answer[i],': ', answer_ai[i], " - ", answer[i] == answer_ai[i])
