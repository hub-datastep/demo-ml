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

dt = '''Плита экструзионная Carbon ECO 1180x580x30 мм (0,267 м3/уп)
Плита экструзионная XPS Carbon Prof (1180x580x100L) 4шт/уп 0,27376м3 (13,1405м3/пал)
Виллатекс Изол С ТПП 3,0 15 м2 материал рулонный гидроизоляционный (R=25мм -5 град, теплостойкость +85 град)
Гидроизоляционный кровельный материалТехноэласт ЭКП(10м2)
Гидроизоляционный материал Техноэласт ЭПП (10м2)
Изделия строительные из камня природного (полнотелые-100) 667*500*100 (ИСКП)
Икопал В ЭКП 5,0 10 м2 материал рулонный кровельный (R=25мм -25 град, теплостойкость +100 град)
Икопал Н ЭПП 4.0 10 м2 материал рулонный гидроизоляционный (R=25мм -25 град, теплостойкость +100 град)
Металлоконструкция навеса над входами, без профнастила и подсистемы вентилируемого фасада. ГОСТ 23118, СНиП 31-01 СП 31-107
Направляющая L-образная КПС 1271 (толщина 1,8), БП,6
Направляющая Т-образная КПС 1270 (80*60 толщина 1,8), БП,6
Плита экструзионная XPS Carbon Prof (1180x580x50L) 8шт/уп 0,27376 m3 (13,1405м3/пал)
Транспортные услуги миксера
Угловой элемент плитки для НФС King Stone "Венский кирпич",KS VG 0518 FK VF,283*137*85 мм,(левый)
Угловой элемент плитки для НФС King Stone "Венский кирпич",KS VG 518 FK VF,280*135*85 мм, шт,(правый
Удлинитель кронштейна несущего, КПС 306-1, вылет 125мм
Удлинитель кронштейна опорного, КПС 306-1, вылет 125мм'''.split('\n')
# q = []
# answer = []
# for i in [i.split('	') for i in dt]:
#     q.append(i[0])
#     answer.append(i[1])

# Создаем объект ModelAI с переданной моделью
ml = ModelAI(model)


answer_ai = label_encoder.inverse_transform(model.predict(dt))  # Пример использования
for i in range(len(answer_ai)):
    print( answer_ai[i])

# m['model'] = ml

# joblib.dump(m, 'model/model_unistroi_LogisticRegression+syntetics+clear_big_class+not_lestnitsa.pkl')
m['model'] = ml
m['label_encoder'] = label_encoder
data_to_save = {
        "model": model,
        "label_encoder": label_encoder,
        "report": m["report"],
        "conf_matrix": m["conf_matrix"],
        "execution_time": m["execution_time"],
        "accuracy": m["accuracy"]
        }
joblib.dump(data_to_save, 'model/model_unistroi_LogisticRegression+syntetics+clear_big_class+not_lestnitsa2.pkl')
# import joblib

# joblib.load( 'model/model_unistroi_LogisticRegression+syntetics+clear_big_class+not_lestnitsa2.pkl')