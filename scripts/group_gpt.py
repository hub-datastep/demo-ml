import re
import pandas as pd
from datetime import datetime
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI


DOCS_PROMPT_TEMPLATE = """# Роль
инженер ПТО (производственно-технического отдела)

# Данные которые получишь на вход:
- список номенклатур из спецификации
- группа с обобщенной формулировкой, к которой подходят все номенклатуры
- список более точных групп из которых надо будет выбрать группу для каждой номенклатуры

Формат входных данных:
{
"nomenclatures": ["номенклатура1", "номенклатура2", "номенклатура3"],
"generalized group":"обобщённая группа",
"precise groups" : ["точная группа1", "точная группа2"]
}


# Задача
выбрать наиболее подходящую точную группу к обобщенной группе и каждой отдельной номенклатуре

# Не забывай что:
1. Посмотри на пример номенклатур в каждом классе:
{json_example}

2. Если номенклатура не подходит к группе писать “ошибка сопоставления”
3. Обращать внимание на параметры в номенклатуре
4. Отвечай кратко
5. Ответ в формате json:
{
 "1": "точная группа для номенклатуры 1",
 "2": "точная группа для номенклатуры 2",
 "3": "точная группа для номенклатуры 3"
}


# Пример
Пример входных данных:
{
  "nomenclatures": ["труба 100*150 противопожарная", "труба 200*150 противопожарная", "труба 200*150 d=25 для кранов и санузлов"],
  "generalized group":"трубы",
  "precise groups" : ["трубы металлические","трубы противопожарные", "трубы сантехнические"]
}


Пример правильного ответа:
{
 "1": "трубы противопожарные",
 "2": "трубы противопожарные",
 "3": "трубы сантехнические"
}


# Вход
{
  "nomenclatures": {nomenclatures},
  "generalized group": {generalized_group},
  "precise groups" : {precise_groups}
} 
"""

def get_emergency_class_chain():
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4-0125-preview-db-assistant",
        temperature=0,
        verbose=False,
    
    )

    prompt = PromptTemplate(
        template=DOCS_PROMPT_TEMPLATE,
        input_variables=["query"]
    )

    return LLMChain(llm=llm, prompt=prompt)

def normalize_resident_request_string(text):
    return (lambda s: re.sub(r'\s+', ' ', re.sub(r'http\S+', '', s.replace('\n', ' ').replace('Прикрепите фото:', ''))))(text)

def get_emergency_class(text):
    text = normalize_resident_request_string(text)
    chain = get_emergency_class_chain()
    answer: str = chain.run(query=text)
    return answer

def save_to_excel(data, column_names=['Emergency', 'Class', 'Truth', 'Time'], file_name='test_case_yk.xlsx'):
    try:
        # Преобразуем массив в DataFrame
        df = pd.DataFrame(data)
        
        # Сохраняем DataFrame в Excel файл
        df = pd.DataFrame(data, columns=column_names)
        df.to_excel(file_name, index=False)
        
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    
def testing():
    s = datetime.now()
    f = open('C:\\Users\\Dmitry\\Desktop\\test_yk.csv', encoding='utf-8').read().strip().replace('\ufeff','').replace('\n"', ';";').split(';";')
    test_case = []
    g = []
    for i in range(0, len(f),2):
        g.append([f[i], 'аварийная' if int(f[i+1]) else 'обычная'])
    # g = g[:2]
    for i in g:
        s_c = datetime.now()
        c = get_emergency_class(i[0])
        if c == i[1]:
            test_case.append([i[0], i[1], '1', datetime.now() - s_c])
        else:
            test_case.append([i[0], i[1], '0', datetime.now() - s_c])
    print(datetime.now() - s)
    return save_to_excel(test_case)

print(testing())
