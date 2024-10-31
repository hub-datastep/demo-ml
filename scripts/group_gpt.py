from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

import os

from dotenv import load_dotenv

load_dotenv()

DOCS_PROMPT_TEMPLATE = """### Роль
инженер ПТО (производственно-технического отдела)

### Данные которые получишь на вход:
- номенклатура из спецификации
- группа с обобщенной формулировкой, к которой подходят все номенклатуры
- список более точных групп из которых надо будет выбрать группу для каждой номенклатуры

Формат входных данных:
{{
"nomenclature": "номенклатура",
"generalized group": "обобщённая группа",
"precise groups" : ["точная группа1", "точная группа2"]
}}


### Задача
выбрать наиболее подходящую точную группу к обобщенной группе и каждой отдельной номенклатуре

### Не забывай:
1. Посмотри на пример номенклатур в каждом классе:
{json_example}
2. Обращать внимание на параметры в номенклатуре
3. Отвечай кратко
4. Ответ в формате json:
{{
 "group": "точная группа для номенклатуры 1"
}}


### Пример
Пример входных данных:
{{
  "nomenclature": "труба 200*150 d=25 для кранов и санузлов",
  "generalized group": "трубы",
  "precise groups" : ["трубы металлические","трубы противопожарные", "трубы сантехнические"]
}}


Пример правильного ответа:
{{
 "group": "трубы сантехнические"
}}


### Вход
{{
  "nomenclature": {nomenclature}
  "generalized group": {generalized_group},
  "precise groups" : {precise_groups}
}}"""

def get_emergency_class_chain():
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME_NOMENCLATURE_CLASSIFIER"),
        temperature=0,
        verbose=False,
    )

    prompt = PromptTemplate(
        template=DOCS_PROMPT_TEMPLATE,
        input_variables=["json_example", "nomenclature", "generalized_group", "precise_groups"]
    )

    return LLMChain(llm=llm, prompt=prompt)


from json import loads
from get_internal_group import get_example, get_test, save_to_excel

nsi = get_example() # {"internal": {"nsi": ["noms1", "noms2"] } }
data = get_test()
print(nsi)
def predict(data: list) -> list:
    strs = []
    for i in data:
        noms = i[0]
        group = i[1]
        try:
            json_example = nsi[group]
            
            precise_groups = list(json_example.keys())

            chain = get_emergency_class_chain()
            c = loads(chain.run(json_example=json_example, nomenclature=noms, generalized_group=group, precise_groups=precise_groups))['group']
            print(noms, group, c)
            strs.append([noms, group, c])
        except:
            strs.append([noms, group, group])
    return strs

output = predict(data)
save_to_excel(data=output, path='output_test/тестирую_llm_0_0.xlsx', columns=["NOM's", 'AI', "LLM"])