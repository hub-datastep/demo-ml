import re
import pandas as pd
import numpy as np

# def read_excel_columns(file_path, columns, sheet):
#         try:
#             df = pd.read_excel(file_path, usecols=columns, sheet_name=sheet)
#         except FileNotFoundError:
#             print(f"Ошибка: Файл '{file_path}' не найден.")
#             return None
#         except ValueError as e:
#             print(f"Ошибка: Некорректные данные в файле или неверные колонки. {e}")
#             return None
#         except Exception as e:
#             print(f"Произошла ошибка: {e}")
#             return None

#         try:
#             result = df.values.tolist()
#             data = np.array(result)
#             # Разделение данных на номенклатуру и классы
#             noms = data[:, 0]  # Номенклатура
#             groups = data[:, 1]  # Классы
#             vids = data[:, 2]  # Виды
#         except Exception as e:
#             print(f"Ошибка при преобразовании данных: {e}")
#             return None

#         return noms, groups 

# nsi = read_excel_columns("c:\\Users\\Dmitry\\Downloads\\LevelGroup_ fixed NSI groups by Dima.xlsx", ["name", "group", "view"], "Sheet0")
# # test = list(read_excel_columns("c:\\Users\\Dmitry\\Downloads\\Classifier_ unistroy UTDs test-cases.xlsx", ["Номенклатура поставщика", "Ожидание группа", "Ожидание номенклатура"], "test-cases")[0])
def process_text(text):
    # 1. Заменяем все символы, кроме русских букв, на пробелы
    clean_text = re.sub(r'[^а-яА-ЯёЁa-zA-Z0-9]', ' ', text)
    
    # 2. Разбиваем строку на слова
    words = clean_text.split()
    
    # 3. Сортируем слова в алфавитном порядке
    sorted_words = [i for i in sorted(words, key=lambda word: word.lower()) if len(i)>2]
    
    return ' '.join(sorted_words)

nsi = ['Алюминиевая металлическая панель, П-образная', 'Выключатель автоматический S201 B10А/1п/ 6,0кА', 'датчики давления (реле, преобразователи) БД Датчик давления БД ПД-Р', 'Изделия из оцинкованной стали Парапет окрашенный 0,57 мм', 'Отлив окрашенный, вальцованный 1 мм', '1-метровый 10G SFP+ кабель прямого подключения TL-SM5220-1M, TP-Link', 'Cветильник Platek Mesh XL', 'Cветильник подвесной для реечных потолков LED Line Pendant 35 1000х35х35 32 Вт', 'Cветильник подвесной для реечных потолков LED Line Pendant 35 1500х35х35 48 Вт', 'Cветильник подвесной для реечных потолков LED Line Pendant 35 500х35х35 16 Вт', 'Cветильник подвесной-Ledalen-LINEAR N4326 3000K 9W-750 мм', 'Cветильник подвесной-Ledalen-LINEAR P1616 3000K 26W-2000', 'Cветильник подвесной-Ledalen-LINEAR P4028 3000K 12W-1000', 'Cветильник подвесной-Ledalen-LINEAR P4028 3000K 13W-750 мм', 'Cветильник подвесной-Ledalen-LINEAR P4028 3000K 19W-1500', 'Cветильник подвесной-Ledalen-LINEAR P4028 3000K 9W-750 мм', 'Cигнализатор потока жидкости DINARM 100мм', 'Cигнализатор потока жидкости DINARM 125мм', 'Cигнализатор потока жидкости DINARM 150мм', 'Cигнализатор потока жидкости DINARM 50мм', 'Cигнализатор потока жидкости DINARM 65мм', 'Cигнализатор потока жидкости DINARM 80мм', 'Cплит-система Haier 1U24TL5FRA-A/AS24TL5HRA-A', 'Cплит-система Haier AD18MS1ERA/5U34HS1ERA', 'Cплит-система Haier AD24MS3ERA/4U30HS3ERA', 'Cплит-система Haier AS 70NHPHRA/1U 70NHPFRA', 'Cплит-система Haier HSU-36HNH03/R2 /HSU-36HUN03/R2', 'Cплит-система кассетного типа Haier AB36ES1ERA(S)/1U36SS1EAB', 'Cплит-система кассетного типа Haier AB48ES1ERA(S) / 1U48LS1EAB(S)', 'Cплит-система на 2 внутренних блока Haier AD18MS1ERA/2U26GS1ERA', 'Cплиттер PoE, Gigabit Ethernet, 24 В, 10 Вт NS-200PS, ICP DAS', 'Cтационарная полка 19" глубиной 400мм ITK', 'Cтационарная полка 19" глубиной 600мм ITK', 'Cтационарная полка глубиной 300мм Hyperline TSH3L-300-RAL9004', 'Cчетчик водяной-ВСХН -100', 'Cчетчик тепла механический ПУЛЬСАР DN 15', 'Cчетчик тепловой ультразвуковой ПУЛЬСАР DN 15', 'Cчетчик тепловой ультразвуковой ПУЛЬСАР DN 20', 'Danfoss реле температуры KP61-6', 'DIN-рейка ABB 12850', 'DIN-рейка DKC 02140', 'DIN-рейка IEK TH35-7,5', 'DIN-рейка IEK YDN10-00100', 'DIN-рейка IEK YDN10-0013', 'DIN-рейка IEK YDN10-0025', 'DIN-рейка IEK YDN10-0040', 'DIN-рейка IEK YDN10-0050', 'DIN-рейка IEK YDN10-0060', 'DIN-рейка перфорированная высота профиля 15 мм', 'DIN-рейка перфорированная высота профиля 35 мм YDN10-00100', 'DIN-рейка перфорированная высота профиля 35 мм YDN10-0013', 'DIN-рейка перфорированная высота профиля 35 мм YDN10-0025', 'DIN-рейка перфорированная высота профиля 35 мм YDN10-0040', 'DIN-рейка перфорированная высота профиля 35 мм YDN10-0060', 'DIN-рейка перфорированная высота профиля 35 мм длина 300мм', 'Itermic-ITF 080.080-1000', 'LED лента NEON 17мм 360 градусов", L=15м, 24В, 6Вт/м, 400Лм/м, IP67 Varton', 'OLFLEX 540-3x2,5 Lapp Kabel', 'OLFLEX 540-3х1,0 Lapp Kabel', 'OLFLEX 540-3х1,5 Lapp Kabel', 'Omada VPN-маршрутизатор с портами 10 Гбит/с, ER8411, TP-Link', 'PILOT-Сетевой фильтр 5 розеток', 'PILOT-Сетевой фильтр 6 розеток', 'POE -инжектор TL-POE15oS V4, TP-Link', 'PoE коммутатор на 4 порта SH-20.4', 'PoE коммутатор на 8 порта SH-20.8', 'POE -удлинители ComOnyx CO-PE-B25-P103v2', 'POE -удлинители Hikvision DS-1H34-0101P', 'POE -удлинители POE - инжектор Midspan-1/650G', 'POE -удлинитель 1 порт 10/100 Мб/с 2хRJ45 (E-PoE/1), 00013563, OSNOVO', 'POE -удлинитель 1 порт 10/100 Мб/с 2хRJ45 (E-PoE/1G), 013564, OSNOVO', 'POE -удлинитель Fast Ethernet E-PoE/1 OSNOVO', 'POE -удлинитель Fast Ethernet E-PoE/1G OSNOVO', 'POE -удлинитель Fast Ethernet E-PoE/1W OSNOVO', 'POE -удлинитель по кабелю PoE по кабелю UTP РоЕ-1.1', 'POE -удлинитель по кабелю UTP DS-1H34-0101P Hikvision', 'PoE-инжектор 65W Gigabit Ethernet на 1 порт FS05-600MP-R OSNOVO', 'PoE-инжектор Midspan-1/650G OSNOVO', 'PoE-инжектор Midspan-2/602G OSNOVO', 'PURMO Aura Comfort WKH 24-100-24-00', 'PURMO Aura Comfort WKH 24-110-24-00', 'PURMO Aura Comfort WKH 24-120-15-00', 'PURMO Aura Comfort WKH 24-120-24-00', 'PURMO Aura Comfort WKH 24-160-24-00', 'PURMO Aura Comfort WKH 24-80-24-00', 'PURMO Aura Comfort WKH 24-90-24-00', 'PURMO Aura Comfort WKH 28-140-24-00', 'SFP-модули оптические Huawei QSFP28-100G-CU1M', 'SFP-модули оптические Huawei QSFP-40G-eSDLC-PAM', 'SFP-модули оптические Huawei SFP-10G-AOC10M', 'SFP-модули оптические Huawei SFP-10G-CU1M', 'SFP-модули оптические Huawei SFP-10G-SR-C', 'SFP-модули оптические TP-Link TL-SM311LS', 'SFP-модули оптические TP-Link TL-SM321A', 'SFP-модули оптические TP-Link TL-SM321B', 'SFP-модули оптические TP-Link TL-SM331T', 'SFP-модули оптические TP-Link TL-SM5110-LR', 'SFP-модуль QSFP-40G-eSDLC-PAM, Huawei', 'SFP-модуль SFP-10G-SR-C, Huawei', 'SFP-модуль TL-SM311LS V3, TP-LINK']
for i in nsi:
    print(i, "||", process_text(i))