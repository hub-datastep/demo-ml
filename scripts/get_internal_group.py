import pandas as pd

def get_example():
    columns = ["name","group","NSI"]
    df = pd.read_excel("datasets/train_dataset_unistroi+syntetics+clear_big_class+not_lestnitsa.xlsx", usecols=columns, sheet_name="main")

    noms = df[columns[0]].astype(str).tolist()
    groups = df[columns[1]].astype(str).tolist()
    nsi = df[columns[2]].astype(str).tolist()

    f = {}

    for i in range(len(noms)):
        if nsi[i] == 'nan':
            continue
        try:
            if len(f[groups[i]][nsi[i]])<=15:
                f[groups[i]][nsi[i]].append(noms[i])
        except:
            try:
                f[groups[i]][nsi[i]] = []
            except:
                f[groups[i]] = {}

    k = {}
    for i in f:
        if len(f[i])>1:
            k[i] = f[i]

    return k
# print(list(get_example().keys()))
def get_test():
    columns = ["NOMs","AI"]
    df = pd.read_excel("output_test/test_model_unistroi_LogisticRegression.xlsx", usecols=columns, sheet_name="Sheet1")
    f = []
    noms = df[columns[0]].astype(str).tolist()
    groups = df[columns[1]].astype(str).tolist()
    for i in range(len(noms)):
        f.append([noms[i], groups[i]])
    return f

def save_to_excel(data, path, columns):
    try:
        df = pd.DataFrame(data, columns=columns)
        df.to_excel(path, index=False)
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False