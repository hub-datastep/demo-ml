import re

def clean_text(text):
    # Удаляем пробелы в начале и конце строки, заменяем множественные пробелы на один
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    return cleaned_text

# Пример использования
text = "  Пример   текста \nс лишними пробелами \nи переносами строк.  "
cleaned_text = clean_text(text)
print(cleaned_text)
