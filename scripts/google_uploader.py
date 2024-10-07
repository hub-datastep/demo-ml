from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import pickle
from datetime import datetime

class GoogleUploader:
    def __init__(self, parent_folder_name, new_folder_name, files, auth_file='drive.pkl'):
        self.auth_file = auth_file
        self.parent_folder_name, self.new_folder_name, self.files = parent_folder_name, new_folder_name, files
        
        # Попробуем загрузить сохранённую авторизацию
        if os.path.exists(self.auth_file):
            try:
                with open(self.auth_file, 'rb') as f:
                    self.drive = pickle.load(f)
                print("Authorization loaded from file.")
            except Exception as e:
                print(f"Failed to load authorization from file: {e}")
                self._authenticate_and_save()
        else:
            self._authenticate_and_save()

    def _authenticate_and_save(self):
        """Авторизуемся через Google и сохраняем сессию в файл."""
        try:
            self.gauth = GoogleAuth()
            self.gauth.LocalWebserverAuth()
            self.drive = GoogleDrive(self.gauth)
            
            # Сохраняем авторизацию в файл
            with open(self.auth_file, 'wb') as f:
                pickle.dump(self.drive, f)
            print("Authorization saved to file.")
        except Exception as e:
            print(f"Failed to authenticate and save session: {e}")

    def create_folder(self, parent_folder_id, folder_name):
        """Создает папку на Google Drive в указанной директории."""
        try:
            folder_metadata = {
                'title': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [{'id': parent_folder_id}]
            }
            folder = self.drive.CreateFile(folder_metadata)
            folder.Upload()
            return folder['id']
        except Exception as _ex:
            return f'Got some trouble creating folder, check your code please! Error: {_ex}'

    def upload_files_to_path(self):
        """
        Загружает файлы в директорию Google Drive по пути (какая-то папка)/(созданная нами директория).
        Если директория уже существует, добавляет к названию директории дату и время.
        Аргументы:
        parent_folder_name - имя существующей папки;
        new_folder_name - имя создаваемой папки;
        files - список файлов для загрузки.
        """
        try:
            # Получить ID папки родителя
            parent_folder_id = self.get_folder_id_by_name(self.parent_folder_name)
            if not parent_folder_id:
                return f'Parent folder "{self.parent_folder_name}" not found.'

            # Проверка, существует ли уже папка с таким именем
            existing_folder_id = self.get_folder_id_by_name(self.new_folder_name)
            if existing_folder_id:
                # Добавляем дату и время, если такая папка существует
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
                self.new_folder_name = f"{self.new_folder_name}_{timestamp}"

            # Создать новую папку внутри родительской
            new_folder_id = self.create_folder(parent_folder_id, self.new_folder_name)
            if not new_folder_id:
                return f'Failed to create folder "{self.new_folder_name}".'

            # Загрузить файлы в созданную директорию
            for file_path in self.files:
                file_name = os.path.basename(file_path)
                file_drive = self.drive.CreateFile({
                    'title': file_name,
                    'parents': [{'id': new_folder_id}]
                })
                file_drive.SetContentFile(file_path)
                file_drive.Upload()
                print(f'File {file_name} was uploaded to {self.new_folder_name}.')

            return 'All files uploaded successfully!'
        except Exception as _ex:
            return f'Got some trouble, check your code please! Error: {_ex}'

    def get_folder_id_by_name(self, folder_name):
        """Ищет папку на Google Drive по имени и возвращает её ID."""
        try:
            folder_list = self.drive.ListFile(
                {'q': f"title = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed=false"}
            ).GetList()
            if folder_list:
                return folder_list[0]['id']
            return None
        except Exception as _ex:
            return f'Got some trouble, check your code please! Error: {_ex}'

# Пример использования
if __name__ == '__main__':
    uploader = GoogleUploader( parent_folder_name='NER',  # Имя существующей папки
        new_folder_name='TestFolder',         # Имя создаваемой папки
        files=['file/Classifier_ unistroy UTDs test-cases (fix)_30.09.2024.xlsx', 'file/Classifier_test-case_unistroi.xlsx']  # Список путей к файлам
    )
    
    # Загрузить файлы в новую папку в Google Drive
    print(uploader.upload_files_to_path(
       ))
