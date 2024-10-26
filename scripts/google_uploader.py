from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
from datetime import datetime

class GoogleUploader:
    def __init__(self, parent_folder_path, new_folder_name, files, auth_file='credentials.json'):
        self.auth_file = auth_file
        self.parent_folder_path, self.new_folder_name, self.files = parent_folder_path, new_folder_name, files

        self._authenticate()

    def _authenticate(self):
        """Авторизуемся через Google и сохраняем учетные данные в файл."""
        try:
            self.gauth = GoogleAuth()
            self.gauth.LoadCredentialsFile(self.auth_file)
            if self.gauth.credentials is None:
                # Аутентифицируемся, если учетные данные отсутствуют
                self.gauth.LocalWebserverAuth()
            elif self.gauth.access_token_expired:
                # Обновляем токен, если он истек
                self.gauth.Refresh()
            else:
                # Инициализируем сохраненные учетные данные
                self.gauth.Authorize()
            self.gauth.SaveCredentialsFile(self.auth_file)
            self.drive = GoogleDrive(self.gauth)
            print("Authentication successful.")
        except Exception as e:
            print(f"Failed to authenticate: {e}")

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
            print(f'Got some trouble creating folder, check your code please! Error: {_ex}')
            return None

    def upload_files_to_path(self):
        """
        Загружает файлы в директорию Google Drive по указанному пути.
        Если директория уже существует, добавляет к названию директории дату и время.
        Аргументы:
        parent_folder_path - путь к существующей папке;
        new_folder_name - имя создаваемой папки;
        files - список файлов для загрузки.
        """
        try:
            # Получить ID папки родителя
            parent_folder_id = self.get_folder_id_by_path(self.parent_folder_path)
            if not parent_folder_id:
                return f'Parent folder "{self.parent_folder_path}" not found.'

            # Проверка, существует ли уже папка с таким именем
            existing_folder_id = self.get_folder_id_by_name(self.new_folder_name, parent_folder_id)
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

    def get_folder_id_by_name(self, folder_name, parent_id='root'):
        """Ищет папку на Google Drive по имени и ID родительской папки, возвращает её ID."""
        try:
            folder_list = self.drive.ListFile(
                {'q': f"'{parent_id}' in parents and title = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed=false"}
            ).GetList()
            if folder_list:
                return folder_list[0]['id']
            return None
        except Exception as _ex:
            print(f'Got some trouble, check your code please! Error: {_ex}')
            return None

    def get_folder_id_by_path(self, folder_path):
        """Ищет папку на Google Drive по пути и возвращает её ID."""
        try:
            folder_names = folder_path.strip('/').split('/')
            parent_id = 'root'
            for folder_name in folder_names:
                folder_id = self.get_folder_id_by_name(folder_name, parent_id)
                if folder_id:
                    parent_id = folder_id
                else:
                    return None
            return parent_id
        except Exception as _ex:
            print(f'Got some trouble, check your code please! Error: {_ex}')
            return None

# Пример использования
if __name__ == '__main__':
    uploader = GoogleUploader(
        parent_folder_path='LevelGroup/ML Data',  # Путь к существующей папке
        new_folder_name='TestFolder',             # Имя создаваемой папки
        files=[
            'datasets\\LevelGroup Материалы НСИ from 01-10-2024.xlsx',
            'datasets\\LevelGroup_ fixed NSI groups by Dima.xlsx'
        ]  # Список путей к файлам
    )
    
    # Загрузить файлы в новую папку в Google Drive
    print(uploader.upload_files_to_path())
