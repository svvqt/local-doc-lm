import json
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

class LocaleManager:
    def __init__(self, filepath="local.json"):
        with open(filepath, "r", encoding="utf-8") as f:
            self._data = json.load(f)
        self.lang = os.getenv("LANGUAGE", "ru")

    def get(self, path: str):
        # Добавляем язык в начало пути
        keys = [self.lang] + path.split(".")
        value = self._data
        
        for key in keys:
            # Проверяем, что текущее значение - словарь, и в нем есть ключ
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                logging.error(f"Ключ '{key}' не найден в пути '{path}'")
                return f"[{path}]" # Возвращаем путь в скобках, чтобы сразу видеть ошибку в UI
        
        return value
        
locale = LocaleManager()