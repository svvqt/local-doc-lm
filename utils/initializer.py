import os
import sys
import time
import subprocess
from ollama import Client, list as list_models, pull
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

def init_system():
    # 1. Запуск Ollama в фоновом режиме (если нужно)
    try:
        list_models()
    except Exception:
        print("Сервер Ollama не запущен. Попытка старта...")
        # Учитываем локальный или системный путь
        env_vars = os.environ.copy()
        subprocess.Popen(
            ["ollama", "serve"],
            env=env_vars,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        time.sleep(5) # Ждем старта

    # 2. Проверка моделей из .env
    client = Client()
    required = os.getenv("MODEL")
    
    try:
        response = client.list()
        target = os.getenv("MODEL")
        installed = [m.model for m in response.models]
        print(f"Доступные модели: {installed}")
        if target in installed:
                print(f"Используется: {target}")
        for model in required:
            if not any(model in m for m in installed):
                pbar = tqdm(total=100, unit='%', desc=f"Загрузка {model}", leave=True)
    
                try:
                    for progress in pull(model, stream=True):
                        # Извлекаем значения безопасно
                        total = progress.get('total')
                        completed = progress.get('completed')

                        # Обновляем только если оба значения — числа
                        if total is not None and completed is not None and total > 0:
                            percent = int((completed / total) * 100)
                        # Чтобы полоска не прыгала назад, если скачиваются разные слои
                            if percent > pbar.n:
                                pbar.n = percent
                                pbar.refresh()
            
                        # Если статус "success", сразу завершаем
                        if progress.get('status') == 'success':
                            pbar.n = 100
                            pbar.refresh()
                            break
                finally:
                    pbar.close()
            
            
    except Exception as e:
        print(f"Ошибка проверки моделей: {e}")
        sys.exit(1)