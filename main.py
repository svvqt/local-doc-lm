import logging
from agent import file_extension, embedding_text, summarize_the_text, finding_the_text, test_use_model
from test_chroma import parents_col
from ollama import ResponseError
from utils.initializer import init_system

# определяем уровень логирования на уровень Warning, чтобы не забивать вывод информацией о запросах ollama
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__=="__main__":
    # иннициализируем систему (загрузка модели, если не установлена)
    init_system()
    history = []
    exist = parents_col.get()
    if (len(exist.get('ids',[]))> 0):
        print(f"Добро пожаловать в систему: сейчас мы работаем с {len(exist.get('ids',[]))} файлами")
    else:
        print(f"Добро пожаловать в систему: сейчас у нас нет файлов")
    while True:
        filename = input("Введите название файла (или exit): ")
        if filename.lower() == 'exit': break
        try:
            # определяем тип файла и возвращаем текст из файла
            content = file_extension(filename)
        except FileNotFoundError:
            logging.error(f"Файл не найден {filename}")
            print("Указанный файл не существует")
            continue
        # векторизация текста и добавление его в бд
        embedding_text(filename, content)
        print("Пересказ текста: \n")
        response = summarize_the_text(filename)
        history.append({'role': 'user', 'content': 'Перескажи этот текст'})
        history.append({'role': 'assistant', 'content': response})
        print(response)
        while True:
            promt = input("Что вы хотите сделать с этим файлом (или back)?: ")
            if promt.lower() == "back": break
            try:
                content = finding_the_text(promt)
            except ResponseError as e:
                if "exceeds the context length" in str(e):
                    logging.error("Большая длина промпта")
                    print("Большая длина промта, попробуйте еще раз")
                else:
                    logging.error(str(e))
                    print(str(e))
                continue
            response = test_use_model(promt, content, history)
            history.append({'role': 'user', 'content': promt})
            history.append({'role': 'assistant', 'content': response['message']['content']})
            print(response['message']['content'])
