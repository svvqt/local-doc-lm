import sys
import logging
import asyncio
from agent import file_extension, embedding_text, summarize_the_text, finding_the_text, test_use_model, delete_the_doc
from test_chroma import parents_col
from ollama import ResponseError
from utils.initializer import init_system
from local import locale

# определяем уровень логирования на уровень Warning, чтобы не забивать вывод информацией о запросах ollama
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

history=[]

async def upload_document():
    filename = input("Введите название файла")
    try:
        # определяем тип файла и возвращаем текст из файла
        content = file_extension(filename)
    except FileNotFoundError:
        logging.error(f"Файл не найден {filename}")
        print("Указанный файл не существует")
    # векторизация текста и добавление его в бд
    await embedding_text(filename, content)
    print("Пересказ текста: \n")
    response = summarize_the_text(filename)
    #history.append({'role': 'user', 'content': 'Перескажи этот текст'})
    #history.append({'role': 'assistant', 'content': response})
    print(response)

def chat_mode():
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

def delete_document():
    filename = input("Введите название файла, который хотите удалить")
    delete_the_doc(filename)

actions = {
        '1': upload_document,
        '2': chat_mode,
        '3': delete_document,
        '4': sys.exit
    }

async def main():
    init_system()
    #history = []
    exist = parents_col.get()
    if (len(exist.get('ids',[]))> 0):
        print(f"Добро пожаловать в систему: сейчас мы работаем с {len(exist.get('ids',[]))} файлами")
    else:
        print(f"Добро пожаловать в систему: сейчас у нас нет файлов")

    while True:
        print(locale.get("menu.header"))
        print(locale.get("menu.upload"))
        print(locale.get("menu.chat"))
        print(locale.get("menu.delete"))
        print(locale.get("menu.exit"))
        
        choice = input("\nВыберите действие: ")

        if choice in actions:
            # Вызываем функцию. Если она асинхронная — используем await
            action = actions[choice]
            if asyncio.iscoroutinefunction(action):
                await action()
            else:
                action()
        else:
            print("⚠️ Неверный ввод, попробуйте снова.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit()





#if __name__=="__main__":
    # иннициализируем систему (загрузка модели, если не установлена)
    # init_system()
    # history = []
    # exist = parents_col.get()
    # if (len(exist.get('ids',[]))> 0):
    #     print(f"Добро пожаловать в систему: сейчас мы работаем с {len(exist.get('ids',[]))} файлами")
    # else:
    #    print(f"Добро пожаловать в систему: сейчас у нас нет файлов")
    #while True:
    #    filename = input("Введите название файла (или exit): ")
    #    if filename.lower() == 'exit': break
    #    try:
    #        # определяем тип файла и возвращаем текст из файла
    #        content = file_extension(filename)
    #    except FileNotFoundError:
    #        logging.error(f"Файл не найден {filename}")
    #        print("Указанный файл не существует")
    #        continue
        # векторизация текста и добавление его в бд
    #    embedding_text(filename, content)
    #    print("Пересказ текста: \n")
    #    response = summarize_the_text(filename)
    #    history.append({'role': 'user', 'content': 'Перескажи этот текст'})
    #    history.append({'role': 'assistant', 'content': response})
    #    print(response)
#        while True:
#            promt = input("Что вы хотите сделать с этим файлом (или back)?: ")
#            if promt.lower() == "back": break
#            try:
#                content = finding_the_text(promt)
#            except ResponseError as e:
#                if "exceeds the context length" in str(e):
#                    logging.error("Большая длина промпта")
#                    print("Большая длина промта, попробуйте еще раз")
#                else:
#                    logging.error(str(e))
#                    print(str(e))
#                continue
#            response = test_use_model(promt, content, history)
#            history.append({'role': 'user', 'content': promt})
#            history.append({'role': 'assistant', 'content': response['message']['content']})
#            print(response['message']['content'])
