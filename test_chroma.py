import chromadb

# иннициализация клиента, сохраняем все наши коллекции в эту папку
client = chromadb.PersistentClient(path="./my_db")

# коллекция для хранения информации из файлов
child_col = client.get_or_create_collection(name="childs")
# коллекция для хранения краткого пересказа наших файлов
parents_col = client.get_or_create_collection(name="parents")
