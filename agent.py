import sys
import os
import pymupdf
from ollama import chat, ChatResponse, embed
from test_chroma import client, child_col, parents_col
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

def test_use_model(user_promt, content: str, history):
    """
    func for chat with model
    
    Parameters
    ----------
    user_promt: str
        instruction of user
    content: str
        text from documents
    history: List[Dict[str, str]]
        context of chat like [{'role': 'user/assistant', 'content': 'text'}]

    Returns
    -------
    response: Ollama object response with answer and metadata
    """
    
    system_instruction = (
        "Ты — аналитик. Твоя задача — отвечать на вопросы по фрагментам документов. "
        "Фрагменты могут быть вырваны из контекста, это нормально. "
        "Если в тексте совсем нет информации по теме, ответь 'Информация не найдена'."
    )
    promt = f"""
    <user_promt>
    {user_promt}
    </user_promt>
    <document_content>
    {content}
    </document_content>

    ИНСТРУКЦИЯ: На основе фрагментов в <document_content> ответь на запрос в <user_promt>. Отвечай сразу по существу..
    """

    message = [{'role': 'system', 'content': system_instruction}] + history + [{'role': 'user', 'content': promt}]
    response: ChatResponse = chat(model=os.getenv("MODEL"), messages=message,
    options={
        'temperature': 0.1,
    }
)
    return response

def file_extension(filename: str) -> str:
    """
    func for detect a right file extension for extraction content from file

    Parameters
    ----------
    filename: str 
        path for file that we want to extract

    Returns
    -------
    content: str
        extracted text from file
    """
    path = Path(filename)

    if path.suffix == ".txt":
        content = read_file_txt(filename)
    elif path.suffix == ".pdf":
        content = read_file_pdf(filename)
    return content


def read_file_txt(filename: str) -> str:
    """
    func for reading a txt files to convert them to string line
    
    Parameters
    ----------
    filename: str
        path for file that we want to read
    
    Returns
    -------
    content: str 
        text from file
    """
    with open(filename, "r", encoding='utf-8') as f:
        content = f.read()
    
    return content


def read_file_pdf(filename: str) -> str:
    """
    func for reading a pdf files to convert them to string line
    
    Parameters
    ----------
    filename: str 
        path for pdf file that we want to read
    
    Returns
    -------
    content: str
        text from file
    """
    content = ""
    doc = pymupdf.open(filename)
    for page in doc:
        content += page.get_text()
    
    return content


def embedding_text(filename: str, content: str):
    """
    Vectorizing our text and embedding him to ChromaDB

    Parameters
    ----------
    filename: str
        path for our file which from we extracted the text
    content: str
        our text

    Returns
    -------
    text about finish and how much chuncks we added to our child collection: str 
    """
    chunk_size = int(os.getenv("CHUNK_SIZE"))
    overlap = int(os.getenv("OVERLAP"))
    chunks = []

    # если документ в коллекции с этим файлом есть то мы не должны его индексировать (если есть файл изменился - то мы это не детектим)
    exist = parents_col.get(where={"source": f"{filename}"})
    if len(exist.get('ids', []))>0:
        return "Индексирование не нужно"

    # пересказ нашего текста, и добавление его в родительскую коллекция
    res: ChatResponse = chat(model=os.getenv("MODEL"), messages=[{"role": "user", "content": f"Перескажи этот текст, одним предложением: {content}"}], options={"temperature": 0.1})
    emb_response = embed(model=os.getenv("EMBEDDING_MODEL"), input=res['message']['content'][:512])
    parents_col.add(
        documents=[res['message']['content']],
        embeddings=[emb_response['embeddings'][0]],
        ids=f"parent_{filename}",
        metadatas=[{"source": f"{filename}", "type": "summary"}]
        )

    # проходимся по тексту разделяего на чанки с захлестом
    for i in range(0, len(content), chunk_size-overlap):
        chunks.append(content[i: i + chunk_size])

    # проходимся по нашим чанкам и добавляем их в коллекцию
    for i, chunk in enumerate(chunks):
        emb_response = embed(model=os.getenv("EMBEDDING_MODEL"), input=chunk)
        vector = emb_response['embeddings'][0]

        child_col.add(
            documents=[chunk],
            embeddings=[vector],
            ids=f"{filename}_{i}",
            metadatas=[{"source": f"{filename}"}]
        )
    return f"Загружено {len(chunks)} фрагментов текста"
        
def finding_the_text(prompt: str) -> str:
    """
    func for finding the info from our documents
    
    Parameters
    ----------
    prompt: str
        user prompt with info that we need to find in doc
    
    Returns
    -------
    All info that we have found in our doc: str
    """
    emb_response = embed(model=os.getenv("EMBEDDING_MODEL"), input=prompt)
    query_vector = emb_response['embeddings'][0]
    results = child_col.query(
        query_embeddings=[query_vector],
        n_results=7
    )
    return " ".join(results['documents'][0])

def summarize_the_text(filename: str):
    """
    func for retelling our file

    Parameters
    ----------
    filename: str
        path to file which we want to retell

    Returns
    -------
    Our retelling: str
    """
    all_docs = parents_col.get(where={"source": f"{filename}"})['documents']
    all_res = []
    for doc in all_docs:
        res = chat(model=os.getenv("MODEL"), messages=[{
            'role': 'user', 
            'content': f'Выдели основной смысл из этого фрагмента {doc}',
        },
        ]
        )
        all_res.append(res['message']['content'])
    return "\n".join(all_res)
