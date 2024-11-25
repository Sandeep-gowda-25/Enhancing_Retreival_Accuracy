import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
from typing import List

def load_data(path):
    reader = pypdf.PdfReader(path)

    file_contents = ''
    for page in reader.pages:
        file_contents = file_contents + page.extract_text()
    return file_contents

def chunk_data(text_input:str):
    splitter = RecursiveCharacterTextSplitter(separators=['\n'],chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text_input)
    return chunks

def process_input(text_input:str):
    word_list = tokenization(text_input)
    word_list = stop_words_removal(word_list)
    word_list = lematization(word_list)

    cleaned_text = ' '.join(word_list)
    return cleaned_text


def tokenization(text_input:str):
    nltk.download('punkt',quiet=True)
    from nltk.tokenize import word_tokenize
    word_list = word_tokenize(text_input)
    return word_list

def stop_words_removal(word_list:List[str]):
    nltk.download('stopwords',quiet=True)
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')
    for i,word in enumerate(word_list):
        if word.lower() in stopwords:
            word_list.pop(i)
    return word_list
    
def lematization(word_list:List[str]):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    return [lemmatizer.lemmatize(word) for word in word_list]