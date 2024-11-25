# Enhancing_Retreival_Accuracy

## Enhancing Data Retreival Accuracy Including Traditional NLP tasks in RAG Applications 

### This contains comparison on Retrieval accuracy with usual approach of embeddings against additional NLP processing
### Edge cases which decides results will be addressed with this (change in ~2% accuracy change observed with this approach)

#### Instead of usual retreival process of embedding source data and then embedding user query to retrieve nearest matches,
#### I have inluded additional NLP processing steps to clean the source data and user query before retrieving the top mathces.

### Code in this repo uses NLP processed data olny for data retrieval but not for further use in the Flow

##### Technologies/Components used
* Python as scripting language
* NLTK package for NLP steps (Tokenization, Stopwords cleaning, Lemmatization)
* PineCone Vector DB for vector storage
* Azure OpenAI Embedding model(text-embedding-ada-002)

##### Steps needed to run this code
* Packages from requirements.txt file can be installed using pip
* Credetentials of embedding model and Pinecode DB needs to be added under .env file or as Environment variables
* Jupyter notebook imports all necessary code from helping fies/folders, same cells can be executed to check on results

## Results:
![image](https://github.com/user-attachments/assets/9ff20a9f-9ff7-4a72-8897-d8fe1693eb9f)

![image](https://github.com/user-attachments/assets/44ade63f-8c33-48b0-8077-b317852ad067)

![image](https://github.com/user-attachments/assets/ea576664-f3dd-445d-812f-6e18059a8192)

![image](https://github.com/user-attachments/assets/d074d069-b182-4751-848c-cd5be3f69971)



