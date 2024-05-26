# PersonalGPT: A Retrieval-Augmented Generation AI System

## Project Overview

PersonalGPT is a Retrieval-Augmented Generation (RAG) based AI application capable of answering questions about me(Ashwanthram). This system integrates data ingestion, a vector database for efficient retrieval, and a generative AI model to provide accurate and contextually relevant responses.

## Project Objectives

1. **Data Ingestion**: Ingest personal data from text files or PDFs and index it into a vector database.
2. **RAG System Integration**: Utilize a RAG approach combining information retrieval from the vector database and generative capabilities of large language models (LLMs).
3. **Follow-Up Questions**: Implement logic to handle follow-up questions, maintaining conversation context.
4. **Interface**: Develop a basic frontend using Streamlit for user interaction with the AI system.


## Setup Instructions

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package installer)
- Git

Additionally, the following Python packages are required:

- streamlit
- langchain
- transformers
- sentence-transformers
- numpy
- pandas
- faiss-cpu
- PyPDF2

### Clone the Repository

```bash
git clone https://github.com/AshwanthramKL/PersonalGPT
cd <repository_directory>
```

Replace `<repository_directory>` with the path to the cloned repository in your local system.

### File Structure
- **PersonalGPT.ipynb:** Jupyter notebook containing the code for extracting personal data,and indexing it into a vector database.
- **streamlit_app.py:** Streamlit application file containing the RAG system integration and frontend interface.
- **personalDocuments/:** Directory containing text files and PDFs for data including:
    * `Ashwanthram_Biodata.txt`: Text file containing Bio data
    * `AshwanthramResume_DS_GenAI.pdf`: Ashwanthram's Resume
    * `BucketList.txt`: Text file containing Ashwanthram's bucket list
    * `Letter_of_Recommendation.txt`: Text file containing a letter of recommendation(MOCKUP DATA)
    * `LinkedIn_Profile.pdf`: Ashwanthram's LinkedIn Profile downloaded as PDF.
    * `Personal_Blog.txt`: Text file containing Ashwanthram's personal blog about himself.
    * `Reading_List.md`: Markdown file containing Ashwanthram's reading list.
    * `WorkoutRoutine.md`: Markdown file containing Ashwanthram's detailed workout routine and calorie goals.
- **vector_db/:** Directory containing the vector database files.

## Running the Jupyter Notebook
Open the Jupyter notebook and run all cells inside `Task: Data Extraction and VectorDB Creation` to execute the data ingestion process. This will extract personal data from text files or PDFs, index it into a vector database, and generate embeddings for retrieval.

### Running the Streamlit Application

To run the Streamlit application, use the following command:

```bash
streamlit run streamlit_app.py
```

### System Architecture
- **Data Ingestion:** Personal data is ingested from text files or PDFs and indexed into a vector database using `InstructorXL` and `FAISS` for efficient retrieval
- **RAG Integration:** The system retrieves relevant information from `FAISS` vector database and uses `GPT-4-1106-preview` to generate responses.
- **Orchestration:** 
    * Orchestration between data retrieval and the generative model is managed using tools from `langchain`.
    * A `RAG Chain` is created to handle the retrieval and generation process and maintain conversational context using a conversational state.
- **Frontend Interface:** A `Streamlit` based frontend for UI.

### Demonstration

Demonstration Video

### Contact
For any questions or support, please contact:

Name: Ashwanthram  
Email: k.l.ashwanthram@gmail.com  
