from langchain_community.vectorstores import FAISS
from InstructorEmbedding import INSTRUCTOR
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

import streamlit as st
import os
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage



# Load the vectordb from local storage
new_db = FAISS.load_local("vectorstore", HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl'))

# Retriever
retriever = new_db.as_retriever(search_kwargs={"k": 7, "hnsw:space": "cosine"})

from dotenv import load_dotenv

# Set api key
load_dotenv("D:\FarmwiseAI\Reddit\.env")

# Load the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# Language Model
llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0, model_kwargs={"top_p": 0.5})

condense_q_system_prompt = """Given a chat history and the latest user question \
which might reference the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
condense_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", condense_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
condense_q_chain = condense_q_prompt | llm | StrOutputParser()

qa_system_prompt = """You are an AI assistant designed to provide comprehensive information about Ashwanthram. Your primary role is to help users understand various aspects of Ashwanthram's life, work, interests, achievements, etc. based on the context provided. Below is a detailed guide on how to interact and provide valuable assistance:

User Profile:
Name: Ashwanthram
Profession: GenAI Data Scientist
Experience: 10 months working with LLMs and RAG to build applications such as an LLM-powered social media recommendation system and a personalized travel planner application.
Interests: Tech, generative AI, health, fitness, longevity, self-help podcasts, and working out.

Interaction Guidelines:
* Be Clear and Informative: Provide detailed and accurate information about Ashwanthram. Avoid overly technical jargon unless necessary.
* Be Supportive and Encouraging: Encourage users to learn more about Ashwanthram's interests and achievements. Offer positive reinforcement.
* Be Detailed and Comprehensive: Provide in-depth information about Ashwanthram's work, interests, and achievements. Include relevant examples and details.

Example Dialogue:
User: Can you tell me about Ashwanthram's professional background?

AI Assistant: Ashwanthram is a GenAI Data Scientist with 9 months of experience working with Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG). He has developed applications such as an LLM-powered social media recommendation system and a personalized travel planner application. He is passionate about continuing to leverage LLMs and work on cutting-edge technologies.

Context:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

def format_docs(documents):
    return "\n\n".join(document.page_content for document in documents)

def condense_question(input: dict):
    if input.get("chat_history"):
        return condense_q_chain
    else:
        return input["question"]

rag_chain = (
    RunnablePassthrough.assign(context=condense_question | retriever | format_docs)
    | qa_prompt
    | llm
)

# Initialize chat history
chat_history = []

# Streamlit app setup
# Set page configuration with title and layout
st.set_page_config(page_title='Ashwanth\'s Personal AI Assistant', layout='wide')

st.title("Ashwanth's Personal AI Assistant")

# Initialize session state for chat history if not already present
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to handle message sending
def send_message():
    user_input = st.session_state.user_input
    if user_input:
        # Process user message
        ai_msg = rag_chain.invoke({"question": user_input, "chat_history": st.session_state['chat_history']})
        st.session_state['chat_history'].extend([HumanMessage(content=user_input), ai_msg])
        # Print chat history for debugging
        print("Updated chat history:", st.session_state['chat_history'])


        # Clear the input box
        st.session_state.user_input = ""

# Use columns for better layout
col1, col2 = st.columns([3, 1])

# Container for chat history
# Initialize session state for chat history if not already present
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Container for user input
with col2:
    with st.container():
        # Text input for user message
        # user_input = st.text_input("Your message:", key="user_input", on_change=send_message, args=())
        user_input = st.text_area("Your message:", key="user_input", on_change=send_message, args=(), height=500)
        
        # Send button
        send_btn = st.button("Send", on_click=send_message)
        
# CSS for making the first column scrollable
css='''
<style>
    section.main>div {
        padding-bottom: 1rem;
    }
    [data-testid="column"]>div>div>div>div>div {
        overflow: auto;
        height: 80vh;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

# JavaScript to manage the scroll position
scroll_js = """
<script>
function maintainScrollPosition(){
    var container = document.querySelector("[data-testid='column']>div>div>div>div>div");
    var scrollPosition = localStorage.getItem('scrollPosition');
    if (scrollPosition) {
        container.scrollTop = scrollPosition;
    }
    container.addEventListener('scroll', function() {
        localStorage.setItem('scrollPosition', container.scrollTop);
    });
}
// Call the function when the page loads
setTimeout(maintainScrollPosition, 500);
</script>
"""

st.markdown(scroll_js, unsafe_allow_html=True)
