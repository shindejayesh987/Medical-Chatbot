import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH="vectorstore/db_faiss"


load_dotenv()  
HF_TOKEN = os.getenv("HF_TOKEN")



@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm


def main():
    st.title("Ask Medibot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Have a medical query? Ask away, and let’s find the answers together!")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                    You are an AI assistant strictly limited to the given context. Your task is to provide accurate, concise, and contextually relevant answers based only on the provided information.

                    - If the answer is not in the context, respond with: "I don't have that information."
                    - Do NOT generate answers beyond the given context.
                    - Do NOT speculate, assume, or fabricate information.
                    - Maintain clarity and directness in your responses.


                    **Special Instructions:**
                    - If the user expresses gratitude (e.g., "thanks", "thank you"), respond with a polite reply such as: "You're welcome! Happy to help."
                    - If the user compliments you (e.g., "good work", "well done"), respond with a friendly acknowledgement like: "Thank you! I appreciate it."
                    - If the user greets you (e.g., "hello", "hi"), respond with a friendly greeting like: "Hello! How can I assist you today?"



                    Context: {context}

                    Question: {question}

                    Provide the most precise answer possible based on the context.
                    """
        
        HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN=os.environ.get("HF_TOKEN")

        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt})

            # result=response["result"]
            # source_documents=response["source_documents"]
            # result_to_show=result+"\nSource Docs:\n"+str(source_documents)
            # #response="Hi, I am MediBot!"
            # st.chat_message('assistant').markdown(result_to_show)
            # st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

            result = response["result"]
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})


        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()