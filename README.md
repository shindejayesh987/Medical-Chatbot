### **MediBot 🤖 - AI-Powered Medical Chatbot**

MediBot is an advanced AI-powered medical chatbot that leverages **Retrieval Augmented Generation (RAG)** to provide accurate and context-aware responses to user queries based on medical documents. The application is built using a modular approach, integrating a **vector database (FAISS)** for memory, the **Mistral model** via **HuggingFace**, and a user-friendly **Streamlit UI** for seamless interactions.

---

## **Table of Contents**

- [Features](#features)
- [Project Layout](#project-layout)
- [Technical Stack](#technical-stack)
- [medichatbot.py - Detailed Explanation](#medichatbotpy---detailed-explanation)
- [Setup Instructions](#setup-instructions)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Conclusion](#conclusion)

---

## **Features**

- 📚 Uses **RAG (Retrieval Augmented Generation)** for contextually accurate responses.
- 🧠 **Vector embeddings** for memory management with **FAISS**.
- 🌐 **Streamlit-based UI** for an interactive chatbot experience.
- 🔍 Handles multiple medical documents efficiently.
- 🔄 Modular design with separate phases for memory, LLM connection, and UI.

---

## **Project Layout**

The project is divided into three main phases:

### **Phase 1: Setup Memory for LLM (Vector Database)**

- Load raw medical PDFs.
- Split them into manageable chunks.
- Generate vector embeddings using **FAISS**.
- Store embeddings for efficient retrieval.

### **Phase 2: Connect Memory with LLM**

- Setup the **Mistral** model using **HuggingFace**.
- Integrate FAISS for memory-based retrieval.
- Create a chain for smooth communication between memory and LLM.

### **Phase 3: Setup UI for the Chatbot**

- Build a **Streamlit UI** for user interactions.
- Load vector store (FAISS) into cache for fast access.
- Implement RAG for accurate and context-aware responses.

---

## **Technical Stack**

- **Langchain:** AI framework for LLM applications.
- **HuggingFace:** Model hosting and inference.
- **Mistral:** LLM model for understanding and generating text.
- **FAISS:** Vector database for efficient similarity search.
- **Streamlit:** Frontend framework for building UI.
- **Python:** Core programming language.
- **VS Code:** IDE for development.

---

## **medichatbot.py - Detailed Explanation**

### **1. Overview**

The `medichatbot.py` script serves as the **backend** for MediBot, handling:

- Loading vector embeddings from FAISS.
- Interacting with the Mistral model to generate responses.
- Managing user queries via Streamlit UI.

---

### **2. Key Components of `medichatbot.py`**

#### **2.1. Environment Setup**

- Uses `.env` for managing API keys and sensitive information.
- Loads vector embeddings stored in the **vectorstore** directory.

**Example `.env` Configuration:**

```bash
API_KEY=your_huggingface_api_key
```

---

#### **2.2. Loading and Storing Vectors**

- Uses **FAISS** to manage vector embeddings for memory.
- Supports multiple documents and efficient similarity search.

**Code Snippet Example:**

```python
import faiss
vector_store = faiss.read_index("vectorstore/vectors.index")
```

---

#### **2.3. Connecting to Mistral (LLM)**

- Utilizes **HuggingFace** API to query the Mistral model.
- Manages responses with context from FAISS-based memory.

**Sample Code for Query:**

```python
response = model.generate(input_ids, max_length=512)
```

---

#### **2.4. Streamlit UI for Chatbot**

- Simple and user-friendly UI using Streamlit.
- Supports real-time chat and displays responses.

**Example UI Code:**

```python
import streamlit as st

st.title("MediBot - AI Medical Chatbot")
user_input = st.text_input("Enter your medical query:")
if st.button("Ask"):
    st.write("Response from MediBot...")
```

---

### **3. Running `medichatbot.py`**

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Run the chatbot:**

```bash
streamlit run medichatbot.py
```

**Access the UI at:** [http://localhost:8501](http://localhost:8501)

---

### **4. Handling Vector Embeddings**

**Generating embeddings:**

```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vector = embeddings.embed("Sample medical text")
```

**Storing embeddings in FAISS:**

```python
import faiss
index = faiss.IndexFlatL2(512)
index.add(vector)
faiss.write_index(index, "vectorstore/vectors.index")
```

---

## **Limitations**

1. **Memory Storage:**

   - **FAISS** is in-memory, limiting scalability for large datasets.

2. **API Dependency:**

   - Requires **HuggingFace API** for Mistral model access.

3. **Performance:**

   - High latency for large documents due to vector search.

4. **Security:**

   - `.env` file must be secured to prevent API key exposure.

5. **Limited UI Features:**
   - Basic UI without user authentication or document uploads.

---

## **Future Enhancements**

1. **Authentication:**

   - Implement OAuth or token-based auth for secure access.

2. **Persistent Storage:**

   - Use **SQLite** or **MongoDB** to store vector embeddings.

3. **Multi-document Handling:**

   - Allow users to upload multiple PDFs dynamically.

4. **Advanced UI:**

   - Include feedback buttons and history in Streamlit.

5. **Asynchronous Processing:**
   - Speed up API requests using async functions.

---

## **Setup Instructions**

### **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/medibot.git
cd medibot
```

---

### **2. Setup Environment**

**Create `.env` file:**

```bash
API_KEY=your_huggingface_api_key
```

**Install Python dependencies:**

```bash
pip install -r requirements.txt
```

---

### **3. Run MediBot**

**Start the Streamlit UI:**

```bash
streamlit run medichatbot.py
```

**Access at:** [http://localhost:8501](http://localhost:8501)

---

### **4. Using FAISS for Vector Storage**

**Create vector embeddings:**

```python
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
vector = embeddings.embed("Medical text")
```

**Save to FAISS:**

```python
import faiss
index = faiss.IndexFlatL2(512)
index.add(vector)
faiss.write_index(index, "vectorstore/vectors.index")
```

---

## **Conclusion**

MediBot is a modern AI-powered medical chatbot that effectively uses RAG with FAISS and Mistral to deliver accurate and context-aware responses. While it has limitations in terms of memory and scalability, future enhancements can significantly broaden its capabilities. Contributions are welcome! 🚀🤖🎉

**Happy chatting with MediBot!** 🩺✨
