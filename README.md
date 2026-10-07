
# College AI Assistant 🎓

College AI Assistant is a RAG-based chatbot that helps students get answers related to college academics, fees, exams, attendance, and other general questions.

I built this project using **Python, LangChain, LangGraph, FAISS, Hugging Face Embeddings, Groq, and Streamlit**.

## What this project does

The main idea of this project is to provide different answers depending on the type of question asked by the student.

For example:

* Academic question → searches the academic handbook
* Fee question → searches the fee structure
* General question → directly asks the LLM

The query classification and routing are handled using **LangGraph**.

---

## How it works

```text
User Question
      ↓
   Classifier
      ↓
 ┌────┼─────┐
 ↓    ↓     ↓
Academic Fee  General
 ↓    ↓       ↓
PDF  PDF     LLM
 ↓    ↓       ↓
 └────┼───────┘
      ↓
   Final Answer
```

For academic and fee questions, the relevant information is retrieved from PDF documents using **FAISS vector search**.

---

## Technologies Used

* Python
* LangChain
* LangGraph
* FAISS
* Hugging Face Embeddings
* Groq LLM
* PyPDF
* Streamlit
* Python-dotenv

---

## Project Structure

```text
College-AI-Assistant/
│
├── app.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── state.py
│   ├── retrievers.py
│   ├── nodes.py
│   └── graph.py
│
├── data/
│   ├── academics_handbook.pdf
│   └── fee_structure.pdf
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Main Features

* Student can select their programme
* Academic questions are handled using RAG
* Fee-related questions are handled using RAG
* General questions are answered using the LLM
* PDF documents are converted into chunks
* Hugging Face embeddings are used for vector representation
* FAISS is used for similarity search
* LangGraph is used for conditional workflow
* Streamlit is used for the frontend

---

## Example Questions

### Academic

```text
What is the minimum attendance required?

How many credits are required?

What are the examination rules?
```

### Fees

```text
What is the semester fee?

What is the late fee?

What is the refund policy?
```

### General

```text
Hello

What is Python?

What can you help me with?
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/College-AI-Assistant.git
```

### 2. Open the project

```bash
cd College-AI-Assistant
```

### 3. Create virtual environment

```bash
python -m venv .venv
```

### 4. Activate it

For Windows:

```bash
.venv\Scripts\activate
```

### 5. Install required packages

```bash
pip install -r requirements.txt
```

### 6. Add Groq API key

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

Do not upload the `.env` file to GitHub.

### 7. Run Streamlit

```bash
streamlit run app.py
```

---

## Future Improvements

I am planning to add more features to this project, such as:

* Persistent FAISS vector database
* Source and page references in answers
* Chat history and conversational memory
* Query rewriting
* PDF upload from Streamlit
* Multiple PDF support
* Better retrieval
* Answer validation
* Response streaming
* Project deployment

---

## What I Learned From This Project

While building this project, I learned how to:

* Build a basic RAG application
* Load and process PDF documents
* Create embeddings
* Use FAISS for similarity search
* Work with LangChain
* Build workflows using LangGraph
* Connect an LLM with retrieved documents
* Create a chatbot interface using Streamlit

---

## Author

**Rishabh Suratiya**

This project is part of my learning journey in **AI/ML and AI Engineering**.

---

## Project Status

🚧 Currently working on improving the project and adding more features.
