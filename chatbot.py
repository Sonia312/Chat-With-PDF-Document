import streamlit as st
import PyPDF2
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import os
import ollama

# Load Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Create or connect to the SQLite database
conn = sqlite3.connect('vector_database.db')
c = conn.cursor()

# Create table to store document vectors and chunks
c.execute('''CREATE TABLE IF NOT EXISTS vectors
             (id INTEGER PRIMARY KEY, chunk TEXT, vector TEXT)''')


def read_pdf(file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            text += page_text
    except Exception as e:
        st.error("An error occurred while reading the PDF")
        st.error(e)
    return text


def chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = ' '.join(words[start:end])
        chunk_embedding = embed([chunk_text])[0].numpy()
        chunk_vector = ','.join(map(str, chunk_embedding))
        c.execute("INSERT INTO vectors (chunk, vector) VALUES (?, ?)", (chunk_text, chunk_vector))
        chunks.append(chunk_text)
        start += chunk_size - overlap
    conn.commit()
    return chunks


def compute_similarity(query_vector, database_vectors):
    similarities = []
    for database_vector in database_vectors:
        database_vector = np.fromstring(database_vector[2], dtype=float, sep=',')
        similarity = cosine_similarity([query_vector], [database_vector])[0][0]
        similarities.append(similarity)
    return similarities


def createprompt(query):
    # Sample query
    query_embedding = embed([query])[0].numpy()

    # Retrieve vectors from the database
    c.execute("SELECT * FROM vectors")
    vectors = c.fetchall()

    # Compute similarity between query and database vectors
    similarities = compute_similarity(query_embedding, vectors)
    print(similarities)

    # Sort vectors based on similarity scores
    unique_similarities = set()
    top_similar_vectors = []
    for vector_id, similarity in sorted(enumerate(similarities, start=1), key=lambda x: x[1], reverse=True):
        if len(top_similar_vectors) == 4:
            break
        if similarity not in unique_similarities:
            unique_similarities.add(similarity)
            top_similar_vectors.append((vector_id, similarity))

    # Print top 4 vectors with highest similarity and their corresponding chunks
    text = ""
    for rank, (vector_id, similarity) in enumerate(top_similar_vectors, start=1):
        # Retrieve chunk corresponding to vector ID
        c.execute("SELECT chunk FROM vectors WHERE id=?", (vector_id,))
        chunk = c.fetchone()[0]
        text = text + f"Retrieved content number: {rank}\n{chunk}\n\n"

    prompt = (f"You are a chatbot. You'll receive a prompt that retrieved content from the vectorDB based on the user's question, and the source.\n Your task is to Give brief and concise answers to the user's new question using the information from the vectorDB without relying on your own knowledge.\
    you will receive a prompt with the the following format:\n{text}User question:\nNew question\n\nContent: {text}\nUser Question: {query}")

    print(text)

    return prompt, text


# Add exit button to delete the database
if st.button("Exit"):
    if os.path.exists('vector_database.db'):
        os.remove('vector_database.db')
    st.stop()

st.title("💬 Chat with your Document")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

### Write Message History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
    else:
        st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

## Upload PDF document
uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file is not None:
    pdf_text = read_pdf(uploaded_file)
    st.write("PDF Uploaded Successfully!")

    # Create chunks from PDF text
    chunks = chunk_text(pdf_text, 100, 25)

    print(chunks)

    # Store PDF text in session state (optional)
    st.session_state["pdf_text"] = pdf_text
    st.session_state["chunks"] = chunks


def generate_response(reference):
    response = ollama.chat(model='llama2', stream=True, messages=st.session_state.messages)
    full_message = ""
    for partial_resp in response:
        token = partial_resp["message"]["content"]
        full_message += token
        yield token
    st.session_state["full_message"] = full_message

    # Print references
    st.write(reference)

    # Clear session state
    st.session_state["messages"] = []
    st.session_state.pop("pdf_text", None)
    st.session_state.pop("chunks", None)
    st.session_state.pop("full_message", None)


if prompt := st.chat_input():
    # Display user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🧑‍💻").write(prompt)

    # Generate prompt and show it
    new_prompt, reference = createprompt(prompt)
    st.session_state.messages.append({"role": "user", "content": new_prompt})
    reference = "\nReferenced text : \n" + reference

    # Display LLAMA response
    st.chat_message("assistant", avatar="🤖").write_stream(generate_response(reference))

# Close the database connection
conn.close()
