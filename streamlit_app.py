__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
import openai
from openai import OpenAI

client_openai = OpenAI()

system_prompt = "You're a helpful assistant who looks answers up for a user in a textbook and returns the answer to the user's question. If the answer is not in the textbook, you say 'I'm sorry, I don't have access to that information.'"
def get_completion(user_prompt, system_prompt, model="gpt-3.5-turbo"):
    try:
        completion = client_openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return completion.choices[0].message.content

    except openai.RateLimitError as e:
        # Check if the specific message is about quota
        if "insufficient_quota" in str(e):
            return "❌ Error: Token credit insufficient. Please top up your OpenAI billing."
        else:
            return "⏳ Error: Rate limit reached. Wait a moment and try again."
            
    except openai.OpenAIError as e:
        return f"⚠️ An OpenAI error occurred: {e}"

# Initialize Chroma client and collection
client_chroma = chromadb.PersistentClient('./mycollection')
collection = client_chroma.get_or_create_collection(name='RAG_Assistant', metadata={'hnsw:space':'cosine'})

# Update previous titles and markdown text
st.title("Similarity Search App")
st.markdown("This app uses Chroma to perform similarity searches on a collection of documents (textbooks about Anthropology of Food) and OpenAI to answer questions based on the search results.")
st.sidebar.title("Configuration")
st.sidebar.markdown("Adjust the settings for your query.")

# Add input text widget for user question
user_question = st.text_area('Ask a question', key='user_question')
# Add number of results to the sidebar
n_results = st.sidebar.number_input('Number of results', min_value=1, max_value=10, value=1)

# Create a button that triggers the action of querying the Chroma Collection
if st.button("Get Answers"):
    st.write(f"Question: {user_question}")
    st.write(f"Number of Results: {n_results}")
    results = collection.query(query_texts=[user_question], n_results=n_results, include=["documents", "metadatas"])

    search_results = []

    for res in results["documents"]:
        for doc, meta in zip(res, results["metadatas"][0]):
           # Format the document text and its metadata
            metadata_str = ", ".join(f"{key}: {value}" for key, value in meta.items())
            search_results.append(f"{doc}\nMetadata: {metadata_str}")
    search_text = "\n\n".join(search_results)

    prompt = f"""Your task is to answer the following user question using the supplied search results.
    User Question: {user_question}
    Search Results: {search_text}
    """
    ## Get and display the response from OpenAI
    response = get_completion(prompt, system_prompt)
    st.write(response)
    
    metadata_prompt = f"""
    Your task is to answer the following user question using the supplied search results. At the end of each search result will be Metadata. Cite the passages, their chunk index, and their URL in your answer.
    User Question: {user_question}
    Search Results: {search_text}
    """
