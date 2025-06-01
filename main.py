# main.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_groq import ChatGroq
import config
from websearch import get_info
import re

load_dotenv()

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    use_web_search: bool = False
    conversation_history: list = []


def get_rag_answer(query: str, conversation_history: list = []):
    cfg = config.read_config()

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vector_store = Chroma(
        collection_name=cfg['chroma_collection'],
        embedding_function=embeddings,
        persist_directory=os.path.join(cfg['chroma_base_dir'], cfg['chroma_collection'])
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)

    if not docs:
        return "I couldn't find relevant information in the documents."

    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="Qwen-Qwq-32b"
    )

    # Format conversation history
    history_str = ""
    if conversation_history:
        history_str = "\n\nPrevious conversation:\n"
        history_str += "\n".join([f"{'User' if i%2==0 else 'Assistant'}: {msg}" 
                                for i, msg in enumerate(conversation_history[-6:])])  # Keep last 3 exchanges

    prompt = f"""
* Important Notes for the Virtual Assistant  
- You are a virtual assistant for students at Ca' Foscari University of Venice.  
- Your role is to provide clear, accurate, and helpful answers based only on the official context provided (from university documents and communications).   
- You will be given a history of your conversation with the user, which will help inform your response. Always consider the history provided to ensure continuity and relevance in your response.   
- Assume students are not familiar with the processes, terminology, or institutional references.

* Key Instructions for Generating Responses:  
- **Understand First**: Carefully read and understand both the student's question and the provided context, including any history of previous interactions, before responding.  
- **Explain Everything Clearly**: If the context includes steps, forms, terms, or references (e.g., phases of a process, document names, codes) that could be unclear, always explain what they mean.  
- **Give Context**: Make sure the student knows where to look for the information or what specific documents or steps they need to focus on.  
- **Do Not Assume**: Do not assume that the student knows anything about the context of university processes, terminology, or documents. Always check if there are any terms that might be unclear to the student.  
- **Use Simple Language**: Avoid jargon or overly formal language. Keep the explanation clear and accessible.  
- If any term in your response might be ambiguous or unfamiliar, briefly define or explain it.  

* When Information Is Missing and If the Context Does Not Contain the Answer:  
- Never mention that you're relying on a context provided—treat all the information as if it were part of your own knowledge.  
- If the answer is not available, say: "I'm really sorry, but I don't have the exact information."  
- Recommend the student contact the appropriate university office for further details. If contact details (such as an email or phone number) are available, provide them to help the student reach the right department.
* **Review the Response**:  
- After generating the answer, re-read your response to check if there are any ambiguous terms or areas where clarity might be needed. If any terms might confuse the student, make sure to explain them.


1- History Conversation: {history_str}

2- Context: {context}

Question: {query}

Answer:"""

    response = llm.invoke(prompt)
    cleaned_content = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL)

    return cleaned_content


async def get_combined_answer(query: str, conversation_history: list = []):
    rag_response = get_rag_answer(query, conversation_history)
    web_response = await get_info(query)
    
    # Format conversation history
    history_str = ""
    if conversation_history:
        history_str = "\n\nPrevious conversation:\n"
        history_str += "\n".join([f"{'User' if i%2==0 else 'Assistant'}: {msg}" 
                                for i, msg in enumerate(conversation_history[-6:])])  # Keep last 3 exchanges
    
    combined_context = f"""
You are a virtual assistant for students at Ca' Foscari University of Venice. Your task is to provide clear, accurate, and helpful answers based only on the university’s official documents, communications, and available web search information.

1. Review History: Start by reviewing the student's previous interactions to ensure continuity. Consider their past questions and responses to understand their needs and any ongoing issues.

2. University Context: The information provided in the university documents or communications is your primary source. Ensure you base your answers on this context. Always clarify terms, processes, or documents that may be unfamiliar to students. Be thorough and explain everything clearly, including definitions for any ambiguous terms.

3. Web Search: If relevant and up-to-date information is required, perform a web search to gather the latest, official details. Provide students with clear and helpful information from the search results. If there’s no available answer, express that politely and suggest the student contact the appropriate university office.

4. Clear, Simple Language: Avoid jargon. Your responses should be accessible and straightforward. If you use a term that might be unfamiliar, make sure to define it briefly.

5. If Information is Missing: If the answer isn’t available in either the university’s context or the web search, say: "I am really sorry, but I do not have the exact information." Provide the relevant university contact information (e.g., email or phone) so the student can reach out for further assistance.


1.History Conversation: {history_str}
2. RAG Response: {rag_response}
3. Web Search Response: {web_response}
    
Now, based on thse information, answer the following question:
    
Question: {query}
"""
    
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="Qwen-Qwq-32b"
    )
    
    final_prompt = f"""
    Based on the information provided from conversaation history, the RAG response and the web search response, please answer the following question:
    
    {combined_context}
    
    Answer the question concisely by combining all the pieces of information:
    """
    
    response = llm.invoke(final_prompt)
    cleaned_content = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL)
    return cleaned_content


@app.post("/ask")
async def ask_question(request: QueryRequest):
    if request.use_web_search:
        answer = await get_combined_answer(request.query, request.conversation_history)
    else:
        answer = get_rag_answer(request.query, request.conversation_history)
    return {"answer": answer}


from websearch import init_crawler

@app.on_event("startup")
async def startup_event():
    await init_crawler()