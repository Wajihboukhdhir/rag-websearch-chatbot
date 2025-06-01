# websearch.py
import os
import asyncio
import aiosqlite

from dotenv import load_dotenv
from langchain_community.utilities import SerpAPIWrapper
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
import chromadb
import re
crawler = None
from crawl4ai import AsyncWebCrawler

async def init_crawler():
    global crawler
    if crawler is None:
        crawler = AsyncWebCrawler()
        await crawler.__aenter__()
# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_key = os.getenv("SERP_API_KEY")

# Crawler Configuration
md_generator = DefaultMarkdownGenerator(
    content_filter=PruningContentFilter(
        threshold=0.6,  
        threshold_type="fixed",
        min_word_threshold=30 
    )
)

config = CrawlerRunConfig(
    markdown_generator=md_generator,
    excluded_tags=["form", "header", "footer", "nav", "aside", "script", "style"],
    remove_overlay_elements=True,
    process_iframes=True,
)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Search the Web 
async def get_web_urls(search_term: str) -> list[str]:
    try:
        serp_api = SerpAPIWrapper(serpapi_api_key=serpapi_key)
        results = await serp_api.aresults(search_term)
        urls = [
            r["link"] for r in results.get("organic_results", [])
            if "link" in r and not r["link"].lower().endswith(".pdf")
        ]
        return urls[:3]
    except Exception as e:
        print(f"❌ SERP API error: {e}")
        return []

# Crawl the Web Pages
async def crawl_urls(urls: list[str]) -> list[str]:
    content_list = []
    for url in urls:
        print(f"🌐 Crawling: {url}")
        try:
            result = await crawler.arun(url, config=config)
            content = result.markdown.strip()
            if content:
                content_list.append(f"# Source: {url}\n\n{content}")
        except Exception as e:
            print(f"❌ Failed to crawl {url}: {e}")
    return content_list

# Generate Answer from Crawled Data
async def get_model_answer(query: str, content: str) -> str:
    if not content:
        return "No usable content was found."

    # Split & embed content
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(content)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    collection_name = "web_content"

    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        client=chroma_client,
        collection_name=collection_name
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    print(context)
    # Prompt the model
    model = ChatGroq(groq_api_key=groq_api_key, model_name="Qwen-Qwq-32b")
    prompt = f"""You are an assistant of Ca'Foscari university of venice for students. Answer the following question using ONLY the provided context.

Question: {query}

Context:
{context}

If the answer is not found in the context, respond: "I couldn't find this information in the sources."

Provide a concise answer:"""

    try:
        response = await model.ainvoke(prompt)
        cleaned_content = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL)

        return cleaned_content.strip()
    except Exception as e:
        print(f"❌ Groq model error: {e}")
        return "Sorry, I couldn't process the query."

# High-Level Async Function
async def async_query(query: str) -> str:
    urls = await get_web_urls(query)
    if not urls:
        return "No relevant websites found."

    content_blocks = await crawl_urls(urls)
    combined_content = "\n\n".join(content_blocks)
    return await get_model_answer(query, combined_content)

# Public Function
async def get_info(query: str) -> str:
    return await async_query(query)
