# LangChain Practice Projects

Practice projects for learning RAG (Retrieval-Augmented Generation) and AI agents with LangChain.

## Projects

### 1. Simple RAG (`rag/simplerag.ipynb`)
A question-answering system over my own documents. It runs fully on my own machine, so it costs nothing.

What we built, step by step:
1. **Loaded data** from a text file, a website, and a PDF (a sample company profile, `Nova_Labs_Company_Profile.pdf`).
2. **Split** the text into small chunks.
3. **Embedded** each chunk (turned it into numbers) with a free HuggingFace model.
4. **Stored** the embeddings in Chroma, a vector database.
5. **Retrieved** the chunks closest to a question.
6. **Answered** using a local LLM (Qwen via Ollama), with a prompt template and a chain.

Example: ask "What is the email of Nova Labs?" and it finds the right part of the document and answers from it.

### 2. Agents (`Agents/agents.ipynb`)
A multi-source agent. The LLM reads the question and picks the right tool on its own, then answers from the tool's result. It runs fully on my machine with a local Ollama model, so it costs nothing.

The agent has three tools:
- **Wikipedia tool** for general knowledge (people, places, history).
- **arXiv tool** for research papers.
- **Website retriever tool** that answers questions about the LangChain tools documentation. The page is loaded, split, embedded (HuggingFace), stored in Chroma, and turned into a tool with `create_retriever_tool`.

What we built, step by step:
1. **Built each tool**: a wrapper for Wikipedia and arXiv, and a retriever tool for the website.
2. **Combined** the tools into one list.
3. **Loaded a local LLM** (Qwen via Ollama) instead of a paid API.
4. **Pulled a ready-made prompt** from LangChain Hub (`hwchase17/openai-functions-agent`).
5. **Created the agent** with `create_openai_tools_agent` and ran it with `AgentExecutor`. With `verbose=True` you can watch which tool it chooses.

Example: "Who was Alan Turing?" uses Wikipedia, and "Find the arxiv paper 1706.03762" uses arXiv.

### 3. Other folders
`api/` and `chatbot/` are empty for now. They are placeholders for later work.

## Setup

```powershell
python -m pip install -r requirements.txt
```

- Use `python -m pip`, because `pip` alone is not on PATH on this machine.
- Put keys in the `.env` file in the project root (for example `OPENAI_API_KEY=`). Never share this file. A local-only setup does not need a key.
- Ollama must be running, with the `qwen2.5:3b` model available.
- The Ollama model runs on CPU (`num_gpu=0`), because the GPU crashed on long prompts.
- In notebooks, run cells from the top after any kernel restart.
- The agents notebook needs `arxiv` older than version 3 (`python -m pip install "arxiv<3"`), because the newest version breaks the LangChain arXiv tool.
- The agents notebook also uses `faiss-cpu`, `langchain`, and `langchainhub`.

---

# Summary of Concepts

## Jupyter Notebook and Kernel
A notebook (`.ipynb`) is code split into cells that you run one at a time, with the output shown below each cell. The **kernel** is the Python process that runs the code, and `ipykernel` is the package that lets Python act as one. Variables stay in memory between cells, so run cells in order. After a restart, run them again from the top.

## Environment and Packages
- `requirements.txt` lists the packages the project needs.
- Install with `python -m pip install -r requirements.txt`. Writing a package in the file does not install it.
- `.env` stores secrets like API keys. `load_dotenv()` reads it, and it searches the notebook's folder and its parent folders.

## RAG (Retrieval-Augmented Generation)
An LLM does not know your private documents. RAG finds the relevant parts of your documents and gives them to the LLM along with the question, so it answers from real data. The steps are: load, split, embed, store, retrieve, answer.

## Document Loaders
Loaders read data and return `Document` objects (text plus metadata).
- `TextLoader` for text files (use `encoding="utf-8"` to avoid garbled characters)
- `WebBaseLoader` for websites
- `PyPDFLoader` for PDFs

## BeautifulSoup (bs4)
`WebBaseLoader` uses bs4 to read a web page's HTML. A `SoupStrainer` picks which tags to keep (for example only `<p>`, or `header` and `footer` too). Images have no text, so the loader skips them.

## Text Splitter (Chunking)
Long text is cut into small pieces so retrieval is precise and the prompt stays short. `RecursiveCharacterTextSplitter` has `chunk_size` (max length of a piece) and `chunk_overlap` (shared text between neighbouring pieces, so no sentence loses its context).

## Embeddings
An embedding model turns text into a list of numbers (a vector) that captures its meaning. Texts with similar meaning get similar vectors. We used the free local model `all-MiniLM-L6-v2`, which gives 384 numbers per text. Always use the same model for the documents and the questions. OpenAI embeddings are paid, but cheap.

## Vector Store (Chroma)
A database that stores chunks together with their vectors and searches by meaning instead of by keyword. `Chroma.from_documents(document, embeddings)` embeds and stores the chunks. `similarity_search(query)` finds the closest ones.

## Retriever
A wrapper around the vector store that returns the most relevant chunks for a question (`db.as_retriever()`). `k` sets how many chunks to return. It only finds chunks and never writes the answer.

## LLM and Ollama
The LLM writes the final answer. Ollama runs models locally, and `langchain-ollama` connects it to LangChain (`ChatOllama`). If Ollama's GPU crashes on long prompts, run on CPU with `num_gpu=0`.

## Prompt Template
A reusable prompt with blanks, such as `{context}` and `{question}`. It is filled in each time, so the instruction stays the same while the data changes.

## Chain
Steps connected with `|`, where each output feeds the next step:
`retriever | format_docs | prompt | llm | StrOutputParser()`

- `format_docs` joins the chunks into one text.
- `StrOutputParser` turns the model reply into plain text.
- The older `create_retrieval_chain` from `langchain.chains` did the same job. It now lives in `langchain_classic`.

## Wrapper
A helper class that hides how an outside service works. `WikipediaAPIWrapper` talks to Wikipedia and returns simple text. A wrapper by itself is not a tool.

## Tools
A tool is a function the LLM can ask to run. It has a name, a description (the LLM reads it to decide when to use the tool), and the function itself. `WikipediaQueryRun` turns the Wikipedia wrapper into a tool. Tools let an LLM get information it was not trained on.

## Agents
An agent is an LLM with tools and a loop. It reads the question, picks a tool, reads the tool's result, and repeats until it can answer. With several tools (Wikipedia, arXiv, a retriever), it picks the best source for each question. Tool calling needs a model that supports it, and small models can be unreliable at it.

## Retriever Tool
A retriever can also be given to an agent as a tool with `create_retriever_tool`. You give it a name and a description, such as "useful for questions about LangChain tools". The agent then searches your own documents whenever the question matches that description.

## Wikipedia vs arXiv
- **Wikipedia**: general knowledge.
- **arXiv**: free archive of research papers, for technical and recent topics.

`top_k_results` sets how many results to return, and `doc_content_chars_max` cuts each one to a maximum length. A short result keeps the prompt small, which matters for a small local model.

## User-Agent (Wikipedia 403)
Wikipedia rejects requests that do not say who is sending them. Without a User-Agent the `wikipedia` library gets a 403 error page, and it crashes with a confusing `JSONDecodeError`. The fix is `wikipedia.wikipedia.USER_AGENT = "..."` before creating the tool.

## LangChain Hub
A place to share ready-made prompts. `Client().pull_prompt("owner/name", dangerously_pull_public_prompt=True)` downloads one. The flag is required because a prompt from a stranger could contain unwanted instructions, so you confirm that you trust it. A prompt is only text, so it works with any model, including Ollama.

## Agent and AgentExecutor
- `create_openai_tools_agent(llm, tools, prompt)` builds the agent: the LLM decides which tool to call. The name says "openai", but it works with any model that supports tool calling, such as Qwen in Ollama.
- `AgentExecutor` runs the loop: ask the LLM, run the chosen tool, give the result back, and repeat until there is an answer.
- The prompt has an `agent_scratchpad` slot where the tool calls and results are kept between steps.
- `tools` must be a flat list. A list inside a list gives `ValueError: Unsupported function`.
- In LangChain 1.x these older classes live in `langchain_classic.agents`.
