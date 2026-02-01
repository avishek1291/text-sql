# Patient Vector Database with LangChain, OpenRouter, and Hugging Face

This project uses **LangChain** framework to integrate **Hugging Face embeddings**, **Chroma vector database**, and **OpenRouter LLM** for creating a semantic vector database for patient medical records. It enables intelligent search and AI-powered question answering over patient data using various LLM models through OpenRouter.

## Features

- **Semantic Search**: Find patients using natural language queries via LangChain retriever
- **LangChain Integration**: Uses LangChain's abstractions for embeddings, vectors, and chains
- **Vector Embeddings**: Hugging Face's `sentence-transformers` for embedding generation
- **Persistent Storage**: Stores embeddings in Chroma vector database with LangChain
- **OpenRouter AI Integration**: Access multiple LLM models through OpenRouter API
- **Multiple Model Support**: Switch between different LLM providers and models
- **Document Management**: Automatic document parsing and metadata extraction

## Architecture

```
Patient Data (patient-data.txt)
    ↓
[LangChain Document Parsing]
    ↓
[Hugging Face Embeddings via LangChain]
    ↓
[Chroma Vector Store (LangChain)]
    ↓
[LangChain RetrievalQA + OpenRouter LLM]
    ↓
[Query Results + AI-Generated Answers]
```

## Tech Stack

- **LangChain**: Framework for building LLM applications
- **Chroma**: Vector database for embeddings
- **Hugging Face**: Sentence transformers for embeddings
- **OpenRouter**: API for accessing multiple LLM models
- **Python**: Implementation language

## Prerequisites

- Python 3.8+
- OpenRouter API key (free account at https://openrouter.ai)
- Internet connection (required for OpenRouter API calls)

## Installation

1. **Clone/Download the project**:
```bash
cd c:\Users\avish\RAG_TUTS\text-sql
```

2. **Install required packages**:
```bash
pip install -r requirements.txt
```

3. **Get OpenRouter API Key**:
   - Sign up at https://openrouter.ai (free account)
   - Get your API key from https://openrouter.ai/keys
   - Set environment variable or pass as parameter

4. **Set Environment Variable** (optional, can also pass in code):
```bash
# Windows PowerShell
$env:OPENROUTER_API_KEY = "your-api-key-here"

# Or create .env file
echo "OPENROUTER_API_KEY=your-api-key-here" > .env
```

## Usage

### Basic Setup and Search

```python
from vector_store import PatientVectorStore
import os

# Set API key
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"

# Initialize the vector store with OpenRouter
vector_store = PatientVectorStore(
    data_file="patient-data.txt",
    db_path="./patient_vector_db",
    model_name="all-MiniLM-L6-v2",
    collection_name="patient_records",
    openrouter_model="openai/gpt-3.5-turbo",
    openrouter_api_key=os.environ.get("OPENROUTER_API_KEY")
)

# Store patient data in vector database
vector_store.store_patient_data()

# Perform semantic search
results = vector_store.semantic_search("Show me patients with high blood pressure")
for result in results:
    print(f"Patient: {result['name']}")
    print(f"Similarity: {result['similarity_score']:.4f}")
```

### Question Answering with OpenRouter

```python
# Ask questions and get AI-powered answers using OpenRouter
result = vector_store.ask_question("Which patients need closer monitoring?")

print(f"Answer: {result['answer']}")
print("Source Documents:")
for doc in result['source_documents']:
    print(f"  - {doc['patient']} ({doc['patient_id']})")
```

### Query Examples

```python
# Different types of queries
queries = [
    "Find patients with elevated temperature",
    "Show me patients with high BMI",
    "Which patients have low oxygen saturation",
    "List patients with high heart rate",
    "Who has normal blood pressure"
]

for query in queries:
    result = vector_store.ask_question(query)
    print(f"{query}: {result['answer']}")
```
```
```

### Run the Complete Demo

```bash
python vector_store.py
```

This will:
1. Prompt for your OpenRouter API key (if not set in environment)
2. Load patient data from `patient-data.txt`
3. Generate embeddings using Hugging Face
4. Store embeddings in Chroma vector database
5. Perform semantic searches
6. Generate AI-powered answers using OpenRouter

## Project Structure

```
text-sql/
├── vector_store.py         # Main vector store implementation
├── chat.py                 # Chat interface (optional)
├── memory.py               # Memory management (optional)
├── patient-data.txt        # Patient records dataset
├── patient-data.json       # Structured patient data (optional)
├── requirements.txt        # Python dependencies
└── patient_vector_db/      # Vector database storage (created automatically)
```

## Configuration

### Vector Store Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_file` | str | `patient-data.txt` | Path to patient data file |
| `db_path` | str | `./patient_vector_db` | Vector database storage location |
| `model_name` | str | `all-MiniLM-L6-v2` | Hugging Face embedding model |
| `collection_name` | str | `patient_records` | Chroma collection name |
| `openrouter_model` | str | `openai/gpt-3.5-turbo` | OpenRouter model identifier |
| `openrouter_api_key` | str | None | OpenRouter API key (or set env var) |

### OpenRouter Models

Popular models available through OpenRouter:

**OpenAI Models**
- `openai/gpt-4-turbo-preview` - Most capable
- `openai/gpt-3.5-turbo` - Fast and cost-effective (default)
- `openai/gpt-3.5-turbo-16k` - Extended context

**Anthropic Models**
- `anthropic/claude-3-opus` - Most capable
- `anthropic/claude-3-sonnet` - Balanced
- `anthropic/claude-3-haiku` - Fast

**Open Source Models**
- `meta-llama/llama-2-70b-chat` - Open source
- `mistralai/mistral-7b-instruct` - Fast open source
- `nousresearch/nous-hermes-2-mixtral-8x7b` - Mixture of experts

**Local & Other Models**
- `google/palm-2-chat-bison` - Google's model
- `aleph-alpha/luminous-supreme` - European alternative
- Many more available on OpenRouter

### Embedding Models

Available Hugging Face models:

- `all-MiniLM-L6-v2` (384 dims) - Fast, recommended for most use cases
- `all-mpnet-base-v2` (768 dims) - Better quality, slower
- `paraphrase-MiniLM-L6-v2` (384 dims) - Balanced
- `all-distilroberta-v1` (768 dims) - Good quality

## API Reference

### PatientVectorStore Class

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_file` | str | `patient-data.txt` | Path to patient data file |
| `db_path` | str | `./patient_vector_db` | Vector database storage location |
| `model_name` | str | `all-MiniLM-L6-v2` | Hugging Face embedding model |
| `collection_name` | str | `patient_records` | Chroma collection name |
| `openrouter_model` | str | `openai/gpt-3.5-turbo` | OpenRouter model to use |
| `openrouter_api_key` | str | None | OpenRouter API key |

#### Methods

**`store_patient_data(force_refresh=False)`**
- Parse and store patient data in vector database
- Args: `force_refresh` (bool) - Clear and rebuild database
- Returns: True if successful

**`semantic_search(query, n_results=3)`**
- Perform semantic search using LangChain retriever
- Args: `query` (str), `n_results` (int)
- Returns: List of search results with similarity scores

**`ask_question(query)`**
- Ask a question using LangChain's RetrievalQA chain with OpenRouter LLM
- Args: `query` (str) - User question
- Returns: Dictionary with answer and source documents

**`get_database_stats()`**
- Get vector database statistics
- Returns: Dictionary with stats

**`parse_patient_records()`**
- Parse patient data and return LangChain Documents
- Returns: List of Document objects with metadata

## Example Output

```
======================================================================
LANGCHAIN PATIENT VECTOR DATABASE SYSTEM
======================================================================

======================================================================
STEP 1: Storing Patient Data in Vector Database
======================================================================
Loading embedding model: all-MiniLM-L6-v2
Initializing Ollama with model: mistral
Parsing patient records...
Parsed: P1001 - John Doe
Parsed: P1002 - Alice Smith
[...]
Successfully parsed 10 patient records

Storing documents in vector database...
Successfully stored 10 patient records

QA chain setup successful

======================================================================
Database Statistics
======================================================================
collection_name.................. patient_records
total_records..................... 10
db_path........................... ./patient_vector_db
embedding_dimension.............. 384
embedding_model.................. all-MiniLM-L6-v2
llm_model......................... mistral
status............................ Active

======================================================================
STEP 2: Semantic Search Examples (using similarity)
======================================================================

Query: Show me patients with high blood pressure
----------------------------------------------------------------------
  1. David Brown (P1005)
     Similarity Score: 0.8234
  2. Michael Johnson (P1009)
     Similarity Score: 0.7891

Query: Which patients have elevated temperature
----------------------------------------------------------------------
  1. Raj Kumar (P1003)
     Similarity Score: 0.8123
  2. Michael Johnson (P1009)
     Similarity Score: 0.7654

======================================================================
STEP 3: Question Answering with Ollama (via LangChain)
======================================================================

Query: Which patients have elevated vital signs?
----------------------------------------------------------------------
Answer:
Based on the patient records, the following patients have elevated vital signs:

1. Raj Kumar (P1003): Febrile (38.1°C), elevated respiratory rate (20), 
   elevated blood pressure (135/88)
2. Michael Johnson (P1009): Febrile (38.3°C), elevated respiratory rate (21),
   elevated blood pressure (145/92)

Source Documents:
  • Raj Kumar (P1003)
  • Michael Johnson (P1009)

======================================================================
Vector Store Setup Complete!
======================================================================
```

## Troubleshooting

### Issue: LangChain import errors

**Solution**: Install all required LangChain packages:
```bash
pip install langchain langchain-community langchain-text-splitters
```

### Issue: OpenRouter API key not found

**Solution**: Set your API key as environment variable:
```bash
# Windows PowerShell
$env:OPENROUTER_API_KEY = "your-api-key-here"

# Or in Python
import os
os.environ["OPENROUTER_API_KEY"] = "your-api-key-here"
```

Get your API key from: https://openrouter.ai/keys

### Issue: OpenRouter API errors

**Solution**: 
- Verify your API key is correct
- Check your account balance/credits
- Ensure internet connection is active
- Check if the model name is correct

### Issue: Memory/Performance with LangChain

**Solution**: 
- Use smaller embedding models
- Reduce chunk size
- Use map_reduce chain for large datasets

### Issue: File not found

**Solution**: Ensure `patient-data.txt` is in the current directory:
```bash
ls patient-data.txt  # Check file exists
pwd  # Verify working directory
```

## Performance Optimization

1. **Batch Processing**: For large datasets, process records in batches
2. **Model Selection**: Use smaller models for faster processing
3. **Database Indexing**: Chroma automatically indexes embeddings
4. **Caching**: Results are cached in the vector database

## Advanced Usage

### Switching Between OpenRouter Models

```python
from vector_store import PatientVectorStore
import os

# Using Claude (Anthropic)
vector_store = PatientVectorStore(
    openrouter_model="anthropic/claude-3-opus",
    openrouter_api_key=os.environ.get("OPENROUTER_API_KEY")
)

# Using Mistral (Open Source)
vector_store = PatientVectorStore(
    openrouter_model="mistralai/mistral-7b-instruct",
    openrouter_api_key=os.environ.get("OPENROUTER_API_KEY")
)

# Using GPT-4
vector_store = PatientVectorStore(
    openrouter_model="openai/gpt-4-turbo-preview",
    openrouter_api_key=os.environ.get("OPENROUTER_API_KEY")
)
```

### Custom Prompt Template

```python
from langchain.prompts import PromptTemplate

# Create custom prompt for specific use cases
custom_prompt = """You are a medical data analyst. Analyze the following patient records 
and provide detailed health insights.

Patient Context:
{context}

Question: {question}

Provide a comprehensive analysis with recommendations:"""

# Can be integrated into RetrievalQA chain
```

### Using .env File

```python
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access API key
api_key = os.environ.get("OPENROUTER_API_KEY")

vector_store = PatientVectorStore(
    openrouter_api_key=api_key
)
```

Create `.env` file:
```
OPENROUTER_API_KEY=sk-your-key-here
```

### Integration with Chat Systems

```python
def medical_chatbot(user_query, vector_store):
    result = vector_store.ask_question(user_query)
    return result['answer']

# Use in a chat loop
while True:
    query = input("Ask a medical question: ")
    answer = medical_chatbot(query, vector_store)
    print(f"Answer: {answer}\n")
```

## License

MIT License - See LICENSE file for details

## Support

For issues or questions, refer to the documentation or create an issue in the repository.

## References

- [Chroma Documentation](https://docs.trychroma.com)
- [Sentence Transformers](https://www.sbert.net)
- [Ollama](https://ollama.ai)
- [Hugging Face Models](https://huggingface.co/models)
