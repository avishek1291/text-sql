"""
Vector database management using LangChain, OpenRouter, and Hugging Face embeddings.
Stores patient data from patient-data.txt in a persistent vector database.
"""

import os
import json
import re
from typing import List, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document


class PatientVectorStore:
    """
    A class to manage patient data embeddings and vector database operations.
    Uses LangChain with Hugging Face embeddings and Ollama for semantic search.
    """
    
    def __init__(
        self,
        data_file: str = "patient-data.txt",
        db_path: str = "./patient_vector_db",
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "patient_records",
        openrouter_model: str = "meta-llama/llama-3.3-70b-instruct",
        openrouter_api_key: str = None
    ):
        """
        Initialize the vector store using LangChain with OpenRouter.
        
        Args:
            data_file: Path to the patient data text file
            db_path: Path to store the vector database
            model_name: Hugging Face model for embeddings
            collection_name: Name of the vector collection
            openrouter_model: OpenRouter model identifier
            openrouter_api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
        """
        self.data_file = data_file
        self.db_path = db_path
        self.collection_name = collection_name
        self.openrouter_model_name = openrouter_model
        
        # Create db directory if it doesn't exist
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize Hugging Face embeddings via LangChain
        print(f"Loading embedding model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"}
        )
        
        # Set API key if provided
        if openrouter_api_key:
            os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
        
        # Verify API key is set
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable "
                "or pass openrouter_api_key parameter."
            )
        
        # Initialize OpenRouter LLM
        print(f"Initializing OpenRouter with model: {openrouter_model}")
        self.llm = ChatOpenAI(
            model_name=openrouter_model,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
            top_p=0.95,
            max_tokens=1024
        )
        
        # Initialize Chroma vector store (will be populated later)
        self.vector_store = None
        self.qa_chain = None
        
        print(f"Vector store initialized with OpenRouter")
    
    def parse_patient_records(self) -> List[Document]:
        """
        Parse patient records from the text file and create LangChain Documents.
        
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by "Record" pattern to identify individual records
            record_blocks = re.split(r'Record \d+', content)[1:]
            
            for block in record_blocks:
                patient_text = block.strip()
                if not patient_text:
                    continue
                
                # Extract patient ID
                patient_id_match = re.search(r'\((P\d+)\)', patient_text)
                patient_id = patient_id_match.group(1) if patient_id_match else "Unknown"
                
                # Extract patient name
                name_match = re.search(r'Patient ([A-Za-z\s]+), a', patient_text)
                patient_name = name_match.group(1).strip() if name_match else "Unknown"
                
                # Extract gender
                gender_match = re.search(r', a (Male|Female)', patient_text)
                gender = gender_match.group(1) if gender_match else "Unknown"
                
                # Extract date of birth
                dob_match = re.search(r'born on (\d{4}-\d{2}-\d{2})', patient_text)
                dob = dob_match.group(1) if dob_match else "Unknown"
                
                # Extract vital signs - Heart Rate
                heart_rate_match = re.search(r'Heart Rate (\d+)', patient_text)
                heart_rate = int(heart_rate_match.group(1)) if heart_rate_match else None
                
                # Extract blood pressure
                bp_match = re.search(r'Blood Pressure (\d+/\d+)', patient_text)
                blood_pressure = bp_match.group(1) if bp_match else "Unknown"
                
                # Extract temperature
                temp_match = re.search(r'Temperature ([\d.]+)°C', patient_text)
                temperature = float(temp_match.group(1)) if temp_match else None
                
                # Extract oxygen saturation
                o2_match = re.search(r'Oxygen Saturation (\d+)', patient_text)
                oxygen_saturation = int(o2_match.group(1)) if o2_match else None
                
                # Extract BMI
                bmi_match = re.search(r'BMI of ([\d.]+)', patient_text)
                bmi = float(bmi_match.group(1)) if bmi_match else None
                
                # Create metadata dictionary
                metadata = {
                    "patient_id": patient_id,
                    "name": patient_name,
                    "gender": gender,
                    "dob": dob,
                    "heart_rate": str(heart_rate) if heart_rate else "N/A",
                    "blood_pressure": blood_pressure,
                    "temperature": str(temperature) if temperature else "N/A",
                    "oxygen_saturation": str(oxygen_saturation) if oxygen_saturation else "N/A",
                    "bmi": str(bmi) if bmi else "N/A"
                }
                
                # Create LangChain Document
                doc = Document(page_content=patient_text, metadata=metadata)
                documents.append(doc)
                print(f"Parsed: {patient_id} - {patient_name}")
            
            print(f"\nSuccessfully parsed {len(documents)} patient records")
            return documents
            
        except FileNotFoundError:
            print(f"Error: File '{self.data_file}' not found.")
            return []
        except Exception as e:
            print(f"Error parsing patient records: {e}")
            return []
    
    
    def store_patient_data(self, force_refresh: bool = False) -> bool:
        """
        Parse patient data and store in Chroma vector database via LangChain.
        
        Args:
            force_refresh: If True, clear and rebuild the database
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if database already exists
            persist_dir = Path(self.db_path)
            existing_db = persist_dir.exists() and len(list(persist_dir.glob("*"))) > 0
            
            if existing_db and not force_refresh:
                print("Loading existing vector database...")
                self.vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=str(self.db_path)
                )
                self._setup_qa_chain()
                print(f"Loaded existing database with {self.vector_store._collection.count()} records")
                return True
            
            # Clear and rebuild if force refresh
            if force_refresh and existing_db:
                print("Clearing existing database...")
                import shutil
                shutil.rmtree(self.db_path)
                Path(self.db_path).mkdir(parents=True, exist_ok=True)
            
            # Parse patient records
            print("\nParsing patient records...")
            documents = self.parse_patient_records()
            
            if not documents:
                print("No patient records to store.")
                return False
            
            # Store documents in Chroma via LangChain
            print("\nStoring documents in vector database...")
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=str(self.db_path)
            )
            
            # Persist the database
            self.vector_store.persist()
            
            # Setup QA chain
            self._setup_qa_chain()
            
            print(f"Successfully stored {len(documents)} patient records")
            return True
            
        except Exception as e:
            print(f"Error storing patient data: {e}")
            return False
    
    def _setup_qa_chain(self):
        """
        Setup the RetrievalQA chain for question answering.
        """
        try:
            if self.vector_store is None:
                print("Vector store not initialized")
                return
            
            # Custom prompt template for medical context
            prompt_template = """You are a helpful medical assistant. Use the following patient records to answer the question.
If you don't know the answer based on the provided context, say so.

Patient Records:
{context}

Question: {question}

Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create retrieval chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            print("QA chain setup successful")
            
        except Exception as e:
            print(f"Error setting up QA chain: {e}")
    
    
    def semantic_search(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Perform semantic search using LangChain's retriever.
        
        Args:
            query: Search query string
            n_results: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        try:
            if self.vector_store is None:
                print("Vector store not initialized. Call store_patient_data() first.")
                return []
            
            # Use similarity search
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=n_results
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "id": doc.metadata.get("patient_id", "Unknown"),
                    "name": doc.metadata.get("name", "Unknown"),
                    "document": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": 1 - score  # Convert distance to similarity
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error performing semantic search: {e}")
            return []
    
    
    def ask_question(self, query: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer using LangChain's RetrievalQA chain.
        
        Args:
            query: User question
            
        Returns:
            Dictionary with query, answer, and source documents
        """
        try:
            if self.qa_chain is None:
                print("QA chain not initialized. Call store_patient_data() first.")
                return {"error": "QA chain not initialized"}
            
            # Run the chain
            result = self.qa_chain({"query": query})
            
            return {
                "query": query,
                "answer": result.get("result", "No answer"),
                "source_documents": [
                    {
                        "patient": doc.metadata.get("name", "Unknown"),
                        "patient_id": doc.metadata.get("patient_id", "Unknown"),
                        "content": doc.page_content[:300] + "..."
                    }
                    for doc in result.get("source_documents", [])
                ]
            }
            
        except Exception as e:
            print(f"Error asking question: {e}")
            return {"error": str(e)}
    
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current vector database.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            if self.vector_store is None:
                return {"status": "Vector store not initialized"}
            
            count = self.vector_store._collection.count()
            embedding_dim = len(self.embeddings.embed_query("test"))
            
            return {
                "collection_name": self.collection_name,
                "total_records": count,
                "db_path": str(self.db_path),
                "embedding_dimension": embedding_dim,
                "embedding_model": "all-MiniLM-L6-v2",
                "llm_provider": "OpenRouter",
                "llm_model": self.openrouter_model_name,
                "status": "Active" if count > 0 else "Empty"
            }
        except Exception as e:
            return {"status": "Error", "error": str(e)}


def main():
    """Main function to demonstrate LangChain vector store with OpenRouter."""
    
    # Get API key from environment or prompt
    api_key = "sk-or-v1-0b19cb0e091705f034fd6d8e8a2b751e9aeba"#os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable or pass it as a parameter")
        print("Get your API key from: https://openrouter.ai/keys")
        api_key = input("Enter your OpenRouter API key: ").strip()
    
    # Initialize vector store with LangChain and OpenRouter
    print("=" * 70)
    print("LANGCHAIN PATIENT VECTOR DATABASE SYSTEM (with OpenRouter)")
    print("=" * 70)
    
    vector_store = PatientVectorStore(
        data_file="patient-data.txt",
        db_path="./patient_vector_db",
        model_name="all-MiniLM-L6-v2",
        collection_name="patient_records",
        openrouter_model="meta-llama/llama-3.3-70b-instruct",  # You can change this model
        openrouter_api_key=api_key
    )
    
    # Store patient data
    print("\n" + "=" * 70)
    print("STEP 1: Storing Patient Data in Vector Database")
    print("=" * 70)
    vector_store.store_patient_data(force_refresh=True)
    
    # Display database stats
    print("\n" + "=" * 70)
    print("Database Statistics")
    print("=" * 70)
    stats = vector_store.get_database_stats()
    for key, value in stats.items():
        print(f"{key:.<25} {value}")
    
    # Example semantic searches
    print("\n" + "=" * 70)
    print("STEP 2: Semantic Search Examples (using similarity)")
    print("=" * 70)
    
    queries = [
        # "Show me patients with high blood pressure",
        # "Which patients have elevated temperature",
        "Find patients with BMI above 26",
        "Find patients with height over 175 cm"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 70)
        results = vector_store.semantic_search(query, n_results=2)
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['name']} ({result['id']})")
            print(f"     Similarity Score: {result['similarity_score']:.4f}")
    
    # Example with LangChain QA chain
    print("\n" + "=" * 70)
    print("STEP 3: Question Answering with OpenRouter (via LangChain)")
    print("=" * 70)
    
    qa_queries = [
        "Which patients have elevated vital signs?",
        "Who has the highest blood pressure?",
        "Which patients are at risk based on their health metrics?"
    ]
    
    for qa_query in qa_queries:
        print(f"\nQuery: {qa_query}")
        print("-" * 70)
        
        result = vector_store.ask_question(qa_query)
        
        print("Answer:")
        print(result.get("answer", "No answer available"))
        
        print("\nSource Documents:")
        for doc in result.get("source_documents", []):
            print(f"  • {doc['patient']} ({doc['patient_id']})")
    
    print("\n" + "=" * 70)
    print("Vector Store Setup Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
