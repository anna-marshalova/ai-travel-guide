import os
import json
from typing import Dict, List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from data_processing import make_chunks, flatten_data

def print_retrieved_items(results, prefix, crop_length=30):
    def crop_text(text):
        text = text.replace("\n", " ")
        if crop_length:
            return text[:crop_length] + "..."
        else:
            return text

    print(
        f"{prefix}: "
        + ", ".join(
            [f"{crop_text(item.page_content)}: {score:.2f}" for item, score in results]
        )
    )


class HierarchicalRetrieval:
    def __init__(self, data: Dict[str, List[str]]=None, 
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 device: str = "cpu",
                 title_top_n: int = 10, 
                 chunks_per_title: int = 2, 
                 total_chunks: int = 5,
                 max_distance: float = 1,
                 title_index_path="vectorstores/title_index",
                 chunk_index_path="vectorstores/chunk_index"):
        """
        :param data: Dictionary with titles as keys and text chunks as values
        :param title_top_n: Number of top titles to retrieve
        :param chunks_per_title: Number of chunks to retrieve per title
        :param total_chunks: Total number of chunks to return after ranking
        """
        self.data = data
        self.title_top_n = title_top_n
        self.chunks_per_title = chunks_per_title
        self.total_chunks = total_chunks
        self.max_distance = max_distance
        self.embedding_model = embedding_model
        self.device = device
        self.title_vectorstore = None
        self.chunk_vectorstore = None
        self.title_index_path=title_index_path
        self.chunk_index_path=chunk_index_path

        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Prepare title and chunk vector stores
        self.title_vectorstore, self.chunk_vectorstore = self.prepare_vector_stores()

    def prepare_vector_stores(self, force_recreate=False, save=True):
        """Prepare vector stores for titles and chunks"""
        
        if not force_recreate:
            if not os.path.exists(self.title_index_path):
                print(f"Title vectorstore not found at {self.title_index_path}. Start indexing.")
            elif not os.path.exists(self.chunk_index_path):
                print(f"Chunk vectorstore not found at {self.title_index_path}. Start indexing.")
            else:
                print("Loading existing vectorstores...")
                title_vectorstore = FAISS.load_local(self.title_index_path, self.embeddings, allow_dangerous_deserialization=True)
                chunk_vectorstore = FAISS.load_local(self.chunk_index_path, self.embeddings, allow_dangerous_deserialization=True)
                return title_vectorstore, chunk_vectorstore
        else:
            print("Start indexing.")
                
        assert self.data is not None, "Data not provded. Please provide data (Dict[str, List[str]])."
        title_texts = list(self.data.keys())
        title_vectorstore = FAISS.from_texts(
            title_texts, 
            self.embeddings
        )
        # Chunks vector store
        all_chunks = []
        chunk_metadata = []
        for title, chunks in self.data.items():
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_metadata.append({"title": title})
        
        chunk_vectorstore = FAISS.from_texts(
            all_chunks, 
            self.embeddings, 
            metadatas=chunk_metadata
        )
        if save:
            self.save_vectorstores(title_vectorstore=title_vectorstore,
                                   chunk_vectorstore=chunk_vectorstore,
                                   title_index_path=self.title_index_path, 
                                   chunk_index_path=self.chunk_index_path)
        return title_vectorstore, chunk_vectorstore
            
    
    def retrieve(self, query: str, verbose: bool = False) -> List[str]:
        """
        Retrieve relevant chunks based on query

        1. First, find top N relevant titles
        2. Then, find M relevant chunks under each title
        3. Rank and return K total chunks
        """
        # Step 1: Retrieve top N titles
        title_results = self.title_vectorstore.similarity_search_with_score(
            query, k=self.title_top_n
        )

        # Collect relevant titles
        if verbose:
            print_retrieved_items(title_results, "Relevant titles", crop_length=None)
        relevant_titles = [result.page_content for result, _ in title_results]

        # Step 2: Retrieve chunks for each title
        all_relevant_chunks = []
        for title in relevant_titles:
            # Filter chunks by title metadata
            title_chunks = self.chunk_vectorstore.similarity_search_with_score(
                query, k=self.chunks_per_title, filter={"title": title}
            )
            all_relevant_chunks.extend(title_chunks)

        if verbose:
            print_retrieved_items(
                all_relevant_chunks, "All relevant chunks", crop_length=30
            )

        # Step 3: Rank and return top K chunks
        final_chunks = sorted(all_relevant_chunks, key=lambda x: x[1])[
            : self.total_chunks
        ]
        final_chunks = [
            chunk for chunk in final_chunks if chunk[1] <= self.max_distance
        ]
        if verbose:
            print_retrieved_items(final_chunks, "Final chunks", crop_length=30)

        return [chunk[0].page_content for chunk in final_chunks]
    
    def save_vectorstores(self, title_vectorstore=None, chunk_vectorstore=None, title_index_path=None, chunk_index_path=None):
        title_vectorstore = title_vectorstore or self.title_vectorstore
        chunk_vectorstore = chunk_vectorstore or self.chunk_vectorstore
        title_index_path = title_index_path or self.title_index_path
        chunk_index_path = chunk_index_path or self.chunk_index_path
        
        for name, path, vectorstore in [('Title', title_index_path, title_vectorstore),
                                        ('Chunk', chunk_index_path, chunk_vectorstore)]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            assert vectorstore, f"{name} vector store not found. Please call `prepare_vector_stores` method."
            vectorstore.save_local(path)
            print(f"{name} saved to {path}.")


# Example usage
def main():
    # # Sample dictionary
    # sample_dict = {
    #     "AI History": [
    #         "Early AI research began in the 1950s",
    #         "Machine learning emerged in the 1980s",
    #         "Deep learning revolution started in 2012"
    #     ],
    #     "Machine Learning": [
    #         "Supervised learning uses labeled data",
    #         "Unsupervised learning finds hidden patterns",
    #         "Reinforcement learning learns through interaction"
    #     ]
    # }

    # Initialize RAG system
    paths = os.listdir("data")
    data = []
    for path in paths:
        with open(f"data/{path}") as f:
            data.append(json.load(f))

    flat_data = flatten_data(data)
    chunked_data = make_chunks(flat_data)
    assert all(len(chunk)<2000 for chunks in chunked_data.values() for chunk in chunks)
    
    #chunked_data = dict(list(chunked_data.items())[:10])
    print(len(chunked_data))

    rag_system = HierarchicalRetrieval(chunked_data)
    # Example query
    query = "Что посмотреть в Шанхае"
    results = rag_system.retrieve(query)

    print("Retrieved Chunks:")
    for chunk in results:
        print("- " + chunk)


if __name__ == "__main__":
    main()
