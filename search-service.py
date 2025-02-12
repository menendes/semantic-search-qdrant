from fastapi import FastAPI, Query
from search_engine import SemanticSearchEngine

# Initialize FastAPI app
app = FastAPI()

# Initialize the search engine
search_engine = SemanticSearchEngine()

@app.get("/search")
def search(query: str = Query(..., description="Enter search query"), top_k: int = 5):
    """
    Perform semantic search using the search engine.
    :param query: The user query
    :param top_k: Number of results to return (default: 5)
    :return: List of relevant customer support issues
    """
    results = search_engine.search(query, top_k)
    return {"query": query, "results": results}

@app.get("/search/filter")
def search_with_filter(query: str = Query(..., description="Enter search query"), category: str = Query(..., description="Filter by category"), top_k: int = 5):
    """
    Perform semantic search with a category filter.
    :param query: The user query
    :param category: Category to filter results
    :param top_k: Number of results to return (default: 5)
    :return: List of relevant customer support issues filtered by category
    """
    results = search_engine.search_with_filter(query, category, top_k)
    return {"query": query, "category": category, "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
