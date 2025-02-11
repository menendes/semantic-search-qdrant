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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
