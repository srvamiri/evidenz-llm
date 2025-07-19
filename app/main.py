from app.modules import uvicorn, FastAPI, BaseModel, List, HTTPException
from app.functions import get_solution

app = FastAPI()

# API Input and Output Model
class TicketInfo(BaseModel):
    ticket_id: str
    category: str
    conversation: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    response: str
    relevant_tickets: List[TicketInfo]

@app.post("/generate", response_model=QueryResponse)
async def generate_query_response(request: QueryRequest):
    try:
        response, relevant_tickets = get_solution(request.query, request.top_k)
        return QueryResponse(response=response, relevant_tickets=relevant_tickets)
    except Exception as e:
        print(f"❌ Error in get_solution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def run_app():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_app()