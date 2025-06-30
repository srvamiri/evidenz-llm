from modules import uvicorn, FastAPI, BaseModel, List, HTTPException
from functions import get_solution
from hugging_face_config import load_hf

load_hf()

app = FastAPI()

# API Input and Output Model
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    response: str
    relevant_tickets: List[str]

# query = "TSS Login funktioniert nicht."
# response, relevant_tickets = get_solution(query, 3)

# for idx, ticket in enumerate(relevant_tickets):
#     print(f"Ticket {idx+1}.")
#     print(f"Ticket-ID: {ticket['ticket_id']}\nCategory: {ticket['category']}\n{ticket['conversation']}\n")

# print("Suggested response:\n", response, end='\n')

@app.post("/generate", response_model=QueryResponse)
async def generate_query_response(request: QueryRequest):
    try:
        response, relevant_tickets = get_solution(request.query, request.top_k)
        return QueryResponse(response=response, relevant_tickets=relevant_tickets)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# def run_app():
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# if __name__ == "__main__":
#     run_app()