from app.modules import np, torch
from app.load_model import load_model
from app.dataset import get_dataset, _embedder

docs, metadata, index = get_dataset()
model, tokenizer = load_model()

def build_rag_prompt(retrieved_context, new_ticket):
    context_block = "\n".join([f"{i+1}. {t['conversation']}" for i, t in enumerate(retrieved_context)])
    prompt = f"""Du bist ein technischer Supportassistent. Basierend auf dem unten beschriebenen Problem und früheren, verwandten Tickets, gib bitte eine hilfreiche und genaue Lösung. Bitte zeig nicht die vorherigen Tickets.

Verwandten Tickets:
{context_block}

Neues Ticket:
{new_ticket}

Loesung:"""
    return prompt

def extract_solution(response_text):
    parts = response_text.split("Loesung:")
    if len(parts) > 1:
        return parts[1].strip()
    else:
        return response_text.strip()
    
# Function to retrieve relevant tickets
def search(query: str, top_k=3):
    query_vec = _embedder.encode([query])
    D, indices = index.search(np.array(query_vec), top_k)
    
    top_tickets = []

    for idx in indices[0]:
          top_tickets.append({
              "ticket_id": metadata[idx]["ID"],
              "category": metadata[idx]["category"],
              "conversation": docs[idx]
          })
    return top_tickets

def generate_response(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode the output
    raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = extract_solution(raw_response)
    return response

def get_solution(query: str, top_k=3):
    # Retrieve relevant tickets
    retrieved_tickets = search(query, top_k)
    if not retrieved_tickets:
        return "Keine relevanten Tickets gefunden.", []
    
    # Build the prompt
    prompt = build_rag_prompt(retrieved_tickets, query)
    
    # Generate the response
    response = generate_response(prompt)
    if not response:
        return "Keine Rückmeldung.", ""

    return response, retrieved_tickets
