from app.modules import st, requests

def generate_response():
    st.title("Support Ticket Assistent")

    user_query = st.text_area("Supportanfrage eingeben")
    top_k = st.slider("Top K Übereinstimmungen", 1, 10, 3)

    if st.button("suchen", use_container_width=True):
        if user_query.strip() == "":
            st.error("Bitte geben Sie eine Supportanfrage ein.")
            return
        with st.spinner("🔍 Verarbeitung..."):
            try:
                # Call the API to generate a response
                response = requests.post(
                    "http://74.82.31.79:8000/generate",
                    json={"query": user_query, "top_k": top_k}
                )
                results = response.json()
                st.subheader("📂 Relevante Tickets:")
                for idx, ticket in enumerate(results.get("relevant_tickets", [])):
                    with st.expander(f"Ticket {idx+1}"):
                        full_text = f"Ticket-ID: {ticket['ticket_id']}\nCategory: {ticket['category']}\n{ticket['conversation']}\n"
                        st.write(full_text.replace("\n", "<br>").replace("\r\n", "<br>"),
                                 unsafe_allow_html=True)

                st.subheader("💬 Lösungsvorschlag:")
                st.write(results.get("response", "Keine Rückmeldung"))
            except requests.exceptions.RequestException as e:
                st.error(f"Fehler bei der API-Anfrage: {e}")
                return

if __name__ == "__main__":
    generate_response()
    
