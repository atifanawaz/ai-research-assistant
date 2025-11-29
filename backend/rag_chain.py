from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

def get_answer_with_citations(question, vectorstore):
    # Step 1: Get relevant docs using Max Marginal Relevance (MMR)
    relevant_docs = vectorstore.max_marginal_relevance_search(
        query=question,
        k=10,
        lambda_mult=0.7
    )

    if not relevant_docs:
        return "No relevant text found in uploaded documents.", []
    
    # Step 2: Initialize LLaMA 3 model via Groq (updated model)
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama3-16k",  # updated to supported model
        temperature=0
    )

    # Step 3: Use LangChain QA chain
    chain = load_qa_chain(llm, chain_type="stuff")

    # Step 4: Run the QA chain
    answer = chain.run(input_documents=relevant_docs, question=question)

    # Step 5: Check for fallback/generic response
    fallback_phrases = [
        "i don't know",
        "not mentioned",
        "not provided in the document",
        "no relevant context"
    ]
    if any(phrase in answer.lower() for phrase in fallback_phrases):
        return answer, []

    # ✅ Step 6: Automatically gather all unique document citations
    citations = []
    seen = set()

    for doc in relevant_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content.strip().replace("\n", " ")

        key = (source, page)
        if key in seen:
            continue
        seen.add(key)

        number = len(citations) + 1

        citations.append({
            "source": source,
            "page": page,
            "content": content[:200],
            "number": number
        })

    # Step 7: Add citation tags [1], [2], ... at the end of the answer
    ref_tags = sorted(set(f"[{entry['number']}]" for entry in citations))
    answer_with_refs = answer.strip() + " " + " ".join(ref_tags)

    return answer_with_refs, citations
