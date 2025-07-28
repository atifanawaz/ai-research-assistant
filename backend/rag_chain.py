from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from config import GROQ_API_KEY


def get_answer_with_citations(question, vectorstore):
    # Step 1: Get relevant docs using max marginal relevance (MMR)
    relevant_docs = vectorstore.max_marginal_relevance_search(
        query=question,
        k=10,
        lambda_mult=0.7  # balance between relevance and diversity
    )

    # Step 2: Initialize LLaMA 3 model via Groq
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192",
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

    from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from config import GROQ_API_KEY


def get_answer_with_citations(question, vectorstore):
    # Step 1: Get relevant docs using max marginal relevance (MMR)
    relevant_docs = vectorstore.max_marginal_relevance_search(
        query=question,
        k=10,
        lambda_mult=0.7  # balance between relevance and diversity
    )

    # Step 2: Initialize LLaMA 3 model via Groq
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192",
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

    # Step 6: Stricter keyword filtering for true "real-world application" pages
    keywords = [
        "ai is used", "ai in", "applications of ai", "real-world applications",
        "used in healthcare", "used in fraud detection", "used in astronomy",
        "used in industry", "ai applications include", "some other applications"
    ]

    citations = []
    seen = set()

        # Ignore citations with these noisy patterns
    ignore_patterns = [
        "available online at", "references", "journal of", "doi", "copyright",
        "this article is", "open access", "all rights reserved"
    ]

    for doc in relevant_docs:
        content_lower = doc.page_content.lower()

        # Keep only chunks that match relevant keywords
        if not any(keyword in content_lower for keyword in keywords):
            continue

        # Skip if chunk contains unwanted metadata-like text
        if any(pattern in content_lower for pattern in ignore_patterns):
            continue


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

    # Step 7: Add citation numbers [1], [2], ... at the end of the answer
    ref_tags = sorted(set(f"[{entry['number']}]" for entry in citations))
    answer_with_refs = answer.strip() + " " + " ".join(ref_tags)

    return answer_with_refs, citations
