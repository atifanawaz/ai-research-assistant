# citations/citation_formatter.py

from collections import defaultdict

def format_citations_grouped(citations):
    grouped = defaultdict(list)
    seen = set()

    for entry in citations:
        source = entry.get("source", "Unknown")
        page = entry.get("page", "N/A")
        content = entry.get("content", "").strip().replace("\n", " ")[:200]
        number = entry.get("number", "?")

        # Avoid repeating same page content
        if (source, page, content) in seen:
            continue
        seen.add((source, page, content))

        grouped[source].append({
            "page": page,
            "content": content,
            "number": number
        })

    output = ""
    for i, (source, entries) in enumerate(grouped.items(), start=1):
        # Show source name (hyperlink if URL)
        
       
        for e in entries:
            output += f"- Page {e['page']}: {e['content']} [^{e['number']}]\n"
            
    return output
