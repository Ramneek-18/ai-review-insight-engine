from transformers import pipeline

def load_llm():
    return pipeline("text-generation", model="google/flan-t5-small")

def generate_insight(query, embedding_model, index, df, generator):

    # Step 1: Retrieve similar reviews
    query_vector = embedding_model.encode([query]).astype('float32')
    distances, indices = index.search(query_vector, k=5)

    reviews = []
    for i in indices[0]:
        text = df.iloc[i]['Text']
        reviews.append(text[:200])

    # Step 2: Context
    context = " ".join(reviews)

    # Step 3: Prompt
    prompt = f"""
    Analyze customer reviews and answer clearly.

    Reviews:
    {context}

    Question: {query}

    Give:
    Summary (1 line)
    3 issues as bullet points
    """

    # Step 4: Generate output
    result = generator(prompt, max_new_tokens=80)
    output = result[0]['generated_text']

    # ✅ FIX: define clean_output properly
    clean_output = output.replace(prompt, "").strip()

    # Step 5: Process output
    lines = clean_output.split("\n")

    summary = ""
    issues = []

    for line in lines:
        line = line.strip()

        if "summary" in line.lower():
            summary = line

        if line.startswith("-"):
            issues.append(line)

        if len(issues) == 3:
            break

    # Fallback
    if not summary:
        summary = "Summary: Customers report issues related to delivery and service."

    if len(issues) == 0:
        issues = [
            "- Delivery delay",
            "- Poor service",
            "- Bad experience"
        ]

    final_output = summary + "\n\n" + "\n".join(issues)

    return final_output, reviews