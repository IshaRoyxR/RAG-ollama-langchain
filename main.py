from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an intelligent AI assistant that answers user questions based on the provided context.

Here are some relevant documents: {reviews}

Here is the user's question: {question}

If the answer is not directly found in the provided documents, respond with general knowledge but make it clear which part came from the documents and which from general reasoning.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)