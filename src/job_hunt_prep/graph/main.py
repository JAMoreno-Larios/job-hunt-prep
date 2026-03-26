"""
main.py

Here we can do a trial run for our agent.

J. A. Moreno
2026
"""

from graph import Graph

def main():
    print("Testing our graph")

    user_query = """
This is the job post:https://recruiting2.ultipro.com/INF1019IRINC/JobBoard/17a8d008-9efe-4e51-8460-47ee205d5229/OpportunityDetail?opportunityId=70528dc0-415e-4c8d-970f-46cf2a253bd0
They are asking: We are looking for people who are personally and/or
professionally passionate about AI. Please briefly explain how you have
put it to work for you in either or both areas of your life.
    """.strip()

    #Initialize graph
    graph = Graph()
    graph.draw_mermaid_png()
    
    for part in graph.run_agent(user_query):
        if part["type"] == "messages":
            # MessagesStreamPart — (message_chunk, metadata) from LLM calls
            msg, metadata = part["data"]
            print(msg.content, end="", flush=True)
    
    user_query = """
They are asking: What are your three main motivators for this position?

Additional instructions: Keep your answer below 500 characters.
    """.strip()

    for part in graph.run_agent(user_query):
        if part["type"] == "messages":
            # MessagesStreamPart — (message_chunk, metadata) from LLM calls
            msg, metadata = part["data"]
            print(msg.content, end="", flush=True)
    
    user_query = """
What was a challenge you found during your past experience and how did you solved it?

User insructions: Keep your answer below 250 characters, avoid technical jargon and use references
no older than 5 years.
    """.strip()

    for part in graph.run_agent(user_query):
        if part["type"] == "messages":
            # MessagesStreamPart — (message_chunk, metadata) from LLM calls
            msg, metadata = part["data"]
            print(msg.content, end="", flush=True)

if __name__ == "__main__":
    main()
