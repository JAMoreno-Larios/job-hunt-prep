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

What's the best way to answer?
    """.strip()

    #Initialize graph
    graph = Graph()
    graph.draw_mermaid_png()
    graph.run_agent(user_query)

if __name__ == "__main__":
    main()
