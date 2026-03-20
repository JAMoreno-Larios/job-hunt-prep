"""
main.py

Here we can do a trial run for our agent.

J. A. Moreno
2026
"""

from graph import graph
from state import JobPrepState

# Generate the workflow diagram
graph.get_graph().draw_mermaid_png(output_file_path="workflow.png")


def main():
    print("Testing our graph")

    #    job_post_url = "https://intel.wd1.myworkdayjobs.com/en-US/External/job/AI-Research-Engineer--Temporary-Position-_JR0281357"
    #   query = "Besides compensation, what are the three most important things you are looking to get out of this opportunity with Intel?"
    job_post_url = "https://recruiting2.ultipro.com/INF1019IRINC/JobBoard/17a8d008-9efe-4e51-8460-47ee205d5229/OpportunityDetail?opportunityId=70528dc0-415e-4c8d-970f-46cf2a253bd0"
    query = "We are looking for people who are personally and/or professionally passionate about AI. Please briefly explain how you have put it to work for you in either or both areas of your life."

    initial_state = {
        'user_query': query,
        'job_post_url': job_post_url
    }
    
 
    for chunk in graph.stream(initial_state):
        print(chunk, end='')

if __name__ == "__main__":
    main()
