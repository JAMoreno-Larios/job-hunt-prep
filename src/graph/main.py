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

    job_post_url = "https://intel.wd1.myworkdayjobs.com/en-US/External/job/AI-Research-Engineer--Temporary-Position-_JR0281357"
    query = "Besides compensation, what are the three most important things you are looking to get out of this opportunity with Intel?"

    initial_state = {
        'user_query': query,
        'job_post_url': job_post_url
    }
    
 
    for chunk in graph.stream(initial_state):
        print(chunk, end='')

if __name__ == "__main__":
    main()
