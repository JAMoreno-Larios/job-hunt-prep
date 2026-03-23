"""
prompts.py


Here we will define all prompts we use to set up our LLM

J. A. Moreno
2026
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, HttpUrl


raw_query = ChatPromptTemplate(
    [
        ("system", 
         """
You are a Human Resources expert.
The user will provide a query containing a job interview query and
the related job post URL.

### USER RAW QUERY
-------------
{raw_query}
-------------
### END USER RAW QUERY

Identify the job_post_url and user_query from the input.
Write it as a valid JSON.

         """.strip()),
    ]
)


distill_query = ChatPromptTemplate(
    [
        ("system", 
         """
You are a Human Resources expert.
You have a job post description and a user query that is a 
job interview question.

### JOB DESCRIPTION
-------------
{job_post}
-------------

### END JOB DESCRIPTION

### USER QUERY
-------------
{user_query}
-------------
### END USER QUERY

Generate a semantic search query that will retrieve the information needed to
answer the query from a vector store that contains the candidate's work
experience (resumes, cv, publications, grant applications). Use consice
wording that can gather as diverse information as possible.
         """.strip()),
    ]
)

draft_answer = ChatPromptTemplate(
    [
        ("system", 
         """
You are a Human Resources expert.
You have a job post description, the user's relevant information,
and a user query that is a job interview question.

### JOB DESCRIPTION
-------------
{job_post}
-------------

### END JOB DESCRIPTION

### USER INFO
-------------
{user_info}
-------------

### END USER INFO

### USER QUERY
-------------
{user_query}
-------------
### END USER QUERY

Draft an answer based on the user query. Answer questions in
first person, using consice language and a profesional tone.
You must use the provided job post and user information
to write the answer.

         """),
    ]
)
