"""
prompts.py


Here we will define all prompts we use to set up our LLM

J. A. Moreno
2026
"""

from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate


raw_query = ChatPromptTemplate(
    [
        ("system", 
         """
You are a Human Resources expert.
The user will provide a job interview query, the job post URL,
and optionally further instructions regarding formatting of the answer.

Identify the job_post_url, user_query, and user_instructions from
the user input. If not provided, use tools.
If user_instructions is not found, do not ask the user for them.
Output as valid JSON.
         """.strip()),
        ("human",
         "{raw_query}")
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

Generate a semantic search phrase that will retrieve the information needed to
answer the query from a vector store that contains the candidate's work
experience (resumes, cv, publications, grant applications). Use consice
wording.
Write as a valid JSON output named distill_query.
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

### USER INSTRUCTIONS
-------------
{user_instructions}
-------------
### END USER INSTRUCTIONS

### USER QUERY
-------------
{user_query}
-------------
### END USER QUERY

Draft an answer based on the user query and experience.
Answer questions in first person, using consice language and a profesional tone.
The answer must be grounded on the user information and be relevant
to the job post and initial query.

         """),
    ]
)

class DistilledQuerySchema(TypedDict):
    """
    Used to define the output for our distilled_query node
    """
    distilled_query: str | None

class ProcessUserInputSchema(TypedDict):
    """
    We take a natural-language user input that contains the
    URL for job_post_url and a job related user_query.
    """
    job_post_url: str | None
    user_query: str | None
