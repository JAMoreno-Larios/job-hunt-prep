# /graph

Here we will have our Agent graph implementation.
We will adhere to the outline described
[here](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph).

## Rough outline

What the agent will do is that, based on the user's ingested work history - 
resumé, curriculum vitae, grant applications, and the like - and a given
job post, can answer to prompts found in job applications, like
'Why do you think you are a good fit in this company?' and 
'List your experience in X field'.

The steps would be:
- Provide job post URL, parse it.
- User creates query
- Search in the user-made database
- Draft an answer
- Ask the user for review
- If accepted, save to disk; if not, accept user feedback and generate new
text.

## Files

The graph files would be:
- `state.py`: Stores the graph state definitions
- `search_node.py`: Implements database search and web parsing.
- `input.py`: Deals with user initial query.
- `response.py`: Takes care of drafting a response with the data we've
gathered so far. Include user feedback and saving routine.
- `graph.py`: This will have the graph compilation code.
