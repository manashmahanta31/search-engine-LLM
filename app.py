import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent,AgentType
# from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.tools import tool
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#from langgraph.prebuilt import create_react_agent, create_openai_tools_agent
import os
from dotenv import load_dotenv

## Arxiv and wikipedia Tools

## 🛠️ CUSTOM ARXIV TOOL PATCH
# Bypasses the broken 'ArxivQueryRun' class to completely avoid the 'Search' object has no attribute 'results' bug.
@tool("Arxiv_Search")
def arxiv(query: str) -> str:
    """Useful for when you need to search and answer questions about scientific or research papers from Arxiv."""
    from langchain_community.utilities import ArxivAPIWrapper
    try:
        wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        return wrapper.run(query)
    except Exception as e:
        return f"Could not find any relevant papers for query: {query}"


# arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
# arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

search=DuckDuckGoSearchRun(name="Search")


st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assisstant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.warning("Please enter your Groq API Key in the sidebar.")
        st.stop()


    llm=ChatGroq(groq_api_key=api_key,model="openai/gpt-oss-120b",streaming=True)
    tools=[search,arxiv,wiki]
# model = Llama3-8b-8192 is decommissioned. Use openai/gpt-oss-120b instead.
    
    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True)

# ZERO_SHOT_REACT_DESCRIPTION don't rely on chat history. It makes a decision based on the current input only.
# If we use CHAT_ZERO_SHOT_REACT_DESCRIPTION, it will rely on the chat history and may not perform well in this case.   
    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
         # Here  st.session_state.messages we are passing entire array list. If we pass just the single prompt to the agent. It will decide which tool to use and how to use it based on the prompt.
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)



# import os
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_community.utilities import WikipediaAPIWrapper
# from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
# from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
# from langchain_core.tools import tool
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.agents import create_react_agent, AgentExecutor

# # Load environment variables
# load_dotenv()

# ## 🛠️ CUSTOM ARXIV TOOL PATCH
# @tool("Arxiv_Search")
# def arxiv(query: str) -> str:
#     """Useful for when you need to search and answer questions about scientific or research papers from Arxiv."""
#     from langchain_community.utilities import ArxivAPIWrapper
#     try:
#         wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
#         return wrapper.run(query)
#     except Exception as e:
#         return f"Could not find any relevant papers for query: {query}"


# ## Wikipedia and DuckDuckGo Tools
# api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
# wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
# search = DuckDuckGoSearchRun(name="Search")

# st.title("🔎 LangChain - Chat with search")

# ## Sidebar for settings
# st.sidebar.title("Settings")
# api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# if "messages" not in st.session_state:
#     st.session_state["messages"] = [
#         {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
#     ]

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg['content'])

# if prompt := st.chat_input(placeholder="What is machine learning?"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)

#     if not api_key:
#         st.warning("Please enter your Groq API Key in the sidebar.")
#         st.stop()

#     # 1. Initialize modern LLM configuration
#     llm = ChatGroq(groq_api_key=api_key, model="openai/gpt-oss-120b", streaming=True)
#     tools = [search, arxiv, wiki]
    
#     # 2. Modern ReAct Prompt Template with a strict system constraint layer
#     react_prompt = ChatPromptTemplate.from_messages([
#         ("system", 
#          "You are a helpful assistant. You have access to tools to look up information. "
#          "If you choose to use a tool, you must write 'Thought:' followed by your reasoning, "
#          "then 'Action:' followed by the tool name on a new line. Do not write text between them. "
#          "Available tools: {tool_names}\nTool descriptions:\n{tools}"),
#         ("human", "{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ])

#     # 3. Create the robust modern ReAct agent engine
#     agent = create_react_agent(llm, tools, react_prompt)
    
#     # 4. Wrap inside a clean Executor loop
#     search_agent = AgentExecutor(
#         agent=agent, 
#         tools=tools, 
#         verbose=True, 
#         handle_parsing_errors=True
#     )
   
#     with st.chat_message("assistant"):
#         st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
#         # Execute using standard input keys
#         result = search_agent.invoke({"input": prompt}, {"callbacks": [st_cb]})
#         response = result["output"]
        
#         st.session_state.messages.append({'role': 'assistant', "content": response})
#         st.write(response)
