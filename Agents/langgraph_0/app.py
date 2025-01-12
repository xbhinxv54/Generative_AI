from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from typing import Annotated
from langgraph.graph import StateGraph,START,END
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages.ai import AIMessage
from PIL import Image
from io import BytesIO



GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] ="true"


llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")


class State(TypedDict):
    messages:Annotated[list,add_messages]
    finished:bool



INSTRUCTION="You are a helpful and very thorough assistant"
WELCOME="How should I help you today"

def chatbot(state:State):
    return {"messages":llm.invoke(state["messages"])}



def human(state:State):
    last_msg=state["messages"][-1]
    print("BOt:",last_msg.content)
    user_input=input("user:")
    if user_input in {"q","quit"}:
        state["finished"]=True
    return state| {"messages": [("user", user_input)]}


def welcome(state:State):
    if state["messages"]:
        output=llm.invoke([INSTRUCTION]+state["messages"])
    else:
        output=AIMessage(content=WELCOME)
    return state | {"messages":[output]}


def exit_chat(state:State):
    if state.get("finished",False):
        return END
    else:
        return "chatbot"

graph_builder=StateGraph(State)
graph_builder.add_node("chatbot",welcome)
graph_builder.add_node("human",human)
graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot","human")
graph_builder.add_conditional_edges("human",exit_chat)

graph=graph_builder.compile()
graph_bytes=graph.get_graph().draw_mermaid_png()

graph_image=Image.open(BytesIO(graph_bytes))

graph_image.show()

state = graph.invoke({"messages": []})

