import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END
from pprint import pprint
from langchain_core.messages.ai import AIMessage
from typing import Literal
from PIL import Image
from io import BytesIO
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from collections.abc import Iterable
from random import randint
from langgraph.prebuilt import InjectedState
from langchain_core.messages.tool import ToolMessage
load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] ="true"

class OrderState(TypedDict):
    messages:Annotated[list,add_messages]

    order:list[str]
    finished:bool


BARISTABOT_SYSINT = (
    "system",  # 'system' indicates the message is a system instruction.
    "You are a BaristaBot, an interactive cafe ordering system. A human will talk to you about the "
    "available products you have and you will answer any questions about menu items (and only about "
    "menu items - no off-topic discussion, but you can chat about the products and their history). "
    "The customer will place an order for 1 or more items from the menu, which you will structure "
    "and send to the ordering system after confirming the order with the human. "
    "\n\n"
    "Add items to the customer's order with add_to_order, and reset the order with clear_order. "
    "To see the contents of the order so far, call get_order (this is shown to you, not the user) "
    "Always confirm_order with the user (double-check) before calling place_order. Calling confirm_order will "
    "display the order items to the user and returns their response to seeing the list. Their response may contain modifications. "
    "Always verify and respond with drink and modifier names from the MENU before adding them to the order. "
    "If you are unsure a drink or modifier matches those on the MENU, ask a question to clarify or redirect. "
    "You only have the modifiers listed on the menu. "
    "Once the customer has finished ordering items, Call confirm_order to ensure it is correct then make "
        "any necessary updates and then call place_order. Once place_order has returned, thank the user and "
    "say goodbye!",
)

WELCOME_MSG = "Welcome to the BaristaBot cafe. Type `q` to quit. How may I serve you today?"

llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

def chatbot(state:OrderState)->OrderState:
    message_history=[BARISTABOT_SYSINT]+state["messages"]
    return {"messages":[llm.invoke(message_history)]}

graph_builder=StateGraph(OrderState)

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_edge(START,"chatbot")

chat_graph=graph_builder.compile()

# user_msg = "Hello, what can you do?"
# state = chat_graph.invoke({"messages": [user_msg]})

# #pprint(state)

# # for msg in state["messages"]:
# #     print(f"{type(msg).__name__}: {msg.content}")

# user_msg = "Oh great, what kinds of latte can you make?"

# state["messages"].append(user_msg)
# state = chat_graph.invoke(state)

# # pprint(state)
# for msg in state["messages"]:
#     print(f"{type(msg).__name__}: {msg.content}")

def human_node(state:OrderState)->OrderState:
    last_msg=state["messages"][-1]
    print("Model:",last_msg.content)
    user_input=input("User:")
    if user_input in {"q","quit","exit","goodbye"}:
        state["finished"]=True
    return state | {"messages": [("user", user_input)]}

def chatbot_with_welcome_msg(state:OrderState):
    if state["messages"]:
        new_output=llm.invoke([BARISTABOT_SYSINT] + state["messages"])
    else:
        new_output=AIMessage(content=WELCOME_MSG)
    return state | {"messages": [new_output]}

def maybe_exit_human_node(state:OrderState):
    if state.get("finished",False):
        return END
    else:
        return "chatbot"

graph_builder=StateGraph(OrderState)


graph_builder.add_node("chatbot",chatbot_with_welcome_msg)
graph_builder.add_node("human",human_node)
graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot", "human")
graph_builder.add_conditional_edges("human",maybe_exit_human_node)

chat_with_human=graph_builder.compile()
graph_bytes=chat_with_human.get_graph().draw_mermaid_png()
# graph_image=Image.open(BytesIO(graph_bytes))

# graph_image.show()

#state = chat_with_human.invoke({"messages": []})

@tool
def get_menu():
    """Provide the latest up-to-date menu."""
    return """
    MENU:
    Coffee Drinks:
    Espresso
    Americano
    Cold Brew

    Coffee Drinks with Milk:
    Latte
    Cappuccino
    Cortado
    Macchiato
    Mocha
    Flat White

    Tea Drinks:
    English Breakfast Tea
    Green Tea
    Earl Grey

    Tea Drinks with Milk:
    Chai Latte
    Matcha Latte
     London Fog

    Other Drinks:
    Steamer
    Hot Chocolate

    Modifiers:
    Milk options: Whole, 2%, Oat, Almond, 2% Lactose Free; Default option: whole
    Espresso shots: Single, Double, Triple, Quadruple; default: Double
    Caffeine: Decaf, Regular; default: Regular
    Hot-Iced: Hot, Iced; Default: Hot
    Sweeteners (option to add one or more): vanilla sweetener, hazelnut sweetener, caramel sauce, chocolate sauce, sugar free vanilla sweetener
    Special requests: any reasonable modification that does not involve items not on the menu, for example: 'extra hot', 'one pump', 'half caff', 'extra foam', etc.

    "dirty" means add a shot of espresso to a drink that doesn't usually have it, like "Dirty Chai Latte".
    "Regular milk" is the same as 'whole milk'.
    "Sweetened" means add some regular sugar, not a sweetener.

    Soy milk has run out of stock today, so soy is not available.
  """
tools=[get_menu]
tool_node=ToolNode(tools)
llm_with_tools=llm.bind_tools(tools)


def maybe_route_to_tools(state:OrderState):
    """Route between human or tool nodes, depending if a tool call is made."""
    if not (msgs := state.get("messages", [])):
        raise ValueError(f"No messages found when parsing state: {state}")
    msg=msgs[-1]
    
    if hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
        return "tools"
    else:
        return "human"
def chatbot_with_tools(state: OrderState) -> OrderState:
    """The chatbot with tools. A simple wrapper around the model's own chat interface."""
    defaults = {"order": [], "finished": False}

    if state["messages"]:
        new_output = llm_with_tools.invoke([BARISTABOT_SYSINT] + state["messages"])
    else:
        new_output = AIMessage(content=WELCOME_MSG)

    # Set up some defaults if not already set, then pass through the provided state,
    # overriding only the "messages" field.
    return defaults | state | {"messages": [new_output]}

@tool
def add_to_order(drink:str,modifiers:Iterable[str]):
    """Adds the specified drink to the customer's order, including any modifiers.

    Returns:
      The updated order in progress."""

@tool
def confirm_order():
    """Confirms orders"""

@tool
def get_order():
    """Returns the order so far"""

@tool
def clear_order():
    """Clears the orders"""

@tool
def place_order():
    """Sends the order to the barista"""

def order_node(state:OrderState)->OrderState:
    """Order state gets manipulated"""
    tool_msg=state.get("messages",[])[-1]
    order=state.get("order",["order"])
    outbound_msg=[]
    order_placed=False


    for tool_call in tool_msg.tool_calls:
        if tool_call["name"] == "add_to_order":

            # Each order item is just a string. This is where it assembled as "drink (modifiers, ...)".
            modifiers = tool_call["args"]["modifiers"]
            modifier_str = ", ".join(modifiers) if modifiers else "no modifiers"

            order.append(f'{tool_call["args"]["drink"]} ({modifier_str})')
            response = "\n".join(order)
        elif tool_call["name"]=="confirm_order":
            print("Your order:")
            if not order:
                print("  (no items)")

            for drink in order:
                print(f"  {drink}")

            response = input("Is this correct? ")
        elif tool_call["name"] == "get_order":

            response = "\n".join(order) if order else "(no order)"
        elif tool_call["name"] == "clear_order":

            order.clear()
            response = None
        elif tool_call["name"] == "place_order":

            order_text = "\n".join(order)
            print("Sending order to kitchen!")
            print(order_text)

            order_placed = True
            response = randint(1, 5)
        else:
            raise NotImplementedError(f'Unknown tool call: {tool_call["name"]}')
        outbound_msg.append(
            ToolMessage(
                content=response,
                name=tool_call["name"],
                tool_call_id=tool_call["id"]

            )
        )
        return {"messages": outbound_msg, "order": order, "finished": order_placed}
    
def maybe_route_to_tools(state:OrderState):
    if not(msgs := state.get("messages", [])):
        raise ValueError(f"No messages found when parsing state: {state}")
    msg=msgs[-1]
    if state.get("finished",False):
        return END
    elif hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
        # Route to `tools` node for any automated tool calls first.
        if any(
            tool["name"] in tool_node.tools_by_name.keys() for tool in msg.tool_calls
        ):
            return "tools"
        else:
            return "ordering"

    else:
        return "human"
    

# Auto-tools will be invoked automatically by the ToolNode
auto_tools = [get_menu]
tool_node = ToolNode(auto_tools)


order_tools = [add_to_order, confirm_order, get_order, clear_order, place_order]

# The LLM needs to know about all of the tools, so specify everything here.
llm_with_tools = llm.bind_tools(auto_tools + order_tools)


graph_builder = StateGraph(OrderState)

# Nodes
graph_builder.add_node("chatbot", chatbot_with_tools)
graph_builder.add_node("human", human_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("ordering", order_node)

# Chatbot -> {ordering, tools, human, END}
graph_builder.add_conditional_edges("chatbot", maybe_route_to_tools)
# Human -> {chatbot, END}
graph_builder.add_conditional_edges("human", maybe_exit_human_node)

# Tools (both kinds) always route back to chat afterwards.
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("ordering", "chatbot")

graph_builder.add_edge(START, "chatbot")
graph_with_order_tools = graph_builder.compile()

config = {"recursion_limit": 100}
state = graph_with_order_tools.invoke({"messages": []}, config)
