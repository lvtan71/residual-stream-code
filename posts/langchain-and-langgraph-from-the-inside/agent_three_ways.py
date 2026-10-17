import inspect
import time
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

# The same calculator agent three ways: a hand-written loop, a LangGraph
# graph built by hand, and one using LangGraph's prebuilt tool node. The
# "model" is a deterministic stub (call the calculator once, then answer),
# so what's compared is the code around the model.
def stub_model(messages):
    last = messages[-1]
    if isinstance(last, HumanMessage):
        return AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": "8113-423+7*755"},
                                                  "id": "call_1"}])
    return AIMessage(content=f"The warehouse holds {last.content} boxes.")

@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression."""
    return str(eval(expression, {"__builtins__": {}}))

# 1. By hand.
def run_by_hand(question, max_steps=5):
    messages = [HumanMessage(question)]
    for _ in range(max_steps):
        reply = stub_model(messages)
        messages.append(reply)
        if not reply.tool_calls:
            return messages
        for call in reply.tool_calls:
            messages.append(ToolMessage(calculator.invoke(call["args"]), tool_call_id=call["id"]))
    return messages

# 2. LangGraph, by hand: the state is a list of messages; nodes return
# updates to it; a conditional edge decides where to go after the model.
def model_node(state: MessagesState):
    return {"messages": [stub_model(state["messages"])]}

def tools_node(state: MessagesState):
    calls = state["messages"][-1].tool_calls
    return {"messages": [ToolMessage(calculator.invoke(c["args"]), tool_call_id=c["id"]) for c in calls]}

def route(state: MessagesState):
    return "tools" if state["messages"][-1].tool_calls else END

def build(checkpointer=None, tools=tools_node):
    g = StateGraph(MessagesState)
    g.add_node("model", model_node)
    g.add_node("tools", tools)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", route, ["tools", END])
    g.add_edge("tools", "model")
    return g.compile(checkpointer=checkpointer)

# 3. LangGraph with its prebuilt pieces (which also handle unknown tools
# and tool errors).
def build_prebuilt():
    g = StateGraph(MessagesState)
    g.add_node("model", model_node)
    g.add_node("tools", ToolNode([calculator]))
    g.add_edge(START, "model")
    g.add_conditional_edges("model", tools_condition)
    g.add_edge("tools", "model")
    return g.compile()

q = "A warehouse holds 8113 boxes, ships 423, then receives 7 deliveries of 755. How many now?"
graph, prebuilt = build(), build_prebuilt()
answers = {run_by_hand(q)[-1].content, graph.invoke({"messages": [HumanMessage(q)]})["messages"][-1].content,
           prebuilt.invoke({"messages": [HumanMessage(q)]})["messages"][-1].content}
print("same final answer:", len(answers) == 1, "->", answers.pop())
lines = lambda *fs: sum(len([l for l in inspect.getsource(f).splitlines() if l.strip() and not l.strip().startswith(("#", '"""'))]) for f in fs)
print(f"lines of code around the model and tool: by hand {lines(run_by_hand)}, "
      f"graph by hand {lines(model_node, tools_node, route, build)}, graph with prebuilt tool node "
      f"{lines(model_node, build_prebuilt)}")

def timeit(f, n=2000):
    f()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - t0) / n * 1e6
saved, runs = build(InMemorySaver()), iter(range(10 ** 9))   # a fresh conversation (thread) for every run
print(f"time per run with an instant model: by hand {timeit(lambda: run_by_hand(q)):.0f} us, "
      f"graph {timeit(lambda: graph.invoke({'messages': [HumanMessage(q)]})):.0f} us, "
      f"graph with a checkpointer {timeit(lambda: saved.invoke({'messages': [HumanMessage(q)]}, {'configurable': {'thread_id': str(next(runs))}})):.0f} us")

# Human approval: the tools node asks for it with interrupt(), which pauses
# the run and saves its state; Command(resume=...) continues it with the
# person's reply, here an edited expression.
def approved_tools(state: MessagesState):
    call = state["messages"][-1].tool_calls[0]
    decision = interrupt({"tool": call["name"], "args": call["args"]})
    args = decision.get("args", call["args"])
    return {"messages": [ToolMessage(calculator.invoke(args), tool_call_id=call["id"])]}
g2 = build(InMemorySaver(), tools=approved_tools)
cfg = {"configurable": {"thread_id": "demo"}}
paused = g2.invoke({"messages": [HumanMessage(q)]}, cfg)
print(f"\npaused for approval: {paused['__interrupt__'][0].value}")
final = g2.invoke(Command(resume={"args": {"expression": "8113-423+7*755+100"}}), cfg)
print("resumed with an edited expression:", final["messages"][-1].content)
history = list(g2.get_state_history(cfg))
print("checkpoints saved:", [(s.metadata.get("step"), s.next) for s in history])

# Replay: run again from an earlier checkpoint, the one where the model had
# just asked for the tool. That forks the thread and pauses again at the
# approval; this time approve the call unchanged.
before_tool = next(s for s in history if s.next == ("tools",))
again = g2.invoke(None, before_tool.config)
print("replaying from step", before_tool.metadata["step"], "-> paused again:", "__interrupt__" in again)
replayed = g2.invoke(Command(resume={"approve": True}), cfg)
print("approved as is this time:", replayed["messages"][-1].content)