from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# LCEL: a prompt, a model and a parser, composed with |. The model is a
# stub that echoes, so the plumbing is all that's measured.
prompt = ChatPromptTemplate.from_messages([("system", "Answer briefly."), ("user", "{question}")])
model = RunnableLambda(lambda pv: AIMessage(content=f"(model saw {len(pv.to_messages())} messages, "
                                                    f"last: {pv.to_messages()[-1].content!r})"))
chain = prompt | model | StrOutputParser()
print(type(chain).__name__, "of", [type(s).__name__ for s in chain.steps])
print(chain.invoke({"question": "What is a KV cache?"}))
print(chain.batch([{"question": "a"}, {"question": "b"}]))

# What | built is ordinary function composition with a shared interface
# (invoke, batch, stream, async versions), equivalent to:
def by_hand(question):
    messages = prompt.invoke({"question": question})
    return StrOutputParser().invoke(model.invoke(messages))
print(by_hand("What is a KV cache?") == chain.invoke({"question": "What is a KV cache?"}))