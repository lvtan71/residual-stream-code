import json
import random
import re
import time
import uuid
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# A small agent, traced. Qwen2.5-1.5B-Instruct with one tool, a calculator,
# on 20 word problems that each need two arithmetic steps on 3- and 4-digit
# numbers. Each problem is run 5 times (temperature 0.7, and the model's
# default repetition penalty of 1.1, stated explicitly). Every model call
# and tool call is recorded as a span-shaped record with OpenTelemetry's
# GenAI attribute names, including the opt-in message contents and each
# call's finish reason; the evaluation reads nothing else but the answer key.
PATH = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct",
                         revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
MODEL = "qwen2.5-1.5b-instruct"
tok = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float16).to(DEVICE).eval()
TOOLS = [{"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic "
          "expression with + - * / and parentheses, and return the number.",
          "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                         "required": ["expression"]}}}]

def calculator(expression):
    if not re.fullmatch(r"[\d\s+\-*/().]+", expression):
        return "error: only numbers and + - * / ( ) are allowed"
    return str(round(eval(expression), 6))                    # safe: the regex admits arithmetic only

rng = random.Random(0)
tasks = []
for _ in range(20):
    a, b, c, d = rng.randint(1000, 9999), rng.randint(100, 999), rng.randint(3, 9), rng.randint(100, 999)
    tasks.append((f"A warehouse holds {a} boxes. It ships out {b} boxes, then receives {c} deliveries of "
                  f"{d} boxes each. How many boxes does it hold now?", a - b + c * d))

def span(trace, name, parent, **attrs):
    s = {"trace_id": trace["id"], "span_id": uuid.uuid4().hex[:16], "parent_id": parent, "name": name,
         "start": time.perf_counter(), "attributes": attrs}
    trace["spans"].append(s)
    return s

def run(question, seed, max_turns=5):
    torch.manual_seed(seed)
    trace = {"id": uuid.uuid4().hex, "spans": []}
    root = span(trace, "invoke_agent warehouse-helper", None, **{"gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "warehouse-helper"})
    messages = [{"role": "system", "content": "Use the calculator tool for every arithmetic step. "
                 "End with: Answer: <number>"}, {"role": "user", "content": question}]
    answer = lenient = final = None
    for _ in range(max_turns):
        prompt = tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, tokenize=False)
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        s = span(trace, f"chat {MODEL}", root["span_id"], **{"gen_ai.operation.name": "chat",
                 "gen_ai.provider.name": "local", "gen_ai.request.model": MODEL, "gen_ai.request.temperature": 0.7,
                 "gen_ai.request.max_tokens": 512})
        out = model.generate(ids, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=1.0, top_k=0,
                             repetition_penalty=1.1)
        new = out[0, ids.shape[1]:]
        text = tok.decode(new, skip_special_tokens=True)
        s["end"] = time.perf_counter()
        stopped = new[-1].item() in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"))
        s["attributes"].update({"gen_ai.usage.input_tokens": ids.shape[1], "gen_ai.usage.output_tokens": len(new),
                                "gen_ai.response.finish_reasons": ["stop" if stopped else "length"],
                                "gen_ai.output.messages": [{"role": "assistant", "content": text}]})
        calls = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
        messages.append({"role": "assistant", "content": text})
        if not calls:
            final = text
            m = re.search(r"Answer:\s*\$?(-?[\d,]+)", text)               # the format we asked for
            answer = int(m.group(1).replace(",", "")) if m else None
            nums = re.findall(r"-?\d[\d,]*", text)                        # or just the last number in the reply
            lenient = int(nums[-1].replace(",", "")) if nums else None
            break
        for c in calls:
            try:
                args = json.loads(c)["arguments"]; expr = args["expression"]
            except (ValueError, KeyError, TypeError):
                args, expr = c, None
            # the model emits no call id, so the harness assigns one
            t = span(trace, "execute_tool calculator", root["span_id"], **{"gen_ai.operation.name": "execute_tool",
                     "gen_ai.tool.name": "calculator", "gen_ai.tool.call.id": uuid.uuid4().hex[:8],
                     "gen_ai.tool.call.arguments": args})
            result = calculator(expr) if expr is not None else "error: malformed tool call"
            t["end"] = time.perf_counter()
            if result.startswith("error"):
                t["attributes"]["error.type"] = "invalid_arguments"; t["status"] = "ERROR"
            else:
                t["attributes"]["gen_ai.tool.call.result"] = result
            messages.append({"role": "tool", "content": result})
    root["end"] = time.perf_counter()
    return trace, answer, lenient, final

results = []
for i, (q, truth) in enumerate(tasks):
    for k in range(5):
        trace, ans, lenient, final = run(q, seed=100 * i + k)
        results.append({"task": i, "truth": truth, "answer": ans, "lenient": lenient, "final": final, "trace": trace})
    print(f"task {i:2d}: {sum(r['answer'] == truth for r in results[-5:])}/5 correct as asked, "
          f"{sum(r['lenient'] == truth for r in results[-5:])}/5 by the last number", flush=True)
json.dump(results, open("traces.json", "w"))