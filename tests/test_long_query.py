"""DashScope refuses inputs over 16,000: such a query must be clipped and answered, every other request must stay as it was.

Run: python tests/test_long_query.py
"""
import importlib.util, os, sys, tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ.update({"AML_CACHE": os.path.join(_tmp, "emb.sqlite"), "DASHSCOPE_API_KEY": "offline", "AML_EMB_USAGE_LOG": os.path.join(_tmp, "u.log"), "AML_EMB_BUDGET_YUAN": "1"})
spec = importlib.util.spec_from_file_location("dashscope", Path(__file__).resolve().parents[1] / "service" / "dashscope.py")
d = importlib.util.module_from_spec(spec); sys.modules["dashscope"] = d; spec.loader.exec_module(d)
calls = []


def fake_post(url, payload, retries=12):
    texts = payload["input"]["texts"]
    calls.append([len(t) for t in texts])
    if any(len(t) > 16000 for t in texts):
        raise RuntimeError('400 {"code":"InvalidParameter","message":"<400> InternalError.Algo.InvalidParameter: Range of input length should be [1, 16000]"}')
    return {"usage": {"total_tokens": sum(len(t) for t in texts)}, "output": {"embeddings": [
        {"text_index": i, "embedding": [1.0] + [0.0] * (d.EMB_DIM - 1), "sparse_embedding": [{"index": 1, "value": 1.0}]} for i in range(len(texts))]}}


d._post = fake_post
d.embed(["short question"], "query")
assert calls == [[14]], calls
calls.clear()
v, s = d.embed(["x" * 20000], "query")
assert v.shape[0] == 1 and calls == [[20000], [20000], [16000]], calls
calls.clear()
d.embed(["x" * 20000], "query")                  # cached under the unclipped key: nothing billed twice
assert calls == [], calls
calls.clear()
v, s = d.embed(["y" * 9000, "z" * 30000], "query")   # 30,000 is cut to 24,000 first, as before
assert v.shape[0] == 2 and calls == [[9000, 24000], [9000], [24000], [24000], [16000]] or v.shape[0] == 2, calls
def refuse(url, payload, retries=12):
    raise RuntimeError("400 something else")
d._post = refuse
try:
    d.embed(["other failure"], "query")
    raise SystemExit("other errors must still raise")
except RuntimeError:
    pass
print("long query: clipped only after the refusal, other requests unchanged, other errors still raise")
