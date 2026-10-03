import json, os
from unittest import mock
os.environ["GEMINI_API_KEY"] = "test"
from streamlit.testing.v1 import AppTest
import llm

# 1. validation: out-of-range dropped, bool/NaN ignored, off-topic refused
v, n = llm.validate({"in_scope": True, "age": 34, "monthly_income": 60000, "utilization": 4.0, "late_30_59": 1.0,
                     "debt_to_income": None, "late_90": True})
assert v == {"age": 34, "income": 60000, "l30": 1}, v
assert any("outside" in x for x in n) and any("unreadable" in x for x in n), n
assert llm.validate({"in_scope": False})[0] == {}
# 2. guardrails
assert "personal" in llm.extract("mail me at a@b.com, age 30")[2]
assert "personal" in llm.extract("account 123456789012 earns 5000")[2]
assert "under" in llm.extract("x" * 700)[2]
# 3. explain: good output, invented number, API failure
f = {"decision": "Accept", "pd": "2.9%", "accept_below": "10.0%", "reject_above": "16.7%", "decision_note": "",
     "raising_risk": [], "lowering_risk": ["Credit utilisation (x0.8)"]}
with mock.patch.object(llm, "call_gemini", return_value="Accept. The 2.9% default probability is under the 10.0% limit."):
    assert llm.explain(f)[1] == "gemini"
with mock.patch.object(llm, "call_gemini", return_value="Accept. Risk is only 1.2% so approve."):
    assert llm.explain(f)[1] == "template"
with mock.patch.object(llm, "call_gemini", side_effect=RuntimeError("down")):
    assert llm.explain(f)[1] == "template"
    assert "unavailable" in llm.extract("34 years old earns 60000")[2]

# 4. full app with mocked Gemini
fake = json.dumps({"in_scope": True, "age": 34, "monthly_income": 60000, "debt_to_income": 0.25, "utilization": 0.4,
                   "late_30_59": 1, "late_60_89": None, "late_90": None, "open_credit_lines": 7, "real_estate_loans": None})
with mock.patch.object(llm, "call_gemini", return_value=fake):
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception, at.exception
    at.text_area(key="intake_text").set_value("34 years old, earns 60000").run()
    [b for b in at.button if b.label == "Fill form from text"][0].click().run()
    assert not at.exception, at.exception
    assert at.session_state["age"] == 34 and at.session_state["income"] == 60000 and at.session_state["util"] == 0.4
    assert at.session_state["re"] == 1  # not mentioned -> unchanged
    print("sidebar age:", at.number_input(key="age").value, "| decision:", at.metric[0].value, at.metric[1].value)
with mock.patch.object(llm, "call_gemini", return_value="Reject. nonsense 99% claim"):
    [b for b in at.button if b.label == "Explain this decision"][0].click().run()
    assert not at.exception, at.exception
print("ALL CHECKS PASSED")
