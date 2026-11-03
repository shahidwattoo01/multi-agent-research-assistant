from research_assistant.models import Critique, FindingDraft, ResearchPlan
from research_assistant.schema import strict_json_schema


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def test_every_object_is_strict():
    for model in (ResearchPlan, Critique, FindingDraft):
        schema = strict_json_schema(model)
        assert "$defs" not in schema
        for node in _walk(schema):
            assert "$ref" not in node
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])


def test_nested_models_are_inlined():
    schema = strict_json_schema(ResearchPlan)
    subtask = schema["properties"]["subtasks"]["items"]
    assert subtask["type"] == "object"
    assert {"question", "rationale", "search_queries"} <= set(subtask["properties"])


def test_unsupported_keywords_dropped_and_values_still_validated():
    schema = strict_json_schema(Critique)
    for node in _walk(schema):
        assert "default" not in node and "title" not in node
    # out-of-range values are clamped by Pydantic after the fact
    c = Critique(score=1.7, approved=True, strengths=[], issues=[], follow_up_tasks=[])
    assert c.score == 1.0
