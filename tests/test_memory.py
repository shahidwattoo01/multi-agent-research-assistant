import threading

from research_assistant.memory import SharedMemory, normalize_url, similarity
from research_assistant.models import Finding, PlannedSubTask, Source, SourceRef, TaskStatus


def _planned(q: str) -> PlannedSubTask:
    return PlannedSubTask(question=q, rationale="", search_queries=[])


def test_similarity_detects_rephrasings():
    assert (
        similarity(
            "How have solar costs changed since 2010?", "how have solar costs changed since 2010"
        )
        > 0.95
    )
    assert similarity("What is the cost of solar power?", "Solar power cost") > 0.6
    assert similarity("Solar panel efficiency", "Offshore wind turbine maintenance") < 0.3


def test_add_tasks_deduplicates_against_ledger():
    mem = SharedMemory()
    first = mem.add_tasks([_planned("How have solar costs changed since 2010?")], origin="planner")
    again = mem.add_tasks(
        [_planned("How have solar costs changed since 2010"), _planned("What drives wind costs?")],
        origin="critic",
        revision=1,
    )
    assert len(first) == 1
    assert [t.question for t in again] == ["What drives wind costs?"]
    assert again[0].origin == "critic" and again[0].revision == 1
    assert mem.stats["duplicate_tasks_skipped"] == 1


def test_failed_tasks_can_be_retried():
    mem = SharedMemory()
    (task,) = mem.add_tasks([_planned("Battery prices 2023")], origin="planner")
    mem.set_status(task.id, TaskStatus.FAILED, error="boom")
    assert mem.add_tasks([_planned("Battery prices 2023")], origin="critic")


def test_status_transitions_and_progress():
    mem = SharedMemory()
    (task,) = mem.add_tasks([_planned("Q1")], origin="planner")
    assert mem.pending_tasks() == [task]
    t = mem.set_status(task.id, TaskStatus.IN_PROGRESS)
    assert t.attempts == 1
    mem.set_status(task.id, TaskStatus.DONE)
    assert mem.progress() == {"pending": 0, "in_progress": 0, "done": 1, "failed": 0}


def test_search_and_page_cache_with_stats():
    mem = SharedMemory()
    assert mem.get_search("Solar  COSTS!") is None
    mem.put_search("solar costs", [Source(url="https://a.org/x/", title="A")])
    assert mem.get_search("Solar  COSTS!")[0].title == "A"
    assert mem.stats["search_cache_hits"] == 1 and mem.stats["search_cache_misses"] == 1
    assert normalize_url("HTTPS://A.org/x/#frag") == normalize_url("https://a.org/x")
    assert normalize_url("https://a.org/x") in mem.known_sources()


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "cache.json"
    mem = SharedMemory(cache_path=path)
    mem.put_search("q", [Source(url="https://a.org", title="A")])
    mem.save_cache()
    reloaded = SharedMemory(cache_path=path)
    assert reloaded.get_search("q")[0].url == "https://a.org"
    reloaded.clear_cache()
    assert not path.exists()


def test_corrupt_cache_is_ignored(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{not json")
    assert SharedMemory(cache_path=path).get_search("q") is None


def test_recall_ranks_relevant_findings_and_excludes_own_task():
    mem = SharedMemory()
    for tid, q in [("t1", "solar module prices"), ("t2", "wind turbine sizes")]:
        mem.publish_finding(
            Finding(
                task_id=tid,
                question=q,
                summary=f"About {q}",
                key_points=[q],
                sources=[SourceRef(url="https://x.org", title="x")],
                confidence=0.9,
                gaps=[],
            )
        )
    assert mem.recall("solar prices")[0].task_id == "t1"
    assert all(f.task_id != "t1" for f in mem.recall("solar prices", exclude_task="t1"))


def test_ledger_is_thread_safe():
    mem = SharedMemory(duplicate_threshold=1.01)  # disable dedup
    threads = [
        threading.Thread(target=lambda i=i: mem.add_tasks([_planned(f"q{i}")], origin="planner"))
        for i in range(50)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(mem.tasks) == 50
