import threading

import pytest
from typing_extensions import TypedDict

from langgraph.config import CONFIG_KEY_RUNNER_SUBMIT_MODE
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

CALLER_IDENT = threading.get_ident()


def _fanout_graph(seen: list[int]) -> StateGraph:
    """Two nodes from START, so the first superstep has 2 tasks.

    This matters: `PregelRunner.tick` has a single-task fast path that already
    runs the task inline, so a linear graph behaves identically with and
    without the submit override. Only a multi-task superstep exercises it.
    """

    class State(TypedDict, total=False):
        a: int
        b: int

    def one(state: State) -> dict:
        seen.append(threading.get_ident())
        return {"a": 1}

    def two(state: State) -> dict:
        seen.append(threading.get_ident())
        return {"b": 2}

    builder = StateGraph(State)
    builder.add_node("one", one)
    builder.add_node("two", two)
    builder.add_edge(START, "one")
    builder.add_edge(START, "two")
    builder.add_edge("one", END)
    builder.add_edge("two", END)
    return builder.compile()


def _run(mode: str | None) -> list[bool]:
    seen: list[int] = []
    config: dict = {"max_concurrency": 1}
    if mode is not None:
        config["configurable"] = {CONFIG_KEY_RUNNER_SUBMIT_MODE: mode}
    graph = _fanout_graph(seen)
    graph.invoke({}, config=config)
    assert len(seen) == 2, f"expected 2 node runs, got {len(seen)}"
    return [ident == CALLER_IDENT for ident in seen]


def test_default_mode_runs_tasks_off_caller_thread() -> None:
    """Baseline: without the knob, a 2-task superstep goes to the pool."""
    assert _run(None) == [False, False]


def test_background_mode_matches_default() -> None:
    assert _run("background") == [False, False]


def test_inline_mode_runs_tasks_on_caller_thread() -> None:
    assert _run("inline") == [True, True]


def test_inline_mode_preserves_results() -> None:
    seen: list[int] = []
    graph = _fanout_graph(seen)
    result = graph.invoke(
        {}, config={"configurable": {CONFIG_KEY_RUNNER_SUBMIT_MODE: "inline"}}
    )
    assert result == {"a": 1, "b": 2}


def test_invalid_mode_raises() -> None:
    seen: list[int] = []
    graph = _fanout_graph(seen)
    with pytest.raises(ValueError, match="Invalid"):
        graph.invoke(
            {}, config={"configurable": {CONFIG_KEY_RUNNER_SUBMIT_MODE: "nope"}}
        )


def test_inline_propagates_node_exceptions() -> None:
    class State(TypedDict, total=False):
        a: int
        b: int

    def boom(state: State) -> dict:
        raise ValueError("node failed")

    builder = StateGraph(State)
    builder.add_node("boom", boom)
    builder.add_node("other", lambda s: {"b": 2})
    builder.add_edge(START, "boom")
    builder.add_edge(START, "other")
    builder.add_edge("boom", END)
    builder.add_edge("other", END)

    with pytest.raises(ValueError, match="node failed"):
        builder.compile().invoke(
            {}, config={"configurable": {CONFIG_KEY_RUNNER_SUBMIT_MODE: "inline"}}
        )


def test_inline_preserves_retry_policy() -> None:
    """Retry must still fire under inline, same as under the pool."""
    attempts = {"n": 0}

    class State(TypedDict, total=False):
        a: int
        b: int

    def flaky(state: State) -> dict:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")
        return {"a": 1}

    builder = StateGraph(State)
    # RetryPolicy does not retry RuntimeError by default, so opt in.
    builder.add_node(
        "flaky",
        flaky,
        retry_policy=RetryPolicy(max_attempts=3, retry_on=lambda e: True),
    )
    builder.add_node("other", lambda s: {"b": 2})
    builder.add_edge(START, "flaky")
    builder.add_edge(START, "other")
    builder.add_edge("flaky", END)
    builder.add_edge("other", END)

    result = builder.compile().invoke(
        {}, config={"configurable": {CONFIG_KEY_RUNNER_SUBMIT_MODE: "inline"}}
    )
    assert result == {"a": 1, "b": 2}
    assert attempts["n"] == 3


@pytest.mark.anyio
async def test_async_rejects_inline_mode() -> None:
    """Async already runs node bodies on the calling event loop, so "inline"
    has no meaning there - it must fail loudly rather than silently no-op."""
    seen: list[int] = []
    graph = _fanout_graph(seen)
    with pytest.raises(ValueError, match="only supported for sync execution"):
        await graph.ainvoke(
            {}, config={"configurable": {CONFIG_KEY_RUNNER_SUBMIT_MODE: "inline"}}
        )
