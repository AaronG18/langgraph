"""Regression tests for the cached `UntrackedValue` channel scan.

`PregelLoop._has_untracked_value_channel()` caches "does any channel hold an
`UntrackedValue`?" instead of rescanning `self.channels` on every `put_writes`.
That is only correct while `self.channels` stays fixed - it is reassigned when a
loop loads channels from a checkpoint, so the resume path is covered explicitly.
"""

from typing import Annotated

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.channels.untracked_value import UntrackedValue
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    kept: str
    dropped: Annotated[str, UntrackedValue]


def _build_app():
    def node(state: State) -> State:
        return {"kept": "kept", "dropped": "dropped"}

    graph = StateGraph(State)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    return graph.compile(checkpointer=InMemorySaver())


def test_untracked_value_not_persisted() -> None:
    """Baseline: untracked channels are still excluded from the checkpoint."""
    app = _build_app()
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)

    values = app.get_state(config).values
    assert values["kept"] == "kept"
    assert "dropped" not in values


def test_untracked_value_not_persisted_after_resume() -> None:
    """A resumed loop reloads `self.channels`, so the cached answer must survive it.

    If the cache were computed against the channels of the first loop only, this
    would start persisting `dropped` after the graph resumes from a checkpoint.
    """

    def node(state: State) -> State:
        answer = interrupt("need input")
        return {"kept": str(answer), "dropped": "dropped"}

    graph = StateGraph(State)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    app = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}

    app.invoke({}, config)
    # second invocation builds a fresh loop that reassigns `self.channels`
    app.invoke(Command(resume="resumed"), config)

    values = app.get_state(config).values
    assert values["kept"] == "resumed"
    assert "dropped" not in values


def test_writes_persisted_when_no_untracked_channel() -> None:
    """Guard the cached-`False` branch: nothing is dropped when no channel is
    untracked."""

    class PlainState(TypedDict):
        kept: str
        other: str

    def node(state: PlainState) -> PlainState:
        return {"kept": "kept", "other": "other"}

    graph = StateGraph(PlainState)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    app = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)

    values = app.get_state(config).values
    assert values == {"kept": "kept", "other": "other"}


@pytest.mark.parametrize("runs", [1, 2, 3])
def test_untracked_value_still_excluded_on_repeated_runs(runs: int) -> None:
    """Repeated invocations on one thread must never leak the untracked channel."""
    app = _build_app()
    config = {"configurable": {"thread_id": "1"}}
    for _ in range(runs):
        app.invoke({}, config)
        assert "dropped" not in app.get_state(config).values
