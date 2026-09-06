"""Regression tests for #6340.

A conditional edge that injects `store` raised
"Missing required config key 'store'" when reached via `update_state`,
even though the same edge worked under `invoke`.

Root cause: `update_state` executes a node's writers directly, outside the
invoke/stream path, and never placed a `Runtime` in `config`. Parameters
injected off the runtime (like `store`) therefore could not resolve.
"""

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

pytestmark = pytest.mark.anyio


class State(TypedDict):
    messages: list[str]


def make_graph(store: BaseStore | None = None, checkpointer: Any = None):
    seen: dict[str, Any] = {}

    def node(state: State) -> dict:
        return {"messages": [*state.get("messages", []), "ran"]}

    def route(state: State, store: BaseStore) -> str:
        seen["store"] = store
        if len(state.get("messages", [])) > 5:
            return END
        return "node"

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_conditional_edges("node", route)
    graph = builder.compile(
        checkpointer=checkpointer or InMemorySaver(),
        store=store if store is not None else InMemoryStore(),
    )
    return graph, seen


def test_invoke_provides_store_to_branch() -> None:
    """Control: under `invoke` the branch already received the store."""
    graph, seen = make_graph()
    graph.invoke({"messages": ["a"]}, config={"configurable": {"thread_id": "t"}})
    assert isinstance(seen.get("store"), BaseStore)


def test_update_state_provides_store_to_branch() -> None:
    """The reported bug: `update_state` must also resolve `store`."""
    graph, seen = make_graph()
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": ["a"]}, config=config)
    seen.clear()

    graph.update_state(config, {"messages": ["b"]})  # must not raise
    assert isinstance(seen.get("store"), BaseStore)


async def test_aupdate_state_provides_store_to_branch() -> None:
    """Async path (the one in the original traceback)."""
    graph, seen = make_graph()
    config = {"configurable": {"thread_id": "t"}}
    await graph.ainvoke({"messages": ["a"]}, config=config)
    seen.clear()

    await graph.aupdate_state(config, {"messages": ["b"]})  # must not raise
    assert isinstance(seen.get("store"), BaseStore)


def test_update_state_branch_without_store_param_unaffected() -> None:
    """Injecting a Runtime must not disturb branches that never ask for
    `store` (guard against a regression from the fix itself)."""
    calls: list[str] = []

    def node(state: State) -> dict:
        return {"messages": [*state.get("messages", []), "ran"]}

    def route_plain(state: State) -> str:
        calls.append("called")
        return END if len(state.get("messages", [])) > 5 else "node"

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_conditional_edges("node", route_plain)
    graph = builder.compile(checkpointer=InMemorySaver(), store=InMemoryStore())

    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": ["a"]}, config=config)
    calls.clear()

    graph.update_state(config, {"messages": ["b"]})
    assert calls == ["called"]
