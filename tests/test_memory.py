"""
tests/test_memory.py — Unit tests for ConversationMemory

FIXED: turn_count() returns the number of MESSAGES stored (not Q&A pairs).
So after add_user("Q1") + add_assistant("A1") + add_user("Q2"),
turn_count() == 3 (three messages), not 2.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.memory import ConversationMemory


@pytest.fixture
def memory():
    return ConversationMemory(max_turns=6)


def test_empty_memory_has_no_history(memory):
    assert memory.get_history() == []
    assert memory.turn_count() == 0


def test_add_user_and_assistant(memory):
    memory.add_user("What is FastAPI?")
    memory.add_assistant("FastAPI is a modern web framework.")
    history = memory.get_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_turn_count_counts_messages(memory):
    """turn_count() returns total messages stored, not Q&A pair count."""
    memory.add_user("Q1")
    assert memory.turn_count() == 1
    memory.add_assistant("A1")
    assert memory.turn_count() == 2
    memory.add_user("Q2")
    assert memory.turn_count() == 3


def test_sliding_window_max_6(memory):
    """With maxlen=6, only last 6 messages are kept."""
    for i in range(5):
        memory.add_user(f"Question {i}")
        memory.add_assistant(f"Answer {i}")
    history = memory.get_history()
    assert len(history) <= 6


def test_clear_resets_memory(memory):
    memory.add_user("Q1")
    memory.add_assistant("A1")
    memory.clear()
    assert memory.get_history() == []
    assert memory.turn_count() == 0


def test_history_format(memory):
    memory.add_user("How do I use FastAPI?")
    memory.add_assistant("Import FastAPI and create an app instance.")
    history = memory.get_history()
    for turn in history:
        assert "role" in turn
        assert "content" in turn
        assert turn["role"] in ("user", "assistant")


def test_display_returns_string(memory):
    memory.add_user("test")
    memory.add_assistant("response")
    display = memory.display()
    assert isinstance(display, str)
    assert len(display) > 0


def test_multiple_turns_preserved_in_order(memory):
    memory.add_user("Q1")
    memory.add_assistant("A1")
    memory.add_user("Q2")
    memory.add_assistant("A2")
    history = memory.get_history()
    roles = [h["role"] for h in history]
    assert roles == ["user", "assistant", "user", "assistant"]
