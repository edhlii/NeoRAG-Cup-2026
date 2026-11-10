"""UI smoke tests offline; không tải model hoặc gọi API."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from neorag.config import CONFIG_DIR
from neorag.pipelines import RAGOutput

st = pytest.importorskip("streamlit")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def test_chat_renders_sources_and_answer():
    st.cache_resource.clear()
    fake = SimpleNamespace(
        prepare=lambda: None,
        index=lambda corpus: None,
        run=lambda query: RAGOutput("Thông tin. [1]", [1]),
    )
    with patch("neorag.pipelines.load_pipeline", return_value=fake):
        app = AppTest.from_file(str(CONFIG_DIR.parent / "app/chat.py")).run(timeout=30)
        app.chat_input[0].set_value("Câu hỏi?").run(timeout=30)
        assert not app.exception
        assert app.session_state["messages"][-1]["content"] == "Thông tin. [1]"
        assert len(app.chat_message) == 2
    st.cache_resource.clear()


def test_chat_transport_error_is_not_an_abstention():
    st.cache_resource.clear()

    def fail(query):
        raise RuntimeError("Đã chạm quota thử nghiệm.")

    fake = SimpleNamespace(prepare=lambda: None, index=lambda corpus: None, run=fail)
    with patch("neorag.pipelines.load_pipeline", return_value=fake):
        app = AppTest.from_file(str(CONFIG_DIR.parent / "app/chat.py")).run(timeout=30)
        app.chat_input[0].set_value("Câu hỏi?").run(timeout=30)
        assert not app.exception
        assert "quota" in app.error[0].value
        assert all(m["role"] != "assistant" for m in app.session_state["messages"])
    st.cache_resource.clear()


def test_chat_keeps_sources_on_rerun_and_can_clear_history():
    st.cache_resource.clear()
    fake = SimpleNamespace(
        prepare=lambda: None,
        index=lambda corpus: None,
        run=lambda query: RAGOutput("Thông tin. [1]", [1]),
    )
    with patch("neorag.pipelines.load_pipeline", return_value=fake):
        app = AppTest.from_file(str(CONFIG_DIR.parent / "app/chat.py")).run(timeout=30)
        app.chat_input[0].set_value("Câu hỏi đầu tiên?").run(timeout=30)
        app.chat_input[0].set_value("Câu hỏi thứ hai?").run(timeout=30)
        assert not app.exception
        assert len(app.expander) == 2
        app.button[0].click().run(timeout=30)
        assert app.session_state["messages"] == []
        assert len(app.chat_message) == 0
    st.cache_resource.clear()
