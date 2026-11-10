"""Chatbot mẫu cho phần Demo Product. Chạy: python -m streamlit run app/chat.py

Đổi pipeline bằng biến môi trường NEORAG_PIPELINE="module.path:ClassName".
Đây chỉ là điểm khởi đầu: giao diện, streaming, lịch sử hội thoại... là phần bạn tự cải thiện.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from neorag.config import CONFIG_DIR
from neorag.data import load_corpus
from neorag.pipelines import load_pipeline

load_dotenv()
PIPELINE = os.getenv("NEORAG_PIPELINE", "neorag.pipelines.baseline:BaselinePipeline")

st.set_page_config(page_title="NeoRAG Chatbot - S2B", page_icon="💬")
st.title("💬 Hỏi đáp cộng đồng S2B")
st.caption("CLB Lập Trình PTIT · CLB Multimedia PTIT · CLB Nhà Sáng Tạo Game PTIT")
st.caption("Mỗi câu hỏi được xử lý độc lập. Hãy ghi rõ CLB, sự kiện hoặc thời điểm bạn muốn hỏi.")


@st.cache_resource(show_spinner="Đang nạp model và xây index (lần đầu có thể mất vài phút)...")
def get_pipeline(spec: str, config_digest: str):
    corpus = load_corpus()
    pipeline = load_pipeline(spec)
    pipeline.prepare()
    pipeline.index(corpus)
    return pipeline, {d.id: d for d in corpus}


config_path = Path(os.getenv("NEORAG_HYBRID_CONFIG") or CONFIG_DIR / "hybrid.toml")
try:
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    pipeline, docs = get_pipeline(PIPELINE, digest)
except Exception:
    st.error("Chưa thể nạp dữ liệu. Hãy kiểm tra dịch vụ tìm kiếm và cấu hình, rồi tải lại trang.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Cuộc trò chuyện mới"):
    st.session_state.messages = []


def show_sources(ids, elapsed):
    with st.expander(f"Tài liệu truy xuất · {elapsed:.1f}s"):
        for doc_id in ids:
            d = docs.get(doc_id)
            if d is not None:
                st.markdown(f"**[{d.id}] {d.club}** · _{d.topic}_\n\n{d.text}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "retrieved_ids" in msg:
            show_sources(msg["retrieved_ids"], msg["elapsed"])

if question := st.chat_input("Hỏi gì đó về ProPTIT, Multimedia hoặc GCC..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        start = time.perf_counter()
        try:
            with st.spinner("Đang tìm câu trả lời..."):
                out = pipeline.run(question)
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
            st.stop()
        except Exception:
            st.error(
                "Không thể hoàn tất câu trả lời. Hãy kiểm tra kết nối hoặc hạn mức API rồi thử lại."
            )
            st.stop()
        st.markdown(out.answer)
        elapsed = time.perf_counter() - start
        show_sources(out.retrieved_ids, elapsed)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": out.answer,
            "retrieved_ids": out.retrieved_ids,
            "elapsed": elapsed,
        }
    )
