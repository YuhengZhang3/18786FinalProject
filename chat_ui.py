
import json, datetime, pathlib, importlib.util, uuid
import streamlit as st

# ── 1. 动态 import 你已有的代理脚本 ──
BACKEND_FILE = pathlib.Path(__file__).parent / "Agent_Langgraph.py"
spec = importlib.util.spec_from_file_location("backend", BACKEND_FILE)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
graph = backend.graph      # 同一张图

# ── 2. 每个浏览器会话用独立 thread_id ──
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
thread_id = st.session_state.thread_id

# ── 3. 历史消息保存在 session_state 方便回显 (跟 MemorySaver 无冲突) ──
if "history" not in st.session_state:
    st.session_state.history = []   # list[dict]: {"role": "user"|"assistant", "content": str}

# ── 4. 页面配置 ──
st.set_page_config(page_title="Flight Finder Chat", page_icon="✈️")
st.title("✈️  Flight Finder (Chat)")

# ── 5. 显示历史对话 ──
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)  # allow <br> etc.

# ── 6. 输入框 ──
user_input = st.chat_input("Type anything about your trip…")

if user_input:
    # 6‑a. 先把用户消息挂到 UI
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 6‑b. 调 LangGraph
    result = graph.invoke(
        {"messages": [("user", user_input)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    assistant_msg = result["messages"][-1].content

    # 6‑c. 尝试解析 JSON
    flights = None
    try:
        data = json.loads(assistant_msg)
        flights = data.get("flights")
    except Exception:
        pass

    # 6‑d. 展示
    with st.chat_message("assistant"):
        if flights:
            # 推荐 + 表格
            def minutes(dur_str):
                h, m = [int(x[:-1]) for x in dur_str.split()]
                return h * 60 + m
            for f in flights:
                f["duration_min"] = minutes(f["duration"])
            flights.sort(key=lambda x: 0.6 * x["price"]/800 +
                                     0.3 * x["duration_min"]/540 +
                                     0.1 * x["stops"]/3)
            best = flights[0]
            st.markdown(
                f"✅ **Recommended:** {best['airline']} {best['flight_number']} "
                f"{best['departure_time']} ➜ {best['arrival_time']}  \n"
                f"💰 {best['formatted_price']} · {best['duration']} · "
                f"{best['stops']} stop(s)"
            )
            st.link_button("Book ↗", best["booking_link"])
            st.markdown("---")
            st.dataframe(flights, use_container_width=True)
            assistant_display = "I've ranked the available flights for you."
        else:
            st.markdown(assistant_msg)  # 普通文本回复
            assistant_display = assistant_msg

    # 6‑e. 存历史
    st.session_state.history.append(
        {"role": "assistant", "content": assistant_display}
    )

    st.rerun()   # 刷新页面，保证顺序正确
