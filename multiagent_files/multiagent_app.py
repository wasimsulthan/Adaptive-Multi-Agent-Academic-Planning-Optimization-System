
import json
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="RAG System", page_icon="📚", layout="wide")


def api_get(path, params=None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        return r.json() if r.ok else None
    except Exception:
        return None

def api_post(path, json_data=None, files=None, timeout=120):
    try:
        r = requests.post(f"{API_BASE}{path}", json=json_data, files=files, timeout=timeout)
        return r.json() if r.ok else None
    except Exception:
        return None

def api_delete(path):
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=10)
        return r.ok
    except Exception:
        return False

def get_docs():
    return api_get("/documents") or []

def get_health():
    return api_get("/health")


for key, default in [
    ("messages", []),
    ("quiz_data", None),
    ("quiz_submitted", False),
    ("user_answers", {}),
    ("quiz_id", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


page = st.sidebar.radio(
    "Navigation",
    [" Documents", " Chat", " Quiz", " Dashboard"],
    label_visibility="collapsed",
)

# Status bar
health = get_health()
if health:
    cols = st.sidebar.columns(3)
    cols[0].caption("DB ✅" if health.get("database") else "DB ❌")
    cols[1].caption("LLM ✅" if health.get("ollama") else "LLM ❌")
    cols[2].caption(f"`{health.get('ollama_model','?')}`")
else:
    st.sidebar.error("API not reachable")

st.sidebar.divider()



if page == " Documents":
    st.title(" Document management")

    # Upload
    st.subheader("Upload a PDF")
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded and st.button(" Ingest document", type="primary"):
        with st.spinner(f"Ingesting {uploaded.name}…"):
            r = requests.post(
                f"{API_BASE}/ingest",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                timeout=300,
            )
            if r.ok:
                data = r.json()
                st.success(f"✅ **{data['filename']}** — {data['total_pages']} pages, {data['total_chunks']} chunks")
                st.rerun()
            else:
                st.error(f"Failed: {r.json().get('detail', 'Unknown error')}")

    st.divider()

    # Document list
    docs = get_docs()
    if docs:
        st.subheader(f"Ingested documents ({len(docs)})")
        for d in docs:
            col1, col2, col3 = st.columns([5, 2, 1])
            col1.write(f"**{d['filename']}** — {d.get('total_pages','?')} pages, {d.get('total_chunks','?')} chunks")
            col2.caption(f"ID: {d['id']} • {d.get('upload_date','')}")
            if col3.button("🗑️", key=f"del_{d['id']}"):
                api_delete(f"/documents/{d['id']}")
                st.rerun()

            # Chapters
            ch_data = api_get(f"/documents/{d['id']}/chapters")
            if ch_data and ch_data.get("chapters"):
                with st.expander(f"📖 {len(ch_data['chapters'])} chapters detected"):
                    for c in ch_data["chapters"]:
                        st.caption(f"**{c['title']}** — pages {c['start_page']}-{c['end_page']}")
    else:
        st.info("No documents yet. Upload a PDF above.")



elif page == " Chat":
    st.title(" Chat with your documents")

    docs = get_docs()
    doc_options = {None: "All documents"}
    for d in docs:
        doc_options[d["id"]] = f"{d['filename']}"

    sel_doc = st.sidebar.selectbox("Search scope", list(doc_options.keys()), format_func=lambda x: doc_options[x])

    if st.sidebar.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()

    # History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("passages"):
                with st.expander(f"📖 {len(msg['passages'])} sources"):
                    for i, p in enumerate(msg["passages"], 1):
                        sc = p.get("rerank_score")
                        st.caption(f"**[{i}]** {p.get('node_type','')} / level {p.get('tree_level','')} "
                                   + (f"— {sc:.3f}" if sc else ""))
                        st.markdown(f"<div style='background:var(--secondary-background-color); "
                                    f"padding:6px 10px; border-radius:6px; font-size:0.85em;'>"
                                    f"{p.get('content','')[:400]}</div>", unsafe_allow_html=True)

    # Input
    if prompt := st.chat_input("Ask a question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            passages = []
            full_resp = ""
            placeholder = st.empty()
            status = st.empty()
            status.caption("🔍 Searching…")

            try:
                import sseclient
                payload = {"question": prompt}
                if sel_doc:
                    payload["doc_id"] = sel_doc
                r = requests.post(f"{API_BASE}/query/stream", json=payload, stream=True, timeout=120,
                                  headers={"Accept": "text/event-stream"})
                for ev in sseclient.SSEClient(r).events():
                    if ev.event == "passages":
                        passages = json.loads(ev.data)
                        status.caption(f"✅ {len(passages)} passages. Generating…")
                    elif ev.event == "token":
                        full_resp += ev.data
                        placeholder.markdown(full_resp + "▌")
                    elif ev.event == "done":
                        placeholder.markdown(full_resp)
                        status.empty()
                    elif ev.event == "error":
                        st.error(ev.data)
                        break
            except Exception as e:
                st.error(f"Error: {e}")
                full_resp = f"{e}"

            if passages:
                with st.expander(f" {len(passages)} sources"):
                    for i, p in enumerate(passages, 1):
                        st.caption(f"**[{i}]** {p.get('node_type','')} / level {p.get('tree_level','')}")

            st.session_state.messages.append({"role": "assistant", "content": full_resp, "passages": passages})




elif page == " Quiz":
    st.title(" Quiz")

    docs = get_docs()
    if not docs:
        st.info("Upload a document first.")
        st.stop()

    # Sidebar config
    doc_map = {d["id"]: d["filename"] for d in docs}
    sel_doc = st.sidebar.selectbox("Document", list(doc_map.keys()), format_func=lambda x: doc_map[x])

    source_type = st.sidebar.radio("Source", ["pages", "chapters"], horizontal=True,
                                   format_func=lambda x: "Pages" if x == "pages" else "Chapters")

    source_value = ""
    if source_type == "pages":
        ch_data = api_get(f"/documents/{sel_doc}/chapters")
        total_p = ch_data.get("total_pages", 0) if ch_data else 0
        st.sidebar.caption(f"{total_p} pages available" if total_p else "")
        source_value = st.sidebar.text_input("Page range", placeholder="1-10, 15")
    else:
        ch_data = api_get(f"/documents/{sel_doc}/chapters")
        chapters = ch_data.get("chapters", []) if ch_data else []
        if chapters:
            sel_ch = st.sidebar.multiselect("Chapters", [c["title"] for c in chapters])
            source_value = ", ".join(sel_ch)
        else:
            st.sidebar.info("No chapters detected.")

    num_q = st.sidebar.slider("Questions", 3, 30, 10)
    difficulty = st.sidebar.select_slider("Difficulty", ["easy", "medium", "hard"], "medium")
    provider = st.sidebar.radio("Provider", ["openai", "anthropic"], horizontal=True)

    if st.sidebar.button("🎯 Generate quiz", type="primary", disabled=not source_value.strip()):
        st.session_state.quiz_data = None
        st.session_state.quiz_submitted = False
        st.session_state.user_answers = {}
        with st.spinner("Generating…"):
            result = api_post("/quiz/generate", {
                "document_id": sel_doc, "source_type": source_type,
                "source_value": source_value, "num_questions": num_q,
                "difficulty": difficulty, "provider": provider,
            })
            if result and "questions" in result:
                st.session_state.quiz_data = result
                st.session_state.quiz_id = result.get("quiz_id")
                st.rerun()
            else:
                st.error("Quiz generation failed.")

    if st.session_state.quiz_data and st.sidebar.button("🔄 New quiz"):
        st.session_state.quiz_data = None
        st.session_state.quiz_submitted = False
        st.session_state.user_answers = {}
        st.rerun()

    # Quiz display
    quiz = st.session_state.quiz_data
    if quiz is None:
        # Show history
        history = api_get("/quizzes", {"doc_id": sel_doc})
        if history:
            st.subheader("Quiz history")
            for h in history[:10]:
                score_str = ""
                if h.get("score") is not None:
                    pct = round(h["score"] / max(1, h["total_attempted"]) * 100, 1)
                    score_str = f" — {h['score']}/{h['total_attempted']} ({pct}%)"
                st.caption(f"**{h['title']}** ({h['difficulty']}){score_str} • {h.get('created_at','')}")
        st.stop()

    questions = quiz.get("questions", [])
    if not questions:
        st.warning("No questions generated.")
        st.stop()

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Questions", len(questions))
    c2.metric("Difficulty", quiz.get("difficulty", "").title())
    if st.session_state.quiz_submitted:
        score = sum(1 for q in questions
                    if st.session_state.user_answers.get(q["question_number"]) == q["correct_answer"])
        c3.metric("Score", f"{score}/{len(questions)}")

    st.divider()

    if not st.session_state.quiz_submitted:
        # Taking the quiz
        with st.form("quiz"):
            for q in questions:
                qn = q["question_number"]
                diff_icons = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
                st.markdown(f"### Q{qn}. {q['question']}")
                st.caption(f"{diff_icons.get(q.get('difficulty',''),'🟡')} {q.get('difficulty','').title()} "
                           + (f"• Page {q['source_page']}" if q.get("source_page") else ""))

                if q["question_type"] == "mcq":
                    ans = st.radio(f"q{qn}", q.get("options", []), key=f"q_{qn}", label_visibility="collapsed")
                    if ans:
                        st.session_state.user_answers[qn] = ans[0] if ans[0] in "ABCD" else ans
                elif q["question_type"] == "true_false":
                    ans = st.radio(f"q{qn}", ["True", "False"], key=f"q_{qn}", label_visibility="collapsed", horizontal=True)
                    st.session_state.user_answers[qn] = ans
                elif q["question_type"] == "fill_blank":
                    ans = st.text_input(f"q{qn}", key=f"q_{qn}", placeholder="Your answer…", label_visibility="collapsed")
                    if ans:
                        st.session_state.user_answers[qn] = ans.strip()
                st.divider()

            if st.form_submit_button("✅ Submit quiz", type="primary", use_container_width=True):
                st.session_state.quiz_submitted = True
                score = 0
                for q in questions:
                    ua = st.session_state.user_answers.get(q["question_number"], "")
                    ca = q["correct_answer"]
                    if q["question_type"] == "fill_blank":
                        if str(ua).lower().strip() == str(ca).lower().strip():
                            score += 1
                    elif ua == ca:
                        score += 1
                if st.session_state.quiz_id:
                    api_post(f"/quiz/{st.session_state.quiz_id}/score",
                             {"score": score, "total_attempted": len(questions)})
                st.rerun()
    else:
        # Results
        score = 0
        for q in questions:
            qn = q["question_number"]
            ua = st.session_state.user_answers.get(qn, "—")
            ca = q["correct_answer"]
            if q["question_type"] == "fill_blank":
                correct = str(ua).lower().strip() == str(ca).lower().strip()
            else:
                correct = ua == ca
            if correct:
                score += 1

            st.markdown(f"### {'✅' if correct else '❌'} Q{qn}. {q['question']}")
            if not correct:
                st.markdown(f"Your answer: ~~{ua}~~ • Correct: **{ca}**")
            if q.get("explanation"):
                st.info(f"💡 {q['explanation']}")
            st.divider()

        pct = round(score / len(questions) * 100, 1)
        grade = "🏆 Excellent!" if pct >= 80 else "👍 Good!" if pct >= 60 else "📚 Keep studying!"
        st.markdown(f"## {grade} Score: {score}/{len(questions)} ({pct}%)")

        c1, c2 = st.columns(2)
        if c1.button("🔄 Retake", use_container_width=True):
            st.session_state.quiz_submitted = False
            st.session_state.user_answers = {}
            st.rerun()
        if c2.button("🆕 New quiz", type="primary", use_container_width=True):
            st.session_state.quiz_data = None
            st.session_state.quiz_submitted = False
            st.session_state.user_answers = {}
            st.rerun()





elif page == " Dashboard":
    st.title(" Performance dashboard")

    docs = get_docs()
    doc_filter = st.sidebar.selectbox(
        "Filter by document",
        [None] + [d["id"] for d in docs],
        format_func=lambda x: "All documents" if x is None else next((d["filename"] for d in docs if d["id"] == x), "?"),
    )

    dash = api_get("/dashboard", {"doc_id": doc_filter} if doc_filter else None)

    if not dash or dash.get("overall", {}).get("total_quizzes", 0) == 0:
        st.info("No completed quizzes yet. Take some quizzes to see your performance analytics.")
        st.stop()

    ov = dash["overall"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Quizzes taken", ov["total_quizzes"])
    m2.metric("Total questions", ov["total_questions"])
    m3.metric("Overall accuracy", f"{ov['overall_accuracy']}%")
    m4.metric("Avg correct/quiz", f"{ov['avg_score_per_quiz']}")

    st.divider()


    timeline = ov.get("quizzes_over_time", [])
    if timeline:
        st.subheader("Score over time")
        import pandas as pd
        df = pd.DataFrame(timeline)
        df["date"] = pd.to_datetime(df["date"])
        st.line_chart(df.set_index("date")["accuracy"], height=250)

    topics = dash.get("topic_performance", [])
    if topics:
        st.subheader("Performance by topic")
        import pandas as pd
        topic_df = pd.DataFrame([
            {"Topic": t["topic"][:40], "Accuracy (%)": t["accuracy"],
             "Correct": t["correct"], "Incorrect": t["incorrect"],
             "Total": t["total_questions"]}
            for t in topics
        ])
        st.bar_chart(topic_df.set_index("Topic")["Accuracy (%)"], height=300)

        with st.expander("Detailed topic breakdown"):
            st.dataframe(topic_df, use_container_width=True, hide_index=True)

    diff_stats = dash.get("difficulty_stats", {})
    if any(diff_stats.get(d, {}).get("total", 0) > 0 for d in ["easy", "medium", "hard"]):
        st.subheader("Performance by difficulty")
        dc1, dc2, dc3 = st.columns(3)
        for col, level, emoji in [(dc1, "easy", "🟢"), (dc2, "medium", "🟡"), (dc3, "hard", "🔴")]:
            d = diff_stats.get(level, {})
            if d.get("total", 0) > 0:
                col.metric(f"{emoji} {level.title()}", f"{d['accuracy']}%",
                           delta=f"{d['correct']}/{d['total']} correct")

    qt_stats = dash.get("question_type_stats", {})
    if any(qt_stats.get(t, {}).get("total", 0) > 0 for t in qt_stats):
        st.subheader("Performance by question type")
        qt_labels = {"mcq": "Multiple choice", "true_false": "True/False", "fill_blank": "Fill in blank"}
        qc = st.columns(len(qt_stats))
        for i, (qtype, data) in enumerate(qt_stats.items()):
            if data.get("total", 0) > 0:
                qc[i].metric(qt_labels.get(qtype, qtype), f"{data['accuracy']}%",
                             delta=f"{data['correct']}/{data['total']}")

    st.divider()


    weak = dash.get("weak_topics", [])
    if weak:
        st.subheader("⚠️ Topics needing improvement")
        for w in weak:
            st.warning(
                f"**{w['topic']}** — {w['accuracy']}% accuracy "
                f"({w['incorrect']} incorrect, {w['gap']}% below target)"
            )

    suggestions = dash.get("improvement_suggestions", [])
    if suggestions:
        st.subheader("💡 Study recommendations")
        for s in suggestions:
            st.markdown(f"- {s}")


    st.divider()
    st.subheader("🤖 AI study coach")
    provider = st.radio("Provider", ["openai", "anthropic"], horizontal=True, key="fb_provider")

    if st.button("Get personalised feedback", type="primary"):
        with st.spinner("Generating feedback…"):
            fb = api_post("/dashboard/feedback", {"doc_id": doc_filter, "provider": provider})
            if fb and fb.get("feedback"):
                st.success(fb["feedback"])
            else:
                st.error("Could not generate feedback. Check your API key.")
