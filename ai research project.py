import html
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta

import arxiv
import faiss
import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Research Agent PRO",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "research_agent.db"
AI_QUERY_DEFAULT = "large language models deep learning transformer"

# =========================================================
# PREMIUM DARK CSS
# =========================================================
st.markdown(
    """
<style>
:root {
    --bg0: #050816;
    --bg1: #0b1020;
    --bg2: #101827;
    --card: rgba(15,23,42,0.72);
    --line: rgba(148,163,184,0.16);
    --text: #e5eefb;
    --muted: #94a3b8;
    --blue: #38bdf8;
    --blue2: #2563eb;
    --green: #22c55e;
    --amber: #f59e0b;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(236,72,153,0.12), transparent 24%),
        linear-gradient(135deg, var(--bg0) 0%, var(--bg1) 45%, #020617 100%);
    color: var(--text);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(4,10,24,0.98), rgba(7,12,28,0.98));
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] * { color: #dbeafe !important; }

.hero {
    border: 1px solid rgba(56,189,248,0.22);
    background: linear-gradient(135deg, rgba(37,99,235,0.24), rgba(14,165,233,0.10)), rgba(15,23,42,0.55);
    border-radius: 28px;
    padding: 28px 30px;
    position: relative;
    overflow: hidden;
    margin-bottom: 20px;
}
.hero:before {
    content: "";
    position: absolute;
    top: -110px; right: -100px;
    width: 280px; height: 280px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(56,189,248,0.20) 0%, transparent 70%);
}
.hero h1 { margin: 0; font-size: 32px; letter-spacing: -0.03em; color: #f8fafc; }
.hero p  { margin: 10px 0 0; color: #93c5fd; font-size: 14px; }
.hero-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
.pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 7px 14px; border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.18);
    background: rgba(2,6,23,0.50);
    color: #dbeafe; font-size: 12px; font-weight: 700;
}

.stat-grid { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 14px; margin: 18px 0 22px; }
.stat-card {
    background: linear-gradient(180deg, rgba(15,23,42,0.85), rgba(2,6,23,0.85));
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 20px; padding: 18px;
}
.stat-label { color: #7dd3fc; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .16em; margin-bottom: 10px; }
.stat-value { font-size: 28px; font-weight: 800; color: #f8fafc; margin: 0; line-height: 1; }
.stat-sub   { margin-top: 8px; color: #94a3b8; font-size: 12px; }

.paper-card {
    background: linear-gradient(180deg, rgba(15,23,42,0.95), rgba(2,6,23,0.93));
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 20px;
    padding: 20px 22px;
    margin-bottom: 16px;
    transition: transform 180ms ease, border-color 180ms ease;
}
.paper-card:hover {
    transform: translateY(-2px);
    border-color: rgba(56,189,248,0.38);
}
.paper-title  { font-size: 16px; font-weight: 800; color: #f8fafc; margin-bottom: 5px; line-height: 1.35; }
.paper-meta   { color: #94a3b8; font-size: 12px; margin-bottom: 10px; }
.paper-summary { color: #cbd5e1; font-size: 13px; line-height: 1.65; margin: 10px 0 6px; }
.paper-kp     { color: #94a3b8; font-size: 12px; line-height: 1.7; margin: 0; padding-left: 16px; }
.paper-kp li  { margin-bottom: 3px; }

.badge {
    display: inline-flex; align-items: center;
    padding: 4px 11px; border-radius: 999px;
    font-size: 11px; font-weight: 800; margin: 0 5px 6px 0; letter-spacing: .02em;
}
.badge-blue   { background: rgba(37,99,235,0.16);  color: #93c5fd; border: 1px solid rgba(37,99,235,0.26); }
.badge-green  { background: rgba(34,197,94,0.16);  color: #86efac; border: 1px solid rgba(34,197,94,0.24); }
.badge-amber  { background: rgba(245,158,11,0.16); color: #fcd34d; border: 1px solid rgba(245,158,11,0.24); }
.badge-slate  { background: rgba(100,116,139,0.16);color: #cbd5e1; border: 1px solid rgba(100,116,139,0.22); }

.open-link {
    display: inline-block; margin-top: 10px;
    color: #7dd3fc; text-decoration: none;
    font-weight: 700; font-size: 13px;
}
.open-link:hover { color: #a5f3fc; text-decoration: underline; }

.log-card {
    background: linear-gradient(180deg, rgba(15,23,42,0.93), rgba(2,6,23,0.92));
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 16px; padding: 14px 18px; margin-bottom: 10px;
}

.footer { text-align: center; color: #94a3b8; padding: 24px 0 8px; font-size: 14px; }

.stButton button {
    background: linear-gradient(135deg, #2563eb, #0ea5e9) !important;
    color: white !important; border: 0 !important;
    border-radius: 14px !important; font-weight: 800 !important;
    padding: .72rem 1rem !important;
}
.stButton button:hover { filter: brightness(1.08) !important; }

.stTextInput input, .stTextArea textarea, .stNumberInput input {
    background: rgba(15,23,42,0.92) !important;
    color: #e5eefb !important; border-radius: 14px !important;
    border: 1px solid rgba(148,163,184,0.22) !important;
}

[data-testid="stTabs"] button { border-radius: 999px !important; font-weight: 800 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# DATABASE
# =========================================================
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def ensure_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT, authors TEXT, abstract TEXT,
            published TEXT, arxiv_url TEXT, pdf_url TEXT, source TEXT,
            summary TEXT, key_points TEXT,
            novelty_score REAL, relevance_score REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS agent_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode TEXT, scheduled_time TEXT, status TEXT,
            started_at TEXT, finished_at TEXT, message TEXT
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY, value TEXT
        )""")
    defaults = {"search_query": AI_QUERY_DEFAULT, "max_papers": "10", "schedule_time": "08:00 AM"}
    for k, v in defaults.items():
        cur.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, v))
    conn.commit()
    conn.close()


def set_setting(key, value):
    ensure_db()
    conn = get_conn()
    conn.execute(
        "INSERT INTO settings (key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value))
    )
    conn.commit()
    conn.close()


def get_setting(key, default=""):
    ensure_db()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else default


def save_run_start(mode, scheduled_time):
    ensure_db()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO agent_runs (mode,scheduled_time,status,started_at,message) VALUES (?,?,?,?,?)",
        (mode, scheduled_time, "running", datetime.now().isoformat(timespec="seconds"), "Running...")
    )
    conn.commit()
    run_id = cur.lastrowid
    conn.close()
    return run_id


def save_run_end(run_id, status, message):
    ensure_db()
    conn = get_conn()
    conn.execute(
        "UPDATE agent_runs SET status=?,finished_at=?,message=? WHERE id=?",
        (status, datetime.now().isoformat(timespec="seconds"), message, run_id)
    )
    conn.commit()
    conn.close()


def get_latest_run():
    ensure_db()
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM agent_runs ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else {}


def get_stats():
    ensure_db()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM papers")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM papers WHERE DATE(created_at)=DATE('now')")
    today = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM agent_runs WHERE status='done'")
    runs = cur.fetchone()[0]
    conn.close()
    return {"total": total, "today": today, "runs": runs}


def get_history_dates(limit=10):
    ensure_db()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT DATE(created_at) AS day FROM papers ORDER BY day DESC LIMIT ?",
        (limit,)
    )
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows


def get_papers_by_date(day):
    ensure_db()
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM papers WHERE DATE(created_at)=? ORDER BY created_at DESC",
        (day,)
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_recent_papers(limit=50):
    ensure_db()
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM papers ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def upsert_paper(paper):
    ensure_db()
    conn = get_conn()
    conn.execute("""
        INSERT INTO papers (paper_id,title,authors,abstract,published,arxiv_url,pdf_url,
            source,summary,key_points,novelty_score,relevance_score)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(paper_id) DO UPDATE SET
            title=excluded.title, authors=excluded.authors, abstract=excluded.abstract,
            published=excluded.published, arxiv_url=excluded.arxiv_url, pdf_url=excluded.pdf_url,
            source=excluded.source, summary=excluded.summary, key_points=excluded.key_points,
            novelty_score=excluded.novelty_score, relevance_score=excluded.relevance_score
    """, (
        paper["paper_id"], paper["title"], paper["authors"], paper["abstract"],
        paper["published"], paper["arxiv_url"], paper["pdf_url"], paper["source"],
        paper.get("summary", ""), paper.get("key_points", ""),
        float(paper.get("novelty_score", 0)), float(paper.get("relevance_score", 0))
    ))
    conn.commit()
    conn.close()


def clear_current_dashboard():
    st.session_state.papers = []
    st.session_state.papers_cleared = True
    st.session_state.banner = "✅ Papers cleared and Stored in history"


def clear_stored_history():
    ensure_db()
    conn = get_conn()
    conn.execute("DELETE FROM papers")
    conn.commit()
    conn.close()


def clear_logs():
    ensure_db()
    conn = get_conn()
    conn.execute("DELETE FROM agent_runs")
    conn.commit()
    conn.close()

# =========================================================
# MODELS
# =========================================================
@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-small")


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


generator = load_generator()
embedder = load_embedder()

# =========================================================
# HELPERS
# =========================================================
AI_KEYWORDS = [
    "transformer", "llm", "language model", "deep learning", "neural", "nlp",
    "machine learning", "artificial intelligence", "ai", "agent"
]


def is_ai_related(text):
    t = (text or "").lower()
    return any(k in t for k in AI_KEYWORDS)


def normalize_title(title):
    return re.sub(r"\s+", " ", (title or "").strip().lower())


def parse_schedule_time(value):
    s = value.strip().upper()
    for fmt in ("%I:%M %p", "%H:%M"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.hour, dt.minute
        except ValueError:
            pass
    raise ValueError("Use 08:00 AM or 13:08")


def next_run_datetime(schedule_value):
    hour, minute = parse_schedule_time(schedule_value)
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def novelty_score(title, abstract):
    text = f"{title} {abstract}".lower()
    score = 4.0
    if any(w in text for w in ["novel", "new", "first", "improve", "efficient"]):
        score += 2.0
    if any(w in text for w in ["survey", "review", "benchmark"]):
        score -= 1.5
    if any(w in text for w in ["transformer", "llm", "language model"]):
        score += 1.0
    if "deep learning" in text:
        score += 0.5
    return max(1.0, min(10.0, score))


def relevance_score(query, text):
    q_vec = embedder.encode([query], normalize_embeddings=True)
    d_vec = embedder.encode([text], normalize_embeddings=True)
    return float(np.dot(q_vec[0], d_vec[0]) * 100.0)


def generate_text(prompt, max_len=140):
    out = generator(prompt, max_length=max_len, do_sample=False)
    return out[0]["generated_text"].strip()


def summarize_abstract(title, abstract):
    prompt = (
        "You are a research analyst. Write a concise 2-sentence summary.\n"
        f"Title: {title}\nAbstract: {abstract[:1600]}\nSummary:"
    )
    try:
        return generate_text(prompt, max_len=120)
    except Exception:
        return "Summary unavailable."


def extract_key_points(title, abstract):
    prompt = (
        "You are a research analyst. List 3 short technical contributions, one per line.\n"
        f"Title: {title}\nAbstract: {abstract[:1600]}\nContributions:"
    )
    try:
        raw = generate_text(prompt, max_len=130)
        lines = [ln.strip("•- ").strip() for ln in raw.splitlines() if ln.strip()]
        return "\n".join(f"• {ln}" for ln in lines[:3]) if lines else "• Contribution details unavailable"
    except Exception:
        return "• Contribution details unavailable"

# =========================================================
# FETCHERS
# =========================================================
def fetch_arxiv(query, max_results):
    papers = []
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        for result in client.results(search):
            title = result.title or ""
            abstract = result.summary or ""
            if not is_ai_related(f"{title} {abstract}"):
                continue
            papers.append({
                "paper_id": result.entry_id.split("/")[-1],
                "title": title,
                "authors": ", ".join(a.name for a in result.authors[:6]),
                "abstract": abstract[:2200],
                "published": result.published.strftime("%Y-%m-%d") if result.published else "",
                "arxiv_url": result.entry_id,
                "pdf_url": result.pdf_url or "",
                "source": "arxiv",
            })
    except Exception:
        pass
    return papers


def fetch_semantic_scholar(query, max_results, retries=3):
    papers = []
    delay = 3
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "paperId,title,authors,abstract,year,externalIds,openAccessPdf,publicationDate"
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
            for item in r.json().get("data", []):
                title = item.get("title", "") or ""
                abstract = item.get("abstract", "") or ""
                if not is_ai_related(f"{title} {abstract}"):
                    continue

                paper_id = "ss_" + (item.get("paperId") or "")
                arxiv_id = (item.get("externalIds") or {}).get("ArXiv", "")
                arxiv_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
                pdf_url = ""
                if item.get("openAccessPdf"):
                    pdf_url = item["openAccessPdf"].get("url", "") or ""
                elif arxiv_id:
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

                papers.append({
                    "paper_id": paper_id,
                    "title": title,
                    "authors": ", ".join(a.get("name", "") for a in item.get("authors", [])[:6]),
                    "abstract": abstract[:2200],
                    "published": item.get("publicationDate") or str(item.get("year") or ""),
                    "arxiv_url": arxiv_url,
                    "pdf_url": pdf_url,
                    "source": "semantic_scholar",
                })
            break
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
    return papers

# =========================================================
# AGENT
# =========================================================
def analyze_paper(paper, query):
    paper = dict(paper)
    paper["summary"] = summarize_abstract(paper["title"], paper["abstract"])
    paper["key_points"] = extract_key_points(paper["title"], paper["abstract"])
    paper["novelty_score"] = novelty_score(paper["title"], paper["abstract"])
    paper["relevance_score"] = relevance_score(query, f"{paper['title']} {paper['abstract']}")
    return paper


def run_agent(mode="manual"):
    ensure_db()
    query = get_setting("search_query", AI_QUERY_DEFAULT)
    max_papers = int(get_setting("max_papers", "10"))
    schedule_time = get_setting("schedule_time", "08:00 AM")
    run_id = save_run_start(mode=mode, scheduled_time=schedule_time)

    try:
        combined = {}
        for p in fetch_arxiv(query, max_papers) + fetch_semantic_scholar(query, max_papers):
            key = normalize_title(p["title"])
            if key not in combined:
                combined[key] = p

        analyzed = []
        for p in combined.values():
            paper = analyze_paper(p, query)
            upsert_paper(paper)
            analyzed.append(paper)

        save_run_end(run_id, "done", f"Saved {len(analyzed)} papers to history")
        return analyzed
    except Exception as e:
        save_run_end(run_id, "error", str(e))
        raise

# =========================================================
# BACKGROUND SCHEDULER
# =========================================================
def scheduler_loop():
    last_run_day = None
    while True:
        try:
            target_hour, target_minute = parse_schedule_time(get_setting("schedule_time", "08:00 AM"))
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")

            if now.hour == target_hour and now.minute == target_minute and last_run_day != today:
                run_agent(mode="scheduled")
                last_run_day = today
        except Exception:
            pass
        time.sleep(5)

# =========================================================
# SESSION STATE
# =========================================================
ensure_db()

if "banner" not in st.session_state:
    st.session_state.banner = ""
if "papers" not in st.session_state:
    st.session_state.papers = get_recent_papers(60)
if "papers_cleared" not in st.session_state:
    st.session_state.papers_cleared = False
if "scheduler_started" not in st.session_state:
    threading.Thread(target=scheduler_loop, daemon=True).start()
    st.session_state.scheduler_started = True

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## ⚙️ Configuration")

schedule_time_input = st.sidebar.text_input(
    "Schedule Time (e.g. 8:00 AM)",
    value=get_setting("schedule_time", "08:00 AM"),
)

search_query_input = st.sidebar.text_area(
    "Search Query",
    value=get_setting("search_query", AI_QUERY_DEFAULT),
    height=80,
)

max_papers_input = st.sidebar.slider(
    "Number of Papers",
    1,
    20,
    int(get_setting("max_papers", "10"))
)

auto_refresh = st.sidebar.toggle("Auto refresh every 30 sec", value=True)

if st.sidebar.button("💾 Save Settings", use_container_width=True):
    try:
        parse_schedule_time(schedule_time_input)
        set_setting("schedule_time", schedule_time_input.strip())
        set_setting("search_query", search_query_input.strip())
        set_setting("max_papers", str(max_papers_input))
        st.sidebar.success("Settings saved.")
    except Exception:
        st.sidebar.error("Use a valid time like 8:00 AM or 13:08.")

st.sidebar.markdown("---")
next_run = next_run_datetime(get_setting("schedule_time", "08:00 AM"))
st.sidebar.markdown(
    f"**Next run:** {next_run.strftime('%Y-%m-%d %I:%M %p')}<br>"
    f"<small>Uses your laptop’s local time.</small>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Danger Zone**")

if st.sidebar.button("🧹 Clear Papers", use_container_width=True):
    clear_current_dashboard()
    st.sidebar.success("Papers cleared and Stored in history")
    st.rerun()

if st.sidebar.button("🗑 Clear Stored History", use_container_width=True):
    clear_stored_history()
    st.sidebar.success("Stored history cleared.")
    st.rerun()

if auto_refresh:
    components.html(
        '<script>setTimeout(()=>window.location.reload(),30000);</script>',
        height=0,
        width=0,
    )

# =========================================================
# HERO + STATS
# =========================================================
latest_run = get_latest_run()
run_status = (latest_run.get("status") or "idle").lower()
stats = get_stats()
scheduled_time = get_setting("schedule_time", "08:00 AM")

st.markdown(f"""
<div class="hero">
    <h1>🚀 AI Research Agent PRO</h1>
    <p>Scheduled fetching · Compact paper cards · Transformer summaries</p>
    <div class="hero-row">
        <span class="pill">⏰ {html.escape(scheduled_time)}</span>
        <span class="pill">📚 {stats["total"]} Papers</span>
        <span class="pill">✨ {stats["today"]} Today</span>
        <span class="pill">🧠 {stats["runs"]} Runs</span>
        <span class="pill">🕒 Local Laptop Time</span>
    </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, label, value, sub in [
    (c1, "Scheduler", run_status.upper(), "Background worker active"),
    (c2, "Saved Papers", stats["total"], "Stored in history"),
    (c3, "Today", stats["today"], "New papers added today"),
    (c4, "Last Run", (latest_run.get("mode") or "—").upper(), latest_run.get("message") or "No run yet"),
]:
    col.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <p class="stat-value">{html.escape(str(value))}</p>
        <div class="stat-sub">{html.escape(str(sub))}</div>
    </div>""", unsafe_allow_html=True)

if run_status == "running":
    st.warning("🤖 AI agent is running... please wait")
elif run_status == "done":
    st.success(latest_run.get("message", "Papers saved"))
elif run_status == "error":
    st.error(latest_run.get("message", "Agent failed"))
else:
    st.info(f"⏳ Waiting for scheduled time: {scheduled_time}")

# =========================================================
# ACTION BAR
# =========================================================
left, right = st.columns(2)

with left:
    if st.button("▶ Run Agent Now", use_container_width=True):
        with st.spinner("Running research agent..."):
            analyzed = run_agent(mode="manual")
        st.session_state.banner = "✅ Papers updated and saved to history"
        st.session_state.papers = analyzed
        st.session_state.papers_cleared = False
        st.rerun()

with right:
    if st.button("🧹 Clear Dashboard", use_container_width=True):
        clear_current_dashboard()
        st.rerun()

if st.session_state.banner:
    st.success(st.session_state.banner)

# =========================================================
# TABS
# =========================================================
tab_papers, tab_history, tab_logs = st.tabs(
    ["📄 Papers", "📚 History", "🧾 Logs"]
)

# ── PAPERS TAB ────────────────────────────────────────────
with tab_papers:
    st.markdown("### Latest Research Papers")

    if st.session_state.papers_cleared:
        recent = []
    else:
        recent = st.session_state.papers
        if not recent:
            recent = get_recent_papers(60)
            st.session_state.papers = recent

    if not recent:
        st.info("No papers currently shown. Run the agent again to repopulate the dashboard.")
    else:
        filter_text = st.text_input("🔍 Filter papers", placeholder="Search by title or author...")
        for idx, p in enumerate(recent, 1):
            if filter_text and filter_text.lower() not in (p["title"] + p["authors"]).lower():
                continue

            novelty = float(p.get("novelty_score") or 0)
            relevance = float(p.get("relevance_score") or 0)

            nov_cls = "badge-green" if novelty >= 8 else "badge-amber" if novelty >= 5 else "badge-slate"
            rel_cls = "badge-green" if relevance >= 70 else "badge-amber" if relevance >= 45 else "badge-slate"
            src_cls = "badge-blue"

            kp_lines = [
                ln.strip("•- ").strip()
                for ln in (p.get("key_points") or "").splitlines()
                if ln.strip()
            ]
            kp_html = "".join(f"<li>{html.escape(k)}</li>" for k in kp_lines[:3]) or "<li>Unavailable</li>"

            summary = html.escape(p.get("summary") or "Summary unavailable.")
            title = html.escape(p["title"] or "")
            authors = html.escape(p["authors"] or "")
            pub = html.escape(p.get("published", "") or "")
            source = html.escape(p.get("source", "") or "")
            link = p.get("arxiv_url") or p.get("pdf_url") or "#"

            st.markdown(f"""
<div class="paper-card">
  <div class="paper-title">{idx}. {title}</div>
  <div class="paper-meta">{authors} · {pub} · {source}</div>
  <span class="badge {nov_cls}">Novelty {novelty:.1f}/10</span>
  <span class="badge {rel_cls}">Relevance {relevance:.0f}%</span>
  <span class="badge {src_cls}">{source}</span>
  <div class="paper-summary"><strong>Summary:</strong> {summary}</div>
  <div style="margin-top:6px;color:#94a3b8;font-size:12px;">
    <strong style="color:#cbd5e1;">Key Points:</strong>
    <ul class="paper-kp">{kp_html}</ul>
  </div>
  <a class="open-link" href="{link}" target="_blank">📄 Open Paper →</a>
</div>""", unsafe_allow_html=True)

# ── HISTORY TAB ───────────────────────────────────────────
with tab_history:
    st.markdown("### 📚 History")

    col_h1, col_h2 = st.columns([3, 1])
    with col_h2:
        if st.button("🗑 Clear Stored History", use_container_width=True):
            clear_stored_history()
            st.rerun()

    dates = get_history_dates(limit=12)
    if not dates:
        st.info("No history yet.")
    else:
        for day in dates:
            day_papers = get_papers_by_date(day)
            with st.expander(f"📅 {day} — {len(day_papers)} papers", expanded=False):
                for i, p in enumerate(day_papers, 1):
                    title = html.escape(p["title"] or "")
                    source = html.escape(p.get("source", "") or "")
                    novelty = float(p.get("novelty_score") or 0)
                    rel = float(p.get("relevance_score") or 0)
                    summary = html.escape(p.get("summary") or "Summary unavailable.")
                    link = p.get("arxiv_url") or p.get("pdf_url") or "#"

                    kp_lines = [
                        ln.strip("•- ").strip()
                        for ln in (p.get("key_points") or "").splitlines()
                        if ln.strip()
                    ]
                    kp_html = "".join(f"<li>{html.escape(k)}</li>" for k in kp_lines[:3]) or "<li>Unavailable</li>"

                    st.markdown(f"""
<div class="paper-card">
  <div class="paper-title">{i}. {title}</div>
  <div class="paper-meta">{source} · Novelty {novelty:.1f}/10 · Relevance {rel:.0f}%</div>
  <div class="paper-summary"><strong>Summary:</strong> {summary}</div>
  <ul class="paper-kp">{kp_html}</ul>
  <a class="open-link" href="{link}" target="_blank">📄 Open Paper →</a>
</div>""", unsafe_allow_html=True)

# ── LOGS TAB ──────────────────────────────────────────────
with tab_logs:
    st.markdown("### 🧾 Agent Logs")

    col_l1, col_l2 = st.columns([3, 1])
    with col_l2:
        if st.button("🗑 Clear Logs", use_container_width=True):
            clear_logs()
            st.rerun()

    conn = get_conn()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM agent_runs ORDER BY id DESC LIMIT 30")
    logs = [dict(r) for r in cur.fetchall()]
    conn.close()

    if not logs:
        st.info("No logs yet.")
    else:
        for row in logs:
            status = (row.get("status") or "").lower()
            badge = "badge-green" if status == "done" else "badge-amber" if status == "running" else "badge-slate"
            st.markdown(f"""
<div class="log-card">
  <span class="badge {badge}">{html.escape(row.get("status", "").upper())}</span>
  <span class="badge badge-blue">{html.escape(row.get("mode", "").upper())}</span>
  <div class="paper-meta" style="margin-top:6px;">{html.escape(row.get("started_at", ""))}</div>
  <div style="color:#cbd5e1;font-size:13px;">{html.escape(row.get("message", ""))}</div>
</div>""", unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">made by <b>Eng Kirollos Ashraf</b></div>
""", unsafe_allow_html=True)
