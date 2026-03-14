#!/usr/bin/env python3

from __future__ import annotations
import os, sys, re, json, math, argparse, tempfile, asyncio
from pathlib import Path
from typing import AsyncIterator, Iterator, Literal
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import tiktoken
import httpx
import psycopg2
from psycopg2.extras import execute_values, Json
from pgvector.psycopg2 import register_vector
from pypdf import PdfReader
from openai import OpenAI
from sklearn.mixture import GaussianMixture
from sentence_transformers import CrossEncoder
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError: pass

console = Console()

OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY", "")
EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
SUMMARISATION_MODEL  = os.getenv("SUMMARISATION_MODEL", "gpt-4o-mini")
QUIZ_LLM_PROVIDER    = os.getenv("QUIZ_LLM_PROVIDER", "openai")
QUIZ_OPENAI_MODEL    = os.getenv("QUIZ_OPENAI_MODEL", "gpt-4o-mini")
QUIZ_ANTHROPIC_MODEL = os.getenv("QUIZ_ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
QUIZ_NUM_QUESTIONS   = int(os.getenv("QUIZ_NUM_QUESTIONS", "10"))
QUIZ_TEMPERATURE     = float(os.getenv("QUIZ_TEMPERATURE", "0.0"))

PG_HOST     = os.getenv("PG_HOST", "localhost")
PG_PORT     = int(os.getenv("PG_PORT", "5432"))
PG_USER     = os.getenv("PG_USER", "")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")
PG_DATABASE = os.getenv("PG_DATABASE", "ragdb")
PG_DSN      = f"host={PG_HOST} port={PG_PORT} dbname={PG_DATABASE} user={PG_USER} password={PG_PASSWORD}"

CHUNK_SIZE  = int(os.getenv("RAPTOR_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("RAPTOR_CHUNK_OVERLAP", "64"))
RAPTOR_MAX  = int(os.getenv("RAPTOR_MAX_TREE_LEVELS", "3"))
RAPTOR_K    = int(os.getenv("RAPTOR_NUM_CLUSTERS", "10"))

RERANKER_MODEL  = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_TOP_K  = int(os.getenv("RERANKER_TOP_K", "3"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "20"))

OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TEMP  = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))
OLLAMA_CTX   = int(os.getenv("OLLAMA_CONTEXT_WINDOW", "8192"))
API_HOST     = os.getenv("API_HOST", "0.0.0.0")
API_PORT     = int(os.getenv("API_PORT", "8000"))

_oai = OpenAI(api_key=OPENAI_API_KEY)

def get_conn():
    c = psycopg2.connect(PG_DSN); register_vector(c); return c

def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS documents(id SERIAL PRIMARY KEY,filename TEXT NOT NULL,upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,total_pages INTEGER,total_chunks INTEGER);
        CREATE TABLE IF NOT EXISTS chunks(id SERIAL PRIMARY KEY,document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,chunk_index INTEGER NOT NULL,content TEXT NOT NULL,token_count INTEGER,metadata JSONB DEFAULT '{}');
        CREATE TABLE IF NOT EXISTS raptor_nodes(id SERIAL PRIMARY KEY,document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,node_type TEXT NOT NULL CHECK(node_type IN('leaf','summary')),tree_level INTEGER NOT NULL DEFAULT 0,content TEXT NOT NULL,token_count INTEGER,children_ids INTEGER[] DEFAULT '{}',source_chunk_ids INTEGER[] DEFAULT '{}',embedding vector(1536),metadata JSONB DEFAULT '{}');
        CREATE INDEX IF NOT EXISTS idx_re ON raptor_nodes USING hnsw(embedding vector_cosine_ops) WITH(m=16,ef_construction=64);
        CREATE INDEX IF NOT EXISTS idx_rd ON raptor_nodes(document_id);
        CREATE TABLE IF NOT EXISTS page_texts(id SERIAL PRIMARY KEY,document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,page_number INTEGER NOT NULL,content TEXT NOT NULL,chapter_title TEXT);
        CREATE INDEX IF NOT EXISTS idx_pt ON page_texts(document_id);
        CREATE TABLE IF NOT EXISTS quizzes(id SERIAL PRIMARY KEY,document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,title TEXT NOT NULL,source_type TEXT NOT NULL,source_value TEXT NOT NULL,num_questions INTEGER NOT NULL,difficulty TEXT DEFAULT 'medium',quiz_data JSONB NOT NULL,score INTEGER,total_attempted INTEGER,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
        """)
    conn.commit()

def insert_doc(conn, fn, pages):
    with conn.cursor() as c:
        c.execute("INSERT INTO documents(filename,total_pages) VALUES(%s,%s) RETURNING id",(fn,pages))
        r=c.fetchone()[0]
    conn.commit(); return r

def insert_chunks_db(conn, did, chunks):
    with conn.cursor() as c:
        rows=[(did,ch["index"],ch["content"],ch["token_count"],Json({})) for ch in chunks]
        r=execute_values(c,"INSERT INTO chunks(document_id,chunk_index,content,token_count,metadata) VALUES %s RETURNING id",rows,fetch=True)
    conn.commit(); return [x[0] for x in r]

def insert_nodes(conn, nodes):
    with conn.cursor() as c:
        rows=[(n["document_id"],n["node_type"],n["tree_level"],n["content"],n["token_count"],n.get("children_ids",[]),n.get("source_chunk_ids",[]),n["embedding"],Json(n.get("metadata",{}))) for n in nodes]
        r=execute_values(c,"INSERT INTO raptor_nodes(document_id,node_type,tree_level,content,token_count,children_ids,source_chunk_ids,embedding,metadata) VALUES %s RETURNING id",rows,fetch=True)
    conn.commit(); return [x[0] for x in r]

def search_sim(conn,emb,k=20,did=None):
    with conn.cursor() as c:
        if did:
            c.execute("SELECT id,node_type,tree_level,content,metadata,1-(embedding<=>%s::vector) AS sim FROM raptor_nodes WHERE document_id=%s ORDER BY embedding<=>%s::vector LIMIT %s",(emb,did,emb,k))
        else:
            c.execute("SELECT id,node_type,tree_level,content,metadata,1-(embedding<=>%s::vector) AS sim FROM raptor_nodes ORDER BY embedding<=>%s::vector LIMIT %s",(emb,emb,k))
        cols=[d[0] for d in c.description]
        return [dict(zip(cols,r)) for r in c.fetchall()]



def extract_pdf(path):
    reader=PdfReader(str(path)); pages=[p.extract_text() or "" for p in reader.pages]; return pages,len(pages)

def _enc():
    try: return tiktoken.encoding_for_model(EMBEDDING_MODEL)
    except: return tiktoken.get_encoding("cl100k_base")

def chunk_text(text,sz=CHUNK_SIZE,ov=CHUNK_OVERLAP):
    enc=_enc(); toks=enc.encode(text); chunks=[]; s=0; i=0
    while s<len(toks):
        e=min(s+sz,len(toks)); ct=toks[s:e]
        chunks.append({"index":i,"content":enc.decode(ct),"token_count":len(ct)}); i+=1; s+=sz-ov
    return chunks

def count_tok(t): return len(_enc().encode(t))

def embed_texts(texts):
    all_e=[]
    for i in range(0,len(texts),2048):
        b=texts[i:i+2048]; r=_oai.embeddings.create(input=b,model=EMBEDDING_MODEL)
        all_e.extend([d.embedding for d in r.data])
    return all_e

def embed_one(t): return embed_texts([t])[0]

def summarise(texts):
    r=_oai.chat.completions.create(model=SUMMARISATION_MODEL,temperature=0,messages=[
        {"role":"system","content":"Summarise the passages concisely. Only use info from them."},
        {"role":"user","content":"Summarise:\n\n"+"\n---\n".join(texts)}],max_tokens=1024)
    return r.choices[0].message.content.strip()

def build_raptor(did,chunks,cids,conn,max_lev=RAPTOR_MAX):
    all_n=[]
    with Progress(SpinnerColumn(),TextColumn("[blue]{task.description}"),transient=True) as prog:
        t=prog.add_task("Embedding leaves…"); texts=[c["content"] for c in chunks]; embs=embed_texts(texts)
        leaves=[]
        for i,(ch,em) in enumerate(zip(chunks,embs)):
            leaves.append({"content":ch["content"],"embedding":em,"token_count":ch["token_count"],"node_type":"leaf","tree_level":0,"source_chunk_ids":[cids[i]],"children_ids":[]})
        rows=[{**n,"document_id":did,"metadata":{}} for n in leaves]
        ids=insert_nodes(conn,rows)
        for n,i in zip(leaves,ids): n["db_id"]=i
        all_n.extend(leaves); prog.update(t,completed=True)
        cur=leaves
        for lev in range(1,max_lev+1):
            t=prog.add_task(f"Level {lev}…")
            if len(cur)<=3: prog.update(t,completed=True); break
            mat=np.array([n["embedding"] for n in cur])
            nc=max(2,min(RAPTOR_K,len(cur)//2))
            gmm=GaussianMixture(n_components=nc,covariance_type="full",random_state=42,max_iter=300)
            gmm.fit(mat); probs=gmm.predict_proba(mat)
            clusters=[[] for _ in range(nc)]
            for idx in range(len(cur)):
                for comp in range(nc):
                    if probs[idx,comp]>0.1: clusters[comp].append(idx)
            clusters=[c for c in clusters if c]
            sums=[]
            for cl in clusters:
                txts=[cur[i]["content"] for i in cl]; cids_cl=[cur[i]["db_id"] for i in cl]
                srcs=list(set(s for i in cl for s in cur[i].get("source_chunk_ids",[])))
                st=summarise(txts); se=embed_texts([st])[0]
                sums.append({"content":st,"embedding":se,"token_count":count_tok(st),"node_type":"summary","tree_level":lev,"children_ids":cids_cl,"source_chunk_ids":srcs})
            rows=[{**n,"document_id":did,"metadata":{"level":lev}} for n in sums]
            ids=insert_nodes(conn,rows)
            for n,i in zip(sums,ids): n["db_id"]=i
            all_n.extend(sums); cur=sums; prog.update(t,completed=True)
    return all_n

# Chapter detection
CHAP_PAT=[re.compile(r"^(chapter\s+[\divxlc]+[\s:.\-–—]*\s*.+)",re.I|re.M),re.compile(r"^(CHAPTER\s+[\dIVXLC]+[\s:.\-–—]*.*)",re.M),re.compile(r"^(\d{1,2}\.\s+[A-Z][a-zA-Z\s]{3,50})$",re.M)]

def _det_chap(text):
    for p in CHAP_PAT:
        m=p.search(text[:500])
        if m: return re.sub(r"\s+"," ",m.group(1).strip())[:120]
    return None

def extract_pages_chapters(path):
    reader=PdfReader(str(path)); bm={}
    try:
        for item in (reader.outline or []):
            if not isinstance(item,list):
                try: pn=reader.get_destination_page_number(item); bm[pn]=item.title.strip()
                except: pass
    except: pass
    pages=[]; cur=None
    for i,pg in enumerate(reader.pages):
        txt=pg.extract_text() or ""
        if i in bm: cur=bm[i]
        else:
            d=_det_chap(txt)
            if d: cur=d
        pages.append({"page_number":i+1,"content":txt,"chapter_title":cur})
    return pages

def store_pages(conn,did,pages):
    with conn.cursor() as c:
        c.execute("DELETE FROM page_texts WHERE document_id=%s",(did,))
        execute_values(c,"INSERT INTO page_texts(document_id,page_number,content,chapter_title) VALUES %s",
                       [(did,p["page_number"],p["content"],p["chapter_title"]) for p in pages])
    conn.commit()


_reranker=None
def get_reranker():
    global _reranker
    if _reranker is None: _reranker=CrossEncoder(RERANKER_MODEL,max_length=512)
    return _reranker

def rerank(query,cands,k=RERANKER_TOP_K):
    if not cands: return []
    m=get_reranker(); pairs=[(query,c["content"]) for c in cands]; scores=m.predict(pairs,show_progress_bar=False)
    for c,s in zip(cands,scores): c["rerank_score"]=float(s)
    return sorted(cands,key=lambda c:c["rerank_score"],reverse=True)[:k]

def retrieve_rerank(q,did=None):
    emb=embed_one(q); conn=get_conn(); cands=search_sim(conn,emb,k=RETRIEVAL_TOP_K,did=did); conn.close()
    return rerank(q,cands) if cands else []

SYS_PROMPT=("You are a helpful assistant. Answer based ONLY on the provided context. "
            "Cite passage numbers like [1]. If context is insufficient, say so.")

def build_prompt(q,passages):
    parts=[f"[{i+1}] [{p.get('node_type','')} L{p.get('tree_level','')}]\n{p['content']}" for i,p in enumerate(passages)]
    return f"--- Context ---\n{chr(10).join(parts)}\n--- End ---\n\nQuestion: {q}\n\nAnswer:"

def ollama_stream(prompt,system=""):
    payload={"model":OLLAMA_MODEL,"prompt":prompt,"system":system,"stream":True,"options":{"temperature":OLLAMA_TEMP,"num_ctx":OLLAMA_CTX}}
    with httpx.stream("POST",f"{OLLAMA_URL}/api/generate",json=payload,timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line: continue
            d=json.loads(line); t=d.get("response","")
            if t: yield t
            if d.get("done"): break

def ollama_gen(prompt,system=""):
    payload={"model":OLLAMA_MODEL,"prompt":prompt,"system":system,"stream":False,"options":{"temperature":OLLAMA_TEMP,"num_ctx":OLLAMA_CTX}}
    r=httpx.post(f"{OLLAMA_URL}/api/generate",json=payload,timeout=120); r.raise_for_status(); return r.json()["response"]

async def ollama_astream(prompt,system=""):
    payload={"model":OLLAMA_MODEL,"prompt":prompt,"system":system,"stream":True,"options":{"temperature":OLLAMA_TEMP,"num_ctx":OLLAMA_CTX}}
    async with httpx.AsyncClient(timeout=120) as cl:
        async with cl.stream("POST",f"{OLLAMA_URL}/api/generate",json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line: continue
                d=json.loads(line); t=d.get("response","")
                if t: yield t
                if d.get("done"): break

def is_ollama_ok():
    try:
        r=httpx.get(f"{OLLAMA_URL}/api/tags",timeout=5); r.raise_for_status()
        return any(OLLAMA_MODEL.split(":")[0] in m["name"] for m in r.json().get("models",[]))
    except: return False

def list_models():
    try: r=httpx.get(f"{OLLAMA_URL}/api/tags",timeout=5); return [m["name"] for m in r.json().get("models",[])]
    except: return []


QUIZ_SYS="""You are an expert quiz generator. Create questions from the provided text ONLY.
Rules: MCQ=4 options A-D; True/False options=["True","False"]; Fill blank uses "___".
Respond with ONLY JSON: {"questions":[{"question_number":1,"question_type":"mcq","question":"...","options":[...],"correct_answer":"A","explanation":"...","difficulty":"medium","source_page":1}]}"""

def _parse_pages(spec,mx):
    ps=set()
    for p in spec.replace(" ","").split(","):
        if "-" in p:
            try: s,e=p.split("-",1); ps.update(range(max(1,int(s)),min(mx,int(e))+1))
            except: pass
        else:
            try: v=int(p); ps.add(v) if 1<=v<=mx else None
            except: pass
    return sorted(ps)

def gen_quiz(did,src_type,src_val,num_q=QUIZ_NUM_QUESTIONS,diff="medium",prov=QUIZ_LLM_PROVIDER):
    conn=get_conn(); ensure_schema(conn)
    if src_type=="pages":
        with conn.cursor() as c:
            c.execute("SELECT COALESCE(MAX(page_number),0) FROM page_texts WHERE document_id=%s",(did,))
            mx=c.fetchone()[0]
        if not mx: conn.close(); raise ValueError(f"No pages for doc {did}. Run reindex.")
        pns=_parse_pages(src_val,mx)
        if not pns: conn.close(); raise ValueError(f"Invalid pages: {src_val}")
        with conn.cursor() as c:
            c.execute("SELECT page_number,content,chapter_title FROM page_texts WHERE document_id=%s AND page_number=ANY(%s) ORDER BY page_number",(did,pns))
            content=[{"page_number":r[0],"content":r[1],"chapter_title":r[2]} for r in c.fetchall()]
        title=f"Quiz — Pages {src_val}"
    else:
        titles=[t.strip() for t in src_val.split(",") if t.strip()]
        conds=["chapter_title ILIKE %s" for _ in titles]; params=[did]+[f"%{t}%" for t in titles]
        with conn.cursor() as c:
            c.execute(f"SELECT page_number,content,chapter_title FROM page_texts WHERE document_id=%s AND ({' OR '.join(conds)}) ORDER BY page_number",params)
            content=[{"page_number":r[0],"content":r[1],"chapter_title":r[2]} for r in c.fetchall()]
        title=f"Quiz — {src_val}"

    content=[p for p in content if p["content"].strip()]
    if not content: conn.close(); raise ValueError("No content found.")
    total=sum(len(p["content"]) for p in content)
    if total>80000:
        trimmed=[]; run=0
        for p in content:
            if run+len(p["content"])>80000: break
            trimmed.append(p); run+=len(p["content"])
        content=trimmed

    ptexts=[]
    for p in content:
        h=f"--- Page {p['page_number']}"+(f" ({p['chapter_title']})" if p.get("chapter_title") else "")+" ---"
        ptexts.append(f"{h}\n{p['content']}")
    user_p=f"Generate {num_q} questions ({diff}).\n\nTEXT:\n\n"+"\n\n".join(ptexts)+f"\n\nGenerate {num_q} as JSON:"

    if prov=="anthropic":
        import anthropic; cl=anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        raw=cl.messages.create(model=QUIZ_ANTHROPIC_MODEL,max_tokens=4096,temperature=QUIZ_TEMPERATURE,system=QUIZ_SYS,messages=[{"role":"user","content":user_p}]).content[0].text.strip()
    else:
        raw=_oai.chat.completions.create(model=QUIZ_OPENAI_MODEL,temperature=QUIZ_TEMPERATURE,messages=[{"role":"system","content":QUIZ_SYS},{"role":"user","content":user_p}],max_tokens=4096,response_format={"type":"json_object"}).choices[0].message.content.strip()

    cleaned=re.sub(r"^```(?:json)?\s*","",raw); cleaned=re.sub(r"\s*```$","",cleaned).strip()
    data=json.loads(cleaned)
    qs=data["questions"] if isinstance(data,dict) and "questions" in data else data
    with conn.cursor() as c:
        c.execute("INSERT INTO quizzes(document_id,title,source_type,source_value,num_questions,difficulty,quiz_data) VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                  (did,title,src_type,src_val,len(qs),diff,Json({"questions":qs})))
        qid=c.fetchone()[0]
    conn.commit(); conn.close()
    return {"quiz_id":qid,"title":title,"num_questions":len(qs),"difficulty":diff,"questions":qs}



def get_completed_quizzes(did=None):
    conn=get_conn()
    with conn.cursor() as c:
        sql="SELECT q.id,q.document_id,d.filename,q.title,q.source_value,q.num_questions,q.difficulty,q.score,q.total_attempted,q.created_at FROM quizzes q JOIN documents d ON q.document_id=d.id WHERE q.score IS NOT NULL"
        if did: sql+=" AND q.document_id=%s"; c.execute(sql+" ORDER BY q.created_at DESC",(did,))
        else: c.execute(sql+" ORDER BY q.created_at DESC")
        rows=c.fetchall()
    conn.close()
    return [{"id":r[0],"doc_id":r[1],"filename":r[2],"title":r[3],"source_value":r[4],"num_q":r[5],"difficulty":r[6],"score":r[7],"attempted":r[8],"date":str(r[9])[:19]} for r in rows]

def compute_analytics(did=None):
    quizzes=get_completed_quizzes(did)
    if not quizzes: return None
    total_q=sum(q["attempted"] or q["num_q"] for q in quizzes)
    total_c=sum(q["score"] or 0 for q in quizzes)
    acc=round(total_c/total_q*100,1) if total_q else 0

    topics={}
    for q in quizzes:
        t=q["source_value"]
        if t not in topics: topics[t]={"total":0,"correct":0}
        topics[t]["total"]+=(q["attempted"] or q["num_q"])
        topics[t]["correct"]+=(q["score"] or 0)
    for t in topics: topics[t]["accuracy"]=round(topics[t]["correct"]/topics[t]["total"]*100,1) if topics[t]["total"] else 0

    weak=[{"topic":t,"accuracy":d["accuracy"],"incorrect":d["total"]-d["correct"],"total":d["total"],"gap":round(70-d["accuracy"],1)} for t,d in topics.items() if d["accuracy"]<70]
    weak.sort(key=lambda w:w["accuracy"])

    suggestions=[]
    for w in weak[:5]:
        suggestions.append(f'Review "{w["topic"]}" — {w["accuracy"]}% accuracy ({w["incorrect"]} wrong). Focus on re-reading this section.')
    if not suggestions: suggestions.append("Great work across all topics!")

    return {"total_quizzes":len(quizzes),"total_questions":total_q,"total_correct":total_c,"accuracy":acc,
            "topics":sorted(topics.items(),key=lambda x:x[1]["accuracy"]),
            "weak":weak,"suggestions":suggestions,"timeline":[{"date":q["date"],"title":q["title"],
            "accuracy":round((q["score"] or 0)/max(1,q["attempted"] or 1)*100,1)} for q in quizzes]}




def cmd_ingest(pdf_path):
    p=Path(pdf_path)
    if not p.exists(): console.print("[red]File not found[/red]"); sys.exit(1)
    console.print(Panel(f"[bold]Ingesting:[/bold] {p.name}",style="cyan"))
    pages,total=extract_pdf(p); full="\n\n".join(pages)
    if not full.strip(): console.print("[red]No text extracted[/red]"); sys.exit(1)
    chunks=chunk_text(full)
    console.print(f"  {total} pages, {len(chunks)} chunks")
    conn=get_conn(); ensure_schema(conn)
    did=insert_doc(conn,p.name,total); cids=insert_chunks_db(conn,did,chunks)
    with conn.cursor() as c: c.execute("UPDATE documents SET total_chunks=%s WHERE id=%s",(len(chunks),did))
    conn.commit()
    console.print("[dim]Building RAPTOR tree…[/dim]")
    nodes=build_raptor(did,chunks,cids,conn)
    console.print("[dim]Storing page texts…[/dim]")
    pg=extract_pages_chapters(p); store_pages(conn,did,pg); conn.close()
    console.print(Panel(f"[green]Done![/green] ID={did}, {len(nodes)} nodes, {len(set(pi['chapter_title'] for pi in pg if pi['chapter_title']))} chapters",style="green"))

def cmd_serve():
    import uvicorn
    console.print(Panel(f"[green]Starting API[/green] on {API_HOST}:{API_PORT}\nDocs: http://localhost:{API_PORT}/docs",title="FastAPI"))
    sys.path.insert(0,str(Path(__file__).parent))
    from src.api import app
    uvicorn.run(app,host=API_HOST,port=API_PORT)

def cmd_ui():
    import subprocess
    candidates=[Path(__file__).parent/"src"/"app.py",Path("src/app.py")]
    ui=next((p for p in candidates if p.exists()),None)
    if not ui: console.print("[red]app.py not found[/red]"); sys.exit(1)
    console.print(Panel("[green]Starting unified UI[/green]\n  http://localhost:8501\n  API must be running on :8000",title="Streamlit"))
    subprocess.run(["streamlit","run",str(ui),"--server.port","8501"])

def cmd_ask(question,did,stream):
    passages=retrieve_rerank(question,did)
    if not passages: console.print("[yellow]No results[/yellow]"); return
    prompt=build_prompt(question,passages)
    if stream:
        ans=""
        with Live(Text(""),console=console,refresh_per_second=15) as live:
            for tok in ollama_stream(prompt,SYS_PROMPT): ans+=tok; live.update(Text(ans))
        console.print()
    else:
        ans=ollama_gen(prompt,SYS_PROMPT); console.print(f"\n{ans}\n")
    table=Table(title="Sources",show_lines=True); table.add_column("#",width=3); table.add_column("Type",width=8); table.add_column("Score",width=8); table.add_column("Preview",max_width=60)
    for i,p in enumerate(passages,1): table.add_row(str(i),p.get("node_type",""),f"{p.get('rerank_score',0):.3f}",p["content"][:120].replace("\n"," ")+"…")
    console.print(table)

def cmd_quiz(did,pages,chapters,num,diff,prov):
    if pages: st,sv="pages",pages
    elif chapters: st,sv="chapters",chapters
    else: console.print("[red]Specify --pages or --chapters[/red]"); return
    try: r=gen_quiz(did,st,sv,num,diff,prov)
    except ValueError as e: console.print(f"[red]{e}[/red]"); return
    console.print(Panel(f"[green]Quiz ID {r['quiz_id']}[/green] — {r['num_questions']} questions",title=r["title"]))
    for q in r["questions"]:
        console.print(f"\n[bold]Q{q['question_number']}[/bold] {q['question']}")
        if q["question_type"]=="mcq":
            for o in q.get("options",[]): console.print(f"  {o}{' ✓' if o[0]==q['correct_answer'] else ''}")
        else: console.print(f"  Answer: {q['correct_answer']}")

def cmd_dashboard(did):
    a=compute_analytics(did)
    if not a: console.print("[yellow]No completed quizzes yet.[/yellow]"); return
    console.print(Panel(f"Quizzes: {a['total_quizzes']} | Questions: {a['total_questions']} | Accuracy: {a['accuracy']}%",title="Dashboard",style="blue"))
    if a["topics"]:
        t=Table(title="Topic Performance",show_lines=True); t.add_column("Topic"); t.add_column("Accuracy"); t.add_column("Correct/Total")
        for name,d in a["topics"]: t.add_row(name[:40],f"{d['accuracy']}%",f"{d['correct']}/{d['total']}")
        console.print(t)
    if a["weak"]:
        console.print("\n[bold red]Weak topics:[/bold red]")
        for w in a["weak"]: console.print(f"  ⚠️  {w['topic']} — {w['accuracy']}% ({w['gap']}% below target)")
    console.print("\n[bold]Suggestions:[/bold]")
    for s in a["suggestions"]: console.print(f"  💡 {s}")

def cmd_chapters(did):
    conn=get_conn(); ensure_schema(conn)
    with conn.cursor() as c:
        c.execute("SELECT chapter_title,MIN(page_number),MAX(page_number),COUNT(*) FROM page_texts WHERE document_id=%s AND chapter_title IS NOT NULL GROUP BY chapter_title ORDER BY MIN(page_number)",(did,))
        rows=c.fetchall()
    conn.close()
    if not rows: console.print("[yellow]No chapters detected.[/yellow]"); return
    t=Table(title=f"Chapters (doc {did})",show_lines=True); t.add_column("Chapter"); t.add_column("Pages"); t.add_column("Count")
    for r in rows: t.add_row(r[0],f"{r[1]}-{r[2]}",str(r[3]))
    console.print(t)

def cmd_health():
    db=False
    try: c=get_conn(); c.close(); db=True
    except: pass
    o=is_ollama_ok()
    console.print(f"  {'✅' if db else '❌'} PostgreSQL")
    console.print(f"  {'✅' if o else '❌'} Ollama ({OLLAMA_MODEL})")
    console.print(f"  {'✅' if OPENAI_API_KEY else '❌'} OpenAI key")
    console.print(f"  {'✅' if ANTHROPIC_API_KEY else '❌'} Anthropic key")


def main():
    p=argparse.ArgumentParser(description="RAG System — Complete",formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python rag_complete.py ingest doc.pdf\n  python rag_complete.py serve\n  python rag_complete.py ui\n  python rag_complete.py ask 'What is X?' --stream\n  python rag_complete.py quiz --doc-id 1 --pages '1-10'\n  python rag_complete.py dashboard")
    s=p.add_subparsers(dest="cmd")
    si=s.add_parser("ingest"); si.add_argument("pdf")
    s.add_parser("serve"); s.add_parser("ui"); s.add_parser("health")
    sa=s.add_parser("ask"); sa.add_argument("question"); sa.add_argument("--doc-id",type=int); sa.add_argument("--stream",action="store_true")
    sq=s.add_parser("quiz"); sq.add_argument("--doc-id",type=int,required=True); sq.add_argument("--pages"); sq.add_argument("--chapters"); sq.add_argument("--num",type=int,default=10); sq.add_argument("--difficulty",default="medium"); sq.add_argument("--provider",default="openai")
    sc=s.add_parser("chapters"); sc.add_argument("doc_id",type=int)
    sd=s.add_parser("dashboard"); sd.add_argument("--doc-id",type=int)
    a=p.parse_args()
    if a.cmd=="ingest": cmd_ingest(a.pdf)
    elif a.cmd=="serve": cmd_serve()
    elif a.cmd=="ui": cmd_ui()
    elif a.cmd=="health": cmd_health()
    elif a.cmd=="ask": cmd_ask(a.question,a.doc_id,a.stream)
    elif a.cmd=="quiz": cmd_quiz(a.doc_id,a.pages,a.chapters,a.num,a.difficulty,a.provider)
    elif a.cmd=="chapters": cmd_chapters(a.doc_id)
    elif a.cmd=="dashboard": cmd_dashboard(getattr(a,"doc_id",None))
    else: p.print_help()

if __name__=="__main__": main()
