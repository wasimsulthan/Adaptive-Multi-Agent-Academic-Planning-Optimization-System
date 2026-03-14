from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.config import OLLAMA_MODEL, RETRIEVAL_TOP_K, RERANKER_TOP_K, QUIZ_LLM_PROVIDER
from src.database import get_connection, insert_document, insert_chunks, update_document_chunk_count
from src.pdf_loader import extract_text_from_pdf
from src.chunker import chunk_text
from src.raptor_indexer import build_raptor_tree
from src.ollama_client import is_ollama_available, list_models
from src.query_pipeline import (
    query as run_query,
    aquery_stream,
    retrieve_and_rerank,
    assemble_context,
)
from src.page_extractor import (
    extract_pages_with_chapters,
    store_page_texts,
    get_document_chapters,
    get_document_page_count,
)
from src.quiz_generator import (
    generate_quiz,
    get_quiz,
    list_quizzes,
    save_quiz_score,
    delete_quiz,
)


app = FastAPI(
    title="RAG System API",
    description="RAPTOR-indexed RAG with cross-encoder re-ranking and Ollama LLM",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    doc_id: int | None = Field(None, description="Restrict search to a specific document")
    retrieval_top_k: int = Field(RETRIEVAL_TOP_K, ge=1, le=100)
    rerank_top_k: int = Field(RERANKER_TOP_K, ge=1, le=50)


class PassageResponse(BaseModel):
    id: int
    node_type: str
    tree_level: int
    content: str
    similarity: float | None = None
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    passages: list[PassageResponse]
    model: str


class DocumentResponse(BaseModel):
    id: int
    filename: str
    total_pages: int | None
    total_chunks: int | None
    upload_date: str


class DocumentDetailResponse(DocumentResponse):
    raptor_stats: dict


class HealthResponse(BaseModel):
    status: str
    database: bool
    ollama: bool
    ollama_model: str
    available_models: list[str]


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check system health: database connectivity and Ollama availability."""
    db_ok = False
    try:
        conn = get_connection()
        conn.close()
        db_ok = True
    except Exception:
        pass

    ollama_ok = is_ollama_available()
    models = list_models()

    return HealthResponse(
        status="ok" if (db_ok and ollama_ok) else "degraded",
        database=db_ok,
        ollama=ollama_ok,
        ollama_model=OLLAMA_MODEL,
        available_models=models,
    )


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    result = run_query(
        question=req.question,
        doc_id=req.doc_id,
        retrieval_top_k=req.retrieval_top_k,
        rerank_top_k=req.rerank_top_k,
    )

    passages = [
        PassageResponse(
            id=p.get("id", 0),
            node_type=p.get("node_type", ""),
            tree_level=p.get("tree_level", 0),
            content=p.get("content", ""),
            similarity=p.get("similarity"),
            rerank_score=p.get("rerank_score"),
        )
        for p in result.passages
    ]

    return QueryResponse(
        question=result.query,
        answer=result.answer,
        passages=passages,
        model=OLLAMA_MODEL,
    )


@app.post("/query/stream")
async def query_stream_endpoint(req: QueryRequest):
    """
    Stream the LLM answer token-by-token via Server-Sent Events.

    Events:
      - type: "passages"  → JSON array of source passages
      - type: "token"     → a single text token
      - type: "done"      → signals end of stream
      - type: "error"     → an error message
    """
    try:
        passages, token_stream = await aquery_stream(
            question=req.question,
            doc_id=req.doc_id,
            retrieval_top_k=req.retrieval_top_k,
            rerank_top_k=req.rerank_top_k,
        )
    except Exception as e:
        async def error_stream():
            yield {"event": "error", "data": str(e)}
        return EventSourceResponse(error_stream())

    async def event_generator():
        # First, send the retrieved passages
        passage_data = [
            {
                "id": p.get("id", 0),
                "node_type": p.get("node_type", ""),
                "tree_level": p.get("tree_level", 0),
                "content": p.get("content", "")[:500],
                "rerank_score": p.get("rerank_score"),
            }
            for p in passages
        ]
        yield {"event": "passages", "data": json.dumps(passage_data)}

        # Then stream tokens
        try:
            async for token in token_stream:
                yield {"event": "token", "data": token}
        except Exception as e:
            yield {"event": "error", "data": str(e)}

        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())

@app.post("/ingest", response_model=DocumentResponse)
async def ingest_endpoint(file: UploadFile = File(...)):
    """Upload and ingest a PDF document."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Extract
        pages, total_pages = extract_text_from_pdf(tmp_path)
        full_text = "\n\n".join(pages)

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF.")


        chunks = chunk_text(full_text)

        conn = get_connection()
        doc_id = insert_document(conn, file.filename, total_pages)
        chunk_db_ids = insert_chunks(conn, doc_id, chunks)
        update_document_chunk_count(conn, doc_id, len(chunks))


        build_raptor_tree(doc_id, chunks, chunk_db_ids, conn)

        page_infos = extract_pages_with_chapters(tmp_path)
        store_page_texts(conn, doc_id, page_infos)

        conn.close()

        return DocumentResponse(
            id=doc_id,
            filename=file.filename,
            total_pages=total_pages,
            total_chunks=len(chunks),
            upload_date="now",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/documents", response_model=list[DocumentResponse])
def list_documents():
    """List all ingested documents."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, filename, total_pages, total_chunks, upload_date "
            "FROM documents ORDER BY upload_date DESC"
        )
        rows = cur.fetchall()
    conn.close()

    return [
        DocumentResponse(
            id=r[0], filename=r[1], total_pages=r[2],
            total_chunks=r[3], upload_date=str(r[4])[:19],
        )
        for r in rows
    ]


@app.get("/documents/{doc_id}", response_model=DocumentDetailResponse)
def get_document(doc_id: int):
    """Get document details including RAPTOR tree statistics."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, filename, total_pages, total_chunks, upload_date "
            "FROM documents WHERE id = %s",
            (doc_id,),
        )
        row = cur.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Document not found")

        cur.execute(
            "SELECT node_type, tree_level, COUNT(*) "
            "FROM raptor_nodes WHERE document_id = %s "
            "GROUP BY node_type, tree_level ORDER BY tree_level",
            (doc_id,),
        )
        stats_rows = cur.fetchall()
    conn.close()

    raptor_stats = {}
    for node_type, level, count in stats_rows:
        raptor_stats[f"level_{level}_{node_type}"] = count

    return DocumentDetailResponse(
        id=row[0], filename=row[1], total_pages=row[2],
        total_chunks=row[3], upload_date=str(row[4])[:19],
        raptor_stats=raptor_stats,
    )


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int):
    """Delete a document and all its chunks/nodes."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM documents WHERE id = %s", (doc_id,))
        if not cur.fetchone():
            conn.close()
            raise HTTPException(status_code=404, detail="Document not found")
        cur.execute("DELETE FROM raptor_nodes WHERE document_id = %s", (doc_id,))
        cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    conn.commit()
    conn.close()
    return {"message": f"Document {doc_id} deleted successfully"}

class QuizGenerateRequest(BaseModel):
    document_id: int = Field(..., description="Document to generate quiz from")
    source_type: str = Field(..., description="'pages' or 'chapters'")
    source_value: str = Field(..., description="Page range like '1-5,10' or chapter titles")
    num_questions: int = Field(10, ge=1, le=50)
    difficulty: str = Field("medium", description="easy, medium, or hard")
    provider: str = Field(QUIZ_LLM_PROVIDER, description="'openai' or 'anthropic'")


class QuizScoreRequest(BaseModel):
    score: int = Field(..., ge=0)
    total_attempted: int = Field(..., ge=1)


@app.get("/documents/{doc_id}/chapters")
def get_chapters_endpoint(doc_id: int):
    """Get all detected chapters for a document."""
    conn = get_connection()
    chapters = get_document_chapters(conn, doc_id)
    page_count = get_document_page_count(conn, doc_id)
    conn.close()
    return {"document_id": doc_id, "total_pages": page_count, "chapters": chapters}


@app.post("/quiz/generate")
def generate_quiz_endpoint(req: QuizGenerateRequest):
    """Generate a quiz from specific pages or chapters."""
    try:
        quiz = generate_quiz(
            doc_id=req.document_id,
            source_type=req.source_type,
            source_value=req.source_value,
            num_questions=req.num_questions,
            difficulty=req.difficulty,
            provider=req.provider,
        )
        return {
            "quiz_id": quiz.db_id,
            "title": quiz.title,
            "num_questions": quiz.num_questions,
            "difficulty": quiz.difficulty,
            "questions": [
                {
                    "question_number": q.question_number,
                    "question_type": q.question_type,
                    "question": q.question,
                    "options": q.options,
                    "correct_answer": q.correct_answer,
                    "explanation": q.explanation,
                    "difficulty": q.difficulty,
                    "source_page": q.source_page,
                }
                for q in quiz.questions
            ],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {str(e)}")


@app.get("/quiz/{quiz_id}")
def get_quiz_endpoint(quiz_id: int):
    """Retrieve a previously generated quiz."""
    quiz = get_quiz(quiz_id)
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    return quiz


@app.get("/quizzes")
def list_quizzes_endpoint(doc_id: int = Query(None)):
    """List all quizzes, optionally filtered by document."""
    return list_quizzes(doc_id=doc_id)


@app.post("/quiz/{quiz_id}/score")
def score_quiz_endpoint(quiz_id: int, req: QuizScoreRequest):
    """Save the user's quiz score."""
    quiz = get_quiz(quiz_id)
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    save_quiz_score(quiz_id, req.score, req.total_attempted)
    return {
        "quiz_id": quiz_id,
        "score": req.score,
        "total_attempted": req.total_attempted,
        "percentage": round(req.score / req.total_attempted * 100, 1),
    }


@app.delete("/quiz/{quiz_id}")
def delete_quiz_endpoint(quiz_id: int):
    """Delete a quiz."""
    quiz = get_quiz(quiz_id)
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    delete_quiz(quiz_id)
    return {"message": f"Quiz {quiz_id} deleted"}

from src.analytics import compute_dashboard, generate_llm_feedback, identify_weak_topics


@app.get("/dashboard")
def dashboard_endpoint(doc_id: int = Query(None)):
    """Get the full student performance dashboard."""
    dash = compute_dashboard(doc_id=doc_id)
    return {
        "overall": {
            "total_quizzes": dash.overall.total_quizzes,
            "total_questions": dash.overall.total_questions,
            "total_correct": dash.overall.total_correct,
            "overall_accuracy": dash.overall.overall_accuracy,
            "avg_score_per_quiz": dash.overall.avg_score_per_quiz,
            "best_topic": dash.overall.best_topic,
            "worst_topic": dash.overall.worst_topic,
            "quizzes_over_time": dash.overall.quizzes_over_time,
        },
        "topic_performance": [
            {
                "topic": tp.topic,
                "total_questions": tp.total_questions,
                "correct": tp.correct,
                "incorrect": tp.incorrect,
                "accuracy": tp.accuracy,
                "difficulty_breakdown": tp.difficulty_breakdown,
            }
            for tp in dash.topic_performance
        ],
        "question_type_stats": dash.question_type_stats,
        "difficulty_stats": dash.difficulty_stats,
        "weak_topics": dash.weak_topics,
        "improvement_suggestions": dash.improvement_suggestions,
    }


class FeedbackRequest(BaseModel):
    doc_id: int | None = None
    provider: str = Field("openai", description="'openai' or 'anthropic'")


@app.post("/dashboard/feedback")
def feedback_endpoint(req: FeedbackRequest):
    """Generate LLM-powered personalised study feedback."""
    dash = compute_dashboard(doc_id=req.doc_id)
    feedback = generate_llm_feedback(
        dash.weak_topics,
        dash.overall.overall_accuracy,
        provider=req.provider,
    )
    return {"feedback": feedback, "weak_topics": dash.weak_topics}
