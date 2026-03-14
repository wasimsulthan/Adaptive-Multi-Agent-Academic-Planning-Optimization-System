from __future__ import annotations

import json
from dataclasses import dataclass, field
from collections import defaultdict

from src.database import get_connection


@dataclass
class TopicPerformance:
    topic: str
    total_questions: int = 0
    correct: int = 0
    incorrect: int = 0
    accuracy: float = 0.0
    difficulty_breakdown: dict = field(default_factory=dict)
    weak_subtopics: list[str] = field(default_factory=list)


@dataclass
class OverallStats:
    total_quizzes: int = 0
    total_questions: int = 0
    total_correct: int = 0
    overall_accuracy: float = 0.0
    avg_score_per_quiz: float = 0.0
    best_topic: str = ""
    worst_topic: str = ""
    quizzes_over_time: list[dict] = field(default_factory=list)


@dataclass
class StudentDashboard:
    overall: OverallStats
    topic_performance: list[TopicPerformance]
    question_type_stats: dict
    difficulty_stats: dict
    weak_topics: list[dict]
    improvement_suggestions: list[str]



def get_all_completed_quizzes(doc_id: int | None = None) -> list[dict]:
    """Get all quizzes that have been scored."""
    conn = get_connection()
    with conn.cursor() as cur:
        if doc_id:
            cur.execute(
                "SELECT q.id, q.document_id, d.filename, q.title, q.source_type, "
                "q.source_value, q.num_questions, q.difficulty, q.quiz_data, "
                "q.score, q.total_attempted, q.created_at "
                "FROM quizzes q JOIN documents d ON q.document_id = d.id "
                "WHERE q.score IS NOT NULL AND q.document_id = %s "
                "ORDER BY q.created_at DESC",
                (doc_id,),
            )
        else:
            cur.execute(
                "SELECT q.id, q.document_id, d.filename, q.title, q.source_type, "
                "q.source_value, q.num_questions, q.difficulty, q.quiz_data, "
                "q.score, q.total_attempted, q.created_at "
                "FROM quizzes q JOIN documents d ON q.document_id = d.id "
                "WHERE q.score IS NOT NULL "
                "ORDER BY q.created_at DESC"
            )
        rows = cur.fetchall()
    conn.close()

    return [
        {
            "id": r[0], "document_id": r[1], "filename": r[2],
            "title": r[3], "source_type": r[4], "source_value": r[5],
            "num_questions": r[6], "difficulty": r[7], "quiz_data": r[8],
            "score": r[9], "total_attempted": r[10],
            "created_at": str(r[11])[:19],
        }
        for r in rows
    ]


def get_quiz_with_answers(quiz_id: int) -> dict | None:
    """Get a quiz with its stored user answers (from quiz_data)."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, document_id, title, source_type, source_value, "
            "num_questions, difficulty, quiz_data, score, total_attempted, created_at "
            "FROM quizzes WHERE id = %s",
            (quiz_id,),
        )
        row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0], "document_id": row[1], "title": row[2],
        "source_type": row[3], "source_value": row[4],
        "num_questions": row[5], "difficulty": row[6],
        "quiz_data": row[7], "score": row[8],
        "total_attempted": row[9], "created_at": str(row[10])[:19],
    }



def compute_topic_performance(quizzes: list[dict]) -> list[TopicPerformance]:
    """
    Compute per-topic (chapter/source) performance from completed quizzes.
    Groups by source_value (chapter name or page range).
    """
    topic_data: dict[str, TopicPerformance] = {}

    for quiz in quizzes:
        topic = quiz.get("source_value", "Unknown")
        if topic not in topic_data:
            topic_data[topic] = TopicPerformance(topic=topic)

        tp = topic_data[topic]
        score = quiz.get("score", 0) or 0
        total = quiz.get("total_attempted", 0) or quiz.get("num_questions", 0)

        tp.total_questions += total
        tp.correct += score
        tp.incorrect += (total - score)

        # Track difficulty
        diff = quiz.get("difficulty", "medium")
        if diff not in tp.difficulty_breakdown:
            tp.difficulty_breakdown[diff] = {"total": 0, "correct": 0}
        tp.difficulty_breakdown[diff]["total"] += total
        tp.difficulty_breakdown[diff]["correct"] += score

    # Compute accuracy
    for tp in topic_data.values():
        if tp.total_questions > 0:
            tp.accuracy = round(tp.correct / tp.total_questions * 100, 1)

    return sorted(topic_data.values(), key=lambda t: t.accuracy)


def compute_question_type_stats(quizzes: list[dict]) -> dict:
    """Compute performance by question type (MCQ, T/F, fill-blank)."""
    stats: dict[str, dict] = {
        "mcq": {"total": 0, "correct": 0},
        "true_false": {"total": 0, "correct": 0},
        "fill_blank": {"total": 0, "correct": 0},
    }

    for quiz in quizzes:
        quiz_data = quiz.get("quiz_data", {})
        questions = quiz_data.get("questions", [])
        user_answers = quiz_data.get("user_answers", {})

        for q in questions:
            qtype = q.get("question_type", "mcq")
            if qtype not in stats:
                stats[qtype] = {"total": 0, "correct": 0}
            stats[qtype]["total"] += 1

            # If user_answers stored, check correctness
            qnum = str(q.get("question_number", ""))
            if qnum in user_answers:
                user_ans = user_answers[qnum]
                correct_ans = q.get("correct_answer", "")
                if qtype == "fill_blank":
                    if user_ans.lower().strip() == correct_ans.lower().strip():
                        stats[qtype]["correct"] += 1
                else:
                    if user_ans == correct_ans:
                        stats[qtype]["correct"] += 1


    for qtype in stats:
        total = stats[qtype]["total"]
        stats[qtype]["accuracy"] = round(
            stats[qtype]["correct"] / total * 100, 1
        ) if total > 0 else 0.0

    return stats


def compute_difficulty_stats(quizzes: list[dict]) -> dict:
    """Compute performance by difficulty level."""
    stats: dict[str, dict] = {
        "easy": {"total": 0, "correct": 0, "quizzes": 0},
        "medium": {"total": 0, "correct": 0, "quizzes": 0},
        "hard": {"total": 0, "correct": 0, "quizzes": 0},
    }

    for quiz in quizzes:
        diff = quiz.get("difficulty", "medium")
        if diff not in stats:
            stats[diff] = {"total": 0, "correct": 0, "quizzes": 0}

        score = quiz.get("score", 0) or 0
        total = quiz.get("total_attempted", 0) or quiz.get("num_questions", 0)

        stats[diff]["total"] += total
        stats[diff]["correct"] += score
        stats[diff]["quizzes"] += 1

    for diff in stats:
        total = stats[diff]["total"]
        stats[diff]["accuracy"] = round(
            stats[diff]["correct"] / total * 100, 1
        ) if total > 0 else 0.0

    return stats


def identify_weak_topics(topic_performance: list[TopicPerformance], threshold: float = 70.0) -> list[dict]:
    weak = []
    for tp in topic_performance:
        if tp.total_questions >= 1 and tp.accuracy < threshold:
            weak.append({
                "topic": tp.topic,
                "accuracy": tp.accuracy,
                "total_questions": tp.total_questions,
                "incorrect": tp.incorrect,
                "gap": round(threshold - tp.accuracy, 1),
            })
    return sorted(weak, key=lambda w: w["accuracy"])


def generate_improvement_suggestions(
    weak_topics: list[dict],
    topic_performance: list[TopicPerformance],
    difficulty_stats: dict,
) -> list[str]:
    suggestions = []

    if not weak_topics and not topic_performance:
        return ["No quiz data available yet. Take some quizzes to see personalised feedback."]


    for wt in weak_topics[:5]:
        suggestions.append(
            f"Review \"{wt['topic']}\" — you scored {wt['accuracy']}% "
            f"({wt['incorrect']} incorrect out of {wt['total_questions']}). "
            f"Focus on re-reading this section and retake the quiz."
        )

    easy = difficulty_stats.get("easy", {})
    hard = difficulty_stats.get("hard", {})

    if easy.get("accuracy", 100) < 80 and easy.get("total", 0) > 0:
        suggestions.append(
            "You're struggling with basic-level questions. Consider re-reading "
            "the fundamentals before attempting harder material."
        )

    if hard.get("accuracy", 0) > 80 and hard.get("total", 0) >= 5:
        suggestions.append(
            "Great performance on hard questions! You might be ready "
            "to explore advanced topics or help others with the material."
        )


    if topic_performance:
        best = max(topic_performance, key=lambda t: t.accuracy)
        worst = min(topic_performance, key=lambda t: t.accuracy)
        if best.topic != worst.topic and worst.accuracy < 70:
            suggestions.append(
                f"Your strongest area is \"{best.topic}\" ({best.accuracy}%). "
                f"Apply similar study techniques to \"{worst.topic}\" ({worst.accuracy}%)."
            )

    if not suggestions:
        suggestions.append("You're performing well across all topics. Keep it up!")

    return suggestions



def generate_llm_feedback(
    weak_topics: list[dict],
    overall_accuracy: float,
    provider: str = "openai",
) -> str:

    if not weak_topics:
        return "Excellent work! You're performing well across all topics. Keep practising to maintain your knowledge."

    topic_summary = "\n".join(
        f"- {wt['topic']}: {wt['accuracy']}% accuracy ({wt['incorrect']} wrong out of {wt['total_questions']})"
        for wt in weak_topics
    )

    prompt = (
        f"A student has an overall quiz accuracy of {overall_accuracy}%. "
        f"They are struggling with these topics:\n{topic_summary}\n\n"
        f"Provide 3-5 specific, encouraging study recommendations. "
        f"Be concrete about what to focus on and suggest study strategies. "
        f"Keep it under 200 words."
    )

    try:
        if provider == "anthropic":
            from src.config import ANTHROPIC_API_KEY, QUIZ_ANTHROPIC_MODEL
            if ANTHROPIC_API_KEY:
                import anthropic
                client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                resp = client.messages.create(
                    model=QUIZ_ANTHROPIC_MODEL, max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text.strip()
        else:
            from src.config import OPENAI_API_KEY, QUIZ_OPENAI_MODEL
            if OPENAI_API_KEY:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                resp = client.chat.completions.create(
                    model=QUIZ_OPENAI_MODEL, temperature=0.5,
                    messages=[
                        {"role": "system", "content": "You are a supportive tutor giving study advice."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=512,
                )
                return resp.choices[0].message.content.strip()
    except Exception:
        pass

    return "\n".join(generate_improvement_suggestions(
        weak_topics,
        [],
        {},
    ))




def compute_dashboard(doc_id: int | None = None) -> StudentDashboard:
    quizzes = get_all_completed_quizzes(doc_id=doc_id)

    # Overall stats
    overall = OverallStats()
    overall.total_quizzes = len(quizzes)
    overall.total_questions = sum(q.get("total_attempted", 0) or q.get("num_questions", 0) for q in quizzes)
    overall.total_correct = sum(q.get("score", 0) or 0 for q in quizzes)
    overall.overall_accuracy = round(
        overall.total_correct / overall.total_questions * 100, 1
    ) if overall.total_questions > 0 else 0.0
    overall.avg_score_per_quiz = round(
        overall.total_correct / overall.total_quizzes, 1
    ) if overall.total_quizzes > 0 else 0.0

    overall.quizzes_over_time = [
        {
            "date": q["created_at"],
            "title": q["title"],
            "score": q.get("score", 0),
            "total": q.get("total_attempted", 0) or q.get("num_questions", 0),
            "accuracy": round(
                (q.get("score", 0) or 0) / max(1, q.get("total_attempted", 0) or q.get("num_questions", 1)) * 100, 1
            ),
            "difficulty": q.get("difficulty", "medium"),
        }
        for q in quizzes
    ]

    topic_performance = compute_topic_performance(quizzes)

    if topic_performance:
        best = max(topic_performance, key=lambda t: t.accuracy)
        worst = min(topic_performance, key=lambda t: t.accuracy)
        overall.best_topic = best.topic
        overall.worst_topic = worst.topic

    question_type_stats = compute_question_type_stats(quizzes)
    difficulty_stats = compute_difficulty_stats(quizzes)


    weak_topics = identify_weak_topics(topic_performance)

    suggestions = generate_improvement_suggestions(
        weak_topics, topic_performance, difficulty_stats
    )

    return StudentDashboard(
        overall=overall,
        topic_performance=topic_performance,
        question_type_stats=question_type_stats,
        difficulty_stats=difficulty_stats,
        weak_topics=weak_topics,
        improvement_suggestions=suggestions,
    )
