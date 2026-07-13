from __future__ import annotations


def render_explanation(prediction: str, concept_names: list[str], probability: float) -> str:
    """Sinh giai thich ngan, chi dua tren concept da kich hoat."""

    if not concept_names:
        return "Model khong tim thay concept noi bat de giai thich quyet dinh."

    joined = ", ".join(concept_names[:5])
    if prediction == "ai_generated":
        return f"Anh co kha nang do AI tao ra ({probability:.2%}) vi kich hoat cac dau hieu: {joined}."
    return f"Anh duoc du doan la anh that ({1.0 - probability:.2%}); cac dau hieu nghi van con lai gom: {joined}."
