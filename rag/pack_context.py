"""Lost-in-the-middle packing for the articles handed to the model.

Ranked retrieval puts the best hit first. Transformers then under-attend the
middle of a long context (Liu et al., 2023). Reorder: best first, second-best
last, the rest in between — and restate the user's story *after* the quotes,
so the query is not buried under six statutes.

Schema-flexible: corpus rows (`code_title`, `article_display`, `text`,
`title`) and thinner retriever dicts (`code`, `article`, `text`) both work.
"""
import config as C

_CODE_KEYS = ("code_title", "code", "slug", "law")
_ART_KEYS = ("article_display", "article", "article_id", "modda")
_TEXT_KEYS = ("text", "body", "content", "quote")
_TITLE_KEYS = ("title", "heading")


def pack_articles(hits: list) -> list:
    """Best first, second-best last. Models under-attend the middle of a list."""
    arts = list(hits or [])
    if len(arts) < 2:
        return arts
    return [arts[0], *arts[2:], arts[1]]


def _pick(hit, keys):
    if not isinstance(hit, dict):
        return ""
    for key in keys:
        value = hit.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _format_hit(hit, max_chars):
    code = _pick(hit, _CODE_KEYS)
    art = _pick(hit, _ART_KEYS)
    title = _pick(hit, _TITLE_KEYS)
    body = _pick(hit, _TEXT_KEYS)
    if max_chars and body:
        body = body[:max_chars]
    head = " | ".join(p for p in (code, art, title) if p) or "modda"
    return f"[{head}]\n{body}"


def format_grounding(hits, user_text: str) -> str:
    """Statutes first (packed); restated user story after the quotes."""
    packed = pack_articles(hits)
    blocks = [_format_hit(h, C.MAX_ARTICLE_CHARS) for h in packed if isinstance(h, dict)]
    ctx = "\n\n".join(blocks)
    story = user_text if user_text is not None else ""
    return f"MODDALAR:\n{ctx}\n\nFOYDALANUVCHI VAZIYATI:\n{story}"
