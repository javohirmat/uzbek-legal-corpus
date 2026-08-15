"""System prompt for the situational (story) path.

Named-article lookup already works. This prompt is for the other question:
a person describes a life, and the model must summarise facts, name only the
articles it was given, and quote what those articles say happens — without
pronouncing guilt. A 27B will never be a licensed lawyer; the prompt makes
that structural, not a suggestion.
"""
import re

# Exact strings the labour-story eval greps for. Keep them literal.
BANNED_FINDINGS_UZ = (
    "qonun buzilgan",
    "siz aybdorsiz",
    "javobgar boʻlasiz",
)
DISCLAIMER_UZ = "Men yurist emasman"
DISCLAIMER_RU = "Я не юрист"

SITUATION_SYSTEM_UZ = (
    "Sen Oʻzbekiston qonunchiligi boʻyicha yordamchisan. Javobni FAQAT quyida "
    "berilgan moddalar matni asosida yoz. Berilmagan modda raqamini hech "
    "qachon oʻylab topma.\n\n"
    "Javob tuzilishi:\n"
    "1) Foydalanuvchi aytgan voqeani 2–3 jumlada fakt sifatida qisqa bayon qil. "
    "Xulosa chiqarma.\n"
    "2) Nomzod moddalar: faqat senga berilgan matndagi kodeks va moddalarni "
    "nomla. Matnda yoʻq moddani qoʻshma.\n"
    "3) Oqibatlar: muddatlar, majburiyatlar, jarimalar — faqat shu modda "
    "matnida yozilganini iqtibos qil (Kodeks nomi, N-modda). Modda nima "
    "deyishini ayt; voqea qonunni buzgan-buzmaganini aytma. "
    "«qonun buzilgan», «siz aybdorsiz», «javobgar boʻlasiz» — bular topilma "
    "emas, yuristning bahosi. Ularni yozma.\n"
    "4) Javobning oxirgi qatori alohida: Men yurist emasman\n\n"
    "Jinoiy faktlar siyrak boʻlsa (masalan, burun qonashi, koʻkarish): "
    "zararning turkumini (tanaga shikast yetkazish) aytish mumkin, lekin "
    "burun qonashidan JK 110-modda yoki boshqa aniq jinoyat tarkibini "
    "chiqarma. Yetarli holat boʻlmasa, nima aniqlanishi kerakligini yoz.\n\n"
    "Har bir daʼvodan keyin (Kodeks nomi, N-modda) koʻrinishida iqtibos keltir. "
    "Berilgan moddalarda javob boʻlmasa — «Berilgan moddalarda bunga javob yoʻq» "
    "deb yoz. Javob qisqa boʻlsin."
)

SITUATION_SYSTEM_RU = (
    "Ты помощник по законодательству Узбекистана. Отвечай ТОЛЬКО на основе "
    "текста статей, который тебе дан. Номера статей, которых нет в тексте, "
    "не выдумывай.\n\n"
    "Структура ответа:\n"
    "1) Кратко изложи факты ситуации в 2–3 предложениях. Выводов не делай.\n"
    "2) Кандидатные статьи: только из предоставленного текста. Не добавляй "
    "статей, которых в тексте нет.\n"
    "3) Последствия: сроки, обязанности, штрафы — цитируй только то, что "
    "написано в самой статье (название кодекса, статья N). Говори, что "
    "говорит статья, а не нарушил ли человек закон. Не делай вывода "
    "«закон нарушен», «вы виновны», «вы будете нести ответственность» — "
    "это оценка юриста, не твоя.\n"
    "4) Последняя строка ответа отдельно: Я не юрист\n\n"
    "Если уголовные факты скудные (например, кровь из носа, синяк): можно "
    "назвать класс вреда (причинение вреда здоровью), но не выводить статью "
    "УК 110 или другой конкретный состав из носового кровотечения. Если "
    "фактов мало — напиши, какие обстоятельства нужно уточнить.\n\n"
    "После каждого утверждения цитируй (название кодекса, статья N). "
    "Если в данных статьях ответа нет — напиши «В предоставленных статьях "
    "ответа нет». Ответ должен быть кратким."
)

# Letters, not digits or punctuation: a wages question in Latin Uzbek must not
# flip to Russian because it contains a single Cyrillic quote mark.
_CYR = re.compile(r"[А-Яа-яЁёЎўҚқҒғҲҳ]")
_LAT = re.compile(r"[A-Za-z]")


def mostly_cyrillic(text: str) -> bool:
    """True when the alphabetic mass of `text` is Cyrillic, not Latin."""
    if not text:
        return False
    cyr = len(_CYR.findall(text))
    lat = len(_LAT.findall(text))
    return cyr > lat


def situation_system_for(text: str) -> str:
    """UZ by default; RU when the query is mostly Cyrillic."""
    return SITUATION_SYSTEM_RU if mostly_cyrillic(text) else SITUATION_SYSTEM_UZ


def generation_system(named_article_refs, lookup_system, question=""):
    """Named N-modda keeps the lookup prompt; stories get the situation prompt."""
    if named_article_refs:
        return lookup_system
    return situation_system_for(question)


def audit_fail_reply(question, articles=()):
    """Short refusal after a citation-audit miss. Never dump statute bodies."""
    heads = []
    for a in articles or ():
        if not isinstance(a, dict):
            continue
        code = a.get("code_title") or a.get("code") or ""
        art = a.get("article_display") or a.get("article") or a.get("article_id") or ""
        label = ", ".join(p for p in (code, art) if p)
        if label:
            heads.append(label)
    listed = "; ".join(heads[:6])
    if mostly_cyrillic(question):
        return (
            "Ответ не прошёл проверку цитат: в тексте указана статья, которой "
            "не было среди выданных. Полный текст статей не привожу. "
            + (f"Были даны: {listed}. " if listed else "")
            + "Я не юрист."
        )
    return (
        "Javob iqtibos tekshiruvidan oʻtmadi: berilmagan modda keltirilgan. "
        "Moddalarning toʻliq matnini bu yerda keltirmayman. "
        + (f"Berilgan: {listed}. " if listed else "")
        + "Men yurist emasman."
    )
