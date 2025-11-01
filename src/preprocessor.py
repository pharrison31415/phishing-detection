import html
import re


def clean_text(s: str) -> str:
    if len(s) == 0:
        return ""
    s = html.unescape(str(s))
    s = re.sub(r"<[^>]+>", " ", s)  # remove HTML
    s = re.sub(r"https?://\S+|www\.\S+", " URL ", s)  # replace URLs
    s = re.sub(r"[^A-Za-z0-9@._\-'\s]", " ", s)  # keep common chars
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s
