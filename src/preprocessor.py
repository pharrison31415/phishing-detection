import html
import re
import pandas as pd


def clean_text(s: str) -> str:
    if len(s) == 0:
        return ""
    s = html.unescape(str(s))
    s = re.sub(r"<[^>]+>", " ", s)  # remove HTML
    s = re.sub(r"https?://\S+|www\.\S+", " URL ", s)  # replace URLs
    s = re.sub(r"[^A-Za-z0-9@._\-'\s]", " ", s)  # keep common chars
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def extract_sender_email_domain(sender):
    if pd.isna(sender):
        return "", ""
    s = str(sender)
    m = EMAIL_RE.search(s)
    if m:
        return m.group(0).lower(), m.group(2).lower()
    return "", ""


def count_urls(s):
    if pd.isna(s):
        return 0
    return len(URL_RE.findall(str(s)))
