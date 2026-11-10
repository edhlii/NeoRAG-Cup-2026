"""Khôi phục dấu bằng n-gram corpus và mở rộng viết tắt ngôn ngữ, không chứa QA."""

import re
import unicodedata
from collections import Counter, defaultdict


def folded(text: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
        if not unicodedata.combining(c)
    )


class QueryNormalizer:
    def __init__(self, texts):
        forms = defaultdict(Counter)
        for text in texts:
            words = re.findall(r"\w+", text.lower())
            for n in range(1, 5):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i : i + n])
                    forms[folded(phrase)][phrase] += 1
        self.forms = {key: counts.most_common(1)[0][0] for key, counts in forms.items()}

    def normalize(self, query: str) -> str:
        query = unicodedata.normalize("NFC", query)
        abbreviations = {
            "dc": "được",
            "đc": "được",
            "ko": "không",
            "khong": "không",
            "k": "không",
            "ntn": "như thế nào",
            "bn": "bao nhiêu",
            "sv": "sinh viên",
            "cntt": "công nghệ thông tin",
            "bcn": "ban chủ nhiệm",
            "clb": "câu lạc bộ",
        }
        words = re.findall(r"\w+|[^\w\s]", query.lower())
        words = [abbreviations.get(word, word) for word in words]
        words = " ".join(words).split()
        result, i = [], 0
        while i < len(words):
            for n in range(min(4, len(words) - i), 0, -1):
                phrase = " ".join(words[i : i + n])
                # Giữ nguyên các từ đã có dấu; chỉ khôi phục những cụm không dấu.
                if phrase == folded(phrase) and phrase in self.forms:
                    result.append(self.forms[phrase])
                    i += n
                    break
            else:
                result.append(words[i])
                i += 1
        return " ".join(result)
