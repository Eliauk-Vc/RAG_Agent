"""Convert generated chat answers to plain text for the web interface."""

import re


def plain_answer(answer: str) -> str:
    """Remove presentation markup and trailing sources, including cached answers."""
    lines = []
    for line in answer.splitlines():
        heading = re.sub(r"^\s{0,3}#{1,6}\s*", "", line).strip()
        heading = heading.strip("*_ ").rstrip(":：").strip("*_ ")
        if re.fullmatch(
            r"references|sources|参考文献|参考资料|参考来源|引用来源|参考|来源",
            heading,
            flags=re.IGNORECASE,
        ):
            break
        if re.fullmatch(r"\s*(?:[-*_]\s*){3,}", line):
            continue
        if re.match(r"^\s*(```|~~~)", line):
            continue
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*>\s?", "", line)
        line = re.sub(r"^\s*[-*+•]\s+", "• ", line)
        line = re.sub(r"^\s*\d+[.)]\s+", "• ", line)
        line = re.sub(r"^\s*(?:\d+|[一二三四五六七八九十]+)[、．]\s*", "• ", line)
        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"!?\[([^\]\n]+)\]\([^\n)]*\)", r"\1", text)
    text = re.sub(r"\[\d+(?:\s*[,，;；\-–]\s*\d+)*\]", "", text)
    for marker in ("**", "__", "~~", "*"):
        escaped = re.escape(marker)
        text = re.sub(escaped + r"(\S(?:[^\n]*?\S)?)" + escaped, r"\1", text)
    text = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", text)
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()
