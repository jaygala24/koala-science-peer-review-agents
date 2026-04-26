from __future__ import annotations

import io
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import replace
from pathlib import Path
from urllib.request import urlopen

from .models import Paper


class PaperReader:
    def __init__(self, koala_api_base: str) -> None:
        self.storage_base = koala_api_base.removesuffix("/api/v1").rstrip("/")

    def enrich(self, paper: Paper, *, max_chars: int = 20000) -> Paper:
        chunks: list[str] = []
        if paper.abstract:
            chunks.append(f"Abstract:\n{paper.abstract}")
        source_text = self.read_tarball(paper)
        if source_text:
            chunks.append(f"Extracted source text:\n{source_text[:max_chars]}")
        elif paper.pdf_url:
            pdf_text = self.read_pdf(paper)
            if pdf_text:
                chunks.append(f"Extracted PDF text:\n{pdf_text[:max_chars]}")
        if not chunks:
            return paper
        return replace(paper, abstract="\n\n".join(chunks)[:max_chars])

    def read_tarball(self, paper: Paper) -> str:
        if not paper.tarball_url:
            return ""
        raw = self.fetch_bytes(self.resolve_url(paper.tarball_url), max_bytes=8_000_000)
        if not raw:
            return ""
        try:
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as archive:
                texts: list[str] = []
                for member in archive.getmembers():
                    name = member.name.lower()
                    if not member.isfile() or not name.endswith((".tex", ".bbl", ".bib", ".txt", ".md")):
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    content = extracted.read(1_000_000).decode("utf-8", errors="ignore")
                    texts.append(strip_latex(content))
                    if sum(len(text) for text in texts) > 30_000:
                        break
                return "\n\n".join(texts)
        except tarfile.TarError:
            return ""

    def read_pdf(self, paper: Paper) -> str:
        if not paper.pdf_url or shutil.which("pdftotext") is None:
            return ""
        raw = self.fetch_bytes(self.resolve_url(paper.pdf_url), max_bytes=20_000_000)
        if not raw:
            return ""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "paper.pdf"
            path.write_bytes(raw)
            try:
                result = subprocess.run(
                    ["pdftotext", str(path), "-"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=45,
                )
            except (OSError, subprocess.TimeoutExpired):
                return ""
        return result.stdout if result.returncode == 0 else ""

    def fetch_bytes(self, url: str, *, max_bytes: int) -> bytes:
        try:
            with urlopen(url, timeout=45) as response:  # noqa: S310 - Koala storage URLs.
                return response.read(max_bytes)
        except OSError:
            return b""

    def resolve_url(self, url: str) -> str:
        if url.startswith("http://") or url.startswith("https://"):
            return url
        return f"{self.storage_base}{url if url.startswith('/') else '/' + url}"


def strip_latex(text: str) -> str:
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\\(cite|ref|label|url|href)(\[[^\]]*\])?\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
