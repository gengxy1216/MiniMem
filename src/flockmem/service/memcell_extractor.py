from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MemCell:
    order: int
    content: str


class RuleMemCellExtractor:
    """Rule-based hard segmentation for long episodes."""

    def __init__(
        self,
        *,
        max_chars_per_cell: int = 900,
        min_chars_per_cell: int = 120,
        max_cells: int = 16,
    ) -> None:
        self.max_chars_per_cell = max(80, int(max_chars_per_cell))
        self.min_chars_per_cell = max(40, int(min_chars_per_cell))
        self.max_cells = max(1, int(max_cells))

    def split(self, text: str) -> list[MemCell]:
        raw = " ".join(str(text or "").replace("\r\n", "\n").split())
        if not raw:
            return []
        parts = self._split_parts(raw)
        if not parts:
            return [MemCell(order=1, content=raw[: self.max_chars_per_cell])]
        cells: list[str] = []
        buf = ""
        for part in parts:
            candidate = part if not buf else f"{buf} {part}"
            if len(candidate) <= self.max_chars_per_cell:
                buf = candidate
                continue
            if buf:
                cells.append(buf.strip())
            if len(part) <= self.max_chars_per_cell:
                buf = part
                continue
            for i in range(0, len(part), self.max_chars_per_cell):
                cells.append(part[i : i + self.max_chars_per_cell].strip())
                if len(cells) >= self.max_cells:
                    return self._to_memcells(cells[: self.max_cells])
            buf = ""
        if buf.strip():
            cells.append(buf.strip())
        merged = self._merge_short_cells(cells)
        if len(merged) > self.max_cells:
            merged = merged[: self.max_cells]
        return self._to_memcells(merged)

    def _split_parts(self, text: str) -> list[str]:
        by_block = [x.strip() for x in re.split(r"\n{2,}", text) if x.strip()]
        parts: list[str] = []
        for block in by_block:
            segs = [
                s.strip()
                for s in re.split(r"[。！？!?；;]|(?<!\d)\.(?!\d)", block)
                if s.strip()
            ]
            if not segs:
                segs = [block.strip()]
            parts.extend(segs)
        return [p for p in parts if p]

    def _merge_short_cells(self, cells: list[str]) -> list[str]:
        if not cells:
            return []
        merged: list[str] = []
        for cell in cells:
            cur = str(cell or "").strip()
            if not cur:
                continue
            if not merged:
                merged.append(cur)
                continue
            prev = merged[-1]
            if len(prev) < self.min_chars_per_cell and len(prev) + len(cur) + 1 <= self.max_chars_per_cell:
                merged[-1] = f"{prev} {cur}".strip()
            else:
                merged.append(cur)
        return merged

    @staticmethod
    def _to_memcells(cells: list[str]) -> list[MemCell]:
        out: list[MemCell] = []
        for idx, cell in enumerate(cells, start=1):
            text = str(cell or "").strip()
            if not text:
                continue
            out.append(MemCell(order=idx, content=text))
        return out
