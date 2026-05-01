"""Structural parsing and hierarchical chunk generation for document retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.state import DocumentUpload

SECTION_TYPES = {"section", "subsection", "appendix"}
ATOMIC_TYPES = {"table", "list", "appendix"}


@dataclass(slots=True)
class StructureNode:
    """Tree node describing a structural unit in a document."""

    node_id: str
    node_type: str
    title: str
    depth: int
    parent: StructureNode | None = None
    is_atomic: bool = False
    content_blocks: list[str] = field(default_factory=list)
    children: list[StructureNode] = field(default_factory=list)
    cross_refs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParentChunk:
    """Parent chunk returned to the LLM after child retrieval."""

    parent_id: str
    section_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class ChildChunk:
    """Child chunk indexed in the vector store."""

    chunk_id: str
    parent_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class HierarchicalDocument:
    """Structured representation of a parsed document."""

    title: str
    root: StructureNode
    parent_chunks: list[ParentChunk]
    child_chunks: list[ChildChunk]


class HierarchicalChunker:
    """Parse documents into parent and child retrieval chunks."""

    _SOURCE_LOCATION_LIMIT = 8
    _TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
    _MARKDOWN_IMAGE_LINE = re.compile(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$")
    _HTML_TAG_LINE = re.compile(r"^\s*</?[a-z][^>]*>\s*$", re.IGNORECASE)
    _PAGE_MARKER_LINE = re.compile(
        r"^(?:KPT\s*)?\d+(?:\.\d+)*\s*-\s*\d+$",
        re.IGNORECASE,
    )
    _HTML_TABLE_MARKER = re.compile(r"<\s*/?\s*(?:table|tr|td|th)\b", re.IGNORECASE)
    _HTML_ROW_SPLIT = re.compile(r"(?=<\s*tr\b)", re.IGNORECASE)
    _DOTTED_LEADER_SUFFIX = re.compile(r"\s*\.{2,}\s*\d+\s*$")
    _HEADING_PATTERNS: tuple[tuple[str, re.Pattern[str], int], ...] = (
        (
            "chapter",
            re.compile(r"^(?:#+\s*)?([0-9]+)\s+(.+)$"),
            1,
        ),
        (
            "section",
            re.compile(r"^(?:#+\s*)?([0-9]+\.[0-9]+)\s+(.+)$"),
            2,
        ),
        (
            "subsection",
            re.compile(r"^(?:#+\s*)?([0-9]+\.[0-9]+\.[0-9]+)\s+(.+)$"),
            3,
        ),
        (
            "appendix",
            re.compile(
                r"^(?:#+\s*)?(Lampiran|Appendix)\s+([A-Za-z0-9]+)\b[:.\-]?\s*(.*)$",
                re.IGNORECASE,
            ),
            1,
        ),
        (
            "chapter",
            re.compile(
                r"^(?:#+\s*)?(BAB|Chapter)\s+([IVXLCDM0-9]+)\b[:.\-]?\s*(.*)$",
                re.IGNORECASE,
            ),
            1,
        ),
        (
            "section",
            re.compile(
                r"^(?:#+\s*)?(Pasal|Article|Section)\s+([A-Za-z0-9.\-]+)\b[:.\-]?\s*(.*)$",
                re.IGNORECASE,
            ),
            2,
        ),
        (
            "subsection",
            re.compile(
                r"^(?:#+\s*)?(Bagian|Subbagian|Paragraf|Paragraph)\s+([A-Za-z0-9.\-]+)\b[:.\-]?\s*(.*)$",
                re.IGNORECASE,
            ),
            3,
        ),
        (
            "clause",
            re.compile(
                r"^(?:#+\s*)?(Ayat|Clause)\s*\(?([A-Za-z0-9]+)\)?\s*[:.\-]?\s*(.*)$",
                re.IGNORECASE,
            ),
            4,
        ),
    )
    _PASAL_REF = re.compile(r"\bPasal\s+([A-Za-z0-9.\-]+)", re.IGNORECASE)
    _AYAT_REF = re.compile(r"\bAyat\s*\(?([A-Za-z0-9]+)\)?", re.IGNORECASE)
    _BAB_REF = re.compile(r"\bBAB\s+([IVXLCDM0-9]+)", re.IGNORECASE)
    _APPENDIX_REF = re.compile(
        r"\b(?:Lampiran|Appendix)\s+([A-Za-z0-9]+)", re.IGNORECASE
    )

    def __init__(
        self,
        parent_max_chars: int = 5000,
        child_max_chars: int = 1200,
        child_overlap_chars: int = 120,
    ):
        self.parent_max_chars = parent_max_chars
        self.child_max_chars = child_max_chars
        self.child_overlap_chars = child_overlap_chars

    def chunk_document(self, document: DocumentUpload) -> HierarchicalDocument:
        """Parse a document and emit parent and child chunks."""
        if not document.extracted_text:
            raise ValueError("Document has no extracted text")

        normalized_text = self._normalize_ocr_text(document.extracted_text)
        title = self._resolve_title(document)
        root = self._parse_tree(normalized_text, title)
        parents = self._select_parent_nodes(root)

        if not parents:
            parents = [root]

        parent_chunks: list[ParentChunk] = []
        child_chunks: list[ChildChunk] = []

        for parent in parents:
            parent_id = self._build_internal_id(document.document_id, parent.node_id)
            parent_metadata = self._build_metadata(
                document=document,
                doc_title=title,
                section_node=parent,
                parent_id=parent_id,
            )
            parent_text = self._truncate_parent_text(self._render_node(parent))
            parent_metadata = self._with_source_locations(
                parent_metadata,
                document.layout_details,
                parent_text,
            )
            parent_chunks.append(
                ParentChunk(
                    parent_id=parent_id,
                    section_id=parent.node_id,
                    text=parent_text,
                    metadata=parent_metadata,
                )
            )

            section_child_chunks = self._build_child_chunks_for_parent(
                document=document,
                doc_title=title,
                parent=parent,
                parent_id=parent_id,
            )
            if not section_child_chunks:
                fallback_text = self._render_node(parent)
                for part_index, part in enumerate(
                    self._split_text(fallback_text, self.child_max_chars),
                    start=1,
                ):
                    metadata = self._build_metadata(
                        document=document,
                        doc_title=title,
                        section_node=parent,
                        parent_id=parent_id,
                    )
                    metadata["chunk_type"] = "paragraph"
                    metadata["is_atomic"] = False
                    child_chunks.append(
                        ChildChunk(
                            chunk_id=f"{parent.node_id}.part_{part_index}",
                            parent_id=parent_id,
                            text=part,
                            metadata=self._with_source_locations(
                                metadata,
                                document.layout_details,
                                part,
                            ),
                        )
                    )
            else:
                child_chunks.extend(section_child_chunks)

        return HierarchicalDocument(
            title=title,
            root=root,
            parent_chunks=parent_chunks,
            child_chunks=child_chunks,
        )

    def _resolve_title(self, document: DocumentUpload) -> str:
        """Infer a stable document title from OCR text or filename."""
        stem = Path(document.filename).stem.replace("_", " ").strip()
        if self._is_title_candidate(document.document_title):
            return self._clean_heading_line(document.document_title or "")

        for line in self._normalize_ocr_text(document.extracted_text or "").splitlines():
            candidate = self._clean_heading_line(line)
            if self._is_title_candidate(candidate):
                return candidate
        return stem or document.filename

    def _parse_tree(self, text: str, title: str) -> StructureNode:
        """Build a hierarchical node tree from raw text blocks."""
        root = StructureNode(node_id="root", node_type="document", title=title, depth=0)
        blocks = self._split_blocks(text)
        stack: list[StructureNode] = [root]
        counters: dict[str, int] = {}

        for block in blocks:
            heading = self._detect_heading(block)
            if heading:
                node_type, identifier, remainder, depth = heading
                while stack and stack[-1].depth >= depth:
                    stack.pop()
                parent = stack[-1] if stack else root
                title = self._clean_heading_line(
                    self._normalize_heading_candidate(block.splitlines()[0])
                )
                node = StructureNode(
                    node_id=self._compose_node_id(
                        parent.node_id,
                        node_type,
                        identifier,
                        counters,
                    ),
                    node_type=node_type,
                    title=title,
                    depth=depth,
                    parent=parent,
                    is_atomic=node_type in ATOMIC_TYPES,
                )
                parent.children.append(node)
                stack.append(node)
                if remainder:
                    node.content_blocks.append(remainder)
                continue

            current = stack[-1]
            block_type = self._detect_atomic_block_type(block)
            if block_type:
                child = StructureNode(
                    node_id=self._compose_node_id(
                        current.node_id,
                        block_type,
                        str(len(current.children) + 1),
                        counters,
                    ),
                    node_type=block_type,
                    title=block.splitlines()[0].strip(),
                    depth=current.depth + 1,
                    parent=current,
                    is_atomic=True,
                    content_blocks=[block],
                )
                current.children.append(child)
            else:
                current.content_blocks.append(block)

        self._populate_cross_references(root)
        return root

    def _split_blocks(self, text: str) -> list[str]:
        """Split OCR output into paragraph-style blocks and isolate inline headings."""
        normalized = self._normalize_ocr_text(text).replace("\r\n", "\n").replace("\r", "\n")
        blocks: list[str] = []
        current_lines: list[str] = []

        def flush() -> None:
            block = "\n".join(current_lines).strip()
            if block:
                blocks.append(block)
            current_lines.clear()

        for raw_line in normalized.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                flush()
                continue
            if current_lines and self._detect_heading(stripped):
                flush()
            current_lines.append(raw_line)

        flush()
        return blocks

    def _detect_heading(self, block: str) -> tuple[str, str, str, int] | None:
        """Classify a block as a structural heading when it matches known markers."""
        block_lines = block.splitlines()
        first_line = self._normalize_heading_candidate(block_lines[0])
        for node_type, pattern, depth in self._HEADING_PATTERNS:
            match = pattern.match(first_line)
            if not match:
                continue
            if node_type in {"chapter", "section", "subsection"} and pattern.pattern.startswith("^(?:#+\\s*)?([0-9]"):
                identifier = self._slug_token(match.group(1)) or node_type
                inline_title = match.group(2).strip()
            else:
                identifier = self._slug_token(match.group(2)) or node_type
                inline_title = match.group(3).strip() if match.lastindex and match.lastindex >= 3 else ""
            remainder_parts: list[str] = []
            if len(block_lines) > 1:
                remainder_parts.append("\n".join(block_lines[1:]).strip())
            if inline_title:
                remainder_parts.insert(0, inline_title)
            remainder = "\n".join(part for part in remainder_parts if part).strip()
            return node_type, identifier, remainder, depth
        return None

    def _detect_atomic_block_type(self, block: str) -> str | None:
        """Mark tables and lists as do-not-split units."""
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            return None
        if (
            self._HTML_TABLE_MARKER.search(block)
            or any("|" in line for line in lines)
            or lines[0].lower().startswith("tabel")
        ):
            return "table"
        if all(
            re.match(r"^(\d+[.)]|[-*])\s+", line)
            for line in lines[: min(len(lines), 4)]
        ):
            return "list"
        return None

    def _populate_cross_references(self, root: StructureNode) -> None:
        """Extract cross-reference metadata for every node."""
        for node in self._walk(root):
            node.cross_refs = self._extract_cross_refs(
                "\n".join([node.title, *node.content_blocks]).strip()
            )

    def _extract_cross_refs(self, text: str) -> list[str]:
        """Normalize inline structural references for later resolution."""
        refs = {
            f"pasal_{self._slug_token(match)}" for match in self._PASAL_REF.findall(text)
        }
        refs.update(
            f"ayat_{self._slug_token(match)}" for match in self._AYAT_REF.findall(text)
        )
        refs.update(f"bab_{self._slug_token(match)}" for match in self._BAB_REF.findall(text))
        refs.update(
            f"lampiran_{self._slug_token(match)}"
            for match in self._APPENDIX_REF.findall(text)
        )
        return sorted(ref for ref in refs if ref)

    def _select_parent_nodes(self, root: StructureNode) -> list[StructureNode]:
        """Prefer section-level units as parent chunks, then fall back to chapters."""
        sections = [node for node in self._walk(root) if node.node_type in SECTION_TYPES]
        if sections:
            return sections
        chapters = [node for node in self._walk(root) if node.node_type == "chapter"]
        if chapters:
            return chapters
        return []

    def _build_child_chunks_for_parent(
        self,
        document: DocumentUpload,
        doc_title: str,
        parent: StructureNode,
        parent_id: str,
    ) -> list[ChildChunk]:
        """Create precise child chunks anchored to a parent chunk."""
        child_chunks: list[ChildChunk] = []

        for paragraph_index, block in enumerate(parent.content_blocks, start=1):
            metadata = self._build_metadata(
                document=document,
                doc_title=doc_title,
                section_node=parent,
                parent_id=parent_id,
            )
            metadata["chunk_type"] = "paragraph"
            metadata["is_atomic"] = False
            for part_index, part in enumerate(
                self._split_text(block, self.child_max_chars),
                start=1,
            ):
                suffix = f"p{paragraph_index}_{part_index}"
                child_chunks.append(
                    ChildChunk(
                        chunk_id=f"{parent.node_id}.{suffix}",
                        parent_id=parent_id,
                        text=part,
                        metadata=self._with_source_locations(
                            metadata,
                            document.layout_details,
                            part,
                        ),
                    )
                )

        for child_node in self._collect_child_nodes(parent):
            metadata = self._build_metadata(
                document=document,
                doc_title=doc_title,
                section_node=child_node,
                parent_id=parent_id,
            )
            metadata["chunk_type"] = child_node.node_type
            metadata["is_atomic"] = child_node.is_atomic
            node_text = self._render_node(child_node)
            parts = [node_text]
            if child_node.node_type == "table":
                parts = self._split_table_text(node_text)
            elif not child_node.is_atomic:
                parts = self._split_text(node_text, self.child_max_chars)
            for part_index, part in enumerate(parts, start=1):
                part_metadata = dict(metadata)
                if child_node.node_type == "table":
                    part_metadata["table_part"] = part_index
                    part_metadata["is_atomic"] = True
                child_chunks.append(
                    ChildChunk(
                        chunk_id=f"{child_node.node_id}.part_{part_index}",
                        parent_id=parent_id,
                        text=part,
                        metadata=self._with_source_locations(
                            part_metadata,
                            document.layout_details,
                            part,
                        ),
                    )
                )
        return child_chunks

    def _collect_child_nodes(self, parent: StructureNode) -> list[StructureNode]:
        """Return clause-level or atomic descendants for a parent section."""
        child_nodes: list[StructureNode] = []
        for child in parent.children:
            if child.node_type in {"clause", "table", "list"}:
                child_nodes.append(child)
                if child.children:
                    child_nodes.extend(self._collect_child_nodes(child))
                continue
            if child.node_type in SECTION_TYPES and child is not parent:
                child_nodes.extend(self._collect_child_nodes(child))
                continue
            if child.children:
                child_nodes.extend(self._collect_child_nodes(child))
        return child_nodes

    def _build_metadata(
        self,
        document: DocumentUpload,
        doc_title: str,
        section_node: StructureNode,
        parent_id: str,
    ) -> dict[str, Any]:
        """Attach breadcrumb and structural metadata to a chunk."""
        chapter = self._find_nearest(section_node, "chapter")
        section = self._find_nearest(section_node, "section")
        clause = self._find_nearest(section_node, "clause")
        appendix = self._find_nearest(section_node, "appendix")
        breadcrumb = " > ".join(node.title for node in self._lineage(section_node))
        chunk_type = section_node.node_type

        return {
            "doc_title": doc_title,
            "document_id": document.document_id,
            "filename": document.filename,
            "chapter": chapter.title if chapter else None,
            "section": section.title if section else None,
            "clause": clause.title if clause else None,
            "appendix": appendix.title if appendix else None,
            "section_id": section_node.node_id,
            "parent_id": parent_id,
            "cross_refs": list(section_node.cross_refs),
            "chunk_type": chunk_type,
            "is_atomic": section_node.is_atomic,
            "breadcrumb": breadcrumb,
            "num_pages": document.num_pages,
        }

    def _with_source_locations(
        self,
        metadata: dict[str, Any],
        layout_details: list[Any] | None,
        text: str,
    ) -> dict[str, Any]:
        """Attach chunk-scoped page and bbox metadata when layout data matches."""
        enriched = dict(metadata)
        source_locations = self._extract_source_locations(layout_details, text)
        if source_locations:
            enriched["source_locations"] = source_locations
        return enriched

    def _extract_source_locations(
        self,
        layout_details: list[Any] | None,
        text: str,
    ) -> list[dict[str, Any]]:
        """Map chunk text to the OCR layout blocks it came from."""
        if not layout_details:
            return []

        query = self._normalize_location_text(text)
        if not query:
            return []
        query_tokens = set(self._TOKEN_PATTERN.findall(query))
        heading_only = self._is_heading_only_location_text(query)

        scored_locations: list[tuple[float, int, dict[str, Any]]] = []
        seen: set[tuple[int, tuple[float, float, float, float]]] = set()
        for page_index, page in enumerate(layout_details):
            for detail in page or []:
                content = (
                    detail.get("content")
                    if isinstance(detail, dict)
                    else getattr(detail, "content", None)
                )
                candidate = self._normalize_location_text(str(content or ""))
                if not candidate or not self._location_text_matches(
                    query,
                    query_tokens,
                    candidate,
                ):
                    continue

                bbox = (
                    detail.get("bbox_2d")
                    if isinstance(detail, dict)
                    else getattr(detail, "bbox_2d", None)
                )
                normalized_bbox = self._normalize_bbox(bbox)
                if normalized_bbox is None:
                    continue

                dedupe_key = (page_index, tuple(normalized_bbox))
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                score = self._score_source_location_match(
                    query_tokens=query_tokens,
                    candidate=candidate,
                    raw_content=str(content or ""),
                    page_index=page_index,
                    detail=detail,
                )
                scored_locations.append(
                    (
                        score,
                        page_index,
                        {
                            "page": page_index,
                            "bbox_2d": normalized_bbox,
                        },
                    )
                )

        if not scored_locations:
            return []

        ranked = sorted(
            scored_locations,
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        if heading_only:
            return [ranked[0][2]]
        return [location for _, _, location in ranked[: self._SOURCE_LOCATION_LIMIT]]

    def _normalize_location_text(self, text: str) -> str:
        """Normalize chunk and layout text for source-location matching."""
        cleaned = self._normalize_ocr_text(text)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip().lower()

    def _location_text_matches(
        self,
        query: str,
        query_tokens: set[str],
        candidate: str,
    ) -> bool:
        """Return whether one OCR layout block likely backs the chunk text."""
        if not candidate or self._is_layout_artifact_line(candidate):
            return False
        if candidate in query or query in candidate:
            return True

        candidate_tokens = set(self._TOKEN_PATTERN.findall(candidate))
        if len(candidate_tokens) < 3 or not query_tokens:
            return False
        shared = len(query_tokens & candidate_tokens)
        coverage = shared / min(len(query_tokens), len(candidate_tokens))
        return shared >= 3 and coverage >= 0.6

    def _score_source_location_match(
        self,
        query_tokens: set[str],
        candidate: str,
        raw_content: str,
        page_index: int,
        detail: Any,
    ) -> float:
        """Rank likely OCR locations, down-weighting table-of-contents artifacts."""
        candidate_tokens = set(self._TOKEN_PATTERN.findall(candidate))
        shared = len(query_tokens & candidate_tokens)
        coverage = shared / max(1, min(len(query_tokens), len(candidate_tokens)))
        score = coverage * 10.0 + min(len(candidate_tokens), 80) * 0.04

        label = (
            detail.get("label")
            if isinstance(detail, dict)
            else getattr(detail, "label", "")
        )
        lowered_label = str(label or "").lower()
        if "toc" in lowered_label or "contents" in lowered_label or "daftar isi" in lowered_label:
            score -= 6.0
        if self._DOTTED_LEADER_SUFFIX.search(raw_content.strip()):
            score -= 5.0
        if page_index <= 2 and len(candidate_tokens) <= 12:
            score -= 2.0
        return score

    def _is_heading_only_location_text(self, text: str) -> bool:
        """Detect short heading snippets that are prone to TOC/body ambiguity."""
        tokens = self._TOKEN_PATTERN.findall(text)
        if not tokens or len(tokens) > 12:
            return False
        if any(marker in text for marker in ("|", "<table", "<td", "<tr")):
            return False
        return not re.search(r"\b(?:cpl\d+|if\d+[a-z]?\d+|sks|mk|semester)\b", text)

    @staticmethod
    def _normalize_bbox(bbox: Any) -> list[float] | None:
        """Return a Qdrant-friendly bbox list when the OCR bbox is valid."""
        if not isinstance(bbox, list | tuple) or len(bbox) != 4:
            return None
        try:
            return [float(value) for value in bbox]
        except (TypeError, ValueError):
            return None

    def _compose_node_id(
        self,
        parent_node_id: str,
        node_type: str,
        identifier: str,
        counters: dict[str, int],
    ) -> str:
        """Build a stable hierarchical identifier for a structure node."""
        prefix = {
            "chapter": "bab",
            "section": "pasal",
            "subsection": "subbagian",
            "clause": "ayat",
            "appendix": "lampiran",
            "table": "table",
            "list": "list",
        }.get(node_type, node_type)
        normalized = self._slug_token(identifier)
        if not normalized:
            counters[prefix] = counters.get(prefix, 0) + 1
            normalized = str(counters[prefix])
        current = f"{prefix}_{normalized}"
        if parent_node_id == "root":
            return current
        return f"{parent_node_id}.{current}"

    def _build_internal_id(self, document_id: str, section_id: str) -> str:
        """Build a storage key unique across documents."""
        return f"{document_id}:{section_id}"

    def _render_node(self, node: StructureNode) -> str:
        """Render a node and all of its descendants as retrieval context."""
        parts = [node.title] if node.title else []
        parts.extend(block for block in node.content_blocks if block.strip())
        for child in node.children:
            rendered = self._render_node(child)
            if rendered:
                parts.append(rendered)
        return "\n\n".join(part for part in parts if part).strip()

    def _truncate_parent_text(self, text: str) -> str:
        """Keep parent sections bounded for prompt context while retaining structure."""
        if len(text) <= self.parent_max_chars:
            return text
        return (
            text[: self.parent_max_chars].rstrip()
            + "\n\n[Section truncated for context window.]"
        )

    def _split_text(self, text: str, limit: int) -> list[str]:
        """Split long text into lightly overlapping windows."""
        cleaned = text.strip()
        if not cleaned:
            return []
        if len(cleaned) <= limit:
            return [cleaned]

        parts: list[str] = []
        start = 0
        while start < len(cleaned):
            end = min(start + limit, len(cleaned))
            chunk = cleaned[start:end]
            if end < len(cleaned):
                split_at = max(chunk.rfind("\n\n"), chunk.rfind(". "), chunk.rfind("\n"))
                if split_at > limit // 2:
                    end = start + split_at + 1
                    chunk = cleaned[start:end]
            parts.append(chunk.strip())
            if end >= len(cleaned):
                break
            start = max(end - self.child_overlap_chars, start + 1)
        return [part for part in parts if part]

    def _split_table_text(self, table_text: str) -> list[str]:
        """Split oversized HTML tables on row boundaries while preserving headers."""
        cleaned = table_text.strip()
        if not cleaned:
            return []
        if len(cleaned) <= self.child_max_chars:
            return [cleaned]
        if not self._HTML_TABLE_MARKER.search(cleaned) or not re.search(
            r"<\s*tr\b",
            cleaned,
            re.IGNORECASE,
        ):
            return self._split_text(cleaned, self.child_max_chars)

        prefix = ""
        first_row = re.search(r"<\s*tr\b", cleaned, re.IGNORECASE)
        if first_row:
            prefix = cleaned[: first_row.start()].strip()

        rows = [
            row.strip()
            for row in self._HTML_ROW_SPLIT.split(cleaned[first_row.start() if first_row else 0 :])
            if row.strip()
        ]
        if not rows:
            return self._split_text(cleaned, self.child_max_chars)

        header_rows: list[str] = []
        for row in rows[:2]:
            if re.search(r"<\s*th\b", row, re.IGNORECASE):
                header_rows.append(row)
        if not header_rows and rows:
            header_rows = [rows[0]]

        parts: list[str] = []
        current_rows: list[str] = []
        for row in rows:
            base_rows = header_rows if parts or current_rows else []
            candidate_rows = [*base_rows, *current_rows, row]
            candidate = "\n".join([part for part in [prefix, *candidate_rows] if part]).strip()
            if current_rows and len(candidate) > self.child_max_chars:
                parts.append(
                    "\n".join([part for part in [prefix, *base_rows, *current_rows] if part]).strip()
                )
                current_rows = [row]
                continue
            current_rows.append(row)

        if current_rows:
            base_rows = header_rows if parts else []
            parts.append(
                "\n".join([part for part in [prefix, *base_rows, *current_rows] if part]).strip()
            )
        return [part for part in parts if part]

    def _walk(self, root: StructureNode) -> list[StructureNode]:
        """Return a depth-first traversal of the tree excluding the root."""
        nodes: list[StructureNode] = []
        for child in root.children:
            nodes.append(child)
            nodes.extend(self._walk(child))
        return nodes

    def _lineage(self, node: StructureNode) -> list[StructureNode]:
        """Return the breadcrumb path from the document root to this node."""
        lineage: list[StructureNode] = []
        current: StructureNode | None = node
        while current is not None and current.node_type != "document":
            lineage.append(current)
            current = current.parent
        return list(reversed(lineage))

    def _find_nearest(self, node: StructureNode, node_type: str) -> StructureNode | None:
        """Return the closest ancestor-or-self with the requested type."""
        current: StructureNode | None = node
        while current is not None:
            if current.node_type == node_type:
                return current
            current = current.parent
        return None

    def _slug_token(self, value: str) -> str:
        """Normalize identifiers for internal IDs."""
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

    def _normalize_ocr_text(self, text: str) -> str:
        """Remove OCR wrapper markup that should not become indexed content."""
        normalized_lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("```") or stripped == "plaintext":
                continue
            if self._is_layout_artifact_line(stripped):
                continue
            normalized_lines.append(line)
        return "\n".join(normalized_lines).strip()

    def _clean_heading_line(self, line: str) -> str:
        """Remove markdown heading markers from stored structural titles."""
        return re.sub(r"^#+\s*", "", line.strip()).strip()

    def _normalize_heading_candidate(self, line: str) -> str:
        """Normalize OCR heading variants before structural detection."""
        candidate = line.strip()
        candidate = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", candidate)
        candidate = self._DOTTED_LEADER_SUFFIX.sub("", candidate)
        return re.sub(r"\s+", " ", candidate).strip()

    def _is_title_candidate(self, line: str | None) -> bool:
        """Return whether a line is usable as a human-readable document title."""
        if not line:
            return False
        candidate = self._clean_heading_line(line)
        if not candidate or len(candidate) > 120:
            return False
        if self._is_layout_artifact_line(candidate):
            return False
        return not (candidate.startswith("|") or candidate.lower().startswith("<table"))

    def _is_layout_artifact_line(self, line: str) -> bool:
        """Detect OCR layout markers that should not become content metadata."""
        stripped = line.strip()
        if not stripped:
            return False
        return bool(
            self._MARKDOWN_IMAGE_LINE.match(stripped)
            or self._HTML_TAG_LINE.match(stripped)
            or self._PAGE_MARKER_LINE.match(stripped)
        )
