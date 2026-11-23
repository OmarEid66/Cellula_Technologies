"""Report Generator Agent tools: a design + formatting toolkit that renders a professional PDF (ReportLab).

`ReportDesigner` holds a theme (fonts, sizes, colours, spacing, page layout, header/footer) and an ordered list of
content elements. Elements are stored as plain specs and turned into ReportLab flowables only when `render()` runs,
so the agent can change the design at any moment (before or after adding content) and it applies to everything
that has no explicit per-element override.

`ReportDesigner.tools()` exposes every capability as a LangChain tool for the agent.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- fonts

# ReportLab's built-in PDF fonts: always available, Latin-1 (WinAnsi) characters only.
_BASE_FAMILIES = {
    "Helvetica": ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique"),
    "Times": ("Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic"),
    "Courier": ("Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique"),
}
_GENERIC_ALIASES = {"sans-serif": "Helvetica", "serif": "Times", "monospace": "Courier", "times-roman": "Times"}

# TrueType families that are registered when their files are found on the machine (Unicode coverage for DejaVu).
_TTF_CANDIDATES = {
    "DejaVu Sans": ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf"),
    "DejaVu Serif": ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf", "DejaVuSerif-Italic.ttf", "DejaVuSerif-BoldItalic.ttf"),
    "DejaVu Sans Mono": ("DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf", "DejaVuSansMono-Oblique.ttf", "DejaVuSansMono-BoldOblique.ttf"),
    "Liberation Sans": ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf", "LiberationSans-Italic.ttf", "LiberationSans-BoldItalic.ttf"),
    "Liberation Serif": ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf", "LiberationSerif-Italic.ttf", "LiberationSerif-BoldItalic.ttf"),
    "Arial": ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"),
    "Times New Roman": ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"),
    "Calibri": ("calibri.ttf", "calibrib.ttf", "calibrii.ttf", "calibriz.ttf"),
    "Verdana": ("verdana.ttf", "verdanab.ttf", "verdanai.ttf", "verdanaz.ttf"),
    "Georgia": ("georgia.ttf", "georgiab.ttf", "georgiai.ttf", "georgiaz.ttf"),
}
_UNICODE_FALLBACKS = ("DejaVu Sans", "Arial", "Liberation Sans")
_FONT_DIRS = [
    os.getenv("FONT_DIR", ""),
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    "~/.fonts",
    "~/.local/share/fonts",
    "/Library/Fonts",
    "/System/Library/Fonts/Supplemental",
    "~/Library/Fonts",
    "C:/Windows/Fonts",
]

_BASE_NAMES = {name for faces in _BASE_FAMILIES.values() for name in faces}
_font_index: dict[str, str] | None = None  # lower-case file name -> path
_registered: dict[str, tuple[str, str, str, str]] = {}  # family -> reportlab face names (normal, bold, italic, bold-italic)


def _index_fonts() -> dict[str, str]:
    global _font_index
    if _font_index is None:
        wanted = {f.lower() for faces in _TTF_CANDIDATES.values() for f in faces}
        found: dict[str, str] = {}
        for d in _FONT_DIRS:
            if not d:
                continue
            root = Path(d).expanduser()
            if not root.is_dir():
                continue
            for dirpath, _dirs, files in os.walk(root):
                for fn in files:
                    low = fn.lower()
                    if low in wanted and low not in found:
                        found[low] = os.path.join(dirpath, fn)
        _font_index = found
    return _font_index


def _register_ttf(family: str) -> bool:
    if family in _registered:
        return True
    files = _TTF_CANDIDATES.get(family)
    if not files:
        return False
    index = _index_fonts()
    paths = [index.get(f.lower()) for f in files]
    if not paths[0]:
        return False
    from reportlab.lib.fonts import addMapping
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    names = []
    for suffix, path in zip(("", "-Bold", "-Italic", "-BoldItalic"), paths):
        name = f"{family}{suffix}"
        pdfmetrics.registerFont(TTFont(name, path or paths[0]))  # missing variants fall back to the regular face
        names.append(name)
    for face in names:
        for bold in (0, 1):
            for italic in (0, 1):
                addMapping(face, bold, italic, names[bold + 2 * italic])
    _registered[family] = tuple(names)  # type: ignore[assignment]
    return True


def available_fonts() -> list[str]:
    fonts = list(_BASE_FAMILIES)
    index = _index_fonts()
    for family, files in _TTF_CANDIDATES.items():
        if files[0].lower() in index:
            fonts.append(family)
    return fonts


def resolve_font_family(name: str) -> str:
    """Canonical family name for a user/LLM-supplied font name; raises ValueError listing the valid ones."""
    key = (name or "").strip().lower()
    key = _GENERIC_ALIASES.get(key, key)
    for fam in available_fonts():
        if fam.lower() == key:
            return fam
    raise ValueError(f"Font '{name}' is not available. Available fonts: {', '.join(available_fonts())}.")


def font_faces(family: str) -> tuple[str, str, str, str]:
    """ReportLab font names (normal, bold, italic, bold-italic) for a canonical family (registers TTFs lazily)."""
    if family in _BASE_FAMILIES:
        return _BASE_FAMILIES[family]
    if not _register_ttf(family):
        raise ValueError(f"Font '{family}' could not be loaded.")
    return _registered[family]


def _unicode_fallback_family() -> str | None:
    for fam in _UNICODE_FALLBACKS:
        if fam in available_fonts():
            return fam
    return None


def _is_latin1_safe(text: str) -> bool:
    try:
        text.encode("cp1252")
        return True
    except UnicodeEncodeError:
        return False


# ---------------------------------------------------------------- colours & text helpers


def parse_color(value: str):
    """Hex ('#0B3C5D', '#fff') or a CSS-style colour name ('navy') -> ReportLab Color."""
    from reportlab.lib.colors import toColor

    v = (value or "").strip()
    if not v:
        raise ValueError("Empty colour.")
    m = re.fullmatch(r"#?([0-9A-Fa-f]{3})", v)
    if m:  # reportlab reads '#222' as the number 0x222, so expand shorthand hex ourselves
        v = "#" + "".join(ch * 2 for ch in m.group(1))
    elif re.fullmatch(r"[0-9A-Fa-f]{6}", v):
        v = "#" + v
    try:
        return toColor(v)
    except Exception as exc:
        raise ValueError(f"Unknown colour '{value}'. Use hex like '#1F4E79' or a name like 'navy'.") from exc


def _luminance(c) -> float:
    def ch(x: float) -> float:
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4

    return 0.2126 * ch(c.red) + 0.7152 * ch(c.green) + 0.0722 * ch(c.blue)


def contrast_ratio(a, b) -> float:
    la, lb = _luminance(a), _luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _hex(c) -> str:
    return "#%02X%02X%02X" % (round(c.red * 255), round(c.green * 255), round(c.blue * 255))


_NUMERIC = re.compile(r"^[\s$€£]*[-+(]?\d[\d,]*(\.\d+)?\s*(%|ms|s|M|B|K|GB|MB)?\)?\s*$")


def inline_markup(text: str) -> str:
    """Escape text for ReportLab and convert light inline markdown: **bold**, *italic*, `code`, {{#hex:coloured}}."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def colored(m: re.Match) -> str:
        try:
            c = parse_color(m.group(1))
        except ValueError:
            return m.group(2)
        return f'<font color="{_hex(c)}">{m.group(2)}</font>'

    text = re.sub(r"\{\{\s*(#?[0-9A-Za-z]+)\s*:\s*(.+?)\}\}", colored, text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\*\w])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\*\w])", r"<i>\1</i>", text)
    return text.replace("\n", "<br/>")


# ---------------------------------------------------------------- markdown -> element specs


def markdown_to_elements(md: str, heading_offset: int = 1) -> list[dict[str, Any]]:
    """Small block-level markdown parser: headings, bullet / numbered lists, pipe tables, rules, paragraphs."""
    lines = md.replace("\r\n", "\n").split("\n")
    out: list[dict[str, Any]] = []
    para: list[str] = []
    i = 0

    def flush():
        if para:
            out.append({"type": "paragraph", "text": " ".join(s.strip() for s in para)})
            para.clear()

    def split_row(line: str) -> list[str]:
        line = line.strip()
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]
        return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line)]

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush()
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush()
            level = min(3, len(m.group(1)) + heading_offset)
            out.append({"type": "heading", "text": m.group(2).strip(" #"), "level": level})
            i += 1
            continue
        if re.fullmatch(r"[-*_]{3,}", stripped):
            flush()
            out.append({"type": "divider"})
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", lines[i + 1].strip()):
            flush()
            headers = split_row(stripped)
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = split_row(lines[i])
                row = (row + [""] * len(headers))[: len(headers)]
                rows.append(row)
                i += 1
            out.append({"type": "table", "headers": headers, "rows": rows})
            continue
        bm = re.match(r"^\s*(?:[-*•]|(\d+)[.)])\s+(.*)$", line)
        if bm:
            flush()
            ordered = bm.group(1) is not None
            items = []
            while i < len(lines):
                m2 = re.match(r"^\s*(?:[-*•]|(\d+)[.)])\s+(.*)$", lines[i])
                if not m2 or (m2.group(1) is not None) != ordered:
                    break
                items.append(m2.group(2).strip())
                i += 1
            out.append({"type": "bullets", "items": items, "ordered": ordered})
            continue
        para.append(line)
        i += 1
    flush()
    return out


# ---------------------------------------------------------------- the designer


@dataclass
class Theme:
    body_font: str = "Helvetica"
    heading_font: str = "Helvetica"
    body_size: float = 10.5
    title_size: float = 26.0
    h1_size: float = 17.0
    h2_size: float = 13.5
    h3_size: float = 11.5
    line_spacing: float = 1.4
    paragraph_spacing: float = 6.0
    heading_space_before: float = 16.0
    justify_body: bool = False
    text_color: str = "#1F2933"
    heading_color: str = "#0B3C5D"
    title_color: str = ""  # empty -> heading colour
    accent_color: str = "#2F80B7"
    table_header_background: str = ""  # empty -> accent colour
    table_header_text: str = "#FFFFFF"
    table_zebra: str = "#F2F6FA"
    page_background: str = ""  # empty -> white
    page_size: str = "A4"
    orientation: str = "portrait"
    margin_mm: float = 20.0
    columns: int = 1
    header_text: str = ""
    footer_text: str = ""
    show_page_numbers: bool = True
    show_header_rule: bool = True


_PAGE_SIZES = ("A3", "A4", "A5", "LETTER", "LEGAL")
Align = Literal["left", "center", "right", "justify"]
_MAX_ELEMENTS = 500
_MAX_TEXT = 12000


class _NoArgs(BaseModel):
    pass


class SetPageLayoutArgs(BaseModel):
    page_size: Optional[Literal["A3", "A4", "A5", "LETTER", "LEGAL"]] = Field(default=None, description="Paper size.")
    orientation: Optional[Literal["portrait", "landscape"]] = None
    margin_mm: Optional[float] = Field(default=None, ge=12, le=40, description="Page margin on all sides, 12-40 mm.")
    columns: Optional[int] = Field(default=None, ge=1, le=2, description="Text columns per page (1 or 2).")


class SetFontsArgs(BaseModel):
    body_font: Optional[str] = Field(default=None, description="Font family for body text, tables and header/footer.")
    heading_font: Optional[str] = Field(default=None, description="Font family for the title and headings.")


class SetFontSizesArgs(BaseModel):
    body: Optional[float] = Field(default=None, ge=7, le=16, description="Body text size in points.")
    title: Optional[float] = Field(default=None, ge=14, le=48)
    h1: Optional[float] = Field(default=None, ge=10, le=36)
    h2: Optional[float] = Field(default=None, ge=9, le=28)
    h3: Optional[float] = Field(default=None, ge=8, le=22)


class SetColorsArgs(BaseModel):
    text: Optional[str] = Field(default=None, description="Body text colour (hex or name).")
    heading: Optional[str] = Field(default=None, description="Heading colour.")
    title: Optional[str] = Field(default=None, description="Title colour (defaults to the heading colour).")
    accent: Optional[str] = Field(default=None, description="Accent colour: rules, bullets, chart bars, callout bars.")
    table_header_background: Optional[str] = None
    table_header_text: Optional[str] = None
    table_zebra: Optional[str] = Field(default=None, description="Alternate table row tint.")
    page_background: Optional[str] = Field(default=None, description="Page background colour, or 'none' for white.")


class SetSpacingArgs(BaseModel):
    line_spacing: Optional[float] = Field(default=None, ge=1.0, le=2.5, description="Line height as a multiple of the font size.")
    paragraph_spacing_pt: Optional[float] = Field(default=None, ge=0, le=30)
    heading_space_before_pt: Optional[float] = Field(default=None, ge=0, le=60)
    justify_body: Optional[bool] = Field(default=None, description="Justify body paragraphs.")


class SetHeaderFooterArgs(BaseModel):
    header_text: Optional[str] = Field(default=None, description="Text on every page header ('' removes it).")
    footer_text: Optional[str] = Field(default=None, description="Text on every page footer ('' removes it).")
    show_page_numbers: Optional[bool] = None
    show_header_rule: Optional[bool] = Field(default=None, description="Thin rule under the header.")


class SetDocumentInfoArgs(BaseModel):
    title: Optional[str] = Field(default=None, description="PDF metadata title.")
    author: Optional[str] = None
    subject: Optional[str] = None


class AddTitleBlockArgs(BaseModel):
    title: str
    subtitle: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    align: Literal["left", "center", "right"] = "left"
    show_rule: bool = True


class AddHeadingArgs(BaseModel):
    text: str
    level: int = Field(default=1, ge=1, le=3)
    color: Optional[str] = None
    font: Optional[str] = None
    font_size: Optional[float] = Field(default=None, ge=8, le=40)
    align: Optional[Literal["left", "center", "right"]] = None
    rule_below: bool = False
    page_break_before: bool = False


class AddParagraphArgs(BaseModel):
    text: str = Field(description="Paragraph text. Inline: **bold**, *italic*, `code`, {{#C0392B:coloured text}}.")
    align: Optional[Align] = None
    bold: bool = False
    italic: bool = False
    underline: bool = False
    color: Optional[str] = None
    font: Optional[str] = None
    font_size: Optional[float] = Field(default=None, ge=6, le=40)
    background_color: Optional[str] = Field(default=None, description="Highlight colour behind the whole paragraph.")
    indent_mm: float = Field(default=0, ge=0, le=60)
    space_before: Optional[float] = Field(default=None, ge=0, le=60)
    space_after: Optional[float] = Field(default=None, ge=0, le=60)
    line_spacing: Optional[float] = Field(default=None, ge=1.0, le=2.5)


class AddBulletsArgs(BaseModel):
    items: list[str] = Field(min_length=1, max_length=60)
    ordered: bool = Field(default=False, description="Numbered list instead of bullets.")
    font_size: Optional[float] = Field(default=None, ge=6, le=24)
    color: Optional[str] = None


class AddTableArgs(BaseModel):
    headers: list[str] = Field(min_length=1, max_length=12)
    rows: list[list[str]] = Field(max_length=200)
    column_widths_percent: Optional[list[float]] = Field(default=None, description="One width per column, e.g. [40, 20, 40].")
    caption: Optional[str] = None
    font_size: Optional[float] = Field(default=None, ge=6, le=14)
    zebra: bool = True
    header_background: Optional[str] = Field(default=None, description="Overrides the theme's table header colour.")


class AddCalloutArgs(BaseModel):
    text: str
    kind: Literal["info", "success", "warning", "quote", "custom"] = "info"
    title: Optional[str] = None
    background_color: Optional[str] = Field(default=None, description="For kind='custom'.")
    border_color: Optional[str] = Field(default=None, description="For kind='custom'.")


class AddBarChartArgs(BaseModel):
    labels: list[str] = Field(min_length=1, max_length=20)
    values: list[float] = Field(min_length=1, max_length=20)
    title: Optional[str] = None
    color: Optional[str] = Field(default=None, description="Bar colour (defaults to the accent colour).")
    value_decimals: int = Field(default=1, ge=0, le=4)
    height_pt: float = Field(default=190, ge=100, le=400)


class AddMarkdownArgs(BaseModel):
    markdown: str = Field(description="Markdown with headings, lists, pipe tables, **bold**. Styled with the current theme.")
    heading_offset: int = Field(default=1, ge=0, le=2, description="'#' becomes heading level 1+offset (max 3).")


class AddDividerArgs(BaseModel):
    color: Optional[str] = None
    thickness_pt: float = Field(default=0.75, ge=0.25, le=6)


class AddSpacerArgs(BaseModel):
    height_pt: float = Field(default=12, ge=1, le=200)


_CALLOUTS = {
    "info": ("#EAF4FB", "#2F80B7"),
    "success": ("#E8F6EE", "#2E9E5B"),
    "warning": ("#FFF6E0", "#D99A00"),
    "quote": ("#F4F4F5", "#8A8F98"),
}


class ReportDesigner:
    """Stateful report builder. Every public tool method returns a short confirmation string (or raises ValueError)."""

    def __init__(self, title: str = "Report", author: str = ""):
        self.theme = Theme()
        self.doc_title = title
        self.doc_author = author
        self.doc_subject = ""
        self.elements: list[dict[str, Any]] = []
        self.log: list[str] = []

    # ------------------------------------------------------------ helpers
    def _note(self, msg: str) -> str:
        self.log.append(msg)
        return msg

    def _add(self, spec: dict[str, Any]) -> None:
        if len(self.elements) >= _MAX_ELEMENTS:
            raise ValueError(f"Report is too long (limit {_MAX_ELEMENTS} elements).")
        self.elements.append(spec)

    @staticmethod
    def _text(value: str, what: str = "text") -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError(f"The {what} is empty.")
        return value[:_MAX_TEXT]

    def has_content(self) -> bool:
        return any(e["type"] not in ("spacer", "divider", "pagebreak") for e in self.elements)

    def _contrast_warning(self) -> str:
        t = self.theme
        bg = parse_color(t.page_background) if t.page_background else parse_color("#FFFFFF")
        notes = []
        for label, col in (("text", t.text_color), ("heading", t.heading_color)):
            ratio = contrast_ratio(parse_color(col), bg)
            if ratio < 4.5:
                notes.append(f"{label} colour {col} on background {_hex(bg)} has low contrast ({ratio:.1f}:1)")
        return (" WARNING: " + "; ".join(notes) + ". Pick darker/lighter colours.") if notes else ""

    # ------------------------------------------------------------ design tools
    def list_fonts(self) -> str:
        fams = available_fonts()
        unicode_note = _unicode_fallback_family()
        extra = f" Text outside Latin-1 automatically uses '{unicode_note}'." if unicode_note else " Only Latin-1 characters are supported."
        return "Available fonts: " + ", ".join(fams) + "." + extra

    def set_page_layout(self, page_size=None, orientation=None, margin_mm=None, columns=None) -> str:
        t = self.theme
        if page_size:
            t.page_size = page_size
        if orientation:
            t.orientation = orientation
        if margin_mm is not None:
            t.margin_mm = float(margin_mm)
        if columns is not None:
            t.columns = int(columns)
        return self._note(f"Page layout: {t.page_size} {t.orientation}, margin {t.margin_mm:g} mm, {t.columns} column(s).")

    def set_fonts(self, body_font=None, heading_font=None) -> str:
        if body_font:
            self.theme.body_font = resolve_font_family(body_font)
            font_faces(self.theme.body_font)
        if heading_font:
            self.theme.heading_font = resolve_font_family(heading_font)
            font_faces(self.theme.heading_font)
        return self._note(f"Fonts: body={self.theme.body_font}, headings={self.theme.heading_font}.")

    def set_font_sizes(self, body=None, title=None, h1=None, h2=None, h3=None) -> str:
        t = self.theme
        for attr, val in (("body_size", body), ("title_size", title), ("h1_size", h1), ("h2_size", h2), ("h3_size", h3)):
            if val is not None:
                setattr(t, attr, float(val))
        return self._note(
            f"Font sizes: body {t.body_size:g}pt, title {t.title_size:g}pt, h1 {t.h1_size:g}pt, h2 {t.h2_size:g}pt, h3 {t.h3_size:g}pt."
        )

    def set_colors(self, text=None, heading=None, title=None, accent=None, table_header_background=None,
                   table_header_text=None, table_zebra=None, page_background=None) -> str:
        t = self.theme
        updates = {
            "text_color": text, "heading_color": heading, "title_color": title, "accent_color": accent,
            "table_header_background": table_header_background, "table_header_text": table_header_text,
            "table_zebra": table_zebra,
        }
        for attr, val in updates.items():
            if val:
                setattr(t, attr, _hex(parse_color(val)))
        if page_background is not None:
            t.page_background = "" if page_background.strip().lower() in ("", "none", "white") else _hex(parse_color(page_background))
        return self._note(
            f"Colours: text {t.text_color}, headings {t.heading_color}, accent {t.accent_color}, "
            f"background {t.page_background or 'white'}." + self._contrast_warning()
        )

    def set_spacing(self, line_spacing=None, paragraph_spacing_pt=None, heading_space_before_pt=None, justify_body=None) -> str:
        t = self.theme
        if line_spacing is not None:
            t.line_spacing = float(line_spacing)
        if paragraph_spacing_pt is not None:
            t.paragraph_spacing = float(paragraph_spacing_pt)
        if heading_space_before_pt is not None:
            t.heading_space_before = float(heading_space_before_pt)
        if justify_body is not None:
            t.justify_body = bool(justify_body)
        return self._note(
            f"Spacing: line height x{t.line_spacing:g}, paragraph gap {t.paragraph_spacing:g}pt, "
            f"space before headings {t.heading_space_before:g}pt, justified={t.justify_body}."
        )

    def set_header_footer(self, header_text=None, footer_text=None, show_page_numbers=None, show_header_rule=None) -> str:
        t = self.theme
        if header_text is not None:
            t.header_text = header_text.strip()[:150]
        if footer_text is not None:
            t.footer_text = footer_text.strip()[:150]
        if show_page_numbers is not None:
            t.show_page_numbers = bool(show_page_numbers)
        if show_header_rule is not None:
            t.show_header_rule = bool(show_header_rule)
        return self._note(
            f"Header='{t.header_text}', footer='{t.footer_text}', page numbers={t.show_page_numbers}."
        )

    def set_document_info(self, title=None, author=None, subject=None) -> str:
        if title:
            self.doc_title = title.strip()[:200]
        if author:
            self.doc_author = author.strip()[:200]
        if subject:
            self.doc_subject = subject.strip()[:200]
        return self._note(f"PDF metadata: title='{self.doc_title}', author='{self.doc_author}'.")

    # ------------------------------------------------------------ content tools
    def add_title_block(self, title, subtitle=None, author=None, date=None, align="left", show_rule=True) -> str:
        title = self._text(title, "title")
        if self.doc_title in ("", "Report"):
            self.doc_title = title[:200]
        self._add({"type": "title", "title": title, "subtitle": subtitle or "", "author": author or "", "date": date or "",
                   "align": align, "rule": show_rule})
        return self._note(f"Added title block '{title[:50]}'.")

    def add_heading(self, text, level=1, color=None, font=None, font_size=None, align=None, rule_below=False,
                    page_break_before=False) -> str:
        spec = {"type": "heading", "text": self._text(text, "heading"), "level": int(level), "rule": rule_below,
                "page_break": page_break_before}
        if color:
            spec["color"] = _hex(parse_color(color))
        if font:
            spec["font"] = resolve_font_family(font)
        if font_size:
            spec["font_size"] = float(font_size)
        if align:
            spec["align"] = align
        self._add(spec)
        return self._note(f"Added heading L{level}: '{spec['text'][:50]}'.")

    def add_paragraph(self, text, align=None, bold=False, italic=False, underline=False, color=None, font=None,
                      font_size=None, background_color=None, indent_mm=0, space_before=None, space_after=None,
                      line_spacing=None) -> str:
        spec: dict[str, Any] = {"type": "paragraph", "text": self._text(text), "bold": bold, "italic": italic,
                                "underline": underline, "indent_mm": float(indent_mm)}
        if align:
            spec["align"] = align
        if color:
            spec["color"] = _hex(parse_color(color))
        if font:
            spec["font"] = resolve_font_family(font)
        if font_size:
            spec["font_size"] = float(font_size)
        if background_color:
            spec["background"] = _hex(parse_color(background_color))
        if space_before is not None:
            spec["space_before"] = float(space_before)
        if space_after is not None:
            spec["space_after"] = float(space_after)
        if line_spacing is not None:
            spec["line_spacing"] = float(line_spacing)
        self._add(spec)
        return self._note(f"Added paragraph ({len(spec['text'])} chars).")

    def add_bullets(self, items, ordered=False, font_size=None, color=None) -> str:
        items = [i.strip() for i in items if i and i.strip()]
        if not items:
            raise ValueError("No list items given.")
        spec: dict[str, Any] = {"type": "bullets", "items": [i[:_MAX_TEXT] for i in items], "ordered": ordered}
        if font_size:
            spec["font_size"] = float(font_size)
        if color:
            spec["color"] = _hex(parse_color(color))
        self._add(spec)
        return self._note(f"Added {'numbered' if ordered else 'bullet'} list with {len(items)} items.")

    def add_table(self, headers, rows, column_widths_percent=None, caption=None, font_size=None, zebra=True,
                  header_background=None) -> str:
        n = len(headers)
        if column_widths_percent is not None:
            if len(column_widths_percent) != n or any(w <= 0 for w in column_widths_percent):
                raise ValueError(f"column_widths_percent needs {n} positive numbers (one per column).")
        norm = [([str(c) for c in r] + [""] * n)[:n] for r in rows]
        spec: dict[str, Any] = {"type": "table", "headers": [str(h) for h in headers], "rows": norm,
                                "widths": column_widths_percent, "caption": caption or "", "zebra": zebra}
        if font_size:
            spec["font_size"] = float(font_size)
        if header_background:
            spec["header_background"] = _hex(parse_color(header_background))
        self._add(spec)
        return self._note(f"Added table {len(norm)} rows x {n} columns.")

    def add_callout(self, text, kind="info", title=None, background_color=None, border_color=None) -> str:
        bg, border = _CALLOUTS.get(kind, _CALLOUTS["info"])
        if kind == "custom":
            bg = _hex(parse_color(background_color)) if background_color else bg
            border = _hex(parse_color(border_color)) if border_color else border
        self._add({"type": "callout", "text": self._text(text), "title": title or "", "bg": bg, "border": border,
                   "italic": kind == "quote"})
        return self._note(f"Added {kind} callout.")

    def add_bar_chart(self, labels, values, title=None, color=None, value_decimals=1, height_pt=190) -> str:
        if len(labels) != len(values):
            raise ValueError("labels and values must have the same length.")
        spec: dict[str, Any] = {"type": "chart", "labels": [str(l) for l in labels], "values": [float(v) for v in values],
                                "title": title or "", "decimals": int(value_decimals), "height": float(height_pt)}
        if color:
            spec["color"] = _hex(parse_color(color))
        self._add(spec)
        return self._note(f"Added bar chart with {len(values)} bars.")

    def add_markdown(self, markdown, heading_offset=1) -> str:
        specs = markdown_to_elements(self._text(markdown, "markdown"), heading_offset)
        if not specs:
            raise ValueError("The markdown contained nothing to add.")
        for s in specs:
            self._add(s)
        kinds: dict[str, int] = {}
        for s in specs:
            kinds[s["type"]] = kinds.get(s["type"], 0) + 1
        return self._note("Added markdown: " + ", ".join(f"{n} {k}" for k, n in kinds.items()) + ".")

    def add_divider(self, color=None, thickness_pt=0.75) -> str:
        spec: dict[str, Any] = {"type": "divider", "thickness": float(thickness_pt)}
        if color:
            spec["color"] = _hex(parse_color(color))
        self._add(spec)
        return self._note("Added divider.")

    def add_spacer(self, height_pt=12) -> str:
        self._add({"type": "spacer", "height": float(height_pt)})
        return self._note(f"Added {height_pt:g}pt spacer.")

    def add_page_break(self) -> str:
        self._add({"type": "pagebreak"})
        return self._note("Added page break.")

    def review_report(self) -> str:
        t = self.theme
        lines = [
            f"Design: {t.page_size} {t.orientation}, {t.columns} column(s), margin {t.margin_mm:g}mm; fonts {t.body_font}/{t.heading_font}; "
            f"sizes body {t.body_size:g} h1 {t.h1_size:g} h2 {t.h2_size:g}; colours text {t.text_color} heading {t.heading_color} "
            f"accent {t.accent_color}; header='{t.header_text}' footer='{t.footer_text}'.",
            f"{len(self.elements)} elements:",
        ]
        for i, e in enumerate(self.elements, 1):
            preview = e.get("text") or e.get("title") or ", ".join(e.get("headers", [])) or ", ".join(e.get("labels", [])) or ""
            if e["type"] == "bullets":
                preview = f"{len(e['items'])} items"
            lines.append(f"{i}. {e['type']}" + (f" (L{e['level']})" if e["type"] == "heading" else "") + (f": {str(preview)[:60]}" if preview else ""))
        warn = self._contrast_warning()
        if warn:
            lines.append(warn.strip())
        if not self.has_content():
            lines.append("WARNING: the report has no content yet.")
        return "\n".join(lines)

    # ------------------------------------------------------------ rendering
    def _face(self, family: str, bold: bool = False, italic: bool = False) -> str:
        n, b, i, bi = font_faces(family)
        return bi if bold and italic else b if bold else i if italic else n

    def _fit_font(self, family: str, text: str) -> str:
        """Built-in fonts cannot draw characters outside Latin-1: switch to a Unicode TTF when the text needs it."""
        if family in _BASE_FAMILIES and not _is_latin1_safe(text):
            fb = _unicode_fallback_family()
            if fb:
                return fb
        return family

    @staticmethod
    def _latin(text: str, family: str) -> str:
        if family in _BASE_FAMILIES and not _is_latin1_safe(text):
            return text.encode("cp1252", "replace").decode("cp1252")
        return text

    def render(self, path: str | Path) -> int:
        """Write the PDF; returns the number of pages."""
        from reportlab.lib import pagesizes
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (BaseDocTemplate, Frame, HRFlowable, KeepTogether, ListFlowable,
                                        ListItem, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)

        t = self.theme
        size = getattr(pagesizes, t.page_size)
        if t.orientation == "landscape":
            size = pagesizes.landscape(size)
        pw, ph = size
        margin = t.margin_mm * mm
        gap = 8 * mm
        ncol = max(1, t.columns)
        frame_w = (pw - 2 * margin - gap * (ncol - 1)) / ncol
        frame_h = ph - 2 * margin
        aligns = {"left": TA_LEFT, "center": TA_CENTER, "right": TA_RIGHT, "justify": TA_JUSTIFY}

        text_c, head_c = parse_color(t.text_color), parse_color(t.heading_color)
        title_c = parse_color(t.title_color or t.heading_color)
        accent_c = parse_color(t.accent_color)
        bg_c = parse_color(t.page_background) if t.page_background else None
        dark_bg = bg_c is not None and _luminance(bg_c) < 0.4
        muted_c = parse_color("#C9CED6" if dark_bg else "#6B7280")

        def mk_style(name: str, family: str, size_pt: float, color, *, bold=False, italic=False, align=TA_LEFT,
                     leading: float | None = None, **kw) -> ParagraphStyle:
            return ParagraphStyle(name, fontName=self._face(family, bold, italic), fontSize=size_pt,
                                  leading=leading or size_pt * t.line_spacing, textColor=color, alignment=align, **kw)

        body = mk_style("body", t.body_font, t.body_size, text_c, align=TA_JUSTIFY if t.justify_body else TA_LEFT,
                        spaceAfter=t.paragraph_spacing)
        head = {
            1: mk_style("h1", t.heading_font, t.h1_size, head_c, bold=True, leading=t.h1_size * 1.25,
                        spaceBefore=t.heading_space_before, spaceAfter=6, keepWithNext=1),
            2: mk_style("h2", t.heading_font, t.h2_size, head_c, bold=True, leading=t.h2_size * 1.25,
                        spaceBefore=t.heading_space_before * 0.75, spaceAfter=4, keepWithNext=1),
            3: mk_style("h3", t.heading_font, t.h3_size, head_c, bold=True, leading=t.h3_size * 1.25,
                        spaceBefore=t.heading_space_before * 0.55, spaceAfter=3, keepWithNext=1),
        }

        def para(text: str, style: ParagraphStyle, *, wrap: tuple[str, str] = ("", "")) -> Paragraph:
            if style.fontName in _BASE_NAMES and not _is_latin1_safe(text):
                fb = _unicode_fallback_family()
                if fb:  # built-in fonts cannot draw this text -> same style in a Unicode font
                    style = ParagraphStyle(style.name + "_u", parent=style, fontName=self._face(
                        fb, "Bold" in style.fontName, any(k in style.fontName for k in ("Italic", "Oblique"))))
                else:
                    text = text.encode("cp1252", "replace").decode("cp1252")
            markup = wrap[0] + inline_markup(text) + wrap[1]
            try:
                return Paragraph(markup, style)
            except Exception:  # malformed nesting from the inline markdown -> fall back to escaped plain text
                plain = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
                return Paragraph(wrap[0] + plain + wrap[1], style)

        def override(base: ParagraphStyle, e: dict, default_family: str, bold=False, italic=False) -> ParagraphStyle:
            kw: dict[str, Any] = {}
            family = e.get("font") or default_family
            fs = e.get("font_size") or base.fontSize
            if e.get("font") or e.get("bold") or e.get("italic"):
                kw["fontName"] = self._face(family, bold or bool(e.get("bold")), italic or bool(e.get("italic")))
            if e.get("font_size"):
                kw["fontSize"] = fs
                kw["leading"] = fs * (e.get("line_spacing") or t.line_spacing)
            elif e.get("line_spacing"):
                kw["leading"] = base.fontSize * e["line_spacing"]
            if e.get("color"):
                kw["textColor"] = parse_color(e["color"])
            if e.get("align"):
                kw["alignment"] = aligns[e["align"]]
            if e.get("background"):
                kw["backColor"] = parse_color(e["background"])
                kw["borderPadding"] = (3, 4, 3, 4)
            if e.get("indent_mm"):
                kw["leftIndent"] = e["indent_mm"] * mm
            if "space_before" in e:
                kw["spaceBefore"] = e["space_before"]
            if "space_after" in e:
                kw["spaceAfter"] = e["space_after"]
            return ParagraphStyle(base.name + "_o", parent=base, **kw)

        flow: list = []
        for e in self.elements:
            kind = e["type"]
            if kind == "title":
                align = aligns[e["align"]]
                ts = mk_style("title", t.heading_font, t.title_size, title_c, bold=True, align=align,
                              leading=t.title_size * 1.2, spaceAfter=4)
                flow.append(para(e["title"], ts))
                if e["subtitle"]:
                    flow.append(para(e["subtitle"], mk_style("sub", t.heading_font, t.title_size * 0.5, muted_c, align=align,
                                                              spaceAfter=4)))
                by = " · ".join(x for x in (e["author"], e["date"]) if x)
                if by:
                    flow.append(para(by, mk_style("by", t.body_font, t.body_size, muted_c, align=align, spaceAfter=4)))
                if e.get("rule", True):
                    flow.append(HRFlowable(width="100%", thickness=2, color=accent_c, spaceBefore=4, spaceAfter=10))
            elif kind == "heading":
                if e.get("page_break"):
                    flow.append(PageBreak())
                base = head[e["level"]]
                style = override(base, e, e.get("font") or t.heading_font, bold=True)
                flow.append(para(e["text"], style))
                if e.get("rule"):
                    flow.append(HRFlowable(width="100%", thickness=0.8, color=accent_c, spaceBefore=0, spaceAfter=6))
            elif kind == "paragraph":
                style = override(body, e, t.body_font)
                wrap = ("<u>", "</u>") if e.get("underline") else ("", "")
                flow.append(para(e["text"], style, wrap=wrap))
            elif kind == "bullets":
                style = override(body, e, t.body_font)
                style.spaceAfter = 2
                items = [ListItem(para(it, style), leftIndent=16) for it in e["items"]]
                lf = ListFlowable(items, bulletType="1" if e["ordered"] else "bullet", start=None if e["ordered"] else "•",
                                  bulletFontName=style.fontName, bulletFontSize=style.fontSize,
                                  bulletColor=parse_color(e["color"]) if e.get("color") else accent_c, leftIndent=16,
                                  bulletFormat="%s." if e["ordered"] else None)
                flow.append(lf)
                flow.append(Spacer(1, t.paragraph_spacing))
            elif kind == "table":
                flow.extend(self._table_flow(e, frame_w, para, mk_style, text_c, accent_c, muted_c, Table, TableStyle, Spacer))
            elif kind == "callout":
                cs = mk_style("callout", t.body_font, t.body_size, text_c, italic=bool(e.get("italic")))
                cell = []
                if e.get("title"):
                    cell.append(para(e["title"], mk_style("callout_t", t.body_font, t.body_size, text_c, bold=True, spaceAfter=2)))
                cell.append(para(e["text"], cs))
                box = Table([[cell]], colWidths=[frame_w])
                box.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), parse_color(e["bg"])),
                    ("LINEBEFORE", (0, 0), (0, -1), 3.5, parse_color(e["border"])),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]))
                flow.extend([box, Spacer(1, t.paragraph_spacing + 4)])
            elif kind == "chart":
                flow.extend(self._chart_flow(e, frame_w, para, mk_style, text_c, accent_c, muted_c, KeepTogether, Spacer))
            elif kind == "divider":
                flow.append(HRFlowable(width="100%", thickness=e.get("thickness", 0.75),
                                       color=parse_color(e["color"]) if e.get("color") else muted_c, spaceBefore=6, spaceAfter=8))
            elif kind == "spacer":
                flow.append(Spacer(1, e["height"]))
            elif kind == "pagebreak":
                flow.append(PageBreak())

        def decorate(canvas, doc):
            canvas.saveState()
            if bg_c is not None:
                canvas.setFillColor(bg_c)
                canvas.rect(0, 0, pw, ph, stroke=0, fill=1)
            fam = self._fit_font(t.body_font, t.header_text + t.footer_text)
            canvas.setFont(self._face(fam), 8.5)
            canvas.setFillColor(muted_c)
            if t.header_text:
                y = ph - margin * 0.55
                canvas.drawString(margin, y, self._latin(t.header_text, fam))
                if t.show_header_rule:
                    canvas.setStrokeColor(muted_c)
                    canvas.setLineWidth(0.4)
                    canvas.line(margin, y - 4, pw - margin, y - 4)
            y = margin * 0.5
            if t.footer_text:
                canvas.drawString(margin, y, self._latin(t.footer_text, fam))
            if t.show_page_numbers:
                canvas.drawRightString(pw - margin, y, f"Page {doc.page}")
            canvas.restoreState()

        frames = [
            Frame(margin + i * (frame_w + gap), margin, frame_w, frame_h, leftPadding=0, rightPadding=0, topPadding=0,
                  bottomPadding=0, id=f"col{i}")
            for i in range(ncol)
        ]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        doc = BaseDocTemplate(str(path), pagesize=size, leftMargin=margin, rightMargin=margin, topMargin=margin,
                              bottomMargin=margin, title=self.doc_title, author=self.doc_author, subject=self.doc_subject,
                              creator="Cellula Report Generator Agent")
        doc.addPageTemplates([PageTemplate(id="main", frames=frames, onPage=decorate)])
        if not flow:
            flow = [Spacer(1, 1)]
        doc.build(flow)
        return doc.page

    # ---- complex flowables
    def _table_flow(self, e, frame_w, para, mk_style, text_c, accent_c, muted_c, Table, TableStyle, Spacer):
        from reportlab.lib.enums import TA_LEFT, TA_RIGHT

        t = self.theme
        fs = e.get("font_size") or max(7.5, t.body_size - 1)
        n = len(e["headers"])
        hdr_bg = parse_color(e.get("header_background") or t.table_header_background or t.accent_color)
        hdr_fg = parse_color(t.table_header_text)
        out = []
        if e.get("caption"):
            out.append(para(e["caption"], mk_style("cap", t.body_font, fs, muted_c, italic=True, spaceAfter=3)))
        if e.get("widths"):
            total = sum(e["widths"])
            widths = [frame_w * w / total for w in e["widths"]]
        else:  # proportional to the longest cell in each column (bounded so no column vanishes or dominates)
            weights = []
            for c in range(n):
                longest = max([len(e["headers"][c])] + [len(r[c]) for r in e["rows"]])
                weights.append(min(max(longest, 8), 40))
            total = sum(weights)
            widths = [frame_w * w / total for w in weights]
        header_style = mk_style("th", t.body_font, fs, hdr_fg, bold=True, leading=fs * 1.25)
        data = [[para(h, header_style) for h in e["headers"]]]
        for r in e["rows"]:
            row = []
            for c in r:
                numeric = bool(_NUMERIC.match(c)) and c.strip() != ""
                st = mk_style("td", t.body_font, fs, text_c, align=TA_RIGHT if numeric else TA_LEFT, leading=fs * 1.3)
                row.append(para(c, st))
            data.append(row)
        tbl = Table(data, colWidths=widths, repeatRows=1)
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), hdr_bg),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, parse_color("#D0D7DE")),
            ("BOX", (0, 0), (-1, -1), 0.6, parse_color("#C3CBD4")),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        if e.get("zebra", True) and len(data) > 2:
            zebra = parse_color(t.table_zebra)
            style.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [parse_color(t.page_background) if t.page_background else parse_color("#FFFFFF"), zebra]))
        tbl.setStyle(TableStyle(style))
        out.extend([tbl, Spacer(1, t.paragraph_spacing + 6)])
        return out

    def _chart_flow(self, e, frame_w, para, mk_style, text_c, accent_c, muted_c, KeepTogether, Spacer):
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        from reportlab.graphics.shapes import Drawing

        t = self.theme
        h = e["height"]
        d = Drawing(frame_w, h)
        chart = VerticalBarChart()
        chart.x, chart.y = 38, 28
        chart.width, chart.height = frame_w - 52, h - 44
        chart.data = [e["values"]]
        chart.categoryAxis.categoryNames = [self._latin(l, "Helvetica")[:24] for l in e["labels"]]
        chart.categoryAxis.labels.fontName = self._face("Helvetica")
        chart.categoryAxis.labels.fontSize = 8
        chart.categoryAxis.labels.fillColor = text_c
        if len(e["labels"]) > 6:
            chart.categoryAxis.labels.angle = 30
            chart.categoryAxis.labels.boxAnchor = "ne"
            chart.categoryAxis.labels.dy = -2
        chart.valueAxis.labels.fontName = self._face("Helvetica")
        chart.valueAxis.labels.fontSize = 8
        chart.valueAxis.labels.fillColor = text_c
        lo, hi = min(e["values"]), max(e["values"])
        chart.valueAxis.valueMin = min(0, lo)
        chart.valueAxis.valueMax = hi + (hi - min(0, lo)) * 0.15 if hi > 0 else 0
        chart.valueAxis.gridStrokeColor = parse_color("#D8DEE4")
        chart.valueAxis.visibleGrid = True
        chart.bars[0].fillColor = parse_color(e["color"]) if e.get("color") else accent_c
        chart.bars[0].strokeColor = None
        chart.barLabelFormat = f"%.{e['decimals']}f"
        chart.barLabels.fontName = self._face("Helvetica")
        chart.barLabels.fontSize = 8
        chart.barLabels.fillColor = text_c
        chart.barLabels.nudge = 7
        chart.barSpacing = 4
        chart.groupSpacing = 12
        d.add(chart)
        items = []
        if e["title"]:
            items.append(para(e["title"], mk_style("chart_t", t.heading_font, t.body_size + 1, text_c, bold=True, spaceAfter=3)))
        items.extend([d, Spacer(1, t.paragraph_spacing + 4)])
        return [KeepTogether(items)]

    # ------------------------------------------------------------ LangChain tools
    def tools(self):
        from langchain_core.tools import StructuredTool

        def tool(func, name, description, schema):
            return StructuredTool.from_function(func=func, name=name, description=description, args_schema=schema)

        return [
            tool(self.list_fonts, "list_fonts", "List the font families that can be used.", _NoArgs),
            tool(self.set_page_layout, "set_page_layout", "Page size, orientation, margins and number of text columns.", SetPageLayoutArgs),
            tool(self.set_fonts, "set_fonts", "Choose the body font and the heading font (call list_fonts to see the options).", SetFontsArgs),
            tool(self.set_font_sizes, "set_font_sizes", "Set body, title and heading (h1-h3) font sizes in points.", SetFontSizesArgs),
            tool(self.set_colors, "set_colors", "Set text, heading, title, accent, table and page-background colours (hex or names).", SetColorsArgs),
            tool(self.set_spacing, "set_spacing", "Line spacing, paragraph spacing, space before headings, justified text.", SetSpacingArgs),
            tool(self.set_header_footer, "set_header_footer", "Header text, footer text, page numbers, header rule.", SetHeaderFooterArgs),
            tool(self.set_document_info, "set_document_info", "PDF metadata: title, author, subject.", SetDocumentInfoArgs),
            tool(self.add_title_block, "add_title_block", "Add the report title block (title, subtitle, author, date).", AddTitleBlockArgs),
            tool(self.add_heading, "add_heading", "Add a heading (level 1-3) with optional colour/font/size/alignment overrides.", AddHeadingArgs),
            tool(self.add_paragraph, "add_paragraph", "Add a paragraph with full text formatting: bold, italic, underline, colour, font, size, alignment, highlight, indent, spacing. Inline **bold**, *italic*, `code` and {{#C0392B:coloured}} also work.", AddParagraphArgs),
            tool(self.add_bullets, "add_bullets", "Add a bullet or numbered list.", AddBulletsArgs),
            tool(self.add_table, "add_table", "Add a styled table (header row, zebra rows, column widths, caption).", AddTableArgs),
            tool(self.add_callout, "add_callout", "Add a highlighted box (info/success/warning/quote/custom) - good for the key answer or a warning.", AddCalloutArgs),
            tool(self.add_bar_chart, "add_bar_chart", "Add a bar chart of numbers that appear in the material (never invent values).", AddBarChartArgs),
            tool(self.add_markdown, "add_markdown", "Insert markdown (headings, lists, tables, bold) styled with the current theme. Best for placing the answer text verbatim.", AddMarkdownArgs),
            tool(self.add_divider, "add_divider", "Add a horizontal rule.", AddDividerArgs),
            tool(self.add_spacer, "add_spacer", "Add vertical space.", AddSpacerArgs),
            tool(self.add_page_break, "add_page_break", "Start a new page.", _NoArgs),
            tool(self.review_report, "review_report", "Show the current design and the outline of the report so far. Call it before finishing.", _NoArgs),
        ]
