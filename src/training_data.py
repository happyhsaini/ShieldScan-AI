from __future__ import annotations

import csv
import io
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .preprocess import clean_text, load_and_prepare_data

LABEL_TO_ID = {"Real": 0, "Suspicious": 1, "Fake": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _read_xlsx_rows(path: str | Path):
    """Read first-sheet cell values with the Python standard library (read-only)."""
    with zipfile.ZipFile(path) as book:
        workbook = ET.fromstring(book.read("xl/workbook.xml"))
        relationships = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
        rel_map = {item.attrib["Id"]: item.attrib["Target"] for item in relationships.findall(f"{{{rel_ns}}}Relationship")}
        shared = []
        if "xl/sharedStrings.xml" in book.namelist():
            strings = ET.fromstring(book.read("xl/sharedStrings.xml"))
            shared = ["".join(text.text or "" for text in item.findall(".//main:t", NS)) for item in strings.findall("main:si", NS)]
        sheet = workbook.find("main:sheets/main:sheet", NS)
        if sheet is None:
            return []
        target = rel_map[sheet.attrib[f"{{{NS['rel']}}}id"]].lstrip("/")
        sheet_path = target if target.startswith("xl/") else f"xl/{target}"
        root = ET.fromstring(book.read(sheet_path))
        rows = []
        for row in root.findall(".//main:sheetData/main:row", NS):
            values = {}
            for cell in row.findall("main:c", NS):
                address = cell.attrib.get("r", "")
                column = re.sub(r"\d", "", address)
                value_node = cell.find("main:v", NS)
                value = "" if value_node is None else value_node.text or ""
                if cell.attrib.get("t") == "s" and value:
                    value = shared[int(value)]
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(text.text or "" for text in cell.findall(".//main:t", NS))
                values[column] = value
            rows.append(values)
        return rows


def _label_from_source(label: str, category: str = "") -> str | None:
    label = str(label).strip().lower()
    category = str(category).strip().lower()
    if category == "word":
        # A lone signal word does not prove a scam; it belongs in the review class.
        return "Suspicious"
    if label in {"real", "safe", "ham", "legitimate", "genuine", "0"}:
        return "Real"
    if label in {"suspicious", "uncertain", "review"}:
        return "Suspicious"
    if label in {"fake", "spam", "phishing", "fraud", "scam", "1", "2"}:
        return "Fake"
    return None


def load_three_class_data(excel_path: str | Path, baseline_path: str | Path, supplemental_path: str | Path, archive_zip_path: str | Path | None = None) -> pd.DataFrame:
    rows: list[tuple[str, str]] = []

    baseline = load_and_prepare_data(str(baseline_path))
    for item in baseline.itertuples(index=False):
        label = "Real" if int(item.label) == 0 else "Fake"
        rows.append((str(item.message), label))

    if archive_zip_path:
        with zipfile.ZipFile(archive_zip_path) as archive:
            csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv") and not name.endswith("/")]
            if not csv_members:
                raise ValueError("The ZIP archive must contain a CSV training file.")
            member = next((name for name in csv_members if Path(name).name.lower() == "spam.csv"), csv_members[0])
            with archive.open(member) as raw_csv:
                source = io.TextIOWrapper(raw_csv, encoding="latin-1", newline="")
                for raw_item in csv.DictReader(source):
                    item = {str(key).lstrip("\ufeffï»¿").strip().lower(): value for key, value in raw_item.items() if key}
                    text = str(item.get("v2") or item.get("message") or item.get("text") or item.get("content") or "").strip()
                    label = _label_from_source(item.get("v1") or item.get("label") or item.get("class") or "")
                    if text and label:
                        rows.append((clean_text(text), label))

    source_rows = _read_xlsx_rows(excel_path)
    if source_rows:
        header = {str(value).strip().lower(): key for key, value in source_rows[0].items()}
        text_col = next((header[k] for k in ("content", "message", "text") if k in header), None)
        category_col = next((header[k] for k in ("category", "type") if k in header), None)
        label_col = next((header[k] for k in ("label", "class", "target") if k in header), None)
        if not text_col or not label_col:
            raise ValueError("The Excel sheet needs Content/Message and Label columns.")
        for item in source_rows[1:]:
            text = str(item.get(text_col, "")).strip()
            label = _label_from_source(item.get(label_col, ""), item.get(category_col, "") if category_col else "")
            if text and label:
                rows.append((clean_text(text), label))

    with open(supplemental_path, encoding="utf-8-sig", newline="") as source:
        for item in csv.DictReader(source):
            text = clean_text(item.get("message", ""))
            label = _label_from_source(item.get("label", ""))
            if text and label:
                rows.append((text, label))

    # Deduplicate exact normalized text. Conflicting source labels become review examples.
    labels_by_text: dict[str, set[str]] = defaultdict(set)
    for text, label in rows:
        if text:
            labels_by_text[text].add(label)
    normalized = [
        (text, next(iter(labels)) if len(labels) == 1 else "Suspicious")
        for text, labels in labels_by_text.items()
    ]
    frame = pd.DataFrame(normalized, columns=["message", "label_name"])
    frame["label"] = frame["label_name"].map(LABEL_TO_ID).astype(int)
    counts = frame["label_name"].value_counts().to_dict()
    print(f"Three-class rows after exact deduplication: {len(frame)} | " + " | ".join(f"{name}: {counts.get(name, 0)}" for name in LABEL_TO_ID))
    if set(frame["label"].unique()) != set(LABEL_TO_ID.values()):
        raise ValueError("Training requires Real, Suspicious, and Fake examples.")
    return frame
