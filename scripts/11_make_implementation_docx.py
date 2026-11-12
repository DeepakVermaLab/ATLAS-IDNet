from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DOCX = PROJECT_ROOT / "implementation" / "docs" / "ATLAS-IDNet_Implementation_Documentation.docx"

LINKS = {
    "Project Drive root": "https://drive.google.com/open?id=17bNFj_4v5KC74o53mLB0cdMR2Xv3omfD",
    "Implementation folder": "https://drive.google.com/open?id=1dbSKgEYjR0g3VZJ9aPb_sMlbvXsJccrs",
    "Results folder": "https://drive.google.com/open?id=1H63dXtNIPHFvr3ICpYjWL7NAILy9Y_2X",
    "Colab bundle ZIP": "https://drive.google.com/open?id=1dc8B98jfYb0z6G7EAIU9JD3faXoSo-dL",
    "Notebooks": "https://drive.google.com/open?id=1MiwKqhWDToEE6bDqskrjTgaKNw2ILiUV",
    "Scripts": "https://drive.google.com/open?id=1nrtW8plW1KxoyyT5W2j5dW77vtyFaTxm",
    "Source package": "https://drive.google.com/open?id=14kNSicp7E4LIVBTooCgUAooZU_wtXi4z",
    "Documentation folder": "https://drive.google.com/open?id=1PEJl3SctssdcEiBgiNGE6tdFcJmD7e4z",
    "Configs": "https://drive.google.com/open?id=1OQqVci8VcBh_kUOaTNQfa1uwgBC4sIXP",
    "Upload utilities": "https://drive.google.com/open?id=1H3xLBmMvQoAZeHGs1Tyd8vMgc6dc3Ok4",
    "Download utilities": "https://drive.google.com/open?id=19gFytQHO6F75x_XTzIlLRCdGFm4tins0",
    "Paper summary folder": "https://drive.google.com/open?id=1_6XtEw4vxMygBa5H6CTlwpCaDWEjCC-G",
    "IEEE tables folder": "https://drive.google.com/open?id=1Ls0mmzqMgfUqsAn-MiSNCYJp7NTcUTaY",
    "Figure: main metrics": "https://drive.google.com/open?id=10JWGmlyZTcl5XrQze8FXIsaPvimUF3go",
    "Figure: ablation delta weighted F1": "https://drive.google.com/open?id=1Jz1N1R7on1A-oBzt6m1qLmfGaOuuWxS_",
    "Figure: latency per sample": "https://drive.google.com/open?id=1L6zAGPE3JK1MGAMJGPkf6dIJ9JLBSYwJ",
    "Figure: throughput": "https://drive.google.com/open?id=1T8jUeJlczNvZCsPt1T9gxEM8Z7hrKi5b",
    "Figure: CICIoT2023 top features": "https://drive.google.com/open?id=1W67mRR5TRtfn9cExsCzQmMMMtVOCm5YM",
    "Figure: NF-UQ-NIDS-v2 top features": "https://drive.google.com/open?id=10ffqhj9jDtxrqWpWzfpRr3PawwIrm6Hi",
}


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in {"top": top, "start": start, "bottom": bottom, "end": end}.items():
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_table_width(table, widths: list[float]) -> None:
    for row in table.rows:
        for idx, width in enumerate(widths):
            row.cells[idx].width = Inches(width)
            tc_pr = row.cells[idx]._tc.get_or_add_tcPr()
            tc_w = tc_pr.first_child_found_in("w:tcW")
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(int(width * 1440)))
            tc_w.set(qn("w:type"), "dxa")


def prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def add_hyperlink(paragraph, text: str, url: str):
    part = paragraph.part
    r_id = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)
    new_run = OxmlElement("w:r")
    r_pr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    r_pr.append(color)
    r_pr.append(underline)
    new_run.append(r_pr)
    text_node = OxmlElement("w:t")
    text_node.text = text
    new_run.append(text_node)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)
    return hyperlink


def add_link_paragraph(doc: Document, label: str, url: str, prefix: str = "") -> None:
    p = doc.add_paragraph()
    if prefix:
        run = p.add_run(prefix)
        run.bold = True
    add_hyperlink(p, label, url)
    p.add_run(f" ({url})")


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = False
    set_table_width(table, widths)
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = header
        set_cell_shading(cell, "E8EEF5")
        set_cell_margins(cell)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9)
    prevent_row_split(table.rows[0])
    for row in rows:
        new_row = table.add_row()
        prevent_row_split(new_row)
        cells = new_row.cells
        for i, value in enumerate(row):
            cells[i].text = value
            set_cell_margins(cells[i], top=60, bottom=60)
            cells[i].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cells[i].paragraphs:
                p.paragraph_format.space_after = Pt(0)
                for r in p.runs:
                    r.font.size = Pt(8)
    doc.add_paragraph()


def style_document(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

    styles = doc.styles
    styles["Normal"].font.name = "Calibri"
    styles["Normal"].font.size = Pt(10.5)
    styles["Normal"].paragraph_format.space_after = Pt(6)
    styles["Normal"].paragraph_format.line_spacing = 1.1
    for name, size, color in [
        ("Heading 1", 15, "2E74B5"),
        ("Heading 2", 12.5, "2E74B5"),
        ("Heading 3", 11.5, "1F4D78"),
    ]:
        style = styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style.font.bold = True


def main() -> None:
    doc = Document()
    style_document(doc)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = title.add_run("ATLAS-IDNet Implementation Documentation")
    r.bold = True
    r.font.size = Pt(22)
    r.font.color.rgb = RGBColor.from_string("0B2545")

    subtitle = doc.add_paragraph()
    subtitle.add_run("Public Drive handover, execution map, code inventory, and result-artifact locations.").italic = True
    doc.add_paragraph(
        "This document records the current implementation package for ATLAS-IDNet. "
        "The Drive links were generated from the uploaded Google Drive copy, and the implementation folder was verified with rclone check: 0 differences found across the active code set."
    )

    doc.add_heading("1. Public Project Links", level=1)
    for key in [
        "Project Drive root",
        "Implementation folder",
        "Results folder",
        "Colab bundle ZIP",
        "Notebooks",
        "Scripts",
        "Source package",
        "Documentation folder",
        "Configs",
        "Upload utilities",
        "Download utilities",
    ]:
        add_link_paragraph(doc, key, LINKS[key])

    doc.add_heading("2. Execution Order and Notebook Purpose", level=1)
    add_table(
        doc,
        ["Step", "Notebook", "Purpose", "Primary output location"],
        [
            ["00", "implementation/notebooks/00_setup_and_environment.ipynb", "Checks runtime, Python imports, GPU/CPU visibility, dataset visibility, and run metadata.", "implementation/results/scaffold_environment_check/ and results/atlas_idnet_main_run/environment_check.json"],
            ["01", "implementation/notebooks/01_eda.ipynb", "Builds dataset-level EDA summaries, label distributions, missing-value summaries, numeric descriptions, and variance plots.", "results/atlas_idnet_main_run/<dataset>/eda/"],
            ["02", "implementation/notebooks/02_preprocess.ipynb", "Creates encoded labels, scaled numeric matrices, train/validation/test splits, metadata, and validation files.", "results/atlas_idnet_main_run/<dataset>/multiclass/preprocessed/"],
            ["03", "implementation/notebooks/03_train_and_evaluate.ipynb", "Trains and evaluates ATLAS-IDNet for multiclass intrusion detection and writes checkpoint, predictions, confusion matrix, and metric files.", "results/atlas_idnet_main_run/<dataset>/multiclass/models/atlas/"],
            ["04", "implementation/notebooks/04_ablation.ipynb", "Runs ATLAS-IDNet component ablations and stores comparative weighted-F1 evidence.", "results/atlas_idnet_main_run/<dataset>/multiclass/ablation/"],
            ["05", "implementation/notebooks/05_latency.ipynb", "Profiles inference latency and throughput across batch settings.", "results/atlas_idnet_main_run/<dataset>/multiclass/latency/"],
            ["06", "implementation/notebooks/06_explainability_agent.ipynb", "Produces feature attribution, branch-attention evidence, and evidence-grounded agentic reports.", "results/atlas_idnet_main_run/<dataset>/multiclass/explainability/ and agentic/"],
            ["08", "implementation/notebooks/08_paper_results_summary.ipynb", "Aggregates paper-ready summary tables and figures from completed runs.", "results/atlas_idnet_main_run/paper_summary/"],
            ["09", "implementation/notebooks/09_ieee_access_results_tables.ipynb", "Exports IEEE Access-ready CSV and LaTeX tables plus narrative result notes.", "results/atlas_idnet_main_run/paper_summary/ieee_access_tables/"],
        ],
        [0.45, 1.95, 2.55, 2.15],
    )

    doc.add_heading("3. Code Structure", level=1)
    add_table(
        doc,
        ["Path", "Contains", "Use in pipeline"],
        [
            ["implementation/src/atlas_idnet/config.py", "Dataset paths, output roots, runtime configuration, and task settings.", "Shared configuration across all stages."],
            ["implementation/src/atlas_idnet/data/", "Dataset discovery, EDA, preprocessing, scaling, label encoding, split validation.", "Input preparation and reproducibility checks."],
            ["implementation/src/atlas_idnet/models/multimodal.py", "ATLAS-IDNet multi-branch architecture, branch encoders, fusion, and classifier head.", "Main proposed model implementation."],
            ["implementation/src/atlas_idnet/training/", "Training loop, metrics, ablation routines, latency profiling, and checkpoint handling.", "Evaluation, ablation, and runtime measurement."],
            ["implementation/src/atlas_idnet/explainability/", "Permutation feature importance, attention summaries, and agentic evidence report generation.", "Explainability and analyst-facing evidence."],
            ["implementation/src/atlas_idnet/reporting/", "Paper summary, IEEE Access tables, and figure-generation utilities.", "Publication-ready output assembly."],
            ["implementation/scripts/", "Command-line wrappers for each notebook stage and reporting stage.", "Allows notebook and script execution with the same Python modules."],
            ["implementation/download/", "Dataset-specific download notebooks and helper configuration.", "Dataset acquisition."],
            ["implementation/upload/", "Drive upload helpers and local dataset status notes.", "Dataset/result transfer support."],
        ],
        [2.2, 2.7, 2.2],
    )

    doc.add_heading("4. Result and Artifact Locations", level=1)
    add_link_paragraph(doc, "Main result run folder", LINKS["Results folder"])
    add_link_paragraph(doc, "Paper summary folder", LINKS["Paper summary folder"])
    add_link_paragraph(doc, "IEEE Access result tables", LINKS["IEEE tables folder"])
    add_table(
        doc,
        ["Artifact group", "Representative files", "Purpose"],
        [
            ["Run metadata and logs", "config.json, run_metadata.json, runtime_provenance.json, logs/*.log", "Records runtime state, configuration, and stage execution logs."],
            ["EDA", "<dataset>_eda_summary.json, label_counts.csv, label_distribution.png, missing_top40.csv, numeric_describe.csv", "Supports dataset characterization and class-distribution reporting."],
            ["Preprocessing", "prepared.npz, scaler.joblib, label_encoder.joblib, prepared_metadata.json, prepared_validation.json", "Stores reusable train/validation/test matrices and preprocessing provenance."],
            ["Model outputs", "atlas_idnet.pt, test_metrics.json, classification_report.csv, confusion_matrix.csv, per_class_detection_metrics.csv, test_predictions.csv, training_history.csv", "Stores checkpoint and all standard evaluation evidence for each dataset."],
            ["Ablation", "ablation_results.csv, atlas_branch_ablation_results.csv", "Quantifies contribution of architecture branches and fusion components."],
            ["Latency", "latency.json, latency_by_batch.csv", "Reports real-time detection latency and throughput."],
            ["Explainability", "atlas_permutation_importance.csv/png, atlas_attention_summary.csv/png, atlas_feature_group_importance.csv", "Stores feature and branch-level interpretability outputs."],
            ["Agentic analysis", "agentic_explanation.json, agentic_explanation_report.md", "Stores evidence-grounded analyst report generated from metrics, attribution, attention, and latency payloads."],
            ["Paper summary", "table_main_atlas_metrics.csv, table_latency_realtime.csv, table_top_atlas_features.csv, figure_*.png", "Provides paper-ready tables and graphs for the manuscript."],
        ],
        [1.55, 3.0, 2.6],
    )

    doc.add_heading("5. Direct Result Figure Links", level=1)
    for key in [
        "Figure: main metrics",
        "Figure: ablation delta weighted F1",
        "Figure: latency per sample",
        "Figure: throughput",
        "Figure: CICIoT2023 top features",
        "Figure: NF-UQ-NIDS-v2 top features",
    ]:
        add_link_paragraph(doc, key, LINKS[key])

    doc.add_heading("6. Verification Notes", level=1)
    doc.add_paragraph(
        "The uploaded active implementation copy was checked from local implementation/ to the Drive implementation folder with sidecar and cache exclusions. "
        "The verification result was 0 differences found and 66 matching files. Generated Python caches, AppleDouble sidecars, .DS_Store files, and .pyc files are intentionally excluded."
    )
    doc.add_paragraph(
        "The Drive result run contains EDA outputs, preprocessing validation files, trained ATLAS-IDNet checkpoints, test predictions, confusion matrices, per-class metrics, ablation tables, latency profiles, explainability artifacts, agentic reports, paper summary graphs, and IEEE Access tables."
    )

    OUTPUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT_DOCX)
    print(OUTPUT_DOCX)


if __name__ == "__main__":
    main()
