"""Combine the saved LR04 interaction PDFs without rerunning any fits."""

from pathlib import Path

import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter, Transformation

from paper_figure_export import copy_pdf_to_paper


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCES = (
    "figures/NGRIP_MIS6_climate_phase_interaction/NGRIP_MIS6_climate_phase_interaction.pdf",
    "Barker2011/figures/Barker2011_climate_phase_interaction/Barker2011_climate_phase_interaction.pdf",
)


def build_figure(project_root=PROJECT_ROOT):
    """Stack NGRIP–MIS6 (a, b) above Barker (c, d), then sync the paper copy."""
    project_root = Path(project_root)
    pages = [PdfReader(project_root / source).pages[0] for source in SOURCES]
    width, gap = 165 / 25.4 * 72, 2  # Preserve readable text at publication width.
    scales = [width / float(page.mediabox.width) for page in pages]
    heights = [float(page.mediabox.height) * scale for page, scale in zip(pages, scales)]
    writer = PdfWriter()
    joined = writer.add_blank_page(width=width, height=sum(heights) + gap)
    for page, scale, bottom in zip(pages, scales, (heights[1] + gap, 0)):
        joined.merge_transformed_page(page, Transformation().scale(scale).translate(ty=bottom))

    output = project_root / "figures/Figure_climate_phase_interaction/Figure_climate_phase_interaction.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        writer.write(handle)
    with pdfium.PdfDocument(output) as document:
        document[0].render(scale=600 / 72).to_pil().save(output.with_suffix(".png"))
    copy_pdf_to_paper(output, project_root)
    return output


if __name__ == "__main__":
    print(build_figure())
