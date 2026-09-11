"""Join the two saved bootstrap PDFs without rerunning simulations."""

from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter, Transformation

from paper_figure_export import copy_pdf_to_paper


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCES = (
    "figures/NGRIP_MIS6_PI_bootstrap/NGRIP_MIS6_PI_bootstrap.pdf",
    "Barker2011/figures/Barker2011_PI_bootstrap/Barker2011_PI_bootstrap.pdf",
)


def build_figure(project_root=PROJECT_ROOT):
    """Place NGRIP–MIS6 on the left and Barker on the right; sync the paper copy."""
    project_root = Path(project_root)
    pages = [PdfReader(project_root / source).pages[0] for source in SOURCES]
    width, gap, heading_height = 165 / 25.4 * 72, 8, 17  # PDF coordinates are points.
    panel_width = (width - gap) / 2
    scales = [panel_width / float(page.mediabox.width) for page in pages]
    heights = [float(page.mediabox.height) * scale for page, scale in zip(pages, scales)]
    height = max(heights) + heading_height
    writer = PdfWriter()
    joined = writer.add_blank_page(width=width, height=height)
    for index, (page, scale) in enumerate(zip(pages, scales)):
        joined.merge_transformed_page(page, Transformation().scale(scale).translate(
            tx=index * (panel_width + gap), ty=0))

    # Add compact catalogue headings in a vector overlay above the source pages.
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "pdf.fonttype": 42}):
        overlay = plt.figure(figsize=(width / 72, height / 72))
        overlay.patch.set_alpha(0)
        for index, heading in enumerate(("(a)  NGRIP–MIS6", "(b)  Barker 2011")):
            overlay.text((index * (panel_width + gap) + 3) / width,
                         (height - 10) / height, heading, fontsize=8, weight="bold")
        stream = BytesIO()
        overlay.savefig(stream, format="pdf", transparent=True)
        plt.close(overlay)
    joined.merge_page(PdfReader(stream).pages[0])

    output = project_root / "figures/Figure_PI_bootstrap/Figure_PI_bootstrap.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        writer.write(handle)
    with pdfium.PdfDocument(output) as document:
        document[0].render(scale=600 / 72).to_pil().save(output.with_suffix(".png"))
    copy_pdf_to_paper(output, project_root)
    return output


if __name__ == "__main__":
    print(build_figure())
