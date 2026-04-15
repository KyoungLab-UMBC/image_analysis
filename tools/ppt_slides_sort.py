from pptx import Presentation

def reorder_slides_by_groups(ppt_path, output_path, n, b):
    """
    n: slides per group
    b: number of groups
    """

    prs = Presentation(ppt_path)
    slides = prs.slides

    total_slides = n * b
    if len(slides) < total_slides:
        raise ValueError("PPT has fewer slides than n × b")

    # Build new order (0-based indices)
    new_order = []
    for i in range(n):
        for j in range(b):
            new_order.append(j * n + i)

    # Reorder slides using internal API
    sldIdLst = slides._sldIdLst
    reordered = [sldIdLst[i] for i in new_order]

    sldIdLst.clear()
    for slide in reordered:
        sldIdLst.append(slide)

    prs.save(output_path)


# =====================
# Example usage
# =====================

ppt_input = r"D:\VS project\Plate 2 - 140 mM.pptx"
ppt_output = r"D:\VS project\Plate 2 - 140 mM_rearranged.pptx"

n = 27  # slides per group
b = 5  # number of groups

reorder_slides_by_groups(ppt_input, ppt_output, n, b)
