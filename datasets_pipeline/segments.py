"""Shared segmentation / span-selection helpers."""

from datasets_pipeline.constants import CHINESE_LABEL


def select_span(labels: list[int], k: int, direction: str = "backward"):
    """1-indexed Chinese-segment choice *k* -> (sel, end, chinese_offset).

    prefix = segments[0:sel]; suffix = segments[sel:end] (consecutive Chinese);
    the k-th Chinese segment (1-indexed) is the anchor.

    Args:
        labels: List of segment labels.
        k: The k-th Chinese segment (1-indexed).
        direction: Search direction, "backward" (extend to the right),
                   "forward" (extend to the left), or "bidirectional".
    """
    chinese_positions = [i for i, l in enumerate(labels) if l == CHINESE_LABEL]
    anchor = chinese_positions[k - 1]

    if direction == "backward":
        sel = anchor
        end = sel
        while end < len(labels) and labels[end] == CHINESE_LABEL:
            end += 1
        chinese_offset = k - 1

    elif direction == "forward":
        end = anchor + 1
        sel = anchor
        while sel >= 0 and labels[sel] == CHINESE_LABEL:
            sel -= 1
        sel += 1

        length = end - sel
        chinese_offset = (k - 1) - (length - 1)

    elif direction == "bidirectional":
        sel = anchor
        while sel >= 0 and labels[sel] == CHINESE_LABEL:
            sel -= 1
        sel += 1

        end = anchor
        while end < len(labels) and labels[end] == CHINESE_LABEL:
            end += 1

        chinese_offset = (k - 1) - (anchor - sel)

    else:
        raise ValueError(f"Unknown direction: {direction}")

    return sel, end, chinese_offset


def chinese_segment_char_prefixes(text_list: list[str], labels: list[int]) -> list[int]:
    """Cumulative character counts over the *Chinese* segments.

    Returns a list of length ``len(labels) + 1`` where ``prefix[i]`` is the
    number of characters covered by Chinese segments strictly before segment
    ``i`` (and ``prefix[-1]`` is the total).  This maps Chinese-segment
    indices to offsets inside the flat per-character pinyin list.
    """
    prefix = [0] * (len(labels) + 1)
    running = 0
    for i, (seg, lab) in enumerate(zip(text_list, labels)):
        prefix[i] = running
        if lab == CHINESE_LABEL:
            running += len(seg)
    prefix[len(labels)] = running
    return prefix
