from typing import List, Optional

def recursive_chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    separators: Optional[List[str]] = None
) -> List[str]:
    """
    Recursively splits text into chunks of at most `chunk_size` characters with `chunk_overlap`.
    Tries splitting on paragraphs (\n\n), newlines (\n), sentence endings (. ), spaces ( ), and characters.
    Preserves semantic blocks without cutting in the middle of sentences or words where possible.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    # Choose first separator that exists in text
    chosen_sep = ""
    for sep in separators:
        if sep == "" or sep in text:
            chosen_sep = sep
            break

    splits = text.split(chosen_sep) if chosen_sep else list(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for item in splits:
        item_len = len(item) + (len(chosen_sep) if current_chunk else 0)
        if current_length + item_len <= chunk_size:
            current_chunk.append(item)
            current_length += item_len
        else:
            if current_chunk:
                merged = chosen_sep.join(current_chunk).strip()
                if merged and (not chunks or merged != chunks[-1]):
                    chunks.append(merged)

                # Keep overlap items from the tail of current_chunk
                overlap_items = []
                overlap_len = 0
                for prev in reversed(current_chunk):
                    if overlap_len + len(prev) + len(chosen_sep) <= chunk_overlap:
                        overlap_items.insert(0, prev)
                        overlap_len += len(prev) + len(chosen_sep)
                    else:
                        break
                current_chunk = overlap_items
                current_length = sum(len(x) for x in current_chunk) + (len(chosen_sep) * max(0, len(current_chunk) - 1))

            # If a single split item is larger than chunk_size, split it with more granular separators
            if len(item) > chunk_size:
                sub_seps = separators[separators.index(chosen_sep) + 1:] if chosen_sep in separators else []
                if sub_seps:
                    sub_chunks = recursive_chunk_text(item, chunk_size, chunk_overlap, sub_seps)
                    for sc in sub_chunks:
                        if sc and (not chunks or sc != chunks[-1]):
                            chunks.append(sc)
                else:
                    # Character slice fallback
                    for start in range(0, len(item), chunk_size - chunk_overlap):
                        part = item[start : start + chunk_size].strip()
                        if part and (not chunks or part != chunks[-1]):
                            chunks.append(part)
            else:
                current_chunk.append(item)
                current_length += len(item)

    if current_chunk:
        merged = chosen_sep.join(current_chunk).strip()
        if merged and (not chunks or merged != chunks[-1]):
            chunks.append(merged)

    return chunks
