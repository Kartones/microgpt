
def unique_chars(docs: list[str]) -> list[str]:
    # uchars: sorted list of all unique characters that appear in the dataset.
    # For a names dataset this will be ['a', 'b', 'c', ..., 'z'] — 26 characters.
    # Sorted so the mapping is deterministic (not dependent on dict insertion order).
    return sorted(set(''.join(docs)))

def get_BOS_token_id(uchars: list[str]) -> int:
    # BOS: "Beginning of Sequence" — a special token with no character equivalent.
    # It serves dual purpose:
    #   - Placed at the START of a sequence to signal "predict the first character"
    #   - Placed at the END of a sequence to signal "stop generating"
    # By reusing the same token for both, the model learns: "when I see BOS, the
    # next thing I predict is the first character; when I predict BOS, I'm done."
    # The BOS token ID is always one past the last character token.
    return len(uchars)

def get_vocabulary_size(uchars: list[str]) -> int:
    # Vocabulary size = number of unique characters + 1 for BOS token.
    return len(uchars) + 1
