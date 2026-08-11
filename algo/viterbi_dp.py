"""
Viterbi DP N-best beam-search for dictionary-constrained Chinese P2C decoding.

Path score = sum(ln(probs)) + prior penalties for single-char vs multi-char word
transitions, modulated by hyperparameters beta_single and beta_word.
"""

import math


def viterbi_nbest(
    candidates: list[dict[str, float]],
    words_at: list[list[tuple[int, str, float]]],
    beta_single: float,
    beta_word: float,
    N: int,
) -> list[tuple[float, list[str]]]:
    """Viterbi DP beam-search for N-best segmentations.

    Path score:
      Single character:  score += log(p_char) + log(beta_single)
      Multi-char word:   score += sum(log(p_chars)) + log(beta_word)

    Args:
        candidates: list of {char -> prob} per position.
        words_at: precomputed list of (length, word_str, model_log_prob_sum)
                  starting at each position, from find_matching_words.
        beta_single: prior for single-char transitions (0 < beta_single <= 1).
        beta_word: prior for multi-char word transitions (0 < beta_word <= 1).
        N: beam width for top-N tracking.

    Returns:
        list of (score, word_list) sorted by score descending.
    """
    L = len(candidates)
    if L == 0:
        return [(0.0, [])]

    ln_beta_single = math.log(beta_single) if beta_single > 0 else float("-inf")
    ln_beta_word = math.log(beta_word) if beta_word > 0 else float("-inf")

    candidates_log = []
    for cand in candidates:
        cand_log = {
            ch: (math.log(prob) if prob > 0 else float("-inf"))
            for ch, prob in cand.items()
        }
        candidates_log.append(cand_log)

    # Use dict at each position to merge identical prefix strings: {prefix_str -> (score, node)}
    paths: list[dict[str, tuple[float, tuple | None]]] = [
        {} for _ in range(L + 1)
    ]
    paths[0][""] = (0.0, None)

    for start in range(L):
        if not paths[start]:
            continue

        # Prune beam at current position to top-N unique text prefixes
        if len(paths[start]) > N:
            sorted_items = sorted(
                paths[start].items(), key=lambda x: x[1][0], reverse=True
            )[:N]
            paths[start] = dict(sorted_items)

        current_paths = paths[start]
        cand_dict = candidates_log[start]

        for prefix_str, (prev_score, prev_node) in current_paths.items():
            for ch, log_prob in cand_dict.items():
                new_score = prev_score + log_prob + ln_beta_single
                next_idx = start + 1
                new_prefix = prefix_str + ch
                next_dict = paths[next_idx]
                new_node = (ch, prev_node)

                if (
                    new_prefix not in next_dict
                    or new_score > next_dict[new_prefix][0]
                ):
                    next_dict[new_prefix] = (new_score, new_node)

            for length, word_str, word_log_prob in words_at[start]:
                next_idx = start + length
                if next_idx > L:
                    continue
                new_score = prev_score + word_log_prob + ln_beta_word
                new_prefix = prefix_str + word_str
                next_dict = paths[next_idx]
                new_node = (word_str, prev_node)

                if (
                    new_prefix not in next_dict
                    or new_score > next_dict[new_prefix][0]
                ):
                    next_dict[new_prefix] = (new_score, new_node)

    results = []
    if paths[L]:
        sorted_final = sorted(
            paths[L].values(), key=lambda x: x[0], reverse=True
        )[:N]
        for score, node in sorted_final:
            words = []
            curr = node
            while curr is not None:
                words.append(curr[0])
                curr = curr[1]
            words.reverse()
            results.append((score, words))

    return results