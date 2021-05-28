def cal_stats(sentences: list[str], labels: list[int], tokenizer) :
    r"""
    It calculates stats and returns stats information and list of the information of sentences and tokenized sentences to check.

    :param sentences: list of raw sentences before tokenization.
    :param labels: list of labels
    :param tokenizer: transformers tokenizer
    :return: stats, sents_tokens_to_check
    """
    total_sents = len(sentences)
    sents_tokens_to_check = []
    label_count = {}
    max_sent_len = 0
    min_sent_len = 999
    n_unk = 0
    total_tokens = 0
    for sentence, label in zip(sentences, labels):
        tokens = tokenizer.tokenize(sentence)

        label_count[label] = label_count.get(label, 0) + 1

        sentence_len = len(tokens)

        max_sent_len = max(max_sent_len, sentence_len)
        min_sent_len = min(min_sent_len, sentence_len)

        total_tokens += sentence_len

        sents_tokens_to_check.append(f"*{sentence_len}token {sentence} | {tokens}\n")
        sents_tokens_to_check.append(f"{'===' * 20}\n")
        if '[UNK]' in tokens:
            n_unk += 1

    stats = f"About sentences\n" \
            f"| total: {total_sents}" \
            f"| max: {max_sent_len}\n" \
            f"| min: {min_sent_len}\n" \
            f"| mean: {round(total_tokens / total_sents, 1)}\n" \
            f"About tokens\n" \
            f"| total tokens: {total_tokens}\n" \
            f"| total unk tokens: {n_unk}({round(n_unk / total_tokens * 100, 1)}%)\n" \
            f"{', '.join([f'label{label}: {label_count.get(label)}({round(label_count.get(label) / total_sents * 100, 1)}%)'for label in label_count.keys()])}"

    return stats, sents_tokens_to_check
