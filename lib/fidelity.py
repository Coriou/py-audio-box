"""Model-free text fidelity, chunk planning and strict WAV assembly contracts."""
import re
import unicodedata
from pathlib import Path


def normalize_text(text):
    """NFC + casefold; ignore punctuation/whitespace. Keep digits, accents and quantities.

    No number expansion, accent folding or semantic rewriting is performed.
    """
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold()))


def sequence_errors(reference, hypothesis):
    """Levenshtein alignment with deterministic substitution/deletion/insertion counts."""
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, hyp in enumerate(hypothesis, 1):
            cost, sub, delete, insert = previous[j - 1]
            diagonal = (cost + (ref != hyp), sub + (ref != hyp), delete, insert)
            cost, sub, delete, insert = previous[j]
            deletion = (cost + 1, sub, delete + 1, insert)
            cost, sub, delete, insert = current[j - 1]
            insertion = (cost + 1, sub, delete, insert + 1)
            current.append(min((diagonal, deletion, insertion), key=lambda x: x[0]))
        previous = current
    distance, substitutions, deletions, insertions = previous[-1]
    return dict(errors=distance, substitutions=substitutions, deletions=deletions,
                insertions=insertions, reference_length=len(reference),
                rate=distance / max(1, len(reference)))


def transcript_metrics(reference, transcript):
    ref, hyp = normalize_text(reference), normalize_text(transcript)
    wer = sequence_errors(ref.split(), hyp.split())
    cer = sequence_errors(ref, hyp)
    alerts = []
    if not hyp:
        alerts.append('empty_transcript')
    # Flag adjacent repeated phrases newly introduced by ASR/output, not script repetition.
    words = hyp.split()
    for size in range(1, min(8, len(words) // 2) + 1):
        for i in range(len(words) - 2 * size + 1):
            phrase = words[i:i + size]
            if phrase == words[i + size:i + 2 * size] and ' '.join(phrase * 2) not in ref:
                alerts.append('unexpected_repetition')
                break
        if 'unexpected_repetition' in alerts:
            break
    return dict(transcript=transcript, raw_transcript=transcript, normalized_reference=ref,
                normalized_transcript=hyp, normalization='nfc-casefold-word-tokens-v1',
                wer=wer['rate'], cer=cer['rate'], word_edits=wer, character_edits=cer,
                intelligibility=round(max(0.0, 1.0 - wer['rate']), 4), alerts=alerts)


def plan_chunks(text, max_chars=200, sentence_pause_ms=300, paragraph_pause_ms=600,
                continuation_pause_ms=100):
    """Exact contiguous original-text spans with hard character limits and pauses.

    Prefer paragraph/sentence boundaries, then spaces. Overlong tokens use a hard
    character boundary; no characters are added, lost or reordered.
    """
    if max_chars < 1 or min(sentence_pause_ms, paragraph_pause_ms, continuation_pause_ms) < 0:
        raise ValueError('chunk limit must be positive and pauses non-negative')
    chunks, start = [], 0
    while start < len(text):
        limit = min(start + max_chars, len(text))
        window = text[start:limit]
        boundaries = list(re.finditer(r'\n\s*\n|[.!?](?:\s+|$)', window))
        if limit == len(text):
            end = limit
        elif boundaries:
            end = start + boundaries[-1].end()
        else:
            spaces = list(re.finditer(r'\s+', window))
            end = start + spaces[-1].end() if spaces else limit
        span = text[start:end]
        if re.search(r'\n\s*\n$', span):
            pause = paragraph_pause_ms
        elif re.search(r'[.!?]\s*$', span):
            pause = sentence_pause_ms
        else:
            pause = continuation_pause_ms
        chunks.append(dict(text=span, start=start, end=end, pause_after_ms=pause))
        start = end
    if chunks:
        chunks[-1]['pause_after_ms'] = 0
    return chunks


def concatenate_wavs(paths, dest, sample_rate=24000, silence_ms=300, pauses_ms=None):
    """Reject mixed rates/channels rather than relabel samples or silently downmix."""
    import numpy as np
    import soundfile as sf
    if not paths or sample_rate <= 0 or silence_ms < 0:
        raise ValueError('nonempty WAV inputs, positive rate and nonnegative silence required')
    if pauses_ms is None:
        pauses_ms = [silence_ms] * (len(paths) - 1)
    if len(pauses_ms) != len(paths) - 1 or any(p < 0 for p in pauses_ms):
        raise ValueError('one nonnegative pause is required per WAV boundary')
    chunks, channels = [], None
    for i, path in enumerate(paths):
        data, rate = sf.read(str(path), dtype='float32', always_2d=True)
        if rate != sample_rate:
            raise ValueError(f'WAV sample rate {rate} does not match assembly rate {sample_rate}; resample explicitly')
        if channels is None:
            channels = data.shape[1]
        elif channels != data.shape[1]:
            raise ValueError('WAV channel counts must match; convert explicitly')
        if not np.isfinite(data).all():
            raise ValueError('WAV contains nonfinite samples')
        if i:
            chunks.append(np.zeros((round(sample_rate * pauses_ms[i-1] / 1000), channels), dtype=np.float32))
        chunks.append(data)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), np.concatenate(chunks, axis=0), sample_rate)
