"""WN18RR text identities for train-only false-negative filtering."""
from pathlib import Path
import re
from .fb_pairs import FB_PAIR_SPAN_RE


def load_wn_entity_text(raw_dir: Path) -> dict[str, str]:
    entities = {}
    for line in (raw_dir / 'entity2text.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        fields = line.strip().split('\t')
        if len(fields) < 2:
            raise ValueError('Malformed WN entity mapping')
        entities[fields[0]] = fields[1]
    for line in (raw_dir / 'train.txt').read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if len(fields) == 3:
            for entity in (fields[0], fields[2]):
                entities.setdefault(entity, 'Unknown')
    return entities


def wn_question_entity_pairs(text: str, entity_texts: set[str]):
    # Generated context relation questions use the shared "relation between"
    # template, possibly with node prefixes; resolve every split against known
    # full entity texts so spaces and embedded "and" remain unambiguous.
    matches = list(FB_PAIR_SPAN_RE.finditer(text))
    if not matches:
        return []
    span = matches[-1].group(1)
    def identities(value):
        value = value.strip()
        candidates = {value}
        for prefix in ('word node ', 'node '):
            if value.lower().startswith(prefix):
                candidates.add(value[len(prefix):].strip())
        return candidates & entity_texts
    pairs = set()
    for match in re.finditer(r' and ', span):
        for head in identities(span[:match.start()]):
            for tail in identities(span[match.end():]):
                pairs.add((head, tail))
    return sorted(pairs)
