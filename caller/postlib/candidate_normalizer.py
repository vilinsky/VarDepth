def normalize_biallelic(start, ref, alt, reference, limit=100000):
    """Canonical bounds for search only; original truth edits remain unchanged."""
    ref, alt = ref.upper(), alt.upper()
    if not ref or not alt or any(b not in 'ACGT' for b in ref + alt):
        raise ValueError('noncanonical_normalization')
    if reference[start:start + len(ref)] != ref:
        raise ValueError('normalization_ref_mismatch')
    while len(ref) > 1 and len(alt) > 1 and ref[-1] == alt[-1]:
        ref, alt = ref[:-1], alt[:-1]
    shifts = 0
    if len(ref) != len(alt):
        while start > 0 and ref[-1] == alt[-1]:
            if shifts >= limit:
                raise ValueError('normalization_limit')
            previous = reference[start - 1]
            if previous not in 'ACGT':
                raise ValueError('noncanonical_normalization')
            ref, alt = previous + ref[:-1], previous + alt[:-1]
            start -= 1
            shifts += 1
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref, alt = ref[1:], alt[1:]
        start += 1
    return start, ref, alt
