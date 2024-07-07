
def decompose_id(identifier):
    components = identifier.split(":")
    assert all(len(c) >= 2 for c in components)
    if all(c[0] == "[" and c[-1] == "]" for c in components):
        components = [c[1:-1] for c in components]
    return components
