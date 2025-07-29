def validate_input(location, budget):
    if not location or not location.strip():
        return False
    if budget < 0:
        return False
    return True