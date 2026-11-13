def add_item(inventory: dict[str, int], name: str, quantity: int) -> dict[str, int]:
    """Return a new inventory with quantity added to name."""
    result = dict(inventory)
    result[name] = result.get(name, 0) + quantity
    return result


def remove_item(inventory: dict[str, int], name: str, quantity: int) -> dict[str, int]:
    """Return a new inventory with quantity taken from name; drop it at 0 or less."""
    result = dict(inventory)
    if name not in result:
        return result
    left = result[name] - quantity
    if left <= 0:
        del result[name]
    else:
        result[name] = left
    return result


def total_items(inventory: dict[str, int]) -> int:
    """Return the sum of all the quantities."""
    return sum(inventory.values())
