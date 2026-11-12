def add_item(inventory: dict[str, int], name: str, quantity: int) -> dict[str, int]:
    """Return a new inventory where name has the given quantity."""
    result = dict(inventory)
    result[name] = quantity
    return result


def remove_item(inventory: dict[str, int], name: str, quantity: int) -> dict[str, int]:
    """Return a new inventory with quantity removed from name."""
    result = dict(inventory)
    if name in result:
        result[name] -= quantity
        if result[name] <= 0:
            del result[name]
    return result


def total_items(inventory: dict[str, int]) -> int:
    """Return the sum of the quantities."""
    return sum(inventory.values())
