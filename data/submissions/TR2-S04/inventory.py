def add_item(inventory: dict[str, int], name: str, quantity: int) -> dict[str, int]:
    """Return a copy of inventory with quantity added to name."""
    updated = {**inventory}
    updated[name] = updated.get(name, 0) + quantity
    return updated


def remove_item(inventory: dict[str, int], name: str, quantity: int) -> dict[str, int]:
    """Return a copy of inventory with quantity removed from name."""
    updated = {**inventory}
    if name in updated:
        remaining = updated[name] - quantity
        if remaining > 0:
            updated[name] = remaining
        else:
            updated.pop(name)
    return updated


def total_items(inventory: dict[str, int]) -> int:
    """Return the total quantity of all items."""
    total = 0
    for quantity in inventory.values():
        total += quantity
    return total
