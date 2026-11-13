def add_item(inventory: dict[str, int], name: str, quantity: int):
    """Add quantity to name."""
    inventory[name] = inventory.get(name, 0) + quantity
    return inventory


def remove_item(inventory: dict[str, int], name: str, quantity: int):
    """Take quantity away from name."""
    if name in inventory:
        inventory[name] -= quantity
        if inventory[name] <= 0:
            del inventory[name]
    return inventory


def total_items(inventory: dict[str, int]):
    """Sum of the quantities."""
    return sum(inventory.values())
