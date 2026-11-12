def add_item(inventory, name, quantity):
    new = dict(inventory)
    new[name] = new.get(name, 0) + quantity
    return new


def remove_item(inventory, name, quantity):
    new = dict(inventory)
    if name in new:
        new[name] = max(new[name] - quantity, 0)
    return new


def total_items(inventory):
    return sum(inventory.values())
