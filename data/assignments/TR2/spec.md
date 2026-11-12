# TR2: Inventory dictionary

Write a module `inventory.py`. An inventory is a dictionary from item names to positive
whole quantities. Every function returns a **new** dictionary (or a number) and must not
change the dictionary it is given.

## `add_item(inventory: dict[str, int], name: str, quantity: int) -> dict[str, int]`
Return a new inventory with `quantity` added to `name`. A new name is inserted with that
quantity; for a name already present the quantities are added together.

## `remove_item(inventory: dict[str, int], name: str, quantity: int) -> dict[str, int]`
Return a new inventory with `quantity` taken away from `name`. If the quantity reaches 0
or less, `name` is removed from the inventory. For a name that is not in the inventory,
see the announcements.

## `total_items(inventory: dict[str, int]) -> int`
Return the sum of all quantities.

## Marking (30 marks)
- Autograder tests: 24 marks.
- Type hints (2 marks): every function has type hints for its parameters and return value.
- Approach (4 marks): no function modifies the dictionary it is given.

## Late policy
Late submissions lose 10 marks. Extensions are decided by the course staff on request.
