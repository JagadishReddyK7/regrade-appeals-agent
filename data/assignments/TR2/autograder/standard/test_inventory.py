from inventory import add_item, remove_item, total_items


def test_add_new():
    assert add_item({}, "apple", 3) == {"apple": 3}


def test_add_existing():
    assert add_item({"apple": 2}, "apple", 3) == {"apple": 5}


def test_remove_partial():
    assert remove_item({"apple": 5}, "apple", 2) == {"apple": 3}


def test_remove_all():
    assert remove_item({"apple": 2}, "apple", 5) == {}


def test_remove_missing():
    assert remove_item({"apple": 2}, "pear", 1) == {"apple": 2}


def test_total():
    assert total_items({"apple": 2, "pear": 3}) == 5
