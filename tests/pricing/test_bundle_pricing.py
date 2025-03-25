import pytest
from pypulate.pricing.bundle_pricing import calculate_bundle_price


def test_calculate_bundle_price_provided_example():
    # Provided example from the docstring
    items = ['item1', 'item2', 'item3']
    item_prices = {'item1': 10, 'item2': 20, 'item3': 30}
    bundle_discounts = {'item1+item2': 0.10, 'item1+item2+item3': 0.20}
    # Total price = 10+20+30 = 60, discount of 20% -> expected price = 60 - 12 = 48
    result = calculate_bundle_price(items, item_prices, bundle_discounts)
    assert result == 48.0


def test_calculate_bundle_price_no_discount_due_to_minimum():
    # Only one item provided, so discount is not applied due to minimum_bundle_size of 2
    items = ['item1']
    item_prices = {'item1': 10, 'item2': 20}
    bundle_discounts = {'item1+item2': 0.15}
    result = calculate_bundle_price(items, item_prices, bundle_discounts)
    assert result == 10.0


def test_calculate_bundle_price_no_matching_discount():
    # Items do not satisfy any bundle discount key
    items = ['item1', 'item2']
    item_prices = {'item1': 10, 'item2': 20, 'item3': 30}
    bundle_discounts = {'item1+item3': 0.25}
    result = calculate_bundle_price(items, item_prices, bundle_discounts)
    # Expected price is simply 10 + 20 = 30
    assert result == 30.0


def test_calculate_bundle_price_multiple_items_with_extra():
    # Test with extra item; discount applies only if the bundle key's items are present.
    items = ['item1', 'item2', 'item3', 'item4']
    item_prices = {'item1': 5, 'item2': 10, 'item3': 15, 'item4': 20}
    bundle_discounts = {'item1+item2+item3': 0.10}
    # Total price = 5+10+15+20 = 50, discount 10% = 5, expected = 45
    result = calculate_bundle_price(items, item_prices, bundle_discounts)
    assert result == 45.0


def test_calculate_bundle_price_duplicate_items():
    # Test when items appear multiple times
    items = ['item1', 'item1', 'item2']
    item_prices = {'item1': 10, 'item2': 20}
    bundle_discounts = {'item1+item2': 0.10}
    # Total = 10*2 + 20 = 40, discount 10% = 4, expected = 36
    result = calculate_bundle_price(items, item_prices, bundle_discounts)
    assert result == 36.0


def test_calculate_bundle_price_custom_minimum_bundle_size():
    # With a custom minimum_bundle_size, discount should not apply if item count is below it.
    items = ['item1', 'item2']
    item_prices = {'item1': 10, 'item2': 20}
    bundle_discounts = {'item1+item2': 0.20}
    # minimum_bundle_size set to 3, so discount is not applied; expected total = 30
    result = calculate_bundle_price(items, item_prices, bundle_discounts, minimum_bundle_size=3)
    assert result == 30.0


def test_calculate_bundle_price_empty_items():
    # With no items, the price should be 0.0
    items = []
    item_prices = {'item1': 10, 'item2': 20}
    bundle_discounts = {'item1+item2': 0.50}
    result = calculate_bundle_price(items, item_prices, bundle_discounts)
    assert result == 0.0


def test_calculate_bundle_price_item_not_priced():
    # Items not present in item_prices should be ignored
    items = ['item1', 'item2', 'item3']
    item_prices = {'item1': 10, 'item2': 20}  # item3 is missing
    bundle_discounts = {'item1+item2': 0.10, 'item1+item2+item3': 0.20}
    # Total should only account for item1 and item2: 10 + 20 = 30, discount from 'item1+item2' applies => 30 - (30 * 0.10) = 27
    result = calculate_bundle_price(items, item_prices, bundle_discounts)
    assert result == 27.0 