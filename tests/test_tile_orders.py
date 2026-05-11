from src.preprocessing.tile_orders import build_tile_order_records, generate_tile_orders, identity_tile_order


def test_generate_tile_orders_length():
    output_tile_orders = generate_tile_orders(3, 5, seed=123)
    assert len(output_tile_orders) == 5
    assert all(len(output_tile_order) == 9 for output_tile_order in output_tile_orders)


def test_identity_tile_order():
    output_tile_order = identity_tile_order(4)
    assert output_tile_order == list(range(16))


def test_build_tile_order_records_uses_unified_seed_for_identity_and_random_records():
    records = build_tile_order_records([1, 2], num_tile_orders=1, seed=123)

    assert {record.tile_order_seed for record in records} == {123}
    assert records[0].grid_side_length == 1
    assert records[0].tile_order_id == 0
    assert records[0].output_tile_order == [0]
