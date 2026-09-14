from schemaguard.compatibility.fixtures import fixture_catalog


def test_fixture_hashes_are_seeded_and_rows_are_disjoint() -> None:
    first = fixture_catalog(1729)
    second = fixture_catalog(1729)
    other = fixture_catalog(1730)
    assert {name: item.hash for name, item in first.items()} == {
        name: item.hash for name, item in second.items()
    }
    assert first["binary_numerical"].hash != other["binary_numerical"].hash
    for fixture in first.values():
        assert set(fixture.train.row_id).isdisjoint(set(fixture.test.row_id))
        assert all((fixture.target == label).sum() >= 5 for label in fixture.classes)


def test_fixture_capabilities_cover_required_cases() -> None:
    fixtures = fixture_catalog()
    assert len(fixtures["binary_numerical"].classes) == 2
    assert len(fixtures["multiclass_numerical"].classes) >= 3
    assert fixtures["mixed_categorical"].capabilities["categorical"]
    assert fixtures["missing_values"].train.isna().sum().sum() > 0
    assert fixtures["unseen_category_inference"].capabilities["unseen_category"]
