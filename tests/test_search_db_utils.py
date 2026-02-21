from memory_retrieval.search.db_utils import deserialize_json_field, serialize_json_field


def test_serialize_json_field_none_returns_empty_json_object() -> None:
    result = serialize_json_field(None)
    assert result == "{}"


def test_serialize_json_field_dict_returns_valid_json_string() -> None:
    result = serialize_json_field({"key": "value"})
    assert '"key"' in result
    assert '"value"' in result


def test_serialize_json_field_preserves_unicode() -> None:
    result = serialize_json_field({"message": "héllo wörld"})
    assert "héllo wörld" in result


def test_deserialize_json_field_none_returns_empty_dict() -> None:
    result = deserialize_json_field(None)
    assert result == {}


def test_deserialize_json_field_empty_string_returns_empty_dict() -> None:
    result = deserialize_json_field("")
    assert result == {}


def test_round_trip_preserves_simple_dict() -> None:
    original = {"key": "value", "count": 42}
    serialized = serialize_json_field(original)
    deserialized = deserialize_json_field(serialized)
    assert deserialized == original


def test_round_trip_preserves_nested_dict() -> None:
    original = {"outer": {"inner": [1, 2, 3]}}
    serialized = serialize_json_field(original)
    deserialized = deserialize_json_field(serialized)
    assert deserialized == original


def test_round_trip_preserves_unicode() -> None:
    original = {"text": "héllo wörld — こんにちは"}
    serialized = serialize_json_field(original)
    deserialized = deserialize_json_field(serialized)
    assert deserialized == original
