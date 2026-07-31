import turbotwix as tw


def _parse(path):
    m = tw.open_twix(path)[-1]
    return m.hdr


def test_parse_protocol_buffers_present(gre_path, epi_path):
    for path in (gre_path, epi_path):
        prot = _parse(path)
        for name in ("Config", "Dicom", "Meas", "MeasYaps", "Phoenix", "Spice"):
            assert name in prot


def test_protocol_buffers_are_parsed_lazily(gre_path):
    prot = _parse(gre_path)
    # Every buffer is listed, but none has been through the regex parsers yet.
    assert set(prot) >= {"Config", "Dicom", "Meas", "MeasYaps", "Phoenix", "Spice"}
    assert all(isinstance(dict.__getitem__(prot, name), bytes) for name in prot)

    assert prot.MeasYaps["alTR"] == [10000]
    assert isinstance(dict.__getitem__(prot, "MeasYaps"), tw.AttrDict)
    assert isinstance(dict.__getitem__(prot, "Meas"), bytes)  # untouched buffers stay raw


def test_protocol_bulk_access_parses_every_buffer(gre_path):
    # dict(p) and {**p} take a fast path that reads the underlying table directly; they
    # must still come back parsed, not as raw bytes.
    assert all(isinstance(v, dict) for v in dict(_parse(gre_path)).values())
    assert all(isinstance(v, dict) for v in {**_parse(gre_path)}.values())

    prot = _parse(gre_path)
    assert all(isinstance(v, dict) for v in prot.values())
    assert all(isinstance(v, dict) for _, v in prot.items())
    assert isinstance(prot.get("Meas"), dict)
    assert prot.get("nope", "default") == "default"


def test_values_are_typed_from_syntax_not_from_key_prefix():
    assert tw.header._cast_value("10000") == 10000
    assert tw.header._cast_value("-3") == -3
    assert tw.header._cast_value("1.5") == 1.5
    assert tw.header._cast_value("0x20") == 32
    assert tw.header._cast_value('"text"') == "text"
    assert tw.header._cast_value("free_form") == "free_form"
    # A key named like a flag does not make its value a bool, and vice versa.
    assert tw.header._cast_value("0") == 0
    assert tw.header._cast_value("2") == 2


def test_identifiers_with_underscores_stay_strings():
    # Python's int()/float() treat `_` as a digit separator, so a bare try/except
    # around float() turns a scanner ID into a number: float(uid) == 6.07e+26.
    uid = "6_0_66327775_20210622_151635_650"
    assert tw.header._cast_value(uid) == uid
    assert tw.header._cast_xprot(uid, "Double") == uid
    assert tw.header._cast_xprot("1_000", "Long") == "1_000"
    # ... while real numbers still parse.
    assert tw.header._cast_xprot("1.25e3", "Double") == 1250.0
    assert tw.header._cast_xprot("20.000000", "Double") == 20.0
    assert tw.header._cast_xprot("", "Long") is None


def test_xprot_values_use_the_declared_tag_type():
    parsed = tw.header._parse_xprot(
        '<ParamLong."lA"> { 5 }\n'
        '<ParamBool."bB"> { "true" }\n'
        '<ParamBool."bC"> { "false" }\n'
        '<ParamString."tD"> { "hi there" }\n'
        '<ParamDouble."dE"> { <Precision> 6  1.25 }\n'
        '<ParamLong."alF"> { 1 2 3 }\n'
    )
    assert parsed == {
        "lA": 5,
        "bB": True,
        "bC": False,
        "tD": "hi there",
        "dE": 1.25,
        "alF": [1, 2, 3],
    }


def test_parse_buffer_strips_ascconv_section_from_xprot_remainder():
    buffer = (
        '<ParamLong."lBefore"> { 1 }\n'
        "### ASCCONV BEGIN ###\n"
        "alTR[0] = 10\n"
        "### ASCCONV END ###\n"
        '<ParamLong."lAfter"> { 2 }\n'
    )
    assert "alTR" not in tw.header._ASCCONV.sub("", buffer)

    prot = tw.header._parse_buffer(buffer)
    assert prot["alTR"] == [10]  # still parsed, from the ascconv branch
    assert prot["lBefore"] == 1
    assert prot["lAfter"] == 2


def test_known_values(gre_path, epi_path):
    assert _parse(gre_path).MeasYaps["alTR"] == [10000]
    assert _parse(epi_path).MeasYaps["alTR"] == [2000000]
    assert _parse(epi_path).Meas["iNoOfFourierLines"] == 80


def test_attrdict_attribute_access(gre_path):
    prot = _parse(gre_path)
    assert prot.MeasYaps is prot["MeasYaps"]
    assert isinstance(prot.MeasYaps, tw.AttrDict)
