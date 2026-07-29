from turbotwix import _format, _read, protocol


def _parse(path):
    mm = _read.open_mmap(path)
    version = _format.detect_version(mm)
    entry = _format.parse_raid_directory(mm, version)[-1]
    return protocol.parse_protocol(mm, entry.offset)


def test_parse_protocol_buffers_present(gre_path, epi_path):
    for path in (gre_path, epi_path):
        prot, hdr_len = _parse(path)
        assert hdr_len > 0
        for name in ("Config", "Dicom", "Meas", "MeasYaps", "Phoenix", "Spice"):
            assert name in prot


def test_parse_protocol_known_values(gre_path, epi_path):
    prot, _ = _parse(gre_path)
    assert prot.MeasYaps["alTR"] == ["10000"]

    prot, _ = _parse(epi_path)
    assert prot.MeasYaps["alTR"] == ["2000000"]
    assert prot.Meas["iNoOfFourierLines"] == 80.0


def test_attrdict_attribute_access(gre_path):
    prot, _ = _parse(gre_path)
    assert prot.MeasYaps is prot["MeasYaps"]
    assert isinstance(prot.MeasYaps, protocol.AttrDict)
