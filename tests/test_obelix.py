def test_public_OBELiX():
    import shutil

    from obelix import OBELiX

    shutil.rmtree("rawdata", ignore_errors=True)
    obelix = OBELiX()
    assert len(obelix) == 599
    assert len(obelix.test_dataset) == 121
    assert len(obelix.train_dataset) == 478
    assert obelix[0]["ID"] == "jqc"
    assert obelix["jqc"]["ID"] == "jqc"


def test_dev_OBELiX():
    import shutil

    from obelix import OBELiX

    obelix = OBELiX("rawdata_dev", dev=True)
    assert len(obelix) == 599
    assert len(obelix.test_dataset) == 121
    assert len(obelix.train_dataset) == 478
    assert obelix[0]["ID"] == "jqc"
    assert obelix["jqc"]["ID"] == "jqc"


def test_custom_OBELiX():
    import shutil

    from obelix import OBELiX

    obelix = OBELiX("data", dev=True)
    assert len(obelix) == 599
    assert len(obelix.test_dataset) == 121
    assert len(obelix.train_dataset) == 478
    assert obelix[0]["ID"] == "jqc"
    assert obelix["jqc"]["ID"] == "jqc"


def test_round_partial():
    from obelix import OBELiX

    obelix = OBELiX()
    obelix_round = obelix.round_partial()
    assert len(obelix_round) == 599
    assert len(obelix_round.with_cifs()) == 321
    for entry in obelix_round.with_cifs():
        for i, site in enumerate(entry["structure"]):
            for k, v in site.species.as_dict().items():
                assert round(v) == v
