import pytest


def test_public_OBELiX():
    import shutil

    from obelix import OBELiX

    obelix = OBELiX("tmp_rawdata")
    assert len(obelix) == 599
    assert len(obelix.test_dataset) == 121
    assert len(obelix.train_dataset) == 478
    assert obelix[0]["ID"] == "jqc"
    assert obelix["jqc"]["ID"] == "jqc"

    shutil.rmtree("tmp_rawdata", ignore_errors=True)


@pytest.mark.dev
def test_dev_OBELiX():
    import shutil

    from obelix import OBELiX

    obelix = OBELiX("rawdata_dev", dev=True)
    assert len(obelix) == 599
    assert len(obelix.test_dataset) == 121
    assert len(obelix.train_dataset) == 478
    assert obelix[0]["ID"] == "jqc"
    assert obelix["jqc"]["ID"] == "jqc"

    shutil.rmtree("tmp_rawdata_dev", ignore_errors=True)


@pytest.mark.dev
def test_custom_OBELiX():
    import shutil
    from pathlib import Path

    from git import Repo

    from obelix import OBELiX

    tmp_repos = Path("tmp_repos")

    tmp_repos.mkdir(exist_ok=True)

    # Make a temporary directory for the test
    Repo.clone_from(
        "git@github.com:NRC-Mila/private-OBELiX.git", tmp_repos / "private_OBELiX"
    )
    Repo.clone_from("git@github.com:NRC-Mila/OBELiX.git", tmp_repos / "OBELiX")

    tmp_repos / "private_OBELiX" / "anon_cifs".rename(
        tmp_repos / "OBELiX" / "data" / "anon_cifs"
    )

    obelix = OBELiX(tmp_repos / "OBELiX" / "data", dev=True)
    assert len(obelix) == 599
    assert len(obelix.test_dataset) == 121
    assert len(obelix.train_dataset) == 478
    assert obelix[0]["ID"] == "jqc"
    assert obelix["jqc"]["ID"] == "jqc"

    shutil.rmtree("tmp_repos", ignore_errors=True)


def test_round_partial():
    import shutil

    from obelix import OBELiX

    obelix = OBELiX("tmp_rawdata")
    obelix_round = obelix.round_partial()
    assert len(obelix_round) == 599
    assert len(obelix_round.with_cifs()) == 321
    for entry in obelix_round.with_cifs():
        for i, site in enumerate(entry["structure"]):
            for k, v in site.species.as_dict().items():
                assert round(v) == v

    shutil.rmtree("tmp_rawdata", ignore_errors=True)
