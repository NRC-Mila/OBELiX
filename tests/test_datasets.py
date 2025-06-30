import shutil
from obelix import LiIon, OBELiX, Laskowski, ShonAndMin, Dataset

def test_LiIon():
    shutil.rmtree("liion_rawdata", ignore_errors=True)
    
    li_raw = LiIon(room_temp_only=False)
    assert len(li_raw.dataframe) == 820
    
    li = LiIon()
    assert len(li.dataframe) == 450
    
    assert 'Reduced Composition' in li.dataframe.columns
    
    obelix_object = OBELiX()
    assert len(li.remove_obelix(obelix_object)) == 314

def test_Laskowski():
    shutil.rmtree("laskowski_rawdata", ignore_errors=True)

    la = Laskowski()
    assert 'Reduced Composition' in la.dataframe.columns
    
    assert len(la.dataframe) == 1346

    obelix_object = OBELiX()
    assert len(la.print_composition_matches_with_missing_doi(obelix_object)) == 6

def test_ShonAndMin():
    shutil.rmtree("shonandmin_rawdata", ignore_errors=True)

    sm_raw = ShonAndMin(clean_data=False, keep_min_conductivity=False)
    assert len(sm_raw.dataframe) == 4032  

    sm = ShonAndMin()
    assert len(sm.dataframe) == 2261 

def test_merge_datasets():
    obelix_object = OBELiX()
    li = LiIon()
    la = Laskowski()
    sm = ShonAndMin()

    assert len(Dataset.merge_datasets(obelix_object, li, la, sm, remove_duplicates=False)) == 4656
    assert len(Dataset.merge_datasets(obelix_object, li, la, sm)) == 3823
    assert len(Dataset.merge_datasets(obelix_object, li, la, remove_duplicates=False)) == 2395
    assert len(Dataset.merge_datasets(obelix_object, li, la)) == 2257

def test_removing_matching_entries():
    li = LiIon()
    la = Laskowski()
    sm = ShonAndMin()
    obelix_object = OBELiX()

    assert len(li.remove_matching_entries(obelix_object.dataframe.copy())) == 306
    li = LiIon()
    assert len(li.remove_matching_entries(la.dataframe.copy())) == 383
    li = LiIon()
    assert len(li.remove_matching_entries(sm.dataframe.copy())) == 439

    la = Laskowski()
    assert len(la.remove_matching_entries(obelix_object.dataframe.copy())) == 1019
    la = Laskowski()
    assert len(la.remove_matching_entries(li.dataframe.copy())) == 1281
    la = Laskowski()
    assert len(la.remove_matching_entries(sm.dataframe.copy())) == 1339

