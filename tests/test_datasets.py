import shutil
from obelix import LiIon, OBELiX, Laskowski, ShonAndMin, Dataset

def test_LiIon():
    shutil.rmtree("lilon_rawdata", ignore_errors=True)
    obelix_object = OBELiX()
    
    li_raw = LiIon(room_temp_only=False)
    assert len(li_raw.dataframe) == 820
    
    li = LiIon()
    assert len(li.dataframe) == 450
    
    assert 'Reduced Composition' in li.dataframe.columns
    
    assert len(li.remove_obelix(obelix_object)) == 314

    assert len(li.remove_matching_entries(obelix_object.dataframe)) == 293
    
    la = Laskowski()
    assert len(li.remove_matching_entries(la.dataframe)) == 371

    sm = ShonAndMin()
    assert len(li.remove_matching_entries(sm.dataframe)) == 437

def test_Laskowski():
    shutil.rmtree("laskowski_rawdata", ignore_errors=True)
    obelix_object = OBELiX()

    la = Laskowski()
    assert 'Reduced Composition' in la.dataframe.columns

    assert len(la.dataframe) == 1346

    assert len(la.remove_obelix(obelix_object)) == 934


    assert len(la.remove_matching_entries(obelix_object.dataframe)) == 934
    
    li = LiIon()
    sm = ShonAndMin()
    assert len(la.remove_matching_entries(li.dataframe)) == 1230
    assert len(la.remove_matching_entries(sm.dataframe)) == 1327

    assert len(la.print_composition_matches_with_missing_doi(obelix_object)) == 6


def test_ShonAndMin():
    shutil.rmtree("SM_rawdata", ignore_errors=True)

    sm_raw = ShonAndMin(clean_data=False, keep_min_conductivity=False)
    assert len(sm_raw.dataframe) == 4032  

    sm = ShonAndMin()
    assert len(sm.dataframe) == 2261 

def test_merge_datasets():

    ob = OBELiX()
    li = LiIon()
    la = Laskowski()
    sm = ShonAndMin()

    assert len(Dataset.merge_datasets(ob, li, la, sm, remove_duplicates=False)) == 5478
    assert len(Dataset.merge_datasets(ob, li, la, sm)) == 4316
    assert len(Dataset.merge_datasets(ob, li, la, remove_duplicates=False)) == 2395
    assert len(Dataset.merge_datasets(ob, li, la)) == 2257




