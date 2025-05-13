import shutil
from obelix import LiIon, OBELiX, Laskowski, ShonAndMin, Dataset

def test_LiIon():
    shutil.rmtree("lilon_rawdata", ignore_errors=True)
    obelix_object = OBELiX()
    li = LiIon()
    assert len(li.dataframe) == 450
    assert 'Reduced Composition' in li.dataframe.columns
    assert len(li.remove_obelix(obelix_object)) == 297

def test_Laskowski():
    shutil.rmtree("laskowski_rawdata", ignore_errors=True)
    obelix_object = OBELiX()

    la = Laskowski()

    assert 'Reduced Composition' in la.dataframe.columns

    assert len(la.dataframe) == 1346
    assert len(la.remove_obelix(obelix_object)) == 824

def test_ShonAndMin():
    shutil.rmtree("SM_rawdata", ignore_errors=True)

    sm = ShonAndMin()

    assert len(sm.dataframe) == 3083

def test_Dataset():
    
    ob = OBELiX()
    li = LiIon ()
    la = Laskowski ()
    sm = ShonAndMin()
    
    df = Dataset.merge_datasets(ob, li, la, sm)
    assert df.shape[0] == 4316  

