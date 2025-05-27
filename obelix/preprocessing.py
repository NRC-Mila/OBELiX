import numpy as np
import re
import pandas as pd
import argparse
import unicodedata

def read_options():

    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--thresh",dest="thresh",type=float, default=None, help="Threshold for ionic conductivity. Default is None.")
    parser.add_argument("-d","--del",dest="delete", action="store_true", default=False, help="Wether to delete close duplicates. Default is False.")

    args = parser.parse_args()

    return args

def is_same_formula(formula_string1, formula_string2):
    f1 = re.findall(r"([A-Za-z]{1,2})([0-9\.]*)\s*", formula_string1)
    f2 = re.findall(r"([A-Za-z]{1,2})([0-9\.]*)\s*", formula_string2)
    if len(f1) != len(f2):
        return False
    for elem, count in f1:
        if (elem, count) not in f2:
            return False
    return True

def remove_overlines(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    out_str = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    out_str = out_str.replace("−", "-")
    
    # Convert exponents like E-9 to E-09
    out_str = re.sub(r'([Ee][\+\-])(\d)(?!\d)', r'\g<1>0\2', out_str)
   
    # Format mantissas to have two decimal places if missing: 3.2E-03 → 3.20E-03
    out_str = re.sub(r'^(\d+\.\d)([Ee][\+\-]\d\d)$', lambda m: f"{float(m.group(1)):.2f}{m.group(2)}", out_str)

    return out_str


def get_paper_from_laskowski(material, cond, laskowski_data, doi_lookup_table, return_idx=False):
    paper_dois = []
    potential_matches = []
    for j, paper_material in enumerate(laskowski_data[:,0]):
        if (material == paper_material or
            is_same_formula(material, paper_material) or
            paper_material + laskowski_data[min(j+1, laskowski_data.shape[0]-1),0] == material or
            paper_material + laskowski_data[min(j+2, laskowski_data.shape[0]-1),0] == material):

            cond_mat = remove_overlines(laskowski_data[j,1])
            

            if cond_mat == "":
                cond_mat = remove_overlines(laskowski_data[j+1,1])

            if cond_mat == "":
                cond_mat = remove_overlines(laskowski_data[j+2,1])
            
            if type(cond) == float:
                try:
                    cond_mat = float(cond_mat)
                except ValueError:
                    continue
            elif type(cond) == str:
                cond = cond.strip().lower()
                cond_mat = cond_mat.strip().lower()
            else:
                print("WARNING: Unknown type for conductivity:", cond)
                continue
                
            paper_doi = laskowski_data[j,-1]
            if paper_doi == "":
                if paper_material + laskowski_data[min(j+1, laskowski_data.shape[0]-1),0] == material and laskowski_data[j+1,-1] != "":
                    paper_doi = laskowski_data[j+1,-1]
                elif paper_material + laskowski_data[min(j+2, laskowski_data.shape[0]-1),0] == material and (laskowski_data[j+1,-1] != "" or laskowski_data[j+2,-1] != ""):
                    if laskowski_data[j+1,-1] == "":
                        paper_doi = laskowski_data[j+2,-1]
                    else:
                        paper_doi = laskowski_data[j+1,-1]
                else:
                    paper_doi = "Missing paper info (Laskowski)"
                    
            if cond == cond_mat:
                if paper_doi == "Missing paper info (Laskowski)":
                    print("WARNING: Found material but paper info is missing:", material)

                if return_idx:
                    paper_dois.extend(re.findall(r"[0-9]+", paper_doi))
                else:
                    paper_dois.extend([doi_lookup_table[doi] for doi in re.findall(r"[0-9]+", paper_doi)])
            else:
                if paper_doi == "Missing paper info (Laskowski)":
                    if return_idx:
                        potential_matches.append(([""], cond_mat))
                    else:
                        potential_matches.append(([paper_doi], cond_mat))
                else:
                    if return_idx:
                        potential_matches.append((re.findall(r"[0-9]+", paper_doi), cond_mat))
                    else:
                        potential_matches.append(([doi_lookup_table[doi] for doi in re.findall(r"[0-9]+", paper_doi)], cond_mat))
    return paper_dois, potential_matches

def get_paper_from_liion(material, cond, liion_data, return_idx=False):
    paper_dois = []
    potential_matches = []
    for j, liion_material in enumerate(liion_data[:,1]):
        if (material == liion_material or is_same_formula(material, liion_material)):
            if return_idx:
                 if cond == float(liion_data[j, 4]):
                     paper_dois.append(str(j))
                 else:
                     potential_matches.append(([str(j)], float(liion_data[j, 4])))
            else:
                 if cond == float(liion_data[j, 4]):
                     paper_dois.append(liion_data[j, 2])
                 else:
                     potential_matches.append(([liion_data[str(j), 2]], float(liion_data[j, 4])))

    return paper_dois, potential_matches

def get_paper_from_unidentified(material, uni_data):
    paper_dois = []
    for j, uni_material in enumerate(uni_data[:,0]):
        if material == uni_material or is_same_formula(material, uni_material):
            paper_dois.append(uni_data[j, 1])
    return paper_dois

def get_paper_from_hand(material, cond, hand_data):
    paper_dois = []

    try:
        float_cond = float(cond)
    except ValueError:
        float_cond = 1e-15
    
    for j, data in enumerate(hand_data[:,[0,1]]):
        hand_material = data[0]
        hand_cond = data[1]

        try:
            float_hand_cond = float(hand_cond)
        except ValueError:
            float_hand_cond = 1e-15
        
        if (material == hand_material or is_same_formula(material, hand_material)) and float_cond == float_hand_cond:
            paper_dois.append(hand_data[j, 2])
    return paper_dois

def add_paper_info(database):

    laskowski_database = "misc/laskowski_semi-fromatted.csv"
    liion_database = "misc/LiIonDatabase.csv"
    uni_datatbase = "misc/unidentified_with_refs.csv"
    doi_lookup_table_file = "misc/doi_lookup_table.csv"
    hand_database = "misc/checked_by_hand.csv"

    homin_data = database.to_numpy()
    laskowski_data = np.genfromtxt(laskowski_database, delimiter=',', dtype=str)
    doi_lookup_table = dict(np.genfromtxt(doi_lookup_table_file, delimiter=',', dtype=str, skip_header=1))
    liion_data = np.genfromtxt(liion_database, delimiter=',', dtype=str, skip_header=1)
    uni_data = np.genfromtxt(uni_datatbase, delimiter=',', dtype=str)
    hand_data = np.genfromtxt(hand_database, delimiter=',', dtype=str, skip_header=1)
    
    new_homin_data = np.concatenate((homin_data,np.ones((homin_data.shape[0],1))), axis=1)
    
    not_found_count = 0
    not_found_exactly_count = 0
    not_found = open("unidentified.txt", "w")
    close_matches = open("close_matches.txt", "w")
    
    for i, mat_info in enumerate(homin_data[:,[0,3]]):
        material = mat_info[0]
        cond = mat_info[1]

        paper_info = get_paper_from_hand(material, cond, hand_data)

        if len(paper_info) > 0:
            new_homin_data[i,-1] = "|".join(paper_info)
            continue
        
        paper_info_laskowski, potential_matches_laskowski = get_paper_from_laskowski(material, cond, laskowski_data, doi_lookup_table)    
        paper_info_LiIon, potential_matches_LiIon = get_paper_from_liion(material, cond, liion_data)
        paper_info_unidentified = get_paper_from_unidentified(material, uni_data)
        
        paper_info = paper_info_laskowski + paper_info_LiIon + paper_info_unidentified
        potential_matches = potential_matches_laskowski + potential_matches_LiIon

        try:
            float_cond = float(cond)
        except ValueError:
            float_cond = 1e-15
        
        if len(paper_info) == 0:

            float_best_match_cond = np.inf
            for doi, match_cond in potential_matches:

                try:
                    float_match_cond = float(match_cond)
                except ValueError:
                    float_match_cond = 1e-15
                
                if abs(float_match_cond - float_cond) <= abs(float_best_match_cond - float_cond):
                    paper_info = ["*" + d for d in doi]
                    float_best_match_cond = float_match_cond
                    best_match_cond = match_cond
                    
            if len(paper_info) == 0:
                not_found_count += 1
                print("No match for:", material, cond)
                print("Matches:", potential_matches)
                print(i, material, file=not_found)
                paper_info = ["Not in databases"]
            else:
                not_found_exactly_count += 1
                # print(i, material, cond, file=close_matches, sep=",")
                print(not_found_exactly_count, i, material,file=close_matches, sep=",")
                print(cond, best_match_cond, float_best_match_cond - float_cond, file=close_matches, sep=",")
                print("Closest match:", ",".join(["=HYPERLINK(\"https://doi.org/" + p.replace("*", "") + "\")" for p in paper_info]), file=close_matches, sep=",")
                print("Potential matches:", file=close_matches)
                for doi, match_cond in potential_matches:
                    print(match_cond, ",".join(["=HYPERLINK(\"https://doi.org/" + p + "\")" for p in doi]), file=close_matches, sep=",")
                print(file=close_matches)
                print(file=close_matches)

        paper_info = list(set([p.lower() for p in paper_info]))
                
        # if len(paper_info) > 1:
        #     print("Multiple dois:", i, material, paper_info)
                
        new_homin_data[i,-1] = "|".join(paper_info)
        
    not_found.close()
    close_matches.close()

    if not_found_exactly_count > 0:
        print("WARNING: %d materials were not found exactly in the databases"%(not_found_exactly_count))
    if not_found_count > 0:
        print("WARNING: %d materials were not found in the databases"%(not_found_count))
    
    new_homin_data = pd.DataFrame(new_homin_data, columns=database.columns.tolist() + ["DOI"])
    
    return new_homin_data

def remove_close_duplicates(df, tol=1e-4):

    cols = ["a", "b", "c", "alpha", "beta", "gamma"]
    
    def is_close(val):
        matches = df.index[(df["Composition"].apply(is_same_formula, args=(val["Composition"],))) & (abs(df[cols] - val[cols])/val[cols] < tol).all(axis=1)]
        return matches.sort_values()[0] != val.name
        
    n_data = len(df)
    dup_rows = df.apply(is_close, axis=1)
    duplicated_df = df[dup_rows]
    df = df[~dup_rows]

    def list_duplicates(val):
        matches = df.index[(df["Composition"].apply(is_same_formula, args=(val["Composition"],))) & (abs(df[cols] - val[cols])/val[cols] < tol).all(axis=1)]
        return matches.tolist()

    n_duplicates = n_data - len(df)
    if n_duplicates > 0:
        print(f"WARNING: The data set contained {n_duplicates} close duplicates!")
        print(duplicated_df.apply(list_duplicates, axis=1))
    n_data = len(df)

    return df
    
def remove_exact_duplicates(df, list_columns=None, index=False):
    # Check duplicates
    n_data = len(df)
    if index:
        dup_rows = df.index.duplicated(keep="first")
    else:
        if list_columns is not None:
            dup_rows = df[list_columns].duplicated(keep="first")
        else:
            dup_rows = df.duplicated(keep="first")
    duplicated_df = df[dup_rows]
    df = df[~dup_rows]

    def list_duplicates(val):
        if index:
            return df[(df.index == val.name)].index.tolist()
        else:
            if list_columns is not None:
                return df[(df[list_columns] == val[list_columns]).all(axis=1)].index.tolist()
            else:
                return df[(df == val).all(axis=1)].index.tolist()
            
    n_duplicates = n_data - len(df)
    if n_duplicates > 0:
        print(f"WARNING: The data set contained {n_duplicates} exact duplicates!")
        print(duplicated_df.apply(list_duplicates, axis=1))
    n_data = len(df)

    return df

def replace_text_IC(cond):
    if cond == '<1E-10' or cond == '<1E-8':
        return 1e-15
    elif type(cond) != float:
        print("WARNING: IC %d is not a float:"%(i), cond)
    return cond

def which_cif_type(row):
    if type(row["close match"]) == str and row["close match"].strip().lower() == "yes":
        return "Close Match"
    elif type(row["Cif ID"]) == str and row["Cif ID"].strip().lower() == "done":
        return "Match"
    else:
        return "No Match"

def main(args):
    
    homin_database = "raw.xlsx"
        
    homin_data = pd.read_excel(homin_database)

    homin_data = homin_data.set_index("ID")

    homin_data = remove_exact_duplicates(homin_data)

    if "DOI" not in homin_data.columns:
        homin_data = add_paper_info(homin_data)
        homin_data.to_excel("raw.xlsx", index=False)
        
    # Rename and reorgnize to columns
    final = homin_data[["True Composition", "Space group #","a", "b", "c", "alpha", "beta", "gamma", "Ionic conductivity (S cm-1)", "DOI"]].copy()
    final["CIF"] = homin_data.apply(which_cif_type, axis=1)
    
    final.rename({"True Composition": "Composition"}, axis=1, inplace=True)
    final.rename({"Space group #": "Space group number"}, axis=1, inplace=True)
    
    final["Ionic conductivity (S cm-1)"] = final["Ionic conductivity (S cm-1)"].apply(replace_text_IC)
        
    if args.thresh is not None:
        final.loc[final["Ionic conductivity (S cm-1)"] < args.thresh, "Ionic conductivity (S cm-1)"] = args.thresh

    print("Checking for duplicate inputs..." )
    final = remove_exact_duplicates(final, ["Composition", "Space group number","a", "b", "c", "alpha", "beta", "gamma"])
    print("WARNING: Entries in the left column were removed.")

    print("Checking for duplicate comp+spg and paper..." )
    final_close = remove_exact_duplicates(final, ["Composition", "Space group number", "DOI"])
    if args.delete:
        final = final_close
        print("WARNING: Entries in the left column were removed.")
    else:
        print("WARNING: Entries should be checked manually. Rerun with -d to remove entries in the left column.")

    print()
    print("Checking for close duplicates...")

    # Just warning, not actually removing
    final_close = remove_close_duplicates(final, 1e-1)
    if args.delete:
        final = final_close
        print("WARNING: Entries in the left column were removed.")
    else:
        print("WARNING: Entries should be checked manually. Rerun with -d to remove entries in the left column.")
    
    print("")
    print("Checking indices...")
    final_index = remove_exact_duplicates(final, index=True)
    
    if len(final) != len(final_index):
        print("WARNING: The data set contained duplicated indices!")
        
    final.to_csv("processed.csv", float_format='%g')
    
if __name__ == "__main__":
    main(read_options())
