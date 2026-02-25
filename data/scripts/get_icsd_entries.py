import os
import re
import time
import urllib.parse
import warnings

import numpy as np
import pandas as pd
import requests
from habanero import Crossref
from ICSDClient import ICSDClient

# Suppress warnings from BeautifulSoup to keep the output clean
warnings.filterwarnings("ignore", category=UserWarning, module="bs4")


def get_bib_info(dois):
    cr = Crossref()
    authors_list = []
    years_list = []
    titles_list = []
    # Split the DOIs and process each one
    for doi in dois.split("|"):
        try:
            result = cr.works(ids=urllib.parse.quote(doi))
            list_authors = result["message"]["author"]
            family_names = [author["family"] for author in list_authors]
            year = result["message"]["created"]["date-parts"][0][0]
            title = result["message"]["title"][0]
            authors_list.extend(family_names)
            years_list.append(year)
            titles_list.append(title)
        except requests.exceptions.HTTPError as e:
            print(f"HTTPError: {e} for DOI: {doi}")
        except Exception as e:
            print(f"Error: {e} for DOI: {doi}")
    return " ".join(authors_list), years_list, " ".join(titles_list)


# Function to get the composition with tolerance
def get_composition(true_comp, z_value, atol=0.05):
    formula = re.findall(r"([A-Za-z]{1,2})([0-9\.]*)\s*", true_comp)
    composition = ""
    for element, amount in formula:
        reduced_amount = float(amount) / z_value if amount else 1.0
        composition += f"{element}:{reduced_amount*(1-atol)}:{reduced_amount*(1+atol)} "
    return composition


# Function to extract numeric parameters from a string
def extract_parameters(cellparamstring):
    params = re.findall(r"\d+\.\d+", cellparamstring)
    return list(map(float, params))


# Function to calculate distance between two sets of parameters
def calculate_distance(row_params, match_params):
    return np.sqrt(
        sum(
            (row_param - match_param) ** 2
            for row_param, match_param in zip(row_params, match_params)
        )
    )


# Function to extract elements from a chemical formula
def extract_elements_from_formula(formula):
    cleaned_formula = re.sub(r"[\s\(\)\.]", "", formula)
    return set(re.findall(r"([A-Za-z]{1,2})", cleaned_formula))


######## Function to perform advanced search with retries
def advanced_search_with_retry(
    client, search_params, property_list, search_type="and", max_retries=3, delay=5
):
    retries = 0
    while retries < max_retries:
        try:
            return client.advanced_search(
                search_params, property_list=property_list, search_type=search_type
            )
        except requests.exceptions.Timeout:
            retries += 1
            print(f"Timeout occurred. Retrying {retries}/{max_retries}...")
            time.sleep(delay)
        except Exception as e:
            return f"error during search: {str(e)}"
    return f"error during search after {max_retries} retries"


client = ICSDClient("ICSD_login", "ICSD_password")

data = pd.read_excel("raw.xlsx")

# Initialize new columns to store search results and comments
data["Matches"] = ""
data["Secondary Matches"] = ""
data["Match Count"] = 0
data["Secondary Match Count"] = 0
data["Final Match"] = ""
data["Final Match Count"] = 0
data["Duplication Comment"] = ""
data["Comment"] = ""


for i, row in data.iterrows():
    try:
        # Generate the composition string with tolerance
        composition = get_composition(row["True Composition"], row["Z"], 0.05)
        # Concatenate cell parameters into a single string
        cellparamstring = f"{row['a']} {row['b']} {row['c']} {row['alpha']} {row['beta']} {row['gamma']}"
        spacegroupnumber = row["Space group #"]
        # Retrieve bibliographic information
        authors, publicationyears, titles = get_bib_info(row["DOI"])
        # Extract elements from the reduced composition
        reduced_composition_elements = extract_elements_from_formula(
            row["Reduced Composition"]
        )

        # Perform the initial search with composition and space group number
        matches = advanced_search_with_retry(
            client,
            {"composition": composition, "spacegroupnumber": spacegroupnumber},
            property_list=["CollectionCode", "StructuredFormula", "CellParameter"],
            search_type="and",
        )

        if isinstance(matches, str):  # Handle error message
            data.at[i, "Matches"] = matches
            continue

        match_count = len(matches) if matches else 0
        data.at[i, "Match Count"] = match_count

        if matches:
            data.at[i, "Matches"] = matches
            secondary_matches = []
            if len(matches) > 1:
                for year in publicationyears:
                    secondary_matches = advanced_search_with_retry(
                        client,
                        {
                            "composition": composition,
                            "cellparameters": cellparamstring,
                            "spacegroupnumber": spacegroupnumber,
                            "publicationyear": year,
                        },
                        property_list=[
                            "CollectionCode",
                            "StructuredFormula",
                            "CellParameter",
                        ],
                        search_type="and",
                    )
                    if secondary_matches and not isinstance(secondary_matches, str):
                        data.at[i, "Secondary Matches"] = secondary_matches
                        data.at[i, "Secondary Match Count"] = len(secondary_matches)
                        break
                else:
                    secondary_matches = advanced_search_with_retry(
                        client,
                        {
                            "composition": composition,
                            "cellparameters": cellparamstring,
                            "spacegroupnumber": spacegroupnumber,
                        },
                        property_list=[
                            "CollectionCode",
                            "StructuredFormula",
                            "CellParameter",
                        ],
                        search_type="and",
                    )
                    if not isinstance(secondary_matches, str):
                        data.at[i, "Secondary Matches"] = secondary_matches
                        data.at[i, "Secondary Match Count"] = (
                            len(secondary_matches) if secondary_matches else 0
                        )
            if not secondary_matches:
                data.at[i, "Secondary Matches"] = matches
                data.at[i, "Secondary Match Count"] = match_count

            match_elements = extract_elements_from_formula(matches[0][1][1])
            if match_elements == reduced_composition_elements:
                data.at[i, "Final Match"] = matches[0]
                data.at[i, "Final Match Count"] = 1
            else:
                data.at[i, "Final Match"] = "not the same elements!"
        else:
            data.at[i, "Matches"] = "no results found"

        if match_count > 1 or data.at[i, "Secondary Match Count"] > 1:
            row_params = [
                row["a"],
                row["b"],
                row["c"],
                row["alpha"],
                row["beta"],
                row["gamma"],
            ]
            match_distances = []

            matches_to_compare = (
                data.at[i, "Secondary Matches"]
                if data.at[i, "Secondary Match Count"] > 1
                else data.at[i, "Matches"]
            )
            for match in matches_to_compare:
                match_params = extract_parameters(match[1][2])
                match_elements = extract_elements_from_formula(match[1][1])
                if match_elements == reduced_composition_elements:
                    distance = calculate_distance(row_params, match_params)
                    match_distances.append((match, distance))

            if match_distances:
                closest_match = min(match_distances, key=lambda x: x[1])[0]
                data.at[i, "Final Match"] = [closest_match]
                data.at[i, "Final Match Count"] = 1
            else:
                all_distances = [
                    (
                        match,
                        calculate_distance(row_params, extract_parameters(match[1][2])),
                    )
                    for match in matches_to_compare
                ]
                closest_match = min(all_distances, key=lambda x: x[1])[0]
                data.at[i, "Final Match"] = [closest_match]
                data.at[i, "Final Match Count"] = 1
        elif match_count == 1 and data.at[i, "Final Match"] != "not the same elements!":
            match_elements = extract_elements_from_formula(matches[0][1][1])
            if match_elements == reduced_composition_elements:
                data.at[i, "Final Match"] = matches[0]
                data.at[i, "Final Match Count"] = 1
            else:
                data.at[i, "Final Match"] = "not the same elements!"

    except Exception as e:
        data.at[i, "Matches"] = f"error during search: {str(e)}"
        data.at[i, "Secondary Matches"] = ""
        data.at[i, "Final Match"] = ""

# Highlight duplicates and add comments
final_matches = data["Final Match"].apply(lambda x: str(x))

# Identify and comment on duplicates
for i, match in enumerate(final_matches):
    if (
        not match
        or match == "no results found"
        or match.startswith("error during search")
        or match == "not the same elements!"
    ):
        continue
    duplicates = final_matches[final_matches == match].index.tolist()
    if len(duplicates) > 1:
        original_row = duplicates[0] + 2
        for idx, dup in enumerate(duplicates):
            if dup == duplicates[0]:
                if len(duplicates) > 1:
                    next_duplicate = duplicates[1] + 2
                    data.at[dup, "Duplication Comment"] = (
                        f"Same as row {next_duplicate}"
                    )
            else:
                data.at[dup, "Duplication Comment"] = f"Same as row {original_row}"

# Add 'Check manually' to the Comment column if there are duplicates or multiple secondary matches
for i, row in data.iterrows():
    if row["Duplication Comment"] or row["Secondary Match Count"] > 1:
        data.at[i, "Comment"] = "Check manually"


# Functions to highlight rows based on certain conditions
def highlight_duplicates(s):
    return ["background-color: yellow" if val else "" for val in s]


def highlight_row(s):
    if s["Duplication Comment"]:
        return ["background-color: yellow"] * len(s)
    else:
        return [""] * len(s)


def highlight_done_row(s):
    return (
        ["background-color: lightgreen"] * len(s)
        if s["Comment"] == "Done"
        else [""] * len(s)
    )


def highlight_not_same_elements_row(s):
    return (
        ["background-color: red"] * len(s)
        if s["Final Match"] == "not the same elements!"
        else [""] * len(s)
    )


def highlight_orange_row(s):
    return (
        ["background-color: orange"] * len(s)
        if s["Final Match"] == "not the same elements!" and s["Duplication Comment"]
        else [""] * len(s)
    )


# Apply the highlight functions to the DataFrame
styled_data = (
    data.style.apply(highlight_row, axis=1)
    .apply(highlight_done_row, axis=1)
    .apply(highlight_not_same_elements_row, axis=1)
    .apply(highlight_orange_row, axis=1)
)

# Fetch CIFs
for i, row in data.iterrows():
    if (
        row["Comment"] != "Check manually"
        and row["Final Match"]
        and row["Final Match"] != "not the same elements!"
        and row["Cif ID"] != "done"
    ):
        search = row["Final Match"]
        try:
            cifs = client.fetch_cifs(search)
            for cif in cifs:
                cif_id = row["ID"]
                cif_filename = f"{cif_id}.cif"
                cif_filepath = os.path.join("ADD_THE_PATH", cif_filename)
                with open(cif_filepath, "w") as f:
                    f.write(cif)
                data.at[i, "Comment"] = "Done"
        except Exception as e:
            print(f"Error fetching CIFs for row {i}: {str(e)}")

# Update styling after changes to comments
styled_data = (
    data.style.apply(highlight_row, axis=1)
    .apply(highlight_done_row, axis=1)
    .apply(highlight_not_same_elements_row, axis=1)
    .apply(highlight_orange_row, axis=1)
)

# Save the updated data
styled_data.to_excel("updated_row.xlsx", index=False)


client.logout()
