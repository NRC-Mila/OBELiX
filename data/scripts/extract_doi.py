import re

from habanero import Crossref
from refextract import extract_references_from_url

references = extract_references_from_url(
    "https://www.rsc.org/suppdata/d2/ee/d2ee03499a/d2ee03499a1.pdf"
)

cr = Crossref()

with open("misc/lookup_table_doi.csv", "w") as f:
    print("ref. number,DOI", file=f)
    for ref in references:
        result = cr.works(query=re.findall("[0-9\.]+\s+(.*)", ref["raw_ref"][0])[0])
        print(
            "%s,%s" % (int(ref["linemarker"][0]), result["message"]["items"][0]["DOI"]),
            file=f,
        )

# print(result['message']['items'][0]['DOI'])
