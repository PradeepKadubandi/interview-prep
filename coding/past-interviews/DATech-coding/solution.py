# pip install --upgrade google-api-python-client google-auth-oauthlib google-auth
from __future__ import annotations
import re
from typing import List, Optional, Union

from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Docs API scope: read-only
SCOPES = ["https://www.googleapis.com/auth/documents.readonly"]


def extract_doc_id_from_url(url: str) -> str:
    """
    Accepts a Google Doc URL in common formats and returns the document ID.
    Works for:
      - https://docs.google.com/document/d/<DOC_ID>/edit
      - https://docs.google.com/document/d/<DOC_ID>
      - with extra query params, anchors, etc.
    """
    m = re.search(r"/document/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        raise ValueError("Could not find a Google Docs document ID in the provided URL.")
    return m.group(1)


def default_user_credentials(token_path: str = "token.json",
                             client_secret_path: str = "credentials.json") -> Credentials:
    """
    Creates/refreshes user OAuth credentials for local scripts.
    - Place your OAuth client file (downloaded from Google Cloud Console) at `credentials.json`.
    - The first run opens a browser for consent and stores a refreshable `token.json`.
    """
    creds: Optional[Credentials] = None
    try:
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    except Exception:
        creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as f:
            f.write(creds.to_json())
    return creds


def _concat_text_from_elements(elements: list) -> str:
    """Concatenate plain text from a list of Structural Elements (paragraphs)."""
    chunks = []
    for e in elements or []:
        if "paragraph" in e:
            for pe in e["paragraph"].get("elements", []):
                text_run = pe.get("textRun")
                if text_run and "content" in text_run:
                    chunks.append(text_run["content"])
        # Tables nested inside cells are possible; ignore here to keep output flat.
    return "".join(chunks).strip()


def parse_table_from_google_doc(
    doc_url: str,
    table_index: int = 0,
    creds: Optional[Credentials] = None,
) -> List[List[str]]:
    """
    Fetches a Google Doc and returns the table at `table_index`
    as a list of lists in **column-major order** (columns -> rows).
    """
    doc_id = extract_doc_id_from_url(doc_url)
    creds = creds or default_user_credentials()

    service = build("docs", "v1", credentials=creds)
    doc = service.documents().get(documentId=doc_id).execute()

    # Find all tables
    tables = []
    for elt in doc.get("body", {}).get("content", []):
        tbl = elt.get("table")
        if tbl:
            tables.append(tbl)

    if not tables or table_index < 0 or table_index >= len(tables):
        raise ValueError(f"No table found at index {table_index}. "
                         f"Document has {len(tables)} table(s).")

    table = tables[table_index]
    row_major: List[List[str]] = []

    for row in table.get("tableRows", []):
        row_values: List[str] = []
        for cell in row.get("tableCells", []):
            text = _concat_text_from_elements(cell.get("content", []))
            row_values.append(text)
        row_major.append(row_values)

    # Convert to column-major (transpose)
    if not row_major:
        return []
    col_major = [list(col) for col in zip(*row_major)]
    return col_major


def print_grid(column_major_data: List[List[Union[str, int]]]):
    assert len(column_major_data) == 3, "The expected format in google doc should have generated data with exactly 3 columns"
    x_values, characters, y_values = column_major_data
    x_values, characters, y_values = x_values[1:], characters[1:], y_values[1:] # Discard the headers (we can also assert the headers if important)

    x_values, y_values = list(map(int, x_values)), list(map(int, y_values))
    max_x, max_y = max(x_values) + 1, max(y_values) + 1

    # walk through the lists and build a dictionary of (x,y) --> character
    location_map = dict()
    for (x,y,c) in zip(x_values, y_values, characters):
        location_map[(x,y)] = c
    
    # Y-axis represents the rows in print output from Y_max-1 to 0 inclusive, X-axis represents the columns from 0 to X_max - 1 inclusive
    # print according to the above
    for y in range(max_y-1, -1, -1):
        for x in range(0, max_x):
            c = location_map.get((x,y), ' ')
            print (c, end = '')
        print ('\n')

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    DOC_URL = "https://docs.google.com/document/d/14Xv8-FRrxaBXrbVvovpszcudvDDI8aQFgSYLpJqaWik/edit?usp=sharing"
    table = parse_table_from_google_doc(DOC_URL, table_index=0)
    # print (table)
    print_grid(table)
