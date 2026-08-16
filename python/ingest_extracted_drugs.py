from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable
import random
import time
import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

load_dotenv(PROJECT_DIR / ".env")
load_dotenv(BASE_DIR / ".env", override=False)


def resolve_project_path(
    env_name: str,
    default_path: Path,
) -> Path:
    value = os.getenv(env_name, "").strip()

    if not value:
        return default_path.resolve()

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_DIR / path

    return path.resolve()


JSON_PATH = resolve_project_path(
    "EXTRACTED_DRUGS_JSON_PATH",
    BASE_DIR / "combined_output.json",
)

PERSIST_DIRECTORY = resolve_project_path(
    "EXTRACTED_DRUGS_CHROMA_DIRECTORY",
    BASE_DIR / "extracted_drugs_chroma_db",
)

COLLECTION_NAME = os.getenv(
    "EXTRACTED_DRUGS_COLLECTION",
    "extracted_drug_rows",
).strip()

EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001",
).strip()

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY",
    "",
).strip()

BATCH_SIZE = int(
    os.getenv(
        "EMBEDDING_BATCH_SIZE",
        "50",
    )
)
BATCHES_PER_WINDOW = int(
    os.getenv(
        "EMBEDDING_BATCHES_PER_WINDOW",
        "2",
    )
)

WINDOW_SLEEP_SECONDS = int(
    os.getenv(
        "EMBEDDING_WINDOW_SLEEP_SECONDS",
        "65",
    )
)

MAX_RETRIES = int(
    os.getenv(
        "EMBEDDING_MAX_RETRIES",
        "6",
    )
)

RECREATE_COLLECTION = (
    os.getenv(
        "RECREATE_EXTRACTED_DRUGS_COLLECTION",
        "true",
    )
    .strip()
    .lower()
    in {
        "1",
        "true",
        "yes",
        "y",
    }
)


# -----------------------------------------------------------------------------
# JSON schema
# -----------------------------------------------------------------------------

EXPECTED_FIELDS = (
    "trade_name",
    "generic_name",
    "active_ingredients",
    "therapeutic_category",
    "usage_instructions",
    "contraindication",
    "disease_category",
    "indications",
    "side_effects",
    "warnings",
    "dosage_form",
    "pack_size",
    "page_ref",
)


SEARCHABLE_FIELDS = (
    "trade_name",
    "generic_name",
    "active_ingredients",
    "therapeutic_category",
    "disease_category",
    "indications",
    "usage_instructions",
    "contraindication",
    "warnings",
)


# -----------------------------------------------------------------------------
# Known OCR / spelling aliases
# -----------------------------------------------------------------------------

GENERIC_NAME_ALIASES = {
    "cimetidine": "CIMETIDINE",

    "esomeprazol": "ESOMEPRAZOLE",
    "esomeprazole": "ESOMEPRAZOLE",

    "metoclopramide hci":
        "METOCLOPRAMIDE HCL",

    "metoclopramide as hci":
        "METOCLOPRAMIDE HCL",

    "metoclopramide hcl anhydrous":
        "METOCLOPRAMIDE HCL",
}


# -----------------------------------------------------------------------------
# Text helpers
# -----------------------------------------------------------------------------

def clean_text(
    value: Any,
) -> str:
    """
    Convert any JSON value into a clean string.
    """

    if value is None:
        return ""

    if isinstance(
        value,
        (dict, list),
    ):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    return " ".join(
        str(value)
        .strip()
        .split()
    )


def normalize_text(
    value: Any,
) -> str:
    """
    Normalize only for comparison and ID generation.

    The original text is still stored in Chroma.
    """

    text = clean_text(
        value
    ).lower()

    # Remove Arabic diacritics.
    text = re.sub(
        r"[\u064B-\u065F\u0670]",
        "",
        text,
    )

    # Normalize some Arabic characters.
    text = re.sub(
        r"[أإآ]",
        "ا",
        text,
    )

    text = text.replace(
        "ى",
        "ي",
    )

    text = text.replace(
        "ؤ",
        "و",
    )

    text = text.replace(
        "ئ",
        "ي",
    )

    text = text.replace(
        "ة",
        "ه",
    )

    text = re.sub(
        r"[^\w\u0600-\u06FF]+",
        " ",
        text,
        flags=re.UNICODE,
    )

    return " ".join(
        text.split()
    )


def canonical_generic_name(
    value: Any,
) -> str:
    """
    Normalize known generic-name OCR variants.

    Important:
    Missing generic names stay empty.
    We do not invent medical information.
    """

    original = clean_text(
        value
    )

    if not original:
        return ""

    normalized = normalize_text(
        original
    )

    return GENERIC_NAME_ALIASES.get(
        normalized,
        original.upper(),
    )


def remove_empty_lines(
    lines: Iterable[str],
) -> str:
    """
    Do not embed labels whose value is empty.
    """

    return "\n".join(
        line
        for line in lines
        if not line.endswith(": ")
    ).strip()


# -----------------------------------------------------------------------------
# IDs
# -----------------------------------------------------------------------------

def stable_document_id(
    json_index: int,
    record: dict[str, str],
) -> str:
    """
    Generate one deterministic Chroma ID
    for every JSON item.

    json_index is included intentionally,
    therefore duplicate JSON items are still
    stored as separate documents.
    """

    parts = [
        str(json_index),

        *(
            normalize_text(
                record.get(
                    field,
                    "",
                )
            )
            for field in EXPECTED_FIELDS
        ),
    ]

    raw_value = "::".join(
        parts
    )

    digest = hashlib.sha256(
        raw_value.encode(
            "utf-8"
        )
    ).hexdigest()[:24]

    return (
        f"drug-json::"
        f"{json_index}::"
        f"{digest}"
    )


# -----------------------------------------------------------------------------
# JSON loading
# -----------------------------------------------------------------------------

def load_json_records() -> list[dict[str, str]]:
    """
    Read the complete JSON array.

    Every array element becomes one record.
    """

    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f"JSON file not found: "
            f"{JSON_PATH}"
        )

    with JSON_PATH.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        payload = json.load(
            file
        )

    if not isinstance(
        payload,
        list,
    ):
        raise ValueError(
            "The JSON root must be an array."
        )

    records: list[
        dict[str, str]
    ] = []

    for json_index, item in enumerate(
        payload,
        start=1,
    ):
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                f"JSON item {json_index} "
                "must be an object."
            )

        record = {
            field: clean_text(
                item.get(
                    field,
                    "",
                )
            )
            for field in EXPECTED_FIELDS
        }

        has_searchable_content = any(
            record[field]
            for field in SEARCHABLE_FIELDS
        )

        if not has_searchable_content:
            raise ValueError(
                f"JSON item {json_index} "
                "contains no searchable "
                "drug information."
            )

        records.append(
            record
        )

    return records


# -----------------------------------------------------------------------------
# Data-quality report
# -----------------------------------------------------------------------------

def print_quality_report(
    records: list[dict[str, str]],
) -> None:
    """
    Report missing values without rejecting
    otherwise usable drug records.
    """

    serialized_records = [
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
        )
        for record in records
    ]

    print(
        "\nJSON quality report"
    )

    print(
        "-" * 60
    )

    print(
        f"Items to ingest: "
        f"{len(records)}"
    )

    distinct_trade_names = {
        record["trade_name"]
        for record in records
        if record["trade_name"]
    }

    print(
        "Distinct trade names: "
        f"{len(distinct_trade_names)}"
    )

    distinct_generic_names = {
        record["generic_name"]
        for record in records
        if record["generic_name"]
    }

    print(
        "Distinct generic names: "
        f"{len(distinct_generic_names)}"
    )

    duplicate_count = (
        len(records)
        - len(
            set(
                serialized_records
            )
        )
    )

    print(
        "Exact duplicate items: "
        f"{duplicate_count}"
    )

    for field in EXPECTED_FIELDS:
        empty_count = sum(
            1
            for record in records
            if not record[field]
        )

        if empty_count:
            print(
                f"Empty {field}: "
                f"{empty_count}/"
                f"{len(records)}"
            )

    print(
        "-" * 60
    )


# -----------------------------------------------------------------------------
# LangChain documents
# -----------------------------------------------------------------------------

def build_document(
    record: dict[str, str],
    json_index: int,
) -> tuple[
    str,
    Document,
]:
    """
    Create one complete LangChain Document
    for one JSON drug record.
    """

    trade_name = record[
        "trade_name"
    ]

    generic_name = record[
        "generic_name"
    ]

    canonical_name = (
        canonical_generic_name(
            generic_name
        )
    )

    page_content = remove_empty_lines(
        [
            (
                "Document Kind: "
                "Complete Drug Record"
            ),

            (
                "Source File: "
                f"{JSON_PATH.name}"
            ),

            (
                "JSON Item Index: "
                f"{json_index}"
            ),

            (
                "Trade Name: "
                f"{trade_name}"
            ),

            (
                "Generic Name: "
                f"{generic_name}"
            ),

            (
                "Canonical Generic Name: "
                f"{canonical_name}"
            ),

            (
                "Active Ingredients: "
                f"{record['active_ingredients']}"
            ),

            (
                "Therapeutic Category: "
                f"{record['therapeutic_category']}"
            ),

            (
                "Disease Category: "
                f"{record['disease_category']}"
            ),

            (
                "Indications: "
                f"{record['indications']}"
            ),

            (
                "Usage Instructions: "
                f"{record['usage_instructions']}"
            ),

            (
                "Contraindications: "
                f"{record['contraindication']}"
            ),

            (
                "Side Effects: "
                f"{record['side_effects']}"
            ),

            (
                "Warnings: "
                f"{record['warnings']}"
            ),

            (
                "Dosage Form: "
                f"{record['dosage_form']}"
            ),

            (
                "Pack Size: "
                f"{record['pack_size']}"
            ),

            (
                "Reference Page: "
                f"{record['page_ref']}"
            ),
        ]
    )

    # Keep metadata scalar for Chroma compatibility.
    metadata = {
        "source":
            JSON_PATH.name,

        "document_kind":
            "complete_drug_record",

        "json_index":
            json_index,

        "trade_name":
            trade_name,

        "generic_name":
            generic_name,

        "canonical_generic_name":
            canonical_name,

        "active_ingredients":
            record[
                "active_ingredients"
            ],

        "therapeutic_category":
            record[
                "therapeutic_category"
            ],

        "disease_category":
            record[
                "disease_category"
            ],

        "dosage_form":
            record[
                "dosage_form"
            ],

        "pack_size":
            record[
                "pack_size"
            ],

        "page_ref":
            record[
                "page_ref"
            ],
    }

    document_id = stable_document_id(
        json_index,
        record,
    )

    document = Document(
        page_content=
            page_content,

        metadata=
            metadata,
    )

    return (
        document_id,
        document,
    )


def build_documents(
    records: list[dict[str, str]],
) -> tuple[
    list[str],
    list[Document],
]:
    """
    Convert all JSON elements into documents.
    """

    ids: list[str] = []

    documents: list[
        Document
    ] = []

    for json_index, record in enumerate(
        records,
        start=1,
    ):
        (
            document_id,
            document,
        ) = build_document(
            record,
            json_index,
        )

        ids.append(
            document_id
        )

        documents.append(
            document
        )

    return (
        ids,
        documents,
    )


# -----------------------------------------------------------------------------
# Chroma persistence
# -----------------------------------------------------------------------------

def delete_collection_if_exists(
    client: chromadb.PersistentClient,
) -> None:
    """
    Recreate only the drug collection.

    Other collections will not be deleted.
    This becomes useful later when we add
    drug_interactions.
    """

    if not RECREATE_COLLECTION:
        return

    existing_names = {
        collection.name
        if hasattr(
            collection,
            "name",
        )
        else str(
            collection
        )

        for collection
        in client.list_collections()
    }

    if COLLECTION_NAME in existing_names:
        client.delete_collection(
            name=
                COLLECTION_NAME
        )

        print(
            "Deleted old collection: "
            f"{COLLECTION_NAME}"
        )


def add_documents_in_batches(
    store: Chroma,
    ids: list[str],
    documents: list[Document],
) -> None:

    if len(ids) != len(documents):
        raise ValueError(
            "IDs count and documents "
            "count must match."
        )

    total_documents = len(
        documents
    )

    batch_number = 0

    for start in range(
        0,
        total_documents,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total_documents,
        )

        batch_ids = ids[
            start:end
        ]

        batch_documents = documents[
            start:end
        ]

        batch_number += 1

        retry_attempt = 0

        while True:

            try:

                print(
                    f"\nEmbedding batch "
                    f"{batch_number}"
                    f" | documents "
                    f"{start + 1}-{end}"
                )

                store.add_documents(
                    documents=
                        batch_documents,

                    ids=
                        batch_ids,
                )

                print(
                    "Stored documents: "
                    f"{end}/"
                    f"{total_documents}"
                )

                break

            except Exception as error:

                error_message = str(
                    error
                )

                is_rate_limit_error = (
                    "429"
                    in error_message
                    or
                    "RESOURCE_EXHAUSTED"
                    in error_message
                    or
                    "quota"
                    in error_message.lower()
                )

                if not is_rate_limit_error:
                    raise

                retry_attempt += 1

                if (
                    retry_attempt
                    > MAX_RETRIES
                ):
                    print(
                        "\nMaximum retries "
                        "reached."
                    )

                    raise

                #
                # Backoff:
                #
                # 20 sec
                # 40 sec
                # 65 sec
                # then max 65 sec
                #
                wait_seconds = min(
                    20
                    * (
                        2
                        ** (
                            retry_attempt
                            - 1
                        )
                    ),
                    65,
                )

                #
                # Small random jitter.
                #
                wait_seconds += (
                    random.uniform(
                        1,
                        5,
                    )
                )

                print(
                    "\nGemini rate limit "
                    "reached."
                )

                print(
                    f"Retry attempt: "
                    f"{retry_attempt}/"
                    f"{MAX_RETRIES}"
                )

                print(
                    "Waiting "
                    f"{wait_seconds:.1f} "
                    "seconds..."
                )

                time.sleep(
                    wait_seconds
                )

        #
        # Proactive rate limiting.
        #
        # With:
        #
        # BATCH_SIZE = 40
        # BATCHES_PER_WINDOW = 2
        #
        # we send at most about
        # 80 documents before waiting.
        #
        if (
            batch_number
            % BATCHES_PER_WINDOW
            == 0
            and
            end
            < total_documents
        ):

            print(
                "\nRate-limit protection:"
            )

            print(
                f"Processed "
                f"{BATCH_SIZE * BATCHES_PER_WINDOW} "
                "documents."
            )

            print(
                "Waiting "
                f"{WINDOW_SLEEP_SECONDS} "
                "seconds before "
                "continuing..."
            )

            time.sleep(
                WINDOW_SLEEP_SECONDS
            )
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY "
            "in the project .env file."
        )

    if BATCH_SIZE <= 0:
        raise ValueError(
            "EMBEDDING_BATCH_SIZE "
            "must be greater than zero."
        )

    if not COLLECTION_NAME:
        raise ValueError(
            "EXTRACTED_DRUGS_COLLECTION "
            "cannot be empty."
        )

    print(
        f"Reading JSON: "
        f"{JSON_PATH}"
    )

    print(
        "Chroma directory: "
        f"{PERSIST_DIRECTORY}"
    )

    print(
        "Collection: "
        f"{COLLECTION_NAME}"
    )

    print(
        "Embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    records = load_json_records()

    print_quality_report(
        records
    )

    (
        document_ids,
        documents,
    ) = build_documents(
        records
    )

    if len(
        document_ids
    ) != len(
        set(
            document_ids
        )
    ):
        raise ValueError(
            "Duplicate Chroma IDs "
            "were generated."
        )

    if len(
        documents
    ) != len(
        records
    ):
        raise ValueError(
            "Every JSON item must "
            "produce exactly one document."
        )

    print(
        "\nDocuments prepared"
    )

    print(
        f"JSON items: "
        f"{len(records)}"
    )

    print(
        f"Documents: "
        f"{len(documents)}"
    )

    embeddings = (
        GoogleGenerativeAIEmbeddings(
            model=
                EMBEDDING_MODEL,

            google_api_key=
                GOOGLE_API_KEY,
        )
    )

    PERSIST_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = (
        chromadb.PersistentClient(
            path=
                str(
                    PERSIST_DIRECTORY
                )
        )
    )

    delete_collection_if_exists(
        client
    )

    store = Chroma(
        client=
            client,

        collection_name=
            COLLECTION_NAME,

        embedding_function=
            embeddings,
    )

    add_documents_in_batches(
        store=
            store,

        ids=
            document_ids,

        documents=
            documents,
    )

    stored_count = (
        client
        .get_collection(
            name=
                COLLECTION_NAME
        )
        .count()
    )

    if stored_count != len(
        documents
    ):
        raise RuntimeError(
            "Stored document count "
            "does not match JSON count. "
            f"Expected {len(documents)}, "
            f"found {stored_count}."
        )

    print(
        "\nDone"
    )

    print(
        f"Collection: "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Stored records: "
        f"{stored_count}"
    )

    print(
        "Database directory: "
        f"{PERSIST_DIRECTORY}"
    )


if __name__ == "__main__":
    main()