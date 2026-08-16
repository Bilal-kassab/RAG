from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

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


def resolve_project_path(env_name: str, default_path: Path) -> Path:
    value = os.getenv(env_name, "").strip()

    if not value:
        return default_path.resolve()

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_DIR / path

    return path.resolve()


JSON_PATH = resolve_project_path(
    "DRUG_INTERACTIONS_JSON_PATH",
    BASE_DIR / "updated_interactions_with_pages.json",
)

# IMPORTANT: use the same physical ChromaDB directory as the drug-profile
# collection. Only the collection name is different.
PERSIST_DIRECTORY = resolve_project_path(
    "EXTRACTED_DRUGS_CHROMA_DIRECTORY",
    BASE_DIR / "extracted_drugs_chroma_db",
)

COLLECTION_NAME = os.getenv(
    "DRUG_INTERACTIONS_COLLECTION",
    "drug_interactions",
).strip()

EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001",
).strip()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

BATCH_SIZE = int(
    os.getenv(
        "INTERACTIONS_EMBEDDING_BATCH_SIZE",
        "50",
    )
)

BATCH_DELAY_SECONDS = float(
    os.getenv(
        "INTERACTIONS_EMBEDDING_BATCH_DELAY_SECONDS",
        "5",
    )
)

MAX_RETRIES = int(
    os.getenv(
        "INTERACTIONS_EMBEDDING_MAX_RETRIES",
        "6",
    )
)

RECREATE_COLLECTION = (
    os.getenv(
        "RECREATE_DRUG_INTERACTIONS_COLLECTION",
        "true",
    )
    .strip()
    .lower()
    in {"1", "true", "yes", "y"}
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def clean_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    return " ".join(str(value).strip().split())


def normalize_text(value: Any) -> str:
    text = clean_text(value).lower()

    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[أإآ]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    text = text.replace("ة", "ه")
    text = re.sub(
        r"[^\w\u0600-\u06FF]+",
        " ",
        text,
        flags=re.UNICODE,
    )

    return " ".join(text.split())


def entity_key(value: Any) -> str:
    normalized = normalize_text(value)

    if not normalized:
        return "unknown"

    digest = hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()[:16]

    return f"entity::{digest}"


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "::".join(
        normalize_text(part)
        for part in parts
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:24]

    return f"{prefix}::{digest}"


def string_list(value: Any) -> list[str]:
    if value is None:
        return []

    if not isinstance(value, list):
        return []

    output: list[str] = []

    for item in value:
        text = clean_text(item)

        if text and text not in output:
            output.append(text)

    return output


def json_string_list(value: Any) -> str:
    return json.dumps(
        string_list(value),
        ensure_ascii=False,
    )


def remove_empty_lines(lines: Iterable[str]) -> str:
    output: list[str] = []

    for line in lines:
        if line.endswith(": "):
            continue

        output.append(line)

    return "\n".join(output).strip()


def reference_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""

    parts: list[str] = []

    for item in value:
        if isinstance(item, dict):
            table_number = clean_text(item.get("table_number"))
            printed_page = clean_text(item.get("printed_page"))
            associated_text = clean_text(item.get("associated_text"))

            segment = remove_empty_lines(
                [
                    f"Table: {table_number}",
                    f"Printed Page: {printed_page}",
                    f"Text: {associated_text}",
                ]
            ).replace("\n", "; ")

            if segment:
                parts.append(segment)
        else:
            text = clean_text(item)
            if text:
                parts.append(text)

    return " | ".join(parts)


# -----------------------------------------------------------------------------
# JSON loading / validation
# -----------------------------------------------------------------------------
def load_entries() -> list[dict[str, Any]]:
    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f"JSON file not found: {JSON_PATH}"
        )

    with JSON_PATH.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError(
            "The interaction JSON root must be an array."
        )

    entries: list[dict[str, Any]] = []

    for entry_index, item in enumerate(
        payload,
        start=1,
    ):
        if not isinstance(item, dict):
            raise ValueError(
                f"JSON item {entry_index} must be an object."
            )

        source = item.get("source")
        entry = item.get("entry")

        if not isinstance(source, dict):
            raise ValueError(
                f"JSON item {entry_index}.source must be an object."
            )

        if not isinstance(entry, dict):
            raise ValueError(
                f"JSON item {entry_index}.entry must be an object."
            )

        entry_name = clean_text(entry.get("entry_name"))

        if not entry_name:
            raise ValueError(
                f"JSON item {entry_index}.entry.entry_name is required."
            )

        interactions = entry.get("interactions", [])

        if interactions is None:
            interactions = []

        if not isinstance(interactions, list):
            raise ValueError(
                f"JSON item {entry_index}.entry.interactions must be an array."
            )

        entries.append(item)

    return entries


def print_quality_report(entries: list[dict[str, Any]]) -> None:
    total_interactions = 0
    entries_with_interactions = 0
    entries_without_interactions = 0
    entry_names: set[str] = set()
    entry_types: set[str] = set()

    for item in entries:
        entry = item["entry"]
        interactions = entry.get("interactions") or []

        total_interactions += len(interactions)

        if interactions:
            entries_with_interactions += 1
        else:
            entries_without_interactions += 1

        entry_name = clean_text(entry.get("entry_name"))
        entry_type = clean_text(entry.get("entry_type"))

        if entry_name:
            entry_names.add(normalize_text(entry_name))

        if entry_type:
            entry_types.add(entry_type)

    print("\nInteraction JSON quality report")
    print("-" * 65)
    print(f"Entries: {len(entries)}")
    print(f"Distinct normalized entry names: {len(entry_names)}")
    print(f"Entries with direct interactions: {entries_with_interactions}")
    print(f"Entries without direct interactions: {entries_without_interactions}")
    print(f"Direct interaction records: {total_interactions}")
    print(f"Distinct raw entry types: {len(entry_types)}")
    print("-" * 65)


# -----------------------------------------------------------------------------
# Document construction
# -----------------------------------------------------------------------------
def build_entry_document(
    item: dict[str, Any],
    entry_index: int,
) -> tuple[str, Document]:
    source = item["source"]
    entry = item["entry"]

    entry_name = clean_text(entry.get("entry_name"))
    entry_type = clean_text(entry.get("entry_type"))
    members = string_list(entry.get("members"))
    supplementary_information = string_list(
        entry.get("supplementary_information")
    )
    global_drug_references = string_list(
        entry.get("global_drug_references")
    )
    global_table_references = reference_text(
        entry.get("global_table_references")
    )
    table_references = reference_text(
        entry.get("table_references")
    )
    interactions = entry.get("interactions") or []

    source_reference = clean_text(source.get("reference"))
    edition = clean_text(source.get("edition"))
    section = clean_text(source.get("section"))
    source_printed_page = clean_text(source.get("printed_page"))
    source_pdf_page = clean_text(source.get("pdf_page"))
    confidence_score = clean_text(entry.get("confidence_score"))
    notes = clean_text(entry.get("notes"))

    content = remove_empty_lines(
        [
            "Document Kind: Drug Interaction Entry",
            f"Source File: {JSON_PATH.name}",
            f"Entry Index: {entry_index}",
            f"Entry Name: {entry_name}",
            f"Entry Type: {entry_type}",
            f"Entry Members: {', '.join(members)}",
            (
                "Supplementary Information: "
                + " | ".join(supplementary_information)
            ),
            (
                "Global Drug References: "
                + ", ".join(global_drug_references)
            ),
            f"Global Table References: {global_table_references}",
            f"Table References: {table_references}",
            f"Direct Interaction Count: {len(interactions)}",
            f"Source Reference: {source_reference}",
            f"Edition: {edition}",
            f"Section: {section}",
            f"Source Printed Page: {source_printed_page}",
            f"Source PDF Page: {source_pdf_page}",
            f"Confidence Score: {confidence_score}",
            f"Notes: {notes}",
        ]
    )

    subject_key = entity_key(entry_name)

    metadata = {
        "source": JSON_PATH.name,
        "document_kind": "interaction_entry",
        "entry_index": entry_index,
        "subject_key": subject_key,
        "subject_name": entry_name,
        "subject_aliases": json.dumps(
            members,
            ensure_ascii=False,
        ),
        "entry_type": entry_type,
        "interaction_count": len(interactions),
        "source_reference": source_reference,
        "edition": edition,
        "section": section,
        "printed_page": source_printed_page,
        "pdf_page": source_pdf_page,
    }

    document_id = stable_id(
        "interaction_entry",
        entry_index,
        entry_name,
        source_pdf_page,
    )

    return document_id, Document(
        page_content=content,
        metadata=metadata,
    )


def build_interaction_document(
    *,
    item: dict[str, Any],
    entry_index: int,
    interaction: dict[str, Any],
    interaction_index: int,
) -> tuple[str, Document]:
    source = item["source"]
    entry = item["entry"]

    entry_name = clean_text(entry.get("entry_name"))
    entry_type = clean_text(entry.get("entry_type"))
    entry_members = string_list(entry.get("members"))

    interacting_entity = interaction.get("interacting_entity") or {}
    applies_to = interaction.get("applies_to") or {}
    direction = interaction.get("interaction_direction") or {}
    effect = interaction.get("effect") or {}
    mechanism = interaction.get("mechanism") or {}
    action = interaction.get("action") or {}
    route = interaction.get("route_or_formulation") or {}

    if not isinstance(interacting_entity, dict):
        interacting_entity = {}
    if not isinstance(applies_to, dict):
        applies_to = {}
    if not isinstance(direction, dict):
        direction = {}
    if not isinstance(effect, dict):
        effect = {}
    if not isinstance(mechanism, dict):
        mechanism = {}
    if not isinstance(action, dict):
        action = {}
    if not isinstance(route, dict):
        route = {}

    related_name = clean_text(interacting_entity.get("name"))
    related_type = clean_text(interacting_entity.get("type"))
    related_members = string_list(interacting_entity.get("members"))

    applies_to_scope = clean_text(applies_to.get("scope"))
    applies_to_members = string_list(applies_to.get("members"))

    causing_entity = clean_text(direction.get("causing_entity"))
    affected_entity = clean_text(direction.get("affected_entity"))

    effect_raw = clean_text(effect.get("raw_text"))
    effect_type = clean_text(effect.get("normalized_type"))

    mechanism_type = clean_text(mechanism.get("type"))
    mechanism_raw = clean_text(mechanism.get("raw_text"))

    action_category = clean_text(action.get("category"))
    action_raw = clean_text(action.get("raw_text"))

    severity = clean_text(interaction.get("severity"))
    evidence = clean_text(interaction.get("evidence"))

    route_raw = clean_text(route.get("raw_text"))
    route_name = clean_text(route.get("route"))
    formulation = clean_text(route.get("formulation"))

    qualifiers = string_list(interaction.get("qualifiers"))
    cross_references = reference_text(interaction.get("cross_references"))

    raw_text = clean_text(interaction.get("raw_text"))
    printed_page = clean_text(interaction.get("printed_page"))
    pdf_page = clean_text(interaction.get("pdf_page"))

    source_reference = clean_text(source.get("reference"))
    edition = clean_text(source.get("edition"))
    section = clean_text(source.get("section"))

    content = remove_empty_lines(
        [
            "Document Kind: Drug Interaction Pair",
            f"Source File: {JSON_PATH.name}",
            f"Entry Index: {entry_index}",
            f"Interaction Index: {interaction_index}",
            f"Entry Name: {entry_name}",
            f"Entry Type: {entry_type}",
            f"Entry Members: {', '.join(entry_members)}",
            f"Interacting Entity: {related_name}",
            f"Interacting Entity Type: {related_type}",
            f"Interacting Entity Members: {', '.join(related_members)}",
            f"Applies To Scope: {applies_to_scope}",
            f"Applies To Members: {', '.join(applies_to_members)}",
            f"Causing Entity: {causing_entity}",
            f"Affected Entity: {affected_entity}",
            f"Effect: {effect_raw}",
            f"Normalized Effect Type: {effect_type}",
            f"Mechanism Type: {mechanism_type}",
            f"Mechanism: {mechanism_raw}",
            f"Action Category: {action_category}",
            f"Action: {action_raw}",
            f"Severity: {severity}",
            f"Evidence: {evidence}",
            f"Route / Formulation Text: {route_raw}",
            f"Route: {route_name}",
            f"Formulation: {formulation}",
            f"Qualifiers: {', '.join(qualifiers)}",
            f"Cross References: {cross_references}",
            f"Raw Interaction Text: {raw_text}",
            f"Printed Page: {printed_page}",
            f"PDF Page: {pdf_page}",
            f"Source Reference: {source_reference}",
            f"Edition: {edition}",
            f"Section: {section}",
        ]
    )

    subject_key = entity_key(entry_name)
    related_key = entity_key(related_name)

    pair_key = "::".join(
        sorted([subject_key, related_key])
    )

    metadata = {
        "source": JSON_PATH.name,
        "document_kind": "interaction_pair",
        "entry_index": entry_index,
        "interaction_index": interaction_index,
        "subject_key": subject_key,
        "subject_name": entry_name,
        "subject_aliases": json.dumps(
            entry_members,
            ensure_ascii=False,
        ),
        "related_key": related_key,
        "related_name": related_name,
        "related_aliases": json.dumps(
            related_members,
            ensure_ascii=False,
        ),
        "pair_key": pair_key,
        "entry_type": entry_type,
        "related_type": related_type,
        "applies_to_scope": applies_to_scope,
        "applies_to_members": json.dumps(
            applies_to_members,
            ensure_ascii=False,
        ),
        "causing_entity": causing_entity,
        "affected_entity": affected_entity,
        "effect_type": effect_type,
        "severity": severity,
        "evidence": evidence,
        "action_category": action_category,
        "printed_page": printed_page,
        "pdf_page": pdf_page,
        "source_reference": source_reference,
    }

    document_id = stable_id(
        "interaction_pair",
        entry_index,
        interaction_index,
        entry_name,
        related_name,
        raw_text,
        pdf_page,
    )

    return document_id, Document(
        page_content=content,
        metadata=metadata,
    )


def build_documents(
    entries: list[dict[str, Any]],
) -> tuple[list[str], list[Document], int, int]:
    ids: list[str] = []
    documents: list[Document] = []

    entry_document_count = 0
    interaction_document_count = 0

    for entry_index, item in enumerate(
        entries,
        start=1,
    ):
        entry_id, entry_document = build_entry_document(
            item,
            entry_index,
        )

        ids.append(entry_id)
        documents.append(entry_document)
        entry_document_count += 1

        interactions = item["entry"].get("interactions") or []

        for interaction_index, interaction in enumerate(
            interactions,
            start=1,
        ):
            if not isinstance(interaction, dict):
                raise ValueError(
                    "Interaction must be an object: "
                    f"entry={entry_index}, interaction={interaction_index}"
                )

            interaction_id, interaction_document = build_interaction_document(
                item=item,
                entry_index=entry_index,
                interaction=interaction,
                interaction_index=interaction_index,
            )

            ids.append(interaction_id)
            documents.append(interaction_document)
            interaction_document_count += 1

    return (
        ids,
        documents,
        entry_document_count,
        interaction_document_count,
    )


# -----------------------------------------------------------------------------
# Chroma persistence
# -----------------------------------------------------------------------------
def delete_collection_if_exists(
    client: chromadb.PersistentClient,
) -> None:
    if not RECREATE_COLLECTION:
        return

    existing_names = {
        collection.name
        if hasattr(collection, "name")
        else str(collection)
        for collection in client.list_collections()
    }

    if COLLECTION_NAME in existing_names:
        client.delete_collection(
            name=COLLECTION_NAME
        )

        print(
            f"Deleted old collection: {COLLECTION_NAME}"
        )


def is_rate_limit_error(error: Exception) -> bool:
    message = str(error).lower()

    return (
        "429" in message
        or "resource_exhausted" in message
        or "quota" in message
        or "rate limit" in message
    )


def add_documents_in_batches(
    store: Chroma,
    ids: list[str],
    documents: list[Document],
) -> None:
    if len(ids) != len(documents):
        raise ValueError(
            "IDs count and documents count must match."
        )

    for start in range(
        0,
        len(documents),
        BATCH_SIZE,
    ):
        end = min(
            start + BATCH_SIZE,
            len(documents),
        )

        attempt = 0

        while True:
            try:
                store.add_documents(
                    documents=documents[start:end],
                    ids=ids[start:end],
                )

                print(
                    "Stored documents: "
                    f"{end}/{len(documents)}"
                )

                break

            except Exception as error:
                attempt += 1

                if (
                    not is_rate_limit_error(error)
                    or attempt > MAX_RETRIES
                ):
                    raise

                wait_seconds = max(
                    BATCH_DELAY_SECONDS,
                    min(
                        60.0,
                        BATCH_DELAY_SECONDS * (2 ** (attempt - 1)),
                    ),
                )

                print(
                    "Embedding rate limit reached. "
                    f"Retry {attempt}/{MAX_RETRIES} "
                    f"after {wait_seconds:.1f}s..."
                )

                time.sleep(wait_seconds)

        if end < len(documents) and BATCH_DELAY_SECONDS > 0:
            time.sleep(BATCH_DELAY_SECONDS)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main() -> None:
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY in the project .env file."
        )

    if not COLLECTION_NAME:
        raise ValueError(
            "DRUG_INTERACTIONS_COLLECTION cannot be empty."
        )

    if BATCH_SIZE <= 0:
        raise ValueError(
            "INTERACTIONS_EMBEDDING_BATCH_SIZE must be greater than zero."
        )

    if BATCH_DELAY_SECONDS < 0:
        raise ValueError(
            "INTERACTIONS_EMBEDDING_BATCH_DELAY_SECONDS cannot be negative."
        )

    if MAX_RETRIES < 0:
        raise ValueError(
            "INTERACTIONS_EMBEDDING_MAX_RETRIES cannot be negative."
        )

    print(f"Reading interaction JSON: {JSON_PATH}")
    print(f"Chroma directory: {PERSIST_DIRECTORY}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Delay between batches: {BATCH_DELAY_SECONDS}s")

    entries = load_entries()

    print_quality_report(entries)

    (
        document_ids,
        documents,
        entry_document_count,
        interaction_document_count,
    ) = build_documents(entries)

    if len(document_ids) != len(set(document_ids)):
        raise ValueError(
            "Duplicate Chroma IDs were generated."
        )

    print("\nDocuments prepared")
    print(f"Entry documents: {entry_document_count}")
    print(f"Interaction documents: {interaction_document_count}")
    print(f"Total documents: {len(documents)}")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    PERSIST_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(PERSIST_DIRECTORY)
    )

    delete_collection_if_exists(client)

    store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    add_documents_in_batches(
        store=store,
        ids=document_ids,
        documents=documents,
    )

    stored_count = client.get_collection(
        name=COLLECTION_NAME
    ).count()

    if stored_count != len(documents):
        raise RuntimeError(
            "Stored document count does not match the prepared document count. "
            f"Expected {len(documents)}, found {stored_count}."
        )

    print("\nDone")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored records: {stored_count}")
    print(f"Database directory: {PERSIST_DIRECTORY}")


if __name__ == "__main__":
    main()
