import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fakes import HashEmbeddings  # noqa: E402
from rag_agents.config import Settings  # noqa: E402
from rag_agents.ingestion import ingest_files  # noqa: E402
from rag_agents.vectorstore import KnowledgeBase  # noqa: E402

SAMPLES = Path(__file__).resolve().parents[1] / "sample_docs"


@pytest.fixture
def settings(tmp_path):
    return Settings(chroma_dir=str(tmp_path / "chroma"), collection_name="test")


@pytest.fixture
def kb(settings):
    return KnowledgeBase(HashEmbeddings(), settings.chroma_dir, settings.collection_name)


@pytest.fixture
def loaded_kb(kb, settings):
    reports = ingest_files(sorted(SAMPLES.glob("*.txt")), kb, settings)
    assert all(r.ok for r in reports)
    return kb
