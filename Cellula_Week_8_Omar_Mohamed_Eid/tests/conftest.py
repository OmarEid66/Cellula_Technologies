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


class FakeTranscriber:
    """Speech-to-text stand-in: returns a fixed transcript and records what it was given."""

    def __init__(self, text="What is the average CNN accuracy across the three papers?"):
        self.text, self.calls = text, []

    def transcribe(self, audio, filename, language=None):
        self.calls.append((len(audio), filename, language))
        return self.text


@pytest.fixture
def transcriber():
    return FakeTranscriber()


@pytest.fixture
def service(settings, transcriber, tmp_path):
    """Full AssistantService on the scripted LLM + hashing embeddings, with the sample papers already ingested."""
    from fakes import HashEmbeddings
    from rag_agents.agents.report import ReportAgent
    from rag_agents.service import AssistantService
    from rag_agents.system import RagSystem
    from test_report import scripted
    from test_workflow import make_llm

    settings.reports_dir = str(tmp_path / "reports")
    llm, _ = make_llm()
    rag = RagSystem(settings=settings, llm=llm, embeddings=HashEmbeddings())
    rag.ingest(sorted(SAMPLES.glob("*.txt")))
    report_llm = scripted([
        [("add_title_block", {"title": "Scripted report"}), ("add_markdown", {"markdown": "Body from the agent"})],
    ])
    return AssistantService(rag=rag, settings=settings, transcriber=transcriber,
                            report_agent=ReportAgent(report_llm, settings))
