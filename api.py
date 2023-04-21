import json
import logging
from itertools import groupby
from typing import Type, Optional, Dict, Any, List
import uuid
import langchain
from langchain.chains import ChatVectorDBChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from pydantic import HttpUrl
from pytube import YouTube
from steamship import File, Task, Tag, SteamshipError, Steamship, MimeTypes, DocTag
from steamship.data import TagValueKey
from steamship.invocable import Config
from steamship.invocable import PackageService, post, get
from steamship_langchain.llms.openai import OpenAIChat
from steamship_langchain.vectorstores import SteamshipVectorStore
import requests

from chat_history import ChatHistory
from prompts import qa_prompt, condense_question_prompt

langchain.llm_cache = None

DEBUG = False


class AskMyCourse(PackageService):
    class AskMyCourseConfig(Config):
        model_name: str = "gpt-3.5-turbo"
        context_window_size: int = 200
        context_window_overlap: int = 50

    config: AskMyCourseConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index_name = self.client.config.workspace_handle + "_index"
        self.qa_chain = self._get_chain()

    @classmethod
    def config_cls(cls) -> Type[Config]:
        return cls.AskMyCourseConfig

    def _get_index(self):
        return SteamshipVectorStore(
            client=self.client,
            index_name=self.index_name,
            embedding="text-embedding-ada-002",
        )

    def _get_chain(self):
        doc_index = self._get_index()

        llm = OpenAIChat(client=self.client, model_name=self.config.model_name, temperature=0, verbose=DEBUG)

        doc_chain = load_qa_chain(
            llm,
            chain_type="stuff",
            prompt=qa_prompt,
            verbose=DEBUG,
        )
        question_chain = LLMChain(
            llm=OpenAIChat(client=self.client, model_name=self.config.model_name, temperature=0, verbose=DEBUG),
            prompt=condense_question_prompt,
        )
        return ChatVectorDBChain(
            vectorstore=doc_index,
            combine_docs_chain=doc_chain,
            question_generator=question_chain,
            return_source_documents=True,
            top_k_docs_for_context=2,
        )

    def _get_lectures(self) -> List[Dict[str, Any]]:
        documents = []
        tags = Tag.query(self.client, 'kind "source" or kind "title" or kind "status"').tags
        for key, tag_group in groupby(sorted(tags, key=lambda x: x.file_id), key=lambda x: x.file_id):
            tag_group = list(tag_group)
            source_tags = [tag for tag in tag_group if tag.kind == "source"]
            status_tags = [tag for tag in tag_group if tag.kind == "status"]
            title_tags = [tag for tag in tag_group if tag.kind == "title"]
            if source_tags and status_tags:
                documents.append(
                    {
                        "source": source_tags[0].name,
                        "status": status_tags[0].name,
                        "title": title_tags[0].name if title_tags else "unknown",
                    }
                )
        return documents

    @get("/lectures", public=True)
    def get_lectures(self) -> List[Dict[str, Any]]:
        return self._get_lectures()

    @get("/sources", public=True)
    def get_sources(self) -> List[Dict[str, Any]]:
        return self._get_lectures()

    def _update_file_status(self, file: File, status: str) -> None:
        file = file.refresh()
        status_tags = [tag for tag in file.tags if tag.kind == "status"]
        for status_tag in status_tags:
            try:
                status_tag.client = self.client
                status_tag.delete()
            except SteamshipError:
                pass

        Tag.create(self.client, file_id=file.id, kind="status", name=status)

    @post("/index_lecture")
    def index_lecture(self, file_id: str, source: str) -> bool:
        file = File.get(self.client, _id=file_id)
        self._update_file_status(file, "Indexing")
        tags = file.blocks[0].tags

        timestamps = [tag for tag in tags if tag.kind == "timestamp"]
        timestamps = sorted(timestamps, key=lambda x: x.start_idx)

        documents = []
        for i in range(
                0, len(timestamps), self.config.context_window_size - self.config.context_window_overlap
        ):
            timestamp_tags_window = timestamps[i: i + self.config.context_window_size]
            page_content = " ".join(tag.name for tag in timestamp_tags_window)
            doc = Document(
                page_content=page_content,
                metadata={
                    "start_time": timestamp_tags_window[0].value["start_time"],
                    "end_time": timestamp_tags_window[-1].value["end_time"],
                    "start_idx": timestamp_tags_window[-1].start_idx,
                    "end_idx": timestamp_tags_window[-1].end_idx,
                    "source": source,
                },
            )
            documents.append(doc)
        self._get_index().add_documents(documents)
        self._update_file_status(file, "Indexed")
        return True

    @post("/index_pdf")
    def index_pdf(self, file_id: str, source: str) -> bool:
        file = File.get(self.client, _id=file_id)
        self._update_file_status(file, "Indexing")

        # For PDFs, we iterate over the blocks (block = page) and then split each chunk of texts into the context
        # window units.

        documents = []

        for block in file.blocks:
            # Load the page_id from the block if it exists
            page_id = None
            for tag in block.tags:
                if tag.name == DocTag.PAGE:
                    page_num = tag.value.get(TagValueKey.NUMBER_VALUE)
                    if page_num is not None:
                        page_id = page_num

            for i in range(0, len(block.text), self.config.context_window_size):
                # Calculate the extent of the window plus the overlap at the edges
                min_range = max(0, i - self.config.context_window_overlap)
                max_range = i + self.config.context_window_size + self.config.context_window_overlap

                # Get the text covering that chunk.
                chunk = block.text[min_range:max_range]

                # Create a Document.
                # TODO(ted): See if there's a way to support the LC Embedding Index abstraction that lets us use Tag here.
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": source,
                        "file_id": file.id,
                        "block_id": block.id,
                        "page": page_id
                    },
                )
                documents.append(doc)

        self._get_index().add_documents(documents)
        self._update_file_status(file, "Indexed")
        return True

    @post("/transcribe_lecture")
    def transcribe_lecture(self, task_id: str, source: str):
        file_create_task = Task.get(self.client, task_id)
        file = File.get(self.client, json.loads(file_create_task.output)["file"]["id"])

        Tag.create(self.client, file_id=file.id, kind="source", name=source)
        try:
            Tag.create(self.client, file_id=file.id, kind="title", name=YouTube(source).title)
        except Exception as e:
            logging.warning(f"Unable to access title of YouTube video {e}")
            Tag.create(self.client, file_id=file.id, kind="title", name=source)


        self._update_file_status(file, "Transcribing")

        blockifier = self.client.use_plugin("s2t-blockifier-default")
        blockify_file_task = file.blockify(blockifier.handle)

        return self.invoke_later(
            method="index_lecture",
            arguments={"file_id": file.id, "source": source},
            wait_on_tasks=[blockify_file_task],
        )

    @post("/blockify_pdf")
    def blockify_pdf(self, file_id: str, source: str):
        file = File.get(self.client, _id=file_id)

        self._update_file_status(file, "Parsing")

        blockifier = self.client.use_plugin("pdf-blockifier")
        blockify_file_task = file.blockify(blockifier.handle)

        return self.invoke_later(
            method="index_pdf",
            arguments={"file_id": file_id, "source": source},
            wait_on_tasks=[blockify_file_task],
        )

    @post("/add_lecture")
    def add_lecture(self, youtube_url: HttpUrl) -> Task:
        file_importer = self.client.use_plugin("youtube-file-importer")
        file_create_task = File.create_with_plugin(
            self.client, plugin_instance=file_importer.handle, url=youtube_url
        )
        return self.invoke_later(
            method="transcribe_lecture",
            arguments={"task_id": file_create_task.task_id, "source": youtube_url},
            wait_on_tasks=[file_create_task],
        )

    @post("/add_pdf")
    def add_pdf(self, pdf_url: HttpUrl) -> Task:
        response = requests.get(pdf_url)
        file = File.create(self.client, content=response.content, mime_type=MimeTypes.PDF)

        # Hacky way to get the last segment of the URL but drop the query & hash
        title = pdf_url.split('/')[-1]
        title = title.split('?')[0]
        title = title.split('#')[0]

        # Tag the title for provenance reporting
        Tag.create(self.client, file_id=file.id, kind="source", name=pdf_url)
        Tag.create(self.client, file_id=file.id, kind="title", name=title)

        return self.invoke_later(
            method="blockify_pdf",
            arguments={"file_id": file.id, "source": pdf_url},
        )

    @post("/add_url")
    def add_url(self, url: HttpUrl) -> Task:
        if "youtube.com" in url:
            return self.add_lecture(url)
        elif "youtu.be" in url:
            return self.add_lecture(url)
        elif ".pdf" in url:
            return self.add_pdf(url)
        else:
            raise SteamshipError(message="Only youtube URLs and URLs of PDF files are currently supported.")

    @post("/answer", public=True)
    def answer(
            self, question: str, chat_session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        chat_session_id = chat_session_id or "default"
        chat_history = ChatHistory(self.client, chat_session_id)

        result = self.qa_chain(
            {"question": question, "chat_history": chat_history.load()}
        )
        if len(result["source_documents"]) == 0:
            return {
                "answer": "No sources found to answer your question. Please try another question.",
                "sources": result["source_documents"],
            }

        answer = result["answer"]
        sources = result["source_documents"]
        chat_history.append(question, answer)

        return {"answer": answer.strip(), "sources": sources}


def test_with_pdf():
    url = "https://www.with.org/tao_te_ching_en.pdf"
    client = Steamship(workspace="tao-test")
    app = AskMyCourse(client)
    task = app.add_pdf(url)
    task.wait()
    print("Waited")
    print(app.get_lectures())


def test_with_video():
    url = "https://www.youtube.com/watch?v=LXDZ6aBjv_I"
    client = Steamship(workspace="youtube-test")
    app = AskMyCourse(client)
    task = app.add_lecture(url)
    print("Waited")
    print(app.get_lectures())


if __name__ == '__main__':
    test_with_pdf()


