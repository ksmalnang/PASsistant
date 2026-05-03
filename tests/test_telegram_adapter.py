"""Telegram adapter tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config.settings import Settings
from src.telegram_bot.adapter import UNSUPPORTED_MESSAGE, WELCOME_MESSAGE, TelegramBotAdapter
from src.utils.state import Citation


class FakeBot:
    def __init__(self):
        self.sent_messages: list[dict[str, object]] = []
        self.sent_actions: list[dict[str, object]] = []
        self.file_bytes = b""

    async def send_message(self, chat_id: int, text: str) -> None:
        self.sent_messages.append({"chat_id": chat_id, "text": text})

    async def send_chat_action(self, chat_id: int, action: str) -> None:
        self.sent_actions.append({"chat_id": chat_id, "action": action})

    async def get_file(self, file_id: str):
        file_bytes = self.file_bytes

        class FakeFile:
            async def download_as_bytearray(self):
                return bytearray(file_bytes)

        return FakeFile()


class FakeChatService:
    def __init__(self):
        self.message_calls: list[tuple[str, str | None]] = []
        self.upload_calls: list[tuple[str, list[tuple[str, bytes]], str | None]] = []
        self.message_response = SimpleNamespace(
            response="Answer",
            intent="general_chat",
            documents_processed=0,
            citations=[],
        )
        self.upload_response = SimpleNamespace(
            response="Processed upload",
            intent="upload_document",
            documents_processed=1,
            citations=[],
        )

    async def handle_chat_message(self, message: str, session_id: str | None = None):
        self.message_calls.append((message, session_id))
        return self.message_response

    async def handle_chat_upload(
        self,
        message: str,
        files: list[tuple[str, bytes]],
        session_id: str | None = None,
    ):
        self.upload_calls.append((message, files, session_id))
        return self.upload_response


def build_settings(**overrides) -> Settings:
    base = {
        "TELEGRAM_ENABLED": True,
        "TELEGRAM_BOT_TOKEN": "token",
        "APP_ENV": "development",
    }
    base.update(overrides)
    return Settings(**base)


def make_update(
    *,
    chat_id: int = 1234,
    user_id: int = 5678,
    text: str | None = None,
    caption: str | None = None,
    document=None,
    photo=None,
    update_id: int = 99,
):
    chat = SimpleNamespace(id=chat_id)
    user = SimpleNamespace(id=user_id)
    message = SimpleNamespace(
        chat=chat,
        from_user=user,
        text=text,
        caption=caption,
        document=document,
        photo=photo,
    )
    return SimpleNamespace(update_id=update_id, effective_message=message)


@pytest.mark.asyncio
async def test_text_update_calls_chat_service():
    bot = FakeBot()
    service = FakeChatService()
    adapter = TelegramBotAdapter(service, bot, build_settings())

    await adapter.handle_update(make_update(text="What is my GPA?"))

    assert service.message_calls == [("What is my GPA?", "telegram:1234")]
    assert bot.sent_actions == [{"chat_id": 1234, "action": "typing"}]
    assert bot.sent_messages == [{"chat_id": 1234, "text": "Answer"}]


@pytest.mark.asyncio
async def test_start_command_returns_welcome_without_calling_agent():
    bot = FakeBot()
    service = FakeChatService()
    adapter = TelegramBotAdapter(service, bot, build_settings())

    await adapter.handle_update(make_update(text="/start"))

    assert service.message_calls == []
    assert bot.sent_messages == [{"chat_id": 1234, "text": WELCOME_MESSAGE}]


@pytest.mark.asyncio
async def test_unsupported_update_replies_with_controlled_message():
    bot = FakeBot()
    service = FakeChatService()
    adapter = TelegramBotAdapter(service, bot, build_settings())

    await adapter.handle_update(make_update())

    assert service.message_calls == []
    assert bot.sent_messages == [{"chat_id": 1234, "text": UNSUPPORTED_MESSAGE}]


@pytest.mark.asyncio
async def test_long_response_is_split_into_multiple_messages():
    bot = FakeBot()
    service = FakeChatService()
    service.message_response = SimpleNamespace(
        response=("A" * 4090) + "\n\n" + ("B" * 40),
        intent="general_chat",
        documents_processed=0,
        citations=[],
    )
    adapter = TelegramBotAdapter(service, bot, build_settings())

    await adapter.handle_update(make_update(text="Long reply"))

    assert len(bot.sent_messages) == 2
    assert bot.sent_messages[0]["chat_id"] == 1234
    assert len(bot.sent_messages[0]["text"]) <= 4096
    assert len(bot.sent_messages[1]["text"]) <= 4096


@pytest.mark.asyncio
async def test_existing_response_sources_are_not_duplicated():
    bot = FakeBot()
    service = FakeChatService()
    service.message_response = SimpleNamespace(
        response="Profile data\n\nSources:\n[1] kurikulum.pdf",
        intent="query_document",
        documents_processed=0,
        citations=[
            Citation(
                id=1,
                filename="kurikulum.pdf",
                section="3.1 Profil Lulusan",
                page=14,
                snippet="Profil lulusan dan deskripsinya",
            )
        ],
    )
    adapter = TelegramBotAdapter(service, bot, build_settings())

    await adapter.handle_update(make_update(text="Show sources"))

    assert "Sources:" in bot.sent_messages[0]["text"]
    assert "[1] kurikulum.pdf" in bot.sent_messages[0]["text"]
    assert bot.sent_messages[0]["text"].count("Sources:") == 1


@pytest.mark.asyncio
async def test_document_update_calls_upload_handler_with_downloaded_bytes():
    bot = FakeBot()
    bot.file_bytes = b"%PDF-1.7"
    service = FakeChatService()
    adapter = TelegramBotAdapter(service, bot, build_settings())
    document = SimpleNamespace(
        file_id="file-1",
        file_name="transcript.pdf",
        mime_type="application/pdf",
        file_size=8,
    )

    await adapter.handle_update(make_update(caption="Check this", document=document))

    assert service.upload_calls == [
        ("Check this", [("transcript.pdf", b"%PDF-1.7")], "telegram:1234")
    ]
    assert bot.sent_actions == [{"chat_id": 1234, "action": "typing"}]
    assert bot.sent_messages == [{"chat_id": 1234, "text": "Processed upload"}]


@pytest.mark.asyncio
async def test_file_size_rejection_avoids_calling_agent():
    bot = FakeBot()
    service = FakeChatService()
    adapter = TelegramBotAdapter(
        service,
        bot,
        build_settings(TELEGRAM_MAX_FILE_BYTES=4),
    )
    document = SimpleNamespace(
        file_id="file-1",
        file_name="transcript.pdf",
        mime_type="application/pdf",
        file_size=10,
    )

    await adapter.handle_update(make_update(document=document))

    assert service.upload_calls == []
    assert bot.sent_messages == [
        {"chat_id": 1234, "text": "File is too large. Maximum supported size is 4 bytes."}
    ]
