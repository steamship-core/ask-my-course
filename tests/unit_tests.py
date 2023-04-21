from transports import ChatMessage, SteamshipWidgetTransport


def test_thing():
    cm = ChatMessage(chat_id="test_id")
    assert cm.get_chat_id() == "test_id"

    cm = ChatMessage(message_id="test_id")
    assert cm.get_message_id() == "test_id"

def test_web_transport():
    c = SteamshipWidgetTransport()
    ib = c.parse_inbound({"question": "What's up"})
    assert ib.get_chat_id() == "default"