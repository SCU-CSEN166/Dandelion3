import json


def parse_messages(file_path):
    """
    Read synthetic WhatsApp messages from a JSON file
    and convert them into standardized records.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        messages = json.load(file)

    parsed_messages = []

    for message in messages:
        parsed_message = {
            "sender": message.get("sender"),
            "timestamp": message.get("timestamp"),
            "text": message.get("message", ""),
            "type": message.get("type", "text")
        }

        parsed_messages.append(parsed_message)

    return parsed_messages


if __name__ == "__main__":
    results = parse_messages("data/synthetic/sample_messages.json")

    for message in results:
        print(message)
