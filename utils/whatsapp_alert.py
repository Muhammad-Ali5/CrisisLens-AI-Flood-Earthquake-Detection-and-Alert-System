import urllib.parse

def generate_whatsapp_link(
phone,
message
):

    encoded_message = urllib.parse.quote(
        message
    )

    return (
        f"https://wa.me/{phone}"
        f"?text={encoded_message}"
    )

