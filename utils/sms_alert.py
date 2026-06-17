def send_sms_alert(
phone,
message
):

    print("\n===================")
    print("SMS ALERT SENT")
    print("===================")

    print("Phone:", phone)

    print("Message:")
    print(message)

    return {
        "success": True
    }
