import os
import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv

load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

def send_email_alert(
receiver_email,
subject,
message
):

    try:

        msg = MIMEMultipart()

        msg["From"] = EMAIL_USER
        msg["To"] = receiver_email
        msg["Subject"] = subject

        msg.attach(
            MIMEText(
                message,
                "plain",
                "utf-8"
            )
        )

        server = smtplib.SMTP(
            "smtp.gmail.com",
            587
        )

        server.starttls()

        server.login(
            EMAIL_USER,
            EMAIL_PASS
        )

        server.sendmail(
            EMAIL_USER,
            receiver_email,
            msg.as_string()
        )

        server.quit()

        return {
            "success": True,
            "message": "Email sent successfully"
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }