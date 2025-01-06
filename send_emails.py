import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.mime.base import MIMEBase
from email import encoders

def send_email(sender_email, sender_password, recipient_emails, subject, body, smtp_server, smtp_port=587, attachment=None):
    """
    Function to send an email using SMTP, with support for multiple recipients and optional attachment.

    Parameters:
        sender_email (str): Sender's email address.
        sender_password (str): Sender's email password or app-specific password.
        recipient_emails (list): List of recipient email addresses.
        subject (str): Subject of the email.
        body (str): Body content of the email.
        smtp_server (str): SMTP server address.
        smtp_port (int): Port number for the SMTP server.
        attachment (str): File path of the attachment (optional).
    """
    try:
        # Create the email object
        message = MIMEMultipart()
        message['From'] = Header(sender_email, 'utf-8')
        message['To'] = Header(", ".join(recipient_emails), 'utf-8')  # Join recipient emails with commas
        message['Subject'] = Header(subject, 'utf-8')

        # Attach the email body
        message.attach(MIMEText(body, 'plain', 'utf-8'))

        # If there is an attachment, add it
        if attachment:
            try:
                with open(attachment, 'rb') as file:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(file.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename="{attachment.split("/")[-1]}"'
                )
                message.attach(part)
            except FileNotFoundError:
                print(f"Attachment not found: {attachment}")
                return

        # Connect to the SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Enable secure connection
        server.login(sender_email, sender_password)

        # Send the email
        server.sendmail(sender_email, recipient_emails, message.as_string())
        print(f"Email sent successfully to: {', '.join(recipient_emails)}")
    except Exception as e:
        print(f"Failed to send email: {e}")
    finally:
        server.quit()