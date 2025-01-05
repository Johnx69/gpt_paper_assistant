import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

def send_email(sender_email, sender_password, recipient_email, subject, body, smtp_server, smtp_port):
    """
    Function to send an email using SMTP.

    Parameters:
        sender_email (str): Sender's email address.
        sender_password (str): Sender's email password or app-specific password.
        recipient_email (str): Recipient's email address.
        subject (str): Subject of the email.
        body (str): Body content of the email.
        smtp_server (str): SMTP server address.
        smtp_port (int): Port number for the SMTP server.
    """
    try:
        # Create the email object
        message = MIMEMultipart()
        message['From'] = Header(sender_email, 'utf-8')
        message['To'] = Header(recipient_email, 'utf-8')
        message['Subject'] = Header(subject, 'utf-8')

        # Attach the email body
        message.attach(MIMEText(body, 'plain', 'utf-8'))

        # Connect to the SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Enable secure connection
        server.login(sender_email, sender_password)

        # Send the email
        server.sendmail(sender_email, recipient_email, message.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")
    finally:
        server.quit()

if __name__ == "__main__":
    sender_email = "yifanli183313@gmail.com"        # sender email
    sender_password = "1833130102"                  # sender passwd
    recipient_email = "17732497527@163.com"         # recipient email
    subject = "Test Email from Python"              # main subject
    body = "Hello, this is a test email sent from Python!"  # main body
    smtp_server = "smtp.gmail.com"                # SMTP server address, e.g., Gmail: smtp.gmail.com
    smtp_port = 587                                 # SMTP port, e.g., Gmail: 587

    send_email(sender_email, sender_password, recipient_email, subject, body, smtp_server, smtp_port)
