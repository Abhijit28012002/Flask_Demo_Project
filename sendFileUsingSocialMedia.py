from twilio.rest import Client
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# Send Whatsapp Message Function
def send_whatsapp_message(to_number, message_body):
    # Twilio Account SID and Auth Token
    account_sid = 'your_account_sid'
    auth_token = 'your_auth_token'

    # Create a Twilio client
    client = Client(account_sid, auth_token)

    # Your Twilio WhatsApp number (you must enable it in your Twilio console)
    from_whatsapp_number = 'whatsapp:+14155238886'

    # Send the message
    message = client.messages.create(
        body=message_body,
        from_=from_whatsapp_number,
        to=f'whatsapp:{to_number}'
    )

    # Print the SID (unique ID) of the message
    print(f'Message sent with SID: {message.sid}')



# Send Email function
def send_email(subject, body, to_email):
    # Sender's email address and password
    sender_email = 'your_email@gmail.com'
    password = 'your_password'

    # Recipient's email address
    receiver_email = to_email

    # Message setup
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = subject

    # Attach the body of the email
    message.attach(MIMEText(body, 'plain'))

    # SMTP server setup (for Gmail)
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587

    # Establish a secure connection to the SMTP server
    context = ssl.create_default_context()

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls(context=context)
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())

    print(f'Email sent to {receiver_email} successfully.')

