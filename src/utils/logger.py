import os
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

class EmailHandler(logging.Handler):
    def __init__(self, mailhost, fromaddr, toaddr, subject, credentials):
        logging.Handler.__init__(self)
        self.mailhost = mailhost
        self.mailport = smtplib.SMTP_PORT
        self.fromaddr = fromaddr
        self.toaddr = toaddr
        self.subject = subject
        self.credentials = credentials

    def emit(self, record):
        try:
            msg = MIMEMultipart()
            msg['From'] = self.fromaddr
            msg['To'] = self.toaddr
            msg['Subject'] = self.subject
            msg.attach(MIMEText(self.format(record)))

            smtp = smtplib.SMTP(self.mailhost, self.mailport)
            smtp.ehlo()
            smtp.starttls()
            smtp.login(self.credentials[0], self.credentials[1])
            smtp.sendmail(self.fromaddr, self.toaddr, msg.as_string())
            smtp.quit()
        except Exception:
            self.handleError(record)


class OperationEndNotifier:
    def __init__(self, subject="Operation Notification"):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(EmailHandler(
            mailhost="smtp.gmail.com",
            fromaddr=os.environ['GMAIL_ADDRESS'], 
            toaddr=os.environ['GMAIL_ADDRESS'],
            subject=subject,
            credentials=(os.environ['GMAIL_ADDRESS'], os.environ['GMAIL_APP_PASSWORD'])))
        
    def notify(self, operation=None, status=None, message=None):
        self.logger.info(f'Operation: {operation}, Status: {status}, Message: {message}')