import smtplib, ssl
from email.mime.text import MIMEText
import yaml

# https://news.mynavi.jp/article/zeropython-51/

class Gmailsender():
    def __init__(self, subject="実行終了通知", mail_to="a109k9i8@gmail.com") -> None:
        self.subject = subject
        self.mail_to = mail_to

    def send(self, body):
        with open('/home/haoki/Documents/vscode-workplaces/lie_detector/src/utils/config.yaml') as f:
            cfg = yaml.safe_load(f)


        # 以下にGmailの設定を書き込む★ --- (*1)
        gmail_account = cfg['account']
        gmail_password = cfg['password']

        # メールデータ(MIME)の作成 --- (*2)
        msg = MIMEText(body, "html")
        msg["Subject"] = self.subject
        msg["To"] = self.mail_to
        msg["From"] = gmail_account

        # Gmailに接続 --- (*3)
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465,
            context=ssl.create_default_context())
        server.login(gmail_account, gmail_password)
        server.send_message(msg) # メールの送信