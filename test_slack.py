# test_slack.py
from dotenv import load_dotenv
import os
from notifier import send_slack

# load .env from current working directory
load_dotenv(dotenv_path=".venv/.env")

print("SLACK_WEBHOOK_URL present?:", bool(os.getenv("SLACK_WEBHOOK_URL")))
send_slack("Test from AutoTrader — BUY alert wiring OK :white_check_mark:")
print("Sent test (check Slack).")
