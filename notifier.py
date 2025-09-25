# notifier.py
import os
import logging
import requests

SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL")

def send_slack(text, username="AutoTrader", channel=None):
    """
    Send a simple Slack message via incoming webhook.
    Keep payload minimal to avoid rate limits.
    """
    if not SLACK_WEBHOOK:
        logging.debug("No SLACK_WEBHOOK configured — skipping Slack notify")
        return
    payload = {"text": text}
    # optionally you can set: payload['username']=username; payload['channel']=channel
    try:
        resp = requests.post(SLACK_WEBHOOK, json=payload, timeout=5)
        if resp.status_code != 200:
            logging.warning("Slack notify non-200: %s %s", resp.status_code, resp.text)
    except Exception:
        logging.exception("Slack notify failed")
