import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv      

load_dotenv()

def send_slack_alert(message: str, channel: str = "#drift-alerts"):
    """Sends an alert message to a specified Slack channel."""
    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        print("Error: SLACK_BOT_TOKEN environment variable not set.")
        return False
    
    client = WebClient(token=token)

    try:
        response=client.chat_postMessage(
            channel=channel,
            text=message,
            mrkdwn=True # Enable markdown formatting
        )
        print(f"Alert sent to Slack channel {channel}")
        return True
    except SlackApiError as e:
        print(f"Error sending alert to Slack: {e.response['error']}")
        return False