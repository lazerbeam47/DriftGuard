# test_slack.py
from dotenv import load_dotenv
import os
from src.alerts.slack_alert import send_slack_alert

load_dotenv()

print("Testing Slack Alert...")

test_message = """
🚨 *DriftGuard Test Alert*

This is a test message to check if Slack integration is working.

If you see this → Integration is successful! ✅
"""

success = send_slack_alert(test_message, channel="#drift-alerts")

if success:
    print("✅ Test alert sent successfully!")
else:
    print("❌ Failed to send alert. Check console for errors.")