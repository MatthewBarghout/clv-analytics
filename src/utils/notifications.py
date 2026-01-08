"""
Email notification system for CLV Analytics.

Sends alerts when data collection fails or other critical events occur.
"""
import logging
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Send email notifications for critical events."""

    def __init__(self):
        """Initialize email notifier with Gmail SMTP settings."""
        # For now, using a simple approach - can be configured via env vars later
        self.enabled = os.getenv("ENABLE_EMAIL_NOTIFICATIONS", "false").lower() == "true"
        self.to_email = os.getenv("NOTIFICATION_EMAIL", "mattbarg@unc.edu")
        self.from_email = os.getenv("SMTP_FROM_EMAIL", "clv-analytics@localhost")
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")

    def send_collection_failure(
        self, error_message: str, batch_time: str = None, games_affected: int = 0
    ):
        """
        Send email alert when odds collection fails.

        Args:
            error_message: The error that occurred
            batch_time: Time the batch was supposed to run
            games_affected: Number of games that lost closing line data
        """
        if not self.enabled:
            logger.info("Email notifications disabled - skipping alert")
            return

        subject = f"🚨 CLV Analytics: Collection Failed"

        if batch_time:
            subject += f" at {batch_time}"

        body = f"""
CLV Analytics Collection Failure Alert
=====================================

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Batch Time: {batch_time or 'Unknown'}
Games Affected: {games_affected}

Error Details:
{error_message}

Action Required:
- Check network connectivity
- Verify API is accessible
- Review logs at /tmp/clv-batch-*.log
- Consider running manual collection if games haven't started yet

System Status:
- Check running batches: launchctl list | grep clvanalytics
- View scheduler logs: tail -f /tmp/clv-scheduler-error.log
- Manual collection: poetry run python -m scripts.collect_odds --closing-only

---
CLV Analytics Automated Alert System
"""

        self._send_email(subject, body)

    def send_batch_success(self, games_collected: int, batch_time: str = None):
        """
        Send success notification (optional - can be disabled to reduce noise).

        Args:
            games_collected: Number of games successfully collected
            batch_time: Time the batch ran
        """
        if not self.enabled:
            return

        # Only send success emails if explicitly requested
        send_success = os.getenv("SEND_SUCCESS_EMAILS", "false").lower() == "true"
        if not send_success:
            return

        subject = f"✅ CLV Analytics: Collection Successful"

        body = f"""
CLV Analytics Collection Success
=================================

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Batch Time: {batch_time or 'Unknown'}
Games Collected: {games_collected}

Status: All closing lines collected successfully.

---
CLV Analytics Automated Alert System
"""

        self._send_email(subject, body)

    def _send_email(self, subject: str, body: str):
        """
        Send email via SMTP.

        Args:
            subject: Email subject
            body: Email body (plain text)
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = self.to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            # Send via SMTP
            if self.smtp_user and self.smtp_password:
                # Use authenticated SMTP (Gmail, etc.)
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.smtp_user, self.smtp_password)
                    server.send_message(msg)
                logger.info(f"Email sent to {self.to_email}: {subject}")
            else:
                # Use local sendmail (macOS default)
                with smtplib.SMTP('localhost') as server:
                    server.send_message(msg)
                logger.info(f"Email sent via local sendmail to {self.to_email}: {subject}")

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            # Don't raise - email failure shouldn't crash the main process
