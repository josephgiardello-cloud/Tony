import os
import argparse
import sys
import pandas as pd  # Open source: pandas (https://pandas.pydata.org/)
from tabulate import tabulate  # Open source: tabulate (https://pypi.org/project/tabulate/)
import uuid
from datetime import datetime
import requests
import yagmail
import getpass
import json
from babel.support import Translations
from accounting import AccountingManager
from donor_tracking import DonorManager
from event_management import EventManager
from volunteer import VolunteerManager
from communications import CommunicationsManager

# Compliance alert and audit trail integration

def send_compliance_alert(event, details=None):
    user = None
    try:
        user = getpass.getuser()
    except Exception:
        user = "unknown"
    timestamp = datetime.now().isoformat()
    msg = f"[{timestamp}] USER: {user} | COMPLIANCE ALERT: {event}"
    if details:
        msg += f" | DETAILS: {details}"
    print(f"ALERT: {msg}")
    with open("alerts.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def log_audit_event(event_type, details):
    user = None
    try:
        user = getpass.getuser()
    except Exception:
        user = "unknown"
    timestamp = datetime.now().isoformat()
    entry = f"[{timestamp}] USER: {user} | TYPE: {event_type} | DETAILS: {details}"
    with open("audit_trail.log", "a", encoding="utf-8") as f:
        f.write(entry + "\n")

# Automated detection and alerting for suspicious or fraudulent activity

def detect_suspicious_activity(app):
    alerts = []
    # Example: ML fraud detection using external API data
    # Placeholder: replace with real ML logic or import from scoring
    if hasattr(app, 'risk_score') and app.risk_score:
        if isinstance(app.risk_score, dict):
            if app.risk_score.get('MLRiskScore') == 'unavailable':
                alerts.append('ML risk score unavailable')
            elif app.risk_score.get('MLRiskScore', 1) < 0.5:
                alerts.append('Low ML risk score detected')
    # Add more rules as needed
    return alerts

def get_notification_roles(event_type):
    try:
        with open("notification_triggers.json", "r", encoding="utf-8") as f:
            triggers = json.load(f)
        return triggers.get(event_type, ["admin"])
    except Exception:
        return ["admin"]

def get_translations(lang_code):
    try:
        return Translations.load('locales', [lang_code])
    except Exception:
        return None

def translate_message(message, lang_code='en'):
    translations = get_translations(lang_code)
    if translations:
        return translations.ugettext(message)
    return message

class GrantApplication:
    def __init__(self, applicant_name, project_title, amount_requested, description, documents=None):
        self.id = str(uuid.uuid4())
        self.applicant_name = applicant_name
        self.project_title = project_title
        self.amount_requested = amount_requested
        self.description = description
        self.documents = documents or []
        self.status = "submitted"  # Possible: submitted, in_review, approved, rejected, needs_revision
        self.submission_date = datetime.utcnow()
        self.reviewer = None
        self.review_notes = ""
        self.risk_score = None
        self.decision_date = None

class GrantManager:
    DEFAULT_STORAGE_PATH = "grant_applications.json"

    def __init__(self, storage_path: str | None = None):
        self.storage_path = storage_path or self.DEFAULT_STORAGE_PATH
        self.applications: dict = {}
        self.load_applications()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_applications(self) -> None:
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for app_id, app_data in data.items():
                app = GrantApplication(
                    app_data["applicant_name"],
                    app_data["project_title"],
                    app_data["amount_requested"],
                    app_data["description"],
                    app_data.get("documents", []),
                )
                app.id = app_id
                app.status = app_data.get("status", "submitted")
                app.submission_date = app_data.get("submission_date")
                app.reviewer = app_data.get("reviewer")
                app.review_notes = app_data.get("review_notes", "")
                app.risk_score = app_data.get("risk_score")
                self.applications[app_id] = app
        except Exception:
            self.applications = {}

    def save_applications(self) -> None:
        data = {}
        for app_id, app in self.applications.items():
            data[app_id] = {
                "applicant_name": app.applicant_name,
                "project_title": app.project_title,
                "amount_requested": app.amount_requested,
                "description": app.description,
                "documents": app.documents,
                "status": app.status,
                "submission_date": str(app.submission_date),
                "reviewer": app.reviewer,
                "review_notes": app.review_notes,
                "risk_score": app.risk_score,
            }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    def submit_application(self, applicant_name, project_title, amount_requested, description, documents=None):
        app = GrantApplication(applicant_name, project_title, amount_requested, description, documents)
        try:
            from ml_risk_model import predict_risk
            app.risk_score = {"MLRiskScore": predict_risk({
                "amount_requested": amount_requested,
                "project_title": project_title,
                "description": description,
            })}
        except Exception:
            app.risk_score = {"MLRiskScore": "unavailable"}
        self.applications[app.id] = app
        self.save_applications()
        send_compliance_alert("Application submitted", f"applicant={applicant_name}, title={project_title}, amount={amount_requested}")
        for alert in detect_suspicious_activity(app):
            send_compliance_alert("Suspicious activity detected", f"app_id={app.id}, {alert}")
        self._notify_submit(app)
        return app.id

    def assign_reviewer(self, app_id, reviewer_name):
        app = self.applications.get(app_id)
        if app:
            app.reviewer = reviewer_name
            app.status = "in_review"
            self.save_applications()
            send_compliance_alert("Reviewer assigned", f"reviewer={reviewer_name}, app_id={app_id}")
            log_audit_event("assign_reviewer", f"app_id={app_id}, reviewer={reviewer_name}")
            return True
        return False

    def review_application(self, app_id, reviewer_name, notes, decision):
        app = self.applications.get(app_id)
        valid_decisions = ["approved", "rejected", "needs_revision"]
        if app and decision in valid_decisions:
            app.reviewer = reviewer_name
            app.review_notes = notes
            app.status = decision
            app.decision_date = datetime.utcnow()
            try:
                from ml_risk_model import predict_risk
                app.risk_score = {"MLRiskScore": predict_risk({
                    "amount_requested": app.amount_requested,
                    "project_title": app.project_title,
                    "description": app.description,
                })}
            except Exception:
                app.risk_score = {"MLRiskScore": "unavailable"}
            self.save_applications()
            log_audit_event("review_application", f"app_id={app_id}, reviewer={reviewer_name}, decision={decision}, notes={notes}")
            self._notify_review(app, reviewer_name, decision)
            return True
        return False

    def update_status(self, app_id, status):
        app = self.applications.get(app_id)
        valid_statuses = ["submitted", "in_review", "approved", "rejected", "needs_revision"]
        if app and status in valid_statuses:
            app.status = status
            if status in ("approved", "rejected", "needs_revision"):
                app.decision_date = datetime.utcnow()
            self.save_applications()
            log_audit_event("update_status", f"app_id={app_id}, status={status}")
            return True
        return False

    def set_status(self, app_id, status):
        return self.update_status(app_id, status)

    def add_review_notes(self, app_id, notes):
        app = self.applications.get(app_id)
        if app:
            app.review_notes = notes
            self.save_applications()
            log_audit_event("add_review_notes", f"app_id={app_id}, notes={notes}")
            return True
        return False

    def upload_supporting_document(self, app_id, file):
        app = self.applications.get(app_id)
        if app:
            app.documents.append(file)
            self.save_applications()
            log_audit_event("upload_document", f"app_id={app_id}, file={file}")
            return True
        return False

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_applications(self, format="csv", out_path=None):
        data = [
            {
                "ID": app.id,
                "Applicant": app.applicant_name,
                "Title": app.project_title,
                "Amount": app.amount_requested,
                "Status": app.status,
                "Reviewer": app.reviewer,
                "RiskScore": app.risk_score,
                "SubmissionDate": app.submission_date,
                "DecisionDate": app.decision_date,
            }
            for app in self.applications.values()
        ]
        df = pd.DataFrame(data)
        if not out_path:
            out_path = f"grant_applications_export.{format}"
        if format == "csv":
            df.to_csv(out_path, index=False)
        elif format == "excel":
            df.to_excel(out_path, index=False)
        elif format == "table":
            try:
                print(tabulate(df, headers="keys", tablefmt="psql"))
            except Exception:
                print(df.to_string())
        return out_path

    # ------------------------------------------------------------------
    # Notifications (fail-safe wrappers)
    # ------------------------------------------------------------------

    def _notify_submit(self, app):
        subject = "Grant Application Submitted"
        body = f"Application '{app.project_title}' submitted. ML Risk: {app.risk_score.get('MLRiskScore') if app.risk_score else 'N/A'}"
        self.notify_roles(subject, body, ["admin", "applicant", "compliance"])
        self.notify_dashboard(body)

    def _notify_review(self, app, reviewer_name, decision):
        subject = "Grant Application Reviewed"
        body = f"Application '{app.project_title}' reviewed by {reviewer_name}. Status: {decision}."
        self.notify_roles(subject, body, ["admin", "applicant", "reviewer", "compliance"])
        self.notify_dashboard(body)

    def send_email_notification(self, subject, body, to_email):
        from flask_mail import Mail, Message
        from flask import Flask
        import os
        from dotenv import load_dotenv
        load_dotenv()
        app = Flask(__name__)
        app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'localhost')
        app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 465))  # Enforce SSL port
        app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME', '')
        app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD', '')
        app.config['MAIL_USE_TLS'] = False
        app.config['MAIL_USE_SSL'] = True  # Enforce SSL
        sender_domain = app.config['MAIL_USERNAME'].split('@')[-1] if app.config['MAIL_USERNAME'] else None
        allowed_domains = os.getenv('ALLOWED_SENDER_DOMAINS', sender_domain)
        if sender_domain and allowed_domains and sender_domain not in allowed_domains:
            with open("notifications.log", "a", encoding="utf-8") as f:
                f.write(f"[SECURITY] Blocked email from unverified sender domain: {sender_domain}\n")
            return
        mail = Mail(app)
        msg = Message(subject, sender=app.config['MAIL_USERNAME'], recipients=[to_email])
        msg.body = body
        try:
            with app.app_context():
                mail.send(msg)
            with open("notifications.log", "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] SECURE EMAIL sent to {to_email}: {subject}\n")
        except Exception as e:
            with open("notifications.log", "a", encoding="utf-8") as f:
                f.write(f"[ERROR] SECURE EMAIL failed: {e}\n")

    def send_webhook_notification(self, event_type, payload):
        webhook_url = os.getenv('WEBHOOK_URL', None)
        if webhook_url:
            try:
                import requests
                resp = requests.post(webhook_url, json={"event": event_type, "payload": payload}, timeout=5)
                with open("notifications.log", "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now().isoformat()}] WEBHOOK sent: {event_type}, status={resp.status_code}\n")
            except Exception as e:
                with open("notifications.log", "a", encoding="utf-8") as f:
                    f.write(f"[ERROR] WEBHOOK failed: {e}\n")

    def notify_event(self, event_type, subject, body, lang_code='en'):
        subject = translate_message(subject, lang_code)
        body = translate_message(body, lang_code)
        roles = get_notification_roles(event_type)
        self.notify_roles(subject, body, roles)
        self.send_webhook_notification(event_type, {"subject": subject, "body": body, "roles": roles})

    def send_sms_notification(self, body, phone_number):
        # Email-to-SMS gateway (industry standard, open source)
        # Example for US carriers: number@vtext.com (Verizon), number@tmomail.net (T-Mobile), number@txt.att.net (AT&T)
        gateways = [
            f"{phone_number}@vtext.com",
            f"{phone_number}@tmomail.net",
            f"{phone_number}@txt.att.net"
        ]
        try:
            yag = yagmail.SMTP(os.getenv('MAIL_USERNAME', ''), os.getenv('MAIL_PASSWORD', ''))
            for gateway in gateways:
                yag.send(gateway, "TONY Notification", body)
            with open("notifications.log", "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] SMS sent to {phone_number}: {body}\n")
        except Exception as e:
            with open("notifications.log", "a", encoding="utf-8") as f:
                f.write(f"[ERROR] SMS notification failed: {e}\n")

    def notify_roles(self, subject, body, roles):
        role_emails = {
            "admin": os.getenv('ADMIN_EMAIL', 'admin@localhost'),
            "applicant": os.getenv('APPLICANT_EMAIL', 'user@localhost'),
            "reviewer": os.getenv('REVIEWER_EMAIL', 'reviewer@localhost'),
            "compliance": os.getenv('COMPLIANCE_EMAIL', 'compliance@localhost')
        }
        role_phones = {
            "admin": os.getenv('ADMIN_PHONE', ''),
            "applicant": os.getenv('APPLICANT_PHONE', ''),
            "reviewer": os.getenv('REVIEWER_PHONE', ''),
            "compliance": os.getenv('COMPLIANCE_PHONE', '')
        }
        for role in roles:
            to_email = role_emails.get(role)
            if to_email:
                self.send_email_notification(subject, body, to_email)
                log_audit_event("notification_email", f"to={to_email}, subject={subject}, body={body}")
            phone = role_phones.get(role)
            if phone:
                self.send_sms_notification(body, phone)
                log_audit_event("notification_sms", f"to={phone}, body={body}")
        self.notify_dashboard(f"{subject}: {body}")
        log_audit_event("notification_dashboard", f"message={subject}: {body}")

    def notify_dashboard(self, message):
        try:
            requests.get(f"http://localhost:5000/notify/{message}")
        except Exception as e:
            with open("notifications.log", "a", encoding="utf-8") as f:
                f.write(f"[ERROR] Dashboard notification failed: {e}\n")

def show_notification_history():
    print("--- Notification History ---")
    try:
        with open("notifications.log", "r", encoding="utf-8") as f:
            for line in f:
                print(line.strip())
    except Exception:
        print("No notifications found.")
    print("--- Audit Trail ---")
    try:
        with open("audit_trail.log", "r", encoding="utf-8") as f:
            for line in f:
                print(line.strip())
    except Exception:
        print("No audit trail found.")

def main():
    parser = argparse.ArgumentParser(description="TONY Nonprofit Platform CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Accounting
    acc_parser = subparsers.add_parser("add-transaction")
    acc_parser.add_argument("amount", type=float)
    acc_parser.add_argument("category")
    acc_parser.add_argument("description")

    # Donor
    donor_parser = subparsers.add_parser("add-donor")
    donor_parser.add_argument("name")
    donor_parser.add_argument("email")
    donor_parser.add_argument("amount", type=float)

    # Event
    event_parser = subparsers.add_parser("create-event")
    event_parser.add_argument("name")
    event_parser.add_argument("date")
    event_parser.add_argument("location")
    event_parser.add_argument("attendees", nargs="*", help="Attendees as name:email pairs")

    # Volunteer
    vol_parser = subparsers.add_parser("schedule-volunteer")
    vol_parser.add_argument("name")
    vol_parser.add_argument("email")
    vol_parser.add_argument("date")
    vol_parser.add_argument("role")

    # Communications
    comm_parser = subparsers.add_parser("send-message")
    comm_parser.add_argument("channel", choices=["email", "sms", "dashboard"])
    comm_parser.add_argument("to")
    comm_parser.add_argument("subject")
    comm_parser.add_argument("body")

    args = parser.parse_args()
    platform = TonyPlatform()

    try:
        if args.command == "add-transaction":
            if args.amount <= 0:
                raise ValueError("Amount must be positive.")
            if not args.category or not args.description:
                raise ValueError("Category and description are required.")
            platform.record_transaction(args.amount, args.category, args.description)
            print("Transaction recorded.")
        elif args.command == "add-donor":
            if not args.name or not args.email or "@" not in args.email:
                raise ValueError("Valid name and email are required.")
            if args.amount <= 0:
                raise ValueError("Donation amount must be positive.")
            platform.add_donor_and_notify(args.name, args.email, args.amount)
            print("Donor added and notified.")
        elif args.command == "create-event":
            if not args.name or not args.date or not args.location:
                raise ValueError("Name, date, and location are required.")
            attendees = []
            for pair in args.attendees:
                if ":" in pair:
                    name, email = pair.split(":", 1)
                    if not name or not email or "@" not in email:
                        raise ValueError(f"Invalid attendee: {pair}")
                    attendees.append({"name": name, "email": email})
            platform.create_event_and_notify(args.name, args.date, args.location, attendees)
            print("Event created and attendees notified.")
        elif args.command == "schedule-volunteer":
            if not args.name or not args.email or "@" not in args.email or not args.date or not args.role:
                raise ValueError("All fields are required and email must be valid.")
            platform.schedule_volunteer_and_notify(args.name, args.email, args.date, args.role)
            print("Volunteer scheduled and notified.")
        elif args.command == "send-message":
            if not args.to or not args.body:
                raise ValueError("Recipient and message body are required.")
            if args.channel == "email" and "@" not in args.to:
                raise ValueError("Email must be valid.")
            if args.channel == "email":
                platform.comms.send_email(args.to, args.subject, args.body)
            elif args.channel == "sms":
                platform.comms.send_sms(args.to, args.body)
            elif args.channel == "dashboard":
                platform.comms.send_dashboard(args.to, args.body)
            print(f"Message sent via {args.channel}.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "history":
        show_notification_history()
    else:
        main()

def check_external_compliance(app):
    alerts = []
    # Example IRS API check
    try:
        ein = getattr(app, 'ein', None)
        if ein:
            irs_resp = requests.get(f'https://api.irs.gov/charities/{ein}')
            if irs_resp.status_code == 200:
                data = irs_resp.json()
                if data.get('revoked', False):
                    alerts.append('IRS revocation detected')
    except Exception:
        alerts.append('IRS compliance check failed')
    # Example state registry check
    try:
        state = getattr(app, 'state', 'NY')
        ein = getattr(app, 'ein', None)
        if ein:
            state_resp = requests.get(f'https://api.state.gov/charities/{state}/{ein}')
            if state_resp.status_code == 200:
                data = state_resp.json()
                if not data.get('registered', True):
                    alerts.append('State registration missing')
    except Exception:
        alerts.append('State compliance check failed')
    # Example fraud database check (stub)
    try:
        if hasattr(app, 'risk_score') and app.risk_score:
            if app.risk_score.get('MLRiskScore', 1) < 0.2:
                alerts.append('Fraud risk detected')
    except Exception:
        alerts.append('Fraud check failed')
    return alerts

class TonyPlatform:
    def __init__(self):
        self.accounting = AccountingManager()
        self.donors = DonorManager()
        self.events = EventManager()
        self.volunteers = VolunteerManager()
        self.comms = CommunicationsManager()

    # Example unified workflow methods
    def add_donor_and_notify(self, name, email, amount):
        donor = self.donors.add_donor(name, email)
        donor.add_donation(amount)
        self.comms.send_email(email, "Thank you for your donation!", f"Dear {name}, thank you for your generous gift of ${amount}.")

    def schedule_volunteer_and_notify(self, name, email, date, role):
        v = self.volunteers.add_volunteer(name, email)
        v.add_shift(date, role)
        self.comms.send_email(email, "Volunteer Shift Scheduled", f"Hi {name}, your shift for {role} on {date} is confirmed.")

    def create_event_and_notify(self, name, date, location, attendees):
        event = self.events.create_event(name, date, location)
        for attendee in attendees:
            event.add_attendee(attendee['name'], attendee['email'])
            self.comms.send_email(attendee['email'], f"Invitation: {name}", f"Dear {attendee['name']}, you are invited to {name} on {date} at {location}.")

    def record_transaction(self, amount, category, description):
        self.accounting.add_transaction(amount, category, description)
