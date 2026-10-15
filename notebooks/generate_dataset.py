"""
generate_dataset.py
-------------------
Generates a synthetic phishing email dataset (~2000 samples)
and saves it to dataset/phishing_emails.csv
"""

import pandas as pd
import numpy as np
import random
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

random.seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────
# PHISHING EMAIL TEMPLATES
# ──────────────────────────────────────────────
PHISHING_TEMPLATES = [
    "Dear Customer, your {bank} account has been suspended due to suspicious activity. "
    "Click here immediately to verify: http://{ip}/login?token={token} "
    "Failure to act within 24 hours will result in permanent account closure. URGENT!",

    "WINNER! Congratulations! You have won ${amount} in the {company} lottery! "
    "To claim your prize, login at http://bit.ly/{short} and provide your bank account details. "
    "Limited time offer - act NOW!",

    "Important security update required for your {bank} account. "
    "Your password must be updated immediately. Visit http://{ip}/update-password "
    "and verify your account. Your account will be locked if you don't act within 48 hours.",

    "Your PayPal account has been limited! To restore access, please verify your identity "
    "at http://tinyurl.com/{short}. Click here to update your account information immediately. "
    "Urgent action required!",

    "Dear {name}, your {service} account has been compromised. "
    "Login now at http://{ip}/secure to reset your password. "
    "Click the link below to verify your email and protect your account. "
    "Failure to verify in 12 hours will result in suspension.",

    "ALERT: Unusual login attempt detected on your account! "
    "Someone tried to access your {bank} account from an unknown location. "
    "Verify your identity immediately at http://bit.ly/{short} "
    "Update your password to secure your account now. URGENT!",

    "Your subscription to {service} will expire today! "
    "Click here to renew your account: http://{ip}/renew?id={token} "
    "Enter your credit card information to continue enjoying our services. "
    "Act now to avoid service interruption!",

    "Congratulations! Your email {email} has been selected as our winner. "
    "You have won a gift card worth ${amount}. "
    "Visit http://goo.gl/{short} and enter your personal details to claim your reward. "
    "This offer expires in 24 hours. Do not miss this limited opportunity!",

    "IMPORTANT: Your bank account requires immediate verification. "
    "We detected suspicious transactions on your {bank} account. "
    "Please login at http://{ip}/verify to confirm your identity and protect your funds. "
    "Click the link to update your account information now. Urgent!",

    "Dear user, your {service} account password will expire in 24 hours. "
    "Update your password immediately at http://bit.ly/{short} "
    "Provide your username, password and credit card to verify your identity. "
    "Click here to login and update your account details now!",

    "Federal Tax Refund Notification: You are eligible for a tax refund of ${amount}. "
    "Submit your bank account details at http://{ip}/tax-refund?ref={token} "
    "Verify your identity with social security number and account information. "
    "Limited processing time - click now to claim your refund!",

    "Your {bank} online banking has been disabled for security reasons. "
    "To reactivate, verify your account at http://tinyurl.com/{short}?id={token} "
    "Enter your account number, PIN and password to restore access. "
    "Account will be permanently closed if not verified within 24 hours!",

    "Security Notice: Multiple failed login attempts detected on your {service} account. "
    "Click here to unlock your account: http://goo.gl/{short} "
    "Verify your email and update your password immediately. "
    "Your account has been temporarily suspended. Act now!",

    "Dear valued customer, your account information needs to be updated. "
    "Please click the link below and provide your {bank} account details: "
    "http://{ip}/account/update?session={token} "
    "This is urgent - failure to comply will result in account suspension. Click here now!",

    "You have received a secure message from {bank}. "
    "Click here to login and view your message: http://bit.ly/{short} "
    "Enter your account credentials to access your secure banking portal. "
    "Warning: Your session will expire in 30 minutes. Login immediately!",
]

# ──────────────────────────────────────────────
# SAFE EMAIL TEMPLATES
# ──────────────────────────────────────────────
SAFE_TEMPLATES = [
    "Hi {name}, just a friendly reminder about our team meeting tomorrow at 10 AM. "
    "Please review the attached agenda and come prepared with your project updates. "
    "Let me know if you have any questions. Looking forward to seeing everyone!",

    "Dear {name}, thank you for your order #{order_id}. "
    "Your package has been shipped and will arrive in 3-5 business days. "
    "Track your shipment at www.example.com/track. "
    "Contact us at support@example.com if you have any questions.",

    "Hello {name}, your monthly newsletter from {company} is here! "
    "This month we have exciting news about our new product launches, "
    "company updates and industry insights. Read more on our blog. "
    "Unsubscribe at any time by clicking the link below.",

    "Hi {name}, your appointment has been confirmed for {date} at {time}. "
    "Please arrive 10 minutes early. If you need to reschedule, "
    "call us at (555) 123-4567 or reply to this email. "
    "We look forward to seeing you!",

    "Dear {name}, attached is the quarterly report for Q{quarter} {year}. "
    "Revenue increased by 15% compared to last quarter. "
    "Please review the financial statements and share your feedback. "
    "Our next board meeting is scheduled for next Friday at 2 PM.",

    "Hi {name}, welcome to {company}! Your account has been successfully created. "
    "You can now access all features on our platform. "
    "Visit our help center at https://help.example.com for getting started guides. "
    "Our support team is available 24/7 at support@example.com.",

    "Hello team, please find the project status update below. "
    "The development phase is 80% complete. QA testing starts next Monday. "
    "Deployment is scheduled for the end of the month. "
    "Please update your tasks in our project management system.",

    "Dear {name}, your premium subscription has been renewed successfully. "
    "Your next billing date is {date}. View your invoice in your account dashboard. "
    "Visit https://www.example.com/account to manage your subscription. "
    "Thank you for being a valued customer!",

    "Hi {name}, I hope this email finds you well. "
    "I wanted to follow up on our conversation from last week about the new project proposal. "
    "Could you send me the final budget estimates by end of this week? "
    "Let me know if you need any clarification. Best regards.",

    "Dear students, the assignment submission deadline has been extended to {date}. "
    "Please submit your work through the university portal. "
    "Office hours this week will be held from 3-5 PM on Tuesday and Thursday. "
    "Email professor@university.edu with any academic questions.",

    "Hi {name}, your password was successfully changed on {date}. "
    "If you did not make this change, please contact our security team immediately "
    "at security@example.com or call (555) 987-6543. "
    "Your account security is our top priority.",

    "Hello {name}, here is your weekly digest from {company}. "
    "Top stories this week: new feature releases, customer success stories, "
    "and upcoming webinars. Register for our free webinar at https://events.example.com. "
    "Have a great week!",

    "Dear {name}, your job application for {position} has been received. "
    "Our HR team will review your application and get back to you within 5-7 business days. "
    "Thank you for your interest in joining {company}. "
    "Visit our careers page for more opportunities.",

    "Hi {name}, the IT department will perform scheduled maintenance this Sunday from 2-4 AM. "
    "Some services may be temporarily unavailable during this window. "
    "Please save your work and log out before the maintenance window. "
    "Contact it-support@company.com if you have any concerns.",

    "Dear {name}, your recent purchase of {product} is complete. "
    "Your receipt is attached to this email for your records. "
    "Rate your experience at https://www.example.com/review. "
    "For returns or exchanges, visit our store or call (555) 234-5678.",
]

# ──────────────────────────────────────────────
# HELPER DATA
# ──────────────────────────────────────────────
BANKS = ["Chase", "Bank of America", "Wells Fargo", "Citibank", "HSBC", "Barclays"]
SERVICES = ["Netflix", "Amazon", "Google", "Microsoft", "Apple", "Dropbox", "Spotify"]
COMPANIES = ["Global Lottery", "Tech Corp", "Amazon", "Microsoft", "Google"]
NAMES = ["John", "Sarah", "Michael", "Emma", "David", "Lisa", "Robert", "Anna"]
POSITIONS = ["Software Engineer", "Data Analyst", "Product Manager", "Designer"]
PRODUCTS = ["Laptop", "Headphones", "Keyboard", "Monitor", "Smart Watch"]
IPS = ["192.168.1.45", "10.0.0.23", "172.16.0.8", "203.45.67.89", "185.23.45.67"]
TOKENS = ["abc123xyz", "xyz789abc", "qwerty456", "pass1234", "secure99"]
SHORTS = ["xK3mP", "aB7nQ", "zR2pW", "mL5sT", "vN8cY"]
EMAILS = ["user123@gmail.com", "john.doe@yahoo.com", "customer@hotmail.com"]


def fill_phishing(template):
    """Fill a phishing template with random values."""
    return template.format(
        bank=random.choice(BANKS),
        service=random.choice(SERVICES),
        company=random.choice(COMPANIES),
        name=random.choice(NAMES),
        email=random.choice(EMAILS),
        ip=random.choice(IPS),
        token=random.choice(TOKENS),
        short=random.choice(SHORTS),
        amount=random.randint(500, 50000),
    )


def fill_safe(template):
    """Fill a safe template with random values."""
    return template.format(
        name=random.choice(NAMES),
        company=random.choice(COMPANIES),
        position=random.choice(POSITIONS),
        product=random.choice(PRODUCTS),
        order_id=random.randint(100000, 999999),
        date=f"{random.randint(1,28)}/{random.randint(1,12)}/2025",
        time=f"{random.randint(9,17)}:{'00' if random.random() > 0.5 else '30'}",
        quarter=random.randint(1, 4),
        year=2025,
    )


def generate_dataset(n_samples=2000):
    """Generate n_samples of phishing/safe emails."""
    records = []
    half = n_samples // 2

    # Generate phishing emails
    for _ in range(half):
        template = random.choice(PHISHING_TEMPLATES)
        text = fill_phishing(template)
        # Add random extra sentences occasionally
        if random.random() > 0.6:
            extras = [
                "Do not ignore this message.",
                "Your account will be closed.",
                "Respond within 24 hours.",
                "This is your final notice.",
                "Click here to verify now.",
            ]
            text += " " + random.choice(extras)
        records.append({"email_text": text, "label": 1})

    # Generate safe emails
    for _ in range(half):
        template = random.choice(SAFE_TEMPLATES)
        text = fill_safe(template)
        records.append({"email_text": text, "label": 0})

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dataset",
        "phishing_emails.csv",
    )
    print("Generating dataset...")
    df = generate_dataset(n_samples=2000)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
