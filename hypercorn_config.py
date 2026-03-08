import os

# Render automatically sets the PORT environment variable.
port = os.getenv("PORT", "10000")
bind = [f"0.0.0.0:{port}"]

# Enable logging to track metrics safely
accesslog = "-"
errorlog = "-"
