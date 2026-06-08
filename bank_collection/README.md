# Clearwater Bank — Research App

Fake banking app for mouse dynamics data collection.

## Setup

1. Make sure Python 3.8+ is installed.

2. Install the one dependency:
   ```
   pip install flask
   ```

3. Run the app:
   ```
   python app.py
   ```

4. Open your browser to:
   ```
   http://127.0.0.1:5000
   ```

## Data

**Mouse data** is saved to:
```
bank-data/{username}-bank/session_{timestamp}.csv
```

Each row: `[elapsed_seconds, elapsed_seconds, button, state, screen_x, screen_y]`

This matches the pynput collector format exactly.

**User account data** is saved to:
```
user-data/{username}.txt
```

## Folder structure

```
bank-app/
├── app.py              # Flask backend
├── requirements.txt
├── templates/
│   └── index.html      # Full single-page app
├── static/
│   └── tracker.js      # Mouse tracking layer
├── user-data/          # Created automatically on first signup
└── bank-data/          # Created automatically on first login
```

## Notes

- Any username/password works at signup; credentials are checked strictly at login.
- Passwords are stored as SHA-256 hashes — never plaintext.
- Checking accounts open with a random balance between $50–$400.
- Savings accounts open with a random balance between $200–$1000.
- Accounts can only be deleted if their balance is exactly $0.00.
- Mouse data flushes to disk on logout and on tab close.
