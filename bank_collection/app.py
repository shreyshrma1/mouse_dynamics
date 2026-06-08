import os
import csv
import time
import hashlib
import random
import string
from flask import Flask, request, session, jsonify, send_from_directory, render_template

app = Flask(__name__, static_folder='static', template_folder='templates')

# Stable secret key - written once to disk so sessions survive restarts
_key_path = 'secret.key'
if os.path.exists(_key_path):
    app.secret_key = open(_key_path).read()
else:
    import secrets
    app.secret_key = secrets.token_hex(32)
    open(_key_path, 'w').write(app.secret_key)

app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True

USER_DATA_DIR = 'user-data'
BANK_DATA_DIR = 'bank-data'
REGISTRY_PATH = os.path.join(USER_DATA_DIR, 'registry.txt')

os.makedirs(USER_DATA_DIR, exist_ok=True)
os.makedirs(BANK_DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# User file helpers
# ---------------------------------------------------------------------------

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def user_path(username):
    return os.path.join(USER_DATA_DIR, f'{username}.txt')

def load_user(username):
    path = user_path(username)
    if not os.path.exists(path):
        return None
    user = {'meta': {}, 'accounts': [], 'transactions': [], 'payees': []}
    section = None
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('[') and line.endswith(']'):
                section = line[1:-1]
                continue
            if section == 'meta':
                k, v = line.split('=', 1)
                user['meta'][k.strip()] = v.strip()
            elif section == 'accounts':
                acc = {}
                for part in line.split('|'):
                    k, v = part.strip().split('=', 1)
                    acc[k.strip()] = v.strip()
                user['accounts'].append(acc)
            elif section == 'transactions':
                txn = {}
                for part in line.split('|'):
                    k, v = part.strip().split('=', 1)
                    txn[k.strip()] = v.strip()
                user['transactions'].append(txn)
            elif section == 'payees':
                payee = {}
                for part in line.split('|'):
                    k, v = part.strip().split('=', 1)
                    payee[k.strip()] = v.strip()
                user['payees'].append(payee)
    return user

def save_user(user):
    username = user['meta']['username']
    path = user_path(username)
    with open(path, 'w') as f:
        f.write('[meta]\n')
        for k, v in user['meta'].items():
            f.write(f'{k}={v}\n')
        f.write('\n[accounts]\n')
        for acc in user['accounts']:
            parts = ' | '.join(f'{k}={v}' for k, v in acc.items())
            f.write(parts + '\n')
        f.write('\n[transactions]\n')
        for txn in user['transactions']:
            parts = ' | '.join(f'{k}={v}' for k, v in txn.items())
            f.write(parts + '\n')
        f.write('\n[payees]\n')
        for payee in user.get('payees', []):
            parts = ' | '.join(f'{k}={v}' for k, v in payee.items())
            f.write(parts + '\n')

def make_id(prefix, length=6):
    return prefix + '_' + ''.join(random.choices(string.digits, k=length))

ROUTING_NUMBER = '021000021'  # shared across all accounts

def make_account_number():
    return ''.join(random.choices(string.digits, k=12))

def make_fake_account_number():
    return ''.join(random.choices(string.digits, k=12))

def make_opened_date(days_ago_min=60, days_ago_max=730):
    import datetime
    days_ago = random.randint(days_ago_min, days_ago_max)
    d = datetime.date.today() - datetime.timedelta(days=days_ago)
    return d.strftime('%b %d, %Y')

def make_account_details(acc_type):
    return {
        'account_no': make_account_number(),
        'routing_no': ROUTING_NUMBER,
        'opened':     make_opened_date(),
        'rate':       '2.35% APY' if acc_type == 'Savings' else 'N/A',
        'limit':      '2500.00',
    }

def make_fake_transactions(checking_id, savings_id, base_time=None):
    """Generate 34 realistic fake transactions spread over ~2 months.
    base_time defaults to now; pass an older timestamp to push history further back."""
    now = base_time or int(time.time())
    day = 86400
    rent_no     = make_fake_account_number()
    electric_no = make_fake_account_number()
    internet_no = make_fake_account_number()
    stream_no   = make_fake_account_number()
    grocery_no  = make_fake_account_number()
    cloud_no    = make_fake_account_number()
    gas_no      = make_fake_account_number()
    car_ins_no  = make_fake_account_number()
    pharmacy_no = make_fake_account_number()
    dinner_no   = make_fake_account_number()
    coffee_no   = make_fake_account_number()
    music_no    = make_fake_account_number()
    lunch_no    = make_fake_account_number()
    return [
        {'id': make_id('txn'), 'ts': str(now - 60*day), 'from': 'external',  'to': checking_id, 'amount': '2400.00', 'note': 'Payroll deposit',        'status': 'completed', 'payee_acc_no': ''},
        {'id': make_id('txn'), 'ts': str(now - 58*day), 'from': checking_id, 'to': 'external',  'amount': '850.00',  'note': 'Rent payment',            'status': 'completed', 'payee_acc_no': rent_no},
        {'id': make_id('txn'), 'ts': str(now - 57*day), 'from': checking_id, 'to': 'external',  'amount': '62.40',   'note': 'Electric bill',           'status': 'completed', 'payee_acc_no': electric_no},
        {'id': make_id('txn'), 'ts': str(now - 55*day), 'from': checking_id, 'to': savings_id,  'amount': '200.00',  'note': 'Savings transfer',        'status': 'completed', 'payee_acc_no': ''},
        {'id': make_id('txn'), 'ts': str(now - 53*day), 'from': checking_id, 'to': 'external',  'amount': '38.99',   'note': 'Streaming subscriptions', 'status': 'completed', 'payee_acc_no': stream_no},
        {'id': make_id('txn'), 'ts': str(now - 51*day), 'from': checking_id, 'to': 'external',  'amount': '124.50',  'note': 'Grocery run',             'status': 'completed', 'payee_acc_no': grocery_no},
        {'id': make_id('txn'), 'ts': str(now - 49*day), 'from': 'external',  'to': checking_id, 'amount': '85.00',   'note': 'Freelance payment',       'status': 'completed', 'payee_acc_no': ''},
        {'id': make_id('txn'), 'ts': str(now - 47*day), 'from': checking_id, 'to': 'external',  'amount': '9.99',    'note': 'Cloud storage',           'status': 'completed', 'payee_acc_no': cloud_no},
        {'id': make_id('txn'), 'ts': str(now - 45*day), 'from': checking_id, 'to': 'external',  'amount': '47.20',   'note': 'Gas station',             'status': 'completed', 'payee_acc_no': gas_no},
        {'id': make_id('txn'), 'ts': str(now - 42*day), 'from': 'external',  'to': checking_id, 'amount': '2400.00', 'note': 'Payroll deposit',         'status': 'completed'},
        {'id': make_id('txn'), 'ts': str(now - 40*day), 'from': checking_id, 'to': 'external',  'amount': '850.00',  'note': 'Rent payment',            'status': 'completed', 'payee_acc_no': rent_no},
        {'id': make_id('txn'), 'ts': str(now - 39*day), 'from': checking_id, 'to': 'external',  'amount': '62.40',   'note': 'Electric bill',           'status': 'completed', 'payee_acc_no': electric_no},
        {'id': make_id('txn'), 'ts': str(now - 37*day), 'from': checking_id, 'to': savings_id,  'amount': '200.00',  'note': 'Savings transfer',        'status': 'completed', 'payee_acc_no': ''},
        {'id': make_id('txn'), 'ts': str(now - 35*day), 'from': checking_id, 'to': 'external',  'amount': '210.00',  'note': 'Car insurance',           'status': 'completed', 'payee_acc_no': car_ins_no},
        {'id': make_id('txn'), 'ts': str(now - 33*day), 'from': checking_id, 'to': 'external',  'amount': '89.30',   'note': 'Pharmacy',                'status': 'completed', 'payee_acc_no': pharmacy_no},
        {'id': make_id('txn'), 'ts': str(now - 31*day), 'from': 'external',  'to': savings_id,  'amount': '500.00',  'note': 'Tax refund',              'status': 'completed', 'payee_acc_no': ''},
        {'id': make_id('txn'), 'ts': str(now - 29*day), 'from': checking_id, 'to': 'external',  'amount': '34.00',   'note': 'Dinner out',              'status': 'completed', 'payee_acc_no': dinner_no},
        {'id': make_id('txn'), 'ts': str(now - 27*day), 'from': checking_id, 'to': 'external',  'amount': '15.49',   'note': 'Coffee shop',             'status': 'completed', 'payee_acc_no': coffee_no},
        {'id': make_id('txn'), 'ts': str(now - 25*day), 'from': checking_id, 'to': 'external',  'amount': '9.99',    'note': 'Music subscription',      'status': 'completed', 'payee_acc_no': music_no},
        {'id': make_id('txn'), 'ts': str(now - 22*day), 'from': 'external',  'to': checking_id, 'amount': '2400.00', 'note': 'Payroll deposit',         'status': 'completed'},
        {'id': make_id('txn'), 'ts': str(now - 20*day), 'from': checking_id, 'to': 'external',  'amount': '850.00',  'note': 'Rent payment',            'status': 'completed', 'payee_acc_no': rent_no},
        {'id': make_id('txn'), 'ts': str(now - 19*day), 'from': checking_id, 'to': 'external',  'amount': '62.40',   'note': 'Electric bill',           'status': 'completed', 'payee_acc_no': electric_no},
        {'id': make_id('txn'), 'ts': str(now - 17*day), 'from': checking_id, 'to': savings_id,  'amount': '200.00',  'note': 'Savings transfer',        'status': 'completed', 'payee_acc_no': ''},
        {'id': make_id('txn'), 'ts': str(now - 15*day), 'from': checking_id, 'to': 'external',  'amount': '143.20',  'note': 'Grocery run',             'status': 'completed', 'payee_acc_no': grocery_no},
        {'id': make_id('txn'), 'ts': str(now - 13*day), 'from': checking_id, 'to': 'external',  'amount': '55.00',   'note': 'Internet bill',           'status': 'completed', 'payee_acc_no': internet_no},
        {'id': make_id('txn'), 'ts': str(now - 11*day), 'from': 'external',  'to': checking_id, 'amount': '120.00',  'note': 'Freelance payment',       'status': 'completed', 'payee_acc_no': ''},
        {'id': make_id('txn'), 'ts': str(now -  9*day), 'from': checking_id, 'to': 'external',  'amount': '28.75',   'note': 'Lunch',                   'status': 'completed', 'payee_acc_no': lunch_no},
        {'id': make_id('txn'), 'ts': str(now -  7*day), 'from': checking_id, 'to': 'external',  'amount': '9.99',    'note': 'Cloud storage',           'status': 'completed', 'payee_acc_no': cloud_no},
        {'id': make_id('txn'), 'ts': str(now -  5*day), 'from': checking_id, 'to': 'external',  'amount': '67.40',   'note': 'Gas station',             'status': 'completed', 'payee_acc_no': gas_no},
        {'id': make_id('txn'), 'ts': str(now -  4*day), 'from': 'external',  'to': checking_id, 'amount': '2400.00', 'note': 'Payroll deposit',         'status': 'completed'},
        {'id': make_id('txn'), 'ts': str(now -  3*day), 'from': checking_id, 'to': 'external',  'amount': '850.00',  'note': 'Rent payment',            'status': 'completed', 'payee_acc_no': rent_no},
        {'id': make_id('txn'), 'ts': str(now -  2*day), 'from': checking_id, 'to': savings_id,  'amount': '200.00',  'note': 'Savings transfer',        'status': 'completed', 'payee_acc_no': ''},
        {'id': make_id('txn'), 'ts': str(now -  1*day), 'from': checking_id, 'to': 'external',  'amount': '44.60',   'note': 'Grocery run',             'status': 'completed', 'payee_acc_no': grocery_no},
        {'id': make_id('txn'), 'ts': str(now),           'from': 'external',  'to': checking_id, 'amount': '18.50',   'note': 'Refund',                  'status': 'completed', 'payee_acc_no': ''},
    ]

# ---------------------------------------------------------------------------
# Registry helpers  (username <-> userN mapping, session counts)
# ---------------------------------------------------------------------------

def load_registry():
    """Returns dict: { 'user1': {'username': 'shrey123', 'sessions': 3}, ... }"""
    registry = {}
    if not os.path.exists(REGISTRY_PATH):
        return registry
    with open(REGISTRY_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: user1 | username=shrey123 | sessions=3
            parts = [p.strip() for p in line.split('|')]
            user_id = parts[0]
            entry = {}
            for part in parts[1:]:
                k, v = part.split('=', 1)
                entry[k.strip()] = v.strip()
            registry[user_id] = entry
    return registry

def save_registry(registry):
    with open(REGISTRY_PATH, 'w') as f:
        for user_id, entry in registry.items():
            parts = [user_id] + [f'{k}={v}' for k, v in entry.items()]
            f.write(' | '.join(parts) + '\n')

def get_user_id(username):
    """Look up the anonymous userN ID for a given username. Returns None if not found."""
    registry = load_registry()
    for user_id, entry in registry.items():
        if entry.get('username') == username:
            return user_id
    return None

def next_user_id():
    """Return the next available userN string (user1, user2, ...)."""
    registry = load_registry()
    n = 1
    while f'user{n}' in registry:
        n += 1
    return f'user{n}'

def increment_sessions(username):
    """Increment session count for a user and return their user_id."""
    registry = load_registry()
    for user_id, entry in registry.items():
        if entry.get('username') == username:
            entry['sessions'] = str(int(entry.get('sessions', 0)) + 1)
            save_registry(registry)
            return user_id
    return None

# ---------------------------------------------------------------------------
# Mouse data helpers
# ---------------------------------------------------------------------------

def session_csv_path(user_id):
    """Build a CSV path using the anonymous user_id (e.g. user1), not the username."""
    folder = os.path.join(BANK_DATA_DIR, f'{user_id}')
    os.makedirs(folder, exist_ok=True)
    ts = int(time.time())
    return os.path.join(folder, f'session_{ts}.csv')

# In-memory buffer keyed by username (for easy lookup during the session),
# but the CSV path uses the anonymous user_id.
mouse_sessions = {}

def get_mouse_session(username, user_id):
    if username not in mouse_sessions:
        mouse_sessions[username] = {
            'path': session_csv_path(user_id),
            'events': [],
            'start': time.time()
        }
    return mouse_sessions[username]

def flush_mouse(username):
    if username not in mouse_sessions:
        return
    ms = mouse_sessions[username]
    if not ms['events']:
        return
    with open(ms['path'], 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(ms['events'])
    ms['events'] = []

# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


# ---------------------------------------------------------------------------
# Routes — auth
# ---------------------------------------------------------------------------

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username', '').strip().lower()
    password = data.get('password', '')
    user = load_user(username)
    if user is None or 'password' not in user.get('meta', {}):
        return jsonify({'error': 'Username not found.'}), 401
    if user['meta']['password'] != hash_password(password):
        return jsonify({'error': 'Incorrect password.'}), 401

    # Backfill fake history for existing users who don't have it yet.
    if len(user['transactions']) < 34:
        checking = next((a['id'] for a in user['accounts'] if 'Checking' in a.get('name','') or a.get('type') == 'Checking'), 'external')
        savings  = next((a['id'] for a in user['accounts'] if 'Savings'  in a.get('name','') or a.get('type') == 'Savings'),  'external')
        fake = make_fake_transactions(checking, savings, base_time=int(time.time()) - 60*86400)
        existing_notes_ts = {(t['note'], t['ts']) for t in user['transactions']}
        to_add = [t for t in fake if (t['note'], t['ts']) not in existing_notes_ts]
        user['transactions'] = to_add + user['transactions']
        save_user(user)

    # Increment session count and get anonymous user_id for data folder
    user_id = increment_sessions(username)
    session['username'] = username
    session['user_id'] = user_id
    get_mouse_session(username, user_id)
    return jsonify({'ok': True})

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get('username', '').strip().lower()
    password = data.get('password', '')
    if not username or not password:
        return jsonify({'error': 'Username and password required.'}), 400
    if os.path.exists(user_path(username)):
        return jsonify({'error': 'Username already taken.'}), 400

    checking_id = make_id('acc')
    savings_id  = make_id('acc')

    fake_txns = make_fake_transactions(checking_id, savings_id)

    checking_details = make_account_details('Checking')
    savings_details  = make_account_details('Savings')
    user = {
        'meta': {'username': username, 'password': hash_password(password)},
        'accounts': [
            {'id': checking_id, 'type': 'Checking', 'name': 'Main Checking', 'balance': '700.00', **checking_details},
            {'id': savings_id,  'type': 'Savings',  'name': 'Main Savings',  'balance': '1500.00', **savings_details},
        ],
        'transactions': fake_txns
    }
    save_user(user)

    # Register the new user with an anonymous user_id
    user_id = next_user_id()
    registry = load_registry()
    registry[user_id] = {'username': username, 'sessions': '1'}
    save_registry(registry)

    session['username'] = username
    session['user_id'] = user_id
    get_mouse_session(username, user_id)
    return jsonify({'ok': True})

@app.route('/logout', methods=['POST'])
def logout():
    username = session.get('username')
    if username:
        flush_mouse(username)
        mouse_sessions.pop(username, None)
        session.clear()
    return jsonify({'ok': True})

# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.route('/api/user', methods=['GET'])
def api_user():
    username = session.get('username')
    if not username:
        return jsonify({'error': 'Not logged in.'}), 401
    user = load_user(username)
    # Sort accounts by saved order if present
    order = user['meta'].get('order', '')
    if order:
        order_list = [x.strip() for x in order.split(',')]
        acc_map = {a['id']: a for a in user['accounts']}
        ordered = [acc_map[i] for i in order_list if i in acc_map]
        # Append any accounts not yet in the order (e.g. newly created)
        ordered += [a for a in user['accounts'] if a['id'] not in order_list]
        user['accounts'] = ordered
    return jsonify({'accounts': user['accounts'], 'transactions': user['transactions'], 'payees': user.get('payees', [])})

@app.route('/api/accounts/reorder', methods=['POST'])
def api_accounts_reorder():
    username = session.get('username')
    if not username:
        return jsonify({'error': 'Not logged in.'}), 401
    order = request.json.get('order', [])  # list of account IDs in new order
    user = load_user(username)
    user['meta']['order'] = ','.join(order)
    save_user(user)
    return jsonify({'ok': True})

@app.route('/api/transfer', methods=['POST'])
def api_transfer():
    username = session.get('username')
    if not username:
        return jsonify({'error': 'Not logged in.'}), 401
    data = request.json
    from_id = data.get('from_id')
    to_id = data.get('to_id')      # account id or 'external'
    amount = data.get('amount')
    note = data.get('note', 'Transfer')
    try:
        amount = round(float(amount), 2)
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid amount.'}), 400
    if amount <= 0:
        return jsonify({'error': 'Amount must be positive.'}), 400
    user = load_user(username)
    from_acc = next((a for a in user['accounts'] if a['id'] == from_id), None)
    if from_acc is None:
        return jsonify({'error': 'Source account not found.'}), 400
    if float(from_acc['balance']) < amount:
        return jsonify({'error': 'Insufficient funds.'}), 400
    from_acc['balance'] = f'{float(from_acc["balance"]) - amount:.2f}'
    if to_id != 'external':
        to_acc = next((a for a in user['accounts'] if a['id'] == to_id), None)
        if to_acc is None:
            return jsonify({'error': 'Destination account not found.'}), 400
        to_acc['balance'] = f'{float(to_acc["balance"]) + amount:.2f}'
    txn = {
        'id':           make_id('txn'),
        'ts':           str(int(time.time())),
        'from':         from_id,
        'to':           to_id,
        'amount':       f'{amount:.2f}',
        'note':         note,
        'status':       'completed',
        'payee_acc_no': '',
    }
    user['transactions'].append(txn)
    save_user(user)
    # Return updated balances directly so frontend has fresh data immediately
    return jsonify({'ok': True, 'accounts': user['accounts']})

@app.route('/api/accounts/add', methods=['POST'])
def api_accounts_add():
    username = session.get('username')
    if not username:
        return jsonify({'error': 'Not logged in.'}), 401
    data = request.json
    acc_type = data.get('type', 'Checking')
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Account name required.'}), 400
    balance = 0.00
    details = make_account_details(acc_type)
    user = load_user(username)
    acc = {
        'id': make_id('acc'),
        'type': acc_type,
        'name': name,
        'balance': f'{balance:.2f}',
        **details
    }
    user['accounts'].append(acc)
    save_user(user)
    return jsonify({'ok': True, 'account': acc})

@app.route('/api/deposit', methods=['POST'])
def api_deposit():
    username = session.get('username')
    if not username:
        return jsonify({'error': 'Not logged in.'}), 401
    data = request.json
    acc_id = data.get('acc_id')
    note = data.get('note', 'Deposit')
    try:
        amount = round(float(data.get('amount')), 2)
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid amount.'}), 400
    if amount <= 0:
        return jsonify({'error': 'Amount must be positive.'}), 400
    user = load_user(username)
    acc = next((a for a in user['accounts'] if a['id'] == acc_id), None)
    if acc is None:
        return jsonify({'error': 'Account not found.'}), 400
    acc['balance'] = f'{float(acc["balance"]) + amount:.2f}'
    txn = {
        'id': make_id('txn'),
        'ts': str(int(time.time())),
        'from': 'external',
        'to': acc_id,
        'amount': f'{amount:.2f}',
        'note': note,
        'status': 'completed'
    }
    user['transactions'].append(txn)
    save_user(user)
    return jsonify({'ok': True})

@app.route('/api/accounts/delete', methods=['POST'])
def api_accounts_delete():
    username = session.get('username')
    if not username:
        return jsonify({'error': 'Not logged in.'}), 401
    data = request.json
    acc_id = data.get('acc_id')
    user = load_user(username)
    acc = next((a for a in user['accounts'] if a['id'] == acc_id), None)
    if acc is None:
        return jsonify({'error': 'Account not found.'}), 400
    if float(acc['balance']) != 0:
        return jsonify({'error': 'Account balance must be $0.00 to delete.'}), 400
    user['accounts'] = [a for a in user['accounts'] if a['id'] != acc_id]
    save_user(user)
    return jsonify({'ok': True})

# ---------------------------------------------------------------------------
# Routes — bill pay / payees
# ---------------------------------------------------------------------------

@app.route('/api/payees', methods=['GET'])
def api_payees():
    username = session.get('username')
    if not username:
        return jsonify({'error': 'Not logged in.'}), 401
    user = load_user(username)
    return jsonify({'payees': user.get('payees', [])})

@app.route('/api/payees/add', methods=['POST'])
def api_payees_add():
    username = session.get('username')
    if not username:
        return jsonify({'error': 'Not logged in.'}), 401
    data = request.json
    name         = data.get('name', '').strip()
    amount       = data.get('amount', '0.00')
    acc_id       = data.get('acc_id', '')
    payee_acc_no = data.get('payee_acc_no', '').strip()
    if not name:
        return jsonify({'error': 'Payee name required.'}), 400
    if not acc_id:
        return jsonify({'error': 'Account required.'}), 400
    try:
        amount = f'{float(amount):.2f}'
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid default amount.'}), 400
    user = load_user(username)
    payee = {
        'id':           make_id('pay'),
        'name':         name,
        'acc_id':       acc_id,
        'amount':       amount,
        'payee_acc_no': payee_acc_no,
    }
    user['payees'].append(payee)
    save_user(user)
    return jsonify({'ok': True, 'payee': payee})

@app.route('/api/payees/delete', methods=['POST'])
def api_payees_delete():
    username = session.get('username')
    if not username:
        return jsonify({'error': 'Not logged in.'}), 401
    pay_id = request.json.get('pay_id')
    user = load_user(username)
    user['payees'] = [p for p in user.get('payees', []) if p['id'] != pay_id]
    save_user(user)
    return jsonify({'ok': True})

@app.route('/api/payees/pay', methods=['POST'])
def api_payees_pay():
    username = session.get('username')
    if not username:
        return jsonify({'error': 'Not logged in.'}), 401
    data = request.json
    pay_id = data.get('pay_id')
    amount = data.get('amount')
    note   = data.get('note', '')
    acc_id_override = data.get('acc_id')  # account selected in modal
    try:
        amount = round(float(amount), 2)
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid amount.'}), 400
    if amount <= 0:
        return jsonify({'error': 'Amount must be positive.'}), 400
    user = load_user(username)
    payee = next((p for p in user.get('payees', []) if p['id'] == pay_id), None)
    if payee is None:
        return jsonify({'error': 'Payee not found.'}), 400
    use_acc_id = acc_id_override or payee['acc_id']
    acc = next((a for a in user['accounts'] if a['id'] == use_acc_id), None)
    if acc is None:
        return jsonify({'error': 'Source account not found.'}), 400
    if float(acc['balance']) < amount:
        return jsonify({'error': 'Insufficient funds.'}), 400
    acc['balance'] = f'{float(acc["balance"]) - amount:.2f}'
    txn = {
        'id':           make_id('txn'),
        'ts':           str(int(time.time())),
        'from':         use_acc_id,
        'to':           'external',
        'amount':       f'{amount:.2f}',
        'note':         note or f'Payment to {payee["name"]}',
        'status':       'completed',
        'payee_acc_no': payee.get('payee_acc_no', ''),
    }
    user['transactions'].append(txn)
    save_user(user)
    return jsonify({'ok': True, 'accounts': user['accounts']})

# ---------------------------------------------------------------------------
# Routes — mouse collection
# ---------------------------------------------------------------------------

@app.route('/collect', methods=['POST'])
def collect():
    username = session.get('username')
    user_id = session.get('user_id')
    if not username or not user_id:
        return jsonify({'error': 'Not logged in.'}), 401
    events = request.json.get('events', [])
    ms = get_mouse_session(username, user_id)
    ms['events'].extend(events)
    if len(ms['events']) >= 1000:
        flush_mouse(username)
    return jsonify({'ok': True})

# ---------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)