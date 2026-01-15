import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ===============================
# 1. SETUP & TRAINING
# ===============================
print("Loading UNSW-NB15 Dataset...")
try:
    train = pd.read_csv("UNSW_NB15_training-set.csv")
    test = pd.read_csv("UNSW_NB15_testing-set.csv")
except FileNotFoundError:
    print("ERROR: Training files not found! Ensure UNSW_NB15 csv files are in the folder.")
    exit()

# Drop unnecessary columns
drop_cols = ['id', 'attack_cat']
train = train.drop(columns=[c for c in drop_cols if c in train.columns])
test = test.drop(columns=[c for c in drop_cols if c in test.columns])

# Encode categorical data
encoders = {}
cat_cols = train.select_dtypes(include=['object']).columns

for col in cat_cols:
    enc = LabelEncoder()
    # Combine train/test to learn all labels safely
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    enc.fit(combined)
    train[col] = enc.transform(train[col].astype(str))
    test[col] = enc.transform(test[col].astype(str))
    encoders[col] = enc

X_train = train.drop('label', axis=1)
y_train = train['label']
X_test = test.drop('label', axis=1)
y_test = test['label']
MODEL_COLUMNS = X_train.columns

print("\nTraining Forensic Model (Random Forest)...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ===============================
# 2. SHOW CALCULATIONS (Text Only)
# ===============================
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

print("\n" + "=" * 30)
print(f" MODEL ACCURACY: {acc * 100:.2f}%")
print("=" * 30)
print("\nDetailed Classification Report:")
print(classification_report(y_test, predictions))


# ===============================
# 3. GUI POPUP LOGIC
# ===============================
def run_live_scan():
    file_path = filedialog.askopenfilename(title="Select Network Log CSV",
                                           filetypes=[("CSV Files", "*.csv")])
    if not file_path: return

    try:
        new_logs = pd.read_csv(file_path)

        # --- SMART ALIGNMENT (Prevents Crashes) ---
        aligned_logs = pd.DataFrame(columns=MODEL_COLUMNS)
        for col in MODEL_COLUMNS:
            if col in new_logs.columns:
                aligned_logs[col] = new_logs[col]
            else:
                aligned_logs[col] = 0  # Fill missing with 0

        # --- APPLY ENCODING ---
        for col, enc in encoders.items():
            if col in aligned_logs.columns:
                aligned_logs[col] = aligned_logs[col].astype(str)
                # Map unknown labels to the first known class to prevent error
                known = set(enc.classes_)
                aligned_logs[col] = aligned_logs[col].apply(lambda x: x if x in known else list(known)[0])
                aligned_logs[col] = enc.transform(aligned_logs[col])

        # --- PREDICT ---
        scan_results = model.predict(aligned_logs)
        threat_count = np.sum(scan_results)

        # --- SHOW POPUP ---
        msg = f"Scan Finished.\n\nTotal Packets: {len(scan_results)}\nTHREATS FOUND: {threat_count}"
        if threat_count > 0:
            messagebox.showwarning("THREAT DETECTED", msg)
        else:
            messagebox.showinfo("CLEAN", msg + "\nSystem is safe.")

    except Exception as e:
        messagebox.showerror("Error", str(e))


# ===============================
# 4. LAUNCH WINDOW
# ===============================
print("\nLaunching Control Panel...")
root = tk.Tk()
root.title("Forensic AI V1.0")
root.geometry("300x150")

tk.Label(root, text="Forensic Scanner", font=("Arial", 14, "bold")).pack(pady=20)
tk.Button(root, text="Load & Scan CSV", command=run_live_scan, bg="darkred", fg="white").pack(pady=10)

root.mainloop()