import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import os
from datetime import datetime


class ForensicScanner:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.MODEL_COLUMNS = None
        self.model_path = "forensic_model.pkl"
        self.setup_gui()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Advanced Forensic AI Scanner v2.0")
        self.root.geometry("450x400")
        self.root.resizable(True, True)

        # Status frame
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", padx=10, pady=5)

        self.status_label = ttk.Label(status_frame, text="Ready - Click 'Load Model' to begin",
                                      font=("Arial", 10))
        self.status_label.pack()

        # Buttons frame
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(btn_frame, text="1. Load/Train Model", command=self.load_or_train_model,
                   style="Accent.TButton").pack(side="left", padx=5)
        ttk.Button(btn_frame, text="2. Scan CSV File", command=self.run_live_scan,
                   state="disabled").pack(side="left", padx=5)
        ttk.Button(btn_frame, text="3. Bulk Scan Folder", command=self.bulk_scan_folder).pack(side="left", padx=5)

        # Results frame
        results_frame = ttk.LabelFrame(self.root, text="Scan Results", padding=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Results text with scrollbar
        self.results_text = tk.Text(results_frame, height=12, width=60, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)

        self.results_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(fill="x", padx=10, pady=5)

    def log_message(self, message):
        """Thread-safe logging to GUI"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.results_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.results_text.see(tk.END)
        self.root.update_idletasks()

    def load_or_train_model(self):
        """Load existing model or train new one"""
        try:
            self.progress.start()
            self.log_message("Loading/Training model...")

            if os.path.exists(self.model_path):
                self.log_message("Loading saved model...")
                model_data = joblib.load(self.model_path)
                self.model, self.scaler, self.encoders, self.MODEL_COLUMNS = model_data
                self.log_message("✓ Model loaded successfully!")
            else:
                self.log_message("Training new model from UNSW-NB15 dataset...")
                self.train_new_model()

            self.log_message(f"✓ Model ready! Features: {len(self.MODEL_COLUMNS)}")
            self.status_label.config(text="Model Loaded - Ready to scan")

            # Enable scan buttons
            btn_frame = self.root.winfo_children()[1]
            for widget in btn_frame.winfo_children():
                if isinstance(widget, ttk.Button) and ("Scan" in widget['text'] or "Load" in widget['text']):
                    widget.config(state="normal")

        except Exception as e:
            self.log_message(f"✗ Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to load/train model:\n{str(e)}")
        finally:
            self.progress.stop()

    def train_new_model(self):
        """Enhanced training with validation and scaling"""
        print("Loading UNSW-NB15 Dataset...")
        try:
            train = pd.read_csv("UNSW_NB15_training-set.csv")
            test = pd.read_csv("UNSW_NB15_testing-set.csv")
        except FileNotFoundError:
            raise FileNotFoundError("UNSW-NB15 CSV files not found! Download from: https://research.unsw.edu.au/")

        # Enhanced preprocessing
        drop_cols = ['id', 'attack_cat']
        train = train.drop(columns=[c for c in drop_cols if c in train.columns])
        test = test.drop(columns=[c for c in drop_cols if c in test.columns])

        # Encode categorical data (IMPROVED: handle unseen categories)
        cat_cols = train.select_dtypes(include=['object']).columns
        combined_data = pd.concat([train[cat_cols], test[cat_cols]], axis=0)

        for col in cat_cols:
            enc = LabelEncoder()
            # Handle NaN and unseen values
            combined_clean = combined_data[col].fillna('missing').astype(str)
            enc.fit(combined_clean)
            train[col] = enc.transform(train[col].fillna('missing').astype(str))
            test[col] = enc.transform(test[col].fillna('missing').astype(str))
            self.encoders[col] = enc

        # Prepare features and targets
        X_train = train.drop('label', axis=1)
        y_train = train['label']
        X_test = test.drop('label', axis=1)
        y_test = test['label']

        self.MODEL_COLUMNS = X_train.columns.tolist()

        # Scale features (NEW: improves model performance)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train enhanced model
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train_scaled, y_train)

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        self.log_message(f"Cross-val accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        # Test accuracy
        predictions = self.model.predict(X_test_scaled)
        acc = accuracy_score(y_test, predictions)
        self.log_message(f"Test accuracy: {acc:.3f} ({acc * 100:.1f}%)")

        # Save model
        joblib.dump((self.model, self.scaler, self.encoders, self.MODEL_COLUMNS), self.model_path)
        self.log_message("✓ Model saved for future use")

    def preprocess_new_data(self, new_logs):
        """Advanced preprocessing for new data"""
        # Smart column alignment with better imputation
        aligned_logs = pd.DataFrame(index=new_logs.index, columns=self.MODEL_COLUMNS)

        for col in self.MODEL_COLUMNS:
            if col in new_logs.columns:
                if col in self.encoders:  # Categorical
                    new_logs[col] = new_logs[col].fillna('missing').astype(str)
                    # Handle unseen categories safely
                    known_classes = set(self.encoders[col].classes_)
                    new_logs[col] = new_logs[col].apply(
                        lambda x: x if x in known_classes else list(known_classes)[0]
                    )
                    aligned_logs[col] = self.encoders[col].transform(new_logs[col])
                else:  # Numeric - use median imputation
                    median_val = new_logs[col].median() if len(new_logs[col].dropna()) > 0 else 0
                    aligned_logs[col] = new_logs[col].fillna(median_val)
            else:
                aligned_logs[col] = 0

        # Scale features
        aligned_logs_scaled = self.scaler.transform(aligned_logs)
        return aligned_logs_scaled

    def run_live_scan(self):
        """Enhanced single file scan"""
        if not self.model:
            messagebox.showwarning("Warning", "Load model first!")
            return

        file_path = filedialog.askopenfilename(
            title="Select Network Log CSV",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        self.scan_file(file_path)

    def bulk_scan_folder(self):
        """NEW: Scan entire folder"""
        if not self.model:
            messagebox.showwarning("Warning", "Load model first!")
            return

        folder_path = filedialog.askdirectory(title="Select Folder with CSV Files")
        if not folder_path:
            return

        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not csv_files:
            messagebox.showinfo("Info", "No CSV files found in folder!")
            return

        total_threats = 0
        total_packets = 0

        self.progress['mode'] = 'determinate'
        self.progress['maximum'] = len(csv_files)
        self.progress['value'] = 0

        for i, filename in enumerate(csv_files):
            file_path = os.path.join(folder_path, filename)
            self.log_message(f"Scanning {filename}...")
            threats, packets = self.scan_file(file_path, show_popup=False)
            total_threats += threats
            total_packets += packets
            self.progress['value'] = i + 1
            self.root.update()

        self.progress['mode'] = 'indeterminate'
        summary = f"BULK SCAN COMPLETE\nFiles: {len(csv_files)}\nTotal Packets: {total_packets:,}\nTOTAL THREATS: {total_threats:,}"
        self.log_message(summary)
        messagebox.showinfo("Bulk Scan Complete", summary)

    def scan_file(self, file_path, show_popup=True):
        """Core scanning logic - FULLY FIXED"""
        try:
            self.log_message(f"Scanning: {os.path.basename(file_path)}")
            new_logs = pd.read_csv(file_path)

            if len(new_logs) == 0:
                self.log_message("Empty file!")
                return 0, 0

            # Preprocess
            aligned_scaled = self.preprocess_new_data(new_logs)

            # Predict with confidence scores
            predictions = self.model.predict(aligned_scaled)
            probabilities = self.model.predict_proba(aligned_scaled)
            confidence_scores = np.max(probabilities, axis=1)

            # FIXED: Properly define all variables with clear logic
            threat_mask = (predictions == 1)
            threat_count = np.sum(threat_mask)
            high_confidence_mask = threat_mask & (confidence_scores > 0.9)
            high_confidence_threats = np.sum(high_confidence_mask)

            # Safe percentage calculation
            threat_pct = (threat_count / len(predictions)) * 100 if len(predictions) > 0 else 0

            # Detailed results
            self.log_message(f"✓ Packets: {len(predictions):,} | Threats: {threat_count:,} ({threat_pct:.1f}%)")
            self.log_message(f"  High-confidence threats: {high_confidence_threats}")

            if show_popup:
                msg = f"SCAN COMPLETE\n\nFile: {os.path.basename(file_path)}\nTotal Packets: {len(predictions):,}\nThreats Found: {threat_count:,} ({threat_pct:.1f}%)\nHigh-confidence: {high_confidence_threats}"
                if threat_count > 0:
                    messagebox.showwarning("⚠️ THREAT DETECTED", msg)
                else:
                    messagebox.showinfo("✅ CLEAN", msg)

            return threat_count, len(predictions)

        except Exception as e:
            error_msg = f"Scan failed: {str(e)}"
            self.log_message(f"✗ {error_msg}")
            if show_popup:
                messagebox.showerror("Scan Error", error_msg)
            return 0, 0

    def run(self):
        """Start the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        """Clean shutdown"""
        if messagebox.askokcancel("Quit", "Save model and quit?"):
            if self.model:
                try:
                    joblib.dump((self.model, self.scaler, self.encoders, self.MODEL_COLUMNS), self.model_path)
                except:
                    pass
            self.root.destroy()


if __name__ == "__main__":
    app = ForensicScanner()
    app.run()
