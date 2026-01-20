import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import os
from datetime import datetime

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ForensicScanner:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.MODEL_COLUMNS = None
        self.model_path = "forensic_model.pkl"

        # Holds details of the last scan
        self.last_scan_details = None  # dict with predictions, probabilities, df, file_path

        self.setup_gui()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Advanced Forensic AI Scanner v3.0")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        # Status frame
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", padx=10, pady=5)

        self.status_label = ttk.Label(
            status_frame,
            text="Ready - Click 'Load/Train Model' to begin",
            font=("Arial", 10),
        )
        self.status_label.pack()

        # Buttons frame
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=10, pady=5)
        self.btn_frame = btn_frame

        ttk.Button(
            btn_frame,
            text="1. Load/Train Model",
            command=self.load_or_train_model,
            style="Accent.TButton",
        ).pack(side="left", padx=5)

        self.scan_btn = ttk.Button(
            btn_frame,
            text="2. Scan CSV File",
            command=self.run_live_scan,
            state="disabled",
        )
        self.scan_btn.pack(side="left", padx=5)

        self.bulk_btn = ttk.Button(
            btn_frame,
            text="3. Bulk Scan Folder",
            command=self.bulk_scan_folder,
            state="disabled",
        )
        self.bulk_btn.pack(side="left", padx=5)

        self.graph_btn = ttk.Button(
            btn_frame,
            text="4. Threat Probability Graph",
            command=self.show_threat_graph,
            state="disabled",
        )
        self.graph_btn.pack(side="left", padx=5)

        self.ip_btn = ttk.Button(
            btn_frame,
            text="5. Threat IPs",
            command=self.show_threat_ips,
            state="disabled",
        )
        self.ip_btn.pack(side="left", padx=5)

        # Results frame
        results_frame = ttk.LabelFrame(self.root, text="Scan Results", padding=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Results text with scrollbar
        self.results_text = tk.Text(results_frame, height=12, width=60, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(
            results_frame, orient="vertical", command=self.results_text.yview
        )
        self.results_text.configure(yscrollcommand=scrollbar.set)

        self.results_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress.pack(fill="x", padx=10, pady=5)

        # Frame for embedded matplotlib figure
        self.figure_frame = ttk.LabelFrame(
            self.root, text="Threat Probability Graph", padding=10
        )
        self.figure_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.canvas = None

    def log_message(self, message):
        """Log to GUI"""
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

                # Expect 4 items; if old file has more, ignore extras
                if isinstance(model_data, (list, tuple)) and len(model_data) >= 4:
                    (
                        self.model,
                        self.scaler,
                        self.encoders,
                        self.MODEL_COLUMNS,
                    ) = model_data[:4]
                else:
                    raise ValueError(
                        "Saved model file format is invalid. Delete forensic_model.pkl and retrain."
                    )

                self.log_message("✓ Model loaded successfully!")
            else:
                self.log_message("Training new model from UNSW-NB15 dataset...")
                self.train_new_model()

            self.log_message(f"✓ Model ready! Features: {len(self.MODEL_COLUMNS)}")
            self.status_label.config(text="Model Loaded - Ready to scan")

            # Enable scan-related buttons
            self.scan_btn.config(state="normal")
            self.bulk_btn.config(state="normal")

        except Exception as e:
            self.log_message(f"✗ Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to load/train model:\n{str(e)}")
        finally:
            self.progress.stop()

    def train_new_model(self):
        """Training with validation and scaling"""
        self.log_message("Loading UNSW-NB15 Dataset...")
        try:
            train = pd.read_csv("UNSW_NB15_training-set.csv")
            test = pd.read_csv("UNSW_NB15_testing-set.csv")
        except FileNotFoundError:
            raise FileNotFoundError(
                "UNSW-NB15 CSV files not found! Place training and testing CSVs next to this script."
            )

        # Drop unused columns
        drop_cols = ["id", "attack_cat"]
        train = train.drop(columns=[c for c in drop_cols if c in train.columns])
        test = test.drop(columns=[c for c in drop_cols if c in test.columns])

        # Encode categorical data
        cat_cols = train.select_dtypes(include=["object"]).columns
        combined_data = pd.concat([train[cat_cols], test[cat_cols]], axis=0)

        self.encoders = {}
        for col in cat_cols:
            enc = LabelEncoder()
            combined_clean = combined_data[col].fillna("missing").astype(str)
            enc.fit(combined_clean)
            train[col] = enc.transform(train[col].fillna("missing").astype(str))
            test[col] = enc.transform(test[col].fillna("missing").astype(str))
            self.encoders[col] = enc

        # Features and targets
        X_train = train.drop("label", axis=1)
        y_train = train["label"]
        X_test = test.drop("label", axis=1)
        y_test = test["label"]

        self.MODEL_COLUMNS = X_train.columns.tolist()

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

        self.model.fit(X_train_scaled, y_train)

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        self.log_message(
            f"Cross-val accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})"
        )

        # Test accuracy
        predictions = self.model.predict(X_test_scaled)
        acc = accuracy_score(y_test, predictions)
        self.log_message(f"Test accuracy: {acc:.3f} ({acc * 100:.1f}%)")

        # Save model
        joblib.dump(
            (self.model, self.scaler, self.encoders, self.MODEL_COLUMNS),
            self.model_path,
        )
        self.log_message("✓ Model saved for future use")

    def preprocess_new_data(self, new_logs):
        """Preprocess new data to match training format"""
        aligned_logs = pd.DataFrame(index=new_logs.index, columns=self.MODEL_COLUMNS)

        for col in self.MODEL_COLUMNS:
            if col in new_logs.columns:
                if col in self.encoders:  # Categorical
                    new_logs[col] = new_logs[col].fillna("missing").astype(str)
                    known_classes = set(self.encoders[col].classes_)

                    def safe_val(x):
                        return x if x in known_classes else list(known_classes)[0]

                    new_logs[col] = new_logs[col].apply(safe_val)
                    aligned_logs[col] = self.encoders[col].transform(new_logs[col])
                else:  # Numeric - median imputation
                    if len(new_logs[col].dropna()) > 0:
                        median_val = new_logs[col].median()
                    else:
                        median_val = 0
                    aligned_logs[col] = new_logs[col].fillna(median_val)
            else:
                aligned_logs[col] = 0

        aligned_logs_scaled = self.scaler.transform(aligned_logs)
        return aligned_logs_scaled

    def run_live_scan(self):
        """Single file scan"""
        if not self.model:
            messagebox.showwarning("Warning", "Load model first!")
            return

        file_path = filedialog.askopenfilename(
            title="Select Network Log CSV",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if not file_path:
            return

        self.scan_file(file_path)

    def bulk_scan_folder(self):
        """Scan entire folder"""
        if not self.model:
            messagebox.showwarning("Warning", "Load model first!")
            return

        folder_path = filedialog.askdirectory(title="Select Folder with CSV Files")
        if not folder_path:
            return

        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        if not csv_files:
            messagebox.showinfo("Info", "No CSV files found in folder!")
            return

        total_threats = 0
        total_packets = 0

        self.progress["mode"] = "determinate"
        self.progress["maximum"] = len(csv_files)
        self.progress["value"] = 0

        for i, filename in enumerate(csv_files):
            file_path = os.path.join(folder_path, filename)
            self.log_message(f"Scanning {filename}...")
            threats, packets = self.scan_file(file_path, show_popup=False)
            total_threats += threats
            total_packets += packets
            self.progress["value"] = i + 1
            self.root.update()

        self.progress["mode"] = "indeterminate"
        summary = (
            "BULK SCAN COMPLETE\n"
            f"Files: {len(csv_files)}\n"
            f"Total Packets: {total_packets:,}\n"
            f"TOTAL THREATS: {total_threats:,}"
        )
        self.log_message(summary)
        messagebox.showinfo("Bulk Scan Complete", summary)

    def scan_file(self, file_path, show_popup=True):
        """Core scanning logic"""
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

            threat_mask = predictions == 1
            threat_count = int(np.sum(threat_mask))
            high_confidence_mask = threat_mask & (confidence_scores > 0.9)
            high_confidence_threats = int(np.sum(high_confidence_mask))

            threat_pct = (
                (threat_count / len(predictions)) * 100 if len(predictions) > 0 else 0
            )

            self.log_message(
                f"✓ Packets: {len(predictions):,} | Threats: {threat_count:,} "
                f"({threat_pct:.1f}%)"
            )
            self.log_message(
                f" High-confidence threats (p>0.9): {high_confidence_threats}"
            )

            # Store details for later use (graph + IP listing)
            self.last_scan_details = {
                "file_path": file_path,
                "df": new_logs,
                "predictions": predictions,
                "probabilities": probabilities,
                "confidence_scores": confidence_scores,
                "threat_mask": threat_mask,
            }

            # Enable graph & IP buttons now that we have data
            self.graph_btn.config(state="normal")
            self.ip_btn.config(state="normal")

            if show_popup:
                msg = (
                    "SCAN COMPLETE\n\n"
                    f"File: {os.path.basename(file_path)}\n"
                    f"Total Packets: {len(predictions):,}\n"
                    f"Threats Found: {threat_count:,} ({threat_pct:.1f}%)\n"
                    f"High-confidence: {high_confidence_threats}"
                )
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

    def show_threat_graph(self):
        """Visualize threat probabilities and save graph image"""
        if not self.last_scan_details:
            messagebox.showinfo(
                "Info", "Run a scan first to generate threat probabilities."
            )
            return

        details = self.last_scan_details
        probabilities = details["probabilities"]
        confidence_scores = details["confidence_scores"]
        threat_mask = details["threat_mask"]
        file_path = details["file_path"]

        # For binary classification (0=benign,1=threat), take probability for class 1
        if probabilities.shape[1] >= 2:
            threat_prob = probabilities[:, 1]
        else:
            threat_prob = confidence_scores

        # Clear any existing figure
        for widget in self.figure_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(7, 3), dpi=100)
        ax = fig.add_subplot(111)

        x = np.arange(len(threat_prob))
        ax.plot(x, threat_prob, label="Threat probability", color="blue", linewidth=1)
        ax.scatter(
            x[threat_mask],
            threat_prob[threat_mask],
            color="red",
            s=10,
            label="Detected threat",
        )

        ax.set_title(f"Threat Probability - {os.path.basename(file_path)}")
        ax.set_xlabel("Packet index")
        ax.set_ylabel("Probability of threat")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right")

        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(fig, master=self.figure_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Save the figure as an image (PNG) next to the input file
        base, _ = os.path.splitext(file_path)
        out_path = base + "_threat_probability.png"
        fig.savefig(out_path, bbox_inches="tight")
        self.log_message(f"Threat probability graph saved as: {out_path}")

    def show_threat_ips(self):
        """List source/destination IPs for each detected threat"""
        if not self.last_scan_details:
            messagebox.showinfo(
                "Info", "Run a scan first to list threat IPs."
            )
            return

        details = self.last_scan_details
        df = details["df"]
        threat_mask = details["threat_mask"]

        if "srcip" not in df.columns or "dstip" not in df.columns:
            messagebox.showinfo(
                "Info",
                "Current log file has no 'srcip' or 'dstip' columns; cannot list IPs.",
            )
            return

        threats_df = df[threat_mask].copy()
        if threats_df.empty:
            messagebox.showinfo("Info", "No threats detected in last scan.")
            return

        self.log_message("Listing source/destination IPs for detected threats...")
        self.results_text.insert(
            tk.END,
            "-" * 60 + "\nTHREAT IP REPORT (last scanned file)\n" + "-" * 60 + "\n",
        )

        # If dataset has attack category, group by it
        attack_col = None
        for candidate in ["attack_cat", "attack_cat_label", "attack_type"]:
            if candidate in threats_df.columns:
                attack_col = candidate
                break

        if attack_col:
            grouped = threats_df.groupby(attack_col)
            for attack_type, group in grouped:
                src_ips = sorted(group["srcip"].astype(str).unique())
                dst_ips = sorted(group["dstip"].astype(str).unique())
                self.results_text.insert(
                    tk.END,
                    f"\nAttack type: {attack_type}\n"
                    f"  Source IPs ({len(src_ips)}): {', '.join(src_ips[:20])}"
                    + ("..." if len(src_ips) > 20 else "")
                    + "\n"
                    f"  Destination IPs ({len(dst_ips)}): {', '.join(dst_ips[:20])}"
                    + ("..." if len(dst_ips) > 20 else "")
                    + "\n",
                )
        else:
            src_ips = sorted(threats_df["srcip"].astype(str).unique())
            dst_ips = sorted(threats_df["dstip"].astype(str).unique())
            self.results_text.insert(
                tk.END,
                "\nThreats (no attack category column found)\n"
                f"  Source IPs ({len(src_ips)}): {', '.join(src_ips[:50])}"
                + ("..." if len(src_ips) > 50 else "")
                + "\n"
                f"  Destination IPs ({len(dst_ips)}): {', '.join(dst_ips[:50])}"
                + ("..." if len(dst_ips) > 50 else "")
                + "\n",
            )

        self.results_text.see(tk.END)

    def run(self):
        """Start the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        """Clean shutdown"""
        if messagebox.askokcancel("Quit", "Save model and quit?"):
            if self.model:
                try:
                    joblib.dump(
                        (self.model, self.scaler, self.encoders, self.MODEL_COLUMNS),
                        self.model_path,
                    )
                except Exception:
                    pass
            self.root.destroy()


if __name__ == "__main__":
    app = ForensicScanner()
    app.run()
