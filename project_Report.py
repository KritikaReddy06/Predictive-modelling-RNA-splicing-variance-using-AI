import os
import pandas as pd
import json
from datetime import datetime

def analyze_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        rows, cols = df.shape
        print(f"CSV: {file_path} — {rows} rows x {cols} cols")
        stats_msg = ""
        for col in df.select_dtypes(include='number').columns[:3]:
            stats_msg += f"{col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}; "
        return f"{os.path.basename(file_path)}: {rows}x{cols}, {stats_msg}"
    except Exception as e:
        return f"{os.path.basename(file_path)}: Error loading CSV ({str(e)})"

def analyze_json(file_path):
    try:
        with open(file_path) as f:
            data = json.load(f)
        keys = list(data.keys())
        print(f"JSON: {file_path} — keys: {keys[:5]}")
        return f"{os.path.basename(file_path)}: keys={keys[:5]}"
    except Exception as e:
        return f"{os.path.basename(file_path)}: Error loading JSON ({str(e)})"

def analyze_md(file_path):
    try:
        with open(file_path, encoding='utf-8') as f:
            first_lines = ''.join([next(f) for _ in range(5)])
        print(f"MD: {file_path} — Preview:\n{first_lines}")
        return f"{os.path.basename(file_path)}: {first_lines[:60].replace('\n', ' ')}"
    except Exception as e:
        return f"{os.path.basename(file_path)}: Error loading MD ({str(e)})"

def analyze_img(file_path):
    try:
        size = os.path.getsize(file_path)
        print(f"IMG: {file_path} — size: {size} bytes")
        return f"{os.path.basename(file_path)}: {size} bytes"
    except Exception as e:
        return f"{os.path.basename(file_path)}: Error reading image ({str(e)})"

def scan_and_report(project_dir):
    report_lines = []
    print(f"\n🔍 Scanning files in directory: {project_dir}\n")
    for root, dirs, files in os.walk(project_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.csv'):
                summary = analyze_csv(file_path)
            elif file.endswith('.json'):
                summary = analyze_json(file_path)
            elif file.endswith('.md'):
                summary = analyze_md(file_path)
            elif file.endswith('.png') or file.endswith('.jpg'):
                summary = analyze_img(file_path)
            else:
                continue
            report_lines.append(f"- {summary}")

    # Write the submission report
    report_content = "# Project File Analysis Report\n"
    report_content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report_content += "## File Summaries\n\n"
    report_content += '\n'.join(report_lines)
    report_content += "\n\n---\nGenerated automatically."

    with open(os.path.join(project_dir, "project_submission_report.md"), "w", encoding='utf-8') as f:
        f.write(report_content)
    print("\n✅ Submission report written to project_submission_report.md\n")

if __name__ == "__main__":
    # Change path below to your actual project folder if needed
    scan_and_report(os.getcwd())
