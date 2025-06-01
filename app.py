from flask import Flask, render_template, jsonify, request, send_file, send_from_directory
import datetime
import subprocess
import json
import csv

app = Flask(__name__)
is_refreshing = False  # Track backend refresh status


last_refresh_time = None
import time
import datetime
import subprocess
import time
from tqdm import tqdm

def refresh_data():
    """Runs the web scraping scripts and updates the last refresh time"""
    global last_refresh_time
    last_refresh_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        time.sleep(2)
        print("Webscraping Step1 in progress...")
        subprocess.run("python webscrapping_compiled.py", shell=True, check=True)
        return {"status": "success", "last_refresh_time": last_refresh_time}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "step": "webscrapping_compiled.py", "message": str(e)}


def refresh_pending_pdufa_data():
    """Runs the web scraping scripts and updates the last refresh time"""
    global last_refresh_time
    last_refresh_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        time.sleep(2)
        print("Pending PDUFA Actions Updates in progress")
        subprocess.run("python grok_Pending_PDUFA_Actions.py", shell=True, check=True)
        return {"status": "success", "last_refresh_time": last_refresh_time}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def read_json_file():
    """Reads the latest scraped data for UI"""
    try:
        file_path = "final_output_json_master_sample.json"
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return []

@app.route("/")
def index():
    sample_data = read_json_file()
    return render_template("index.html", records=sample_data, last_refresh_time=last_refresh_time)

@app.route("/refresh", methods=["POST"])
def refresh():
    global is_refreshing
    if is_refreshing:
        return jsonify({"status": "error", "message": "Refresh already in progress!"})

    is_refreshing = True  # Set flag to indicate refresh started

    result = refresh_data()
    is_refreshing = False  # Reset flag after completion

    if result["status"] == "success":
        sample_data = read_json_file()
        return jsonify({
            "status": "success",
            "records": sample_data,
            "last_refresh_time": result["last_refresh_time"]
        })
    else:
        return jsonify(result)

@app.route("/get_refresh_status")
def get_refresh_status():
    global is_refreshing
    return jsonify({"is_refreshing": is_refreshing})  # Frontend can check status


@app.route("/refresh_pending_pdufa", methods=["POST"])
def refresh_pending_pdufa():
    """Refresh data, process it, and return latest sample data"""
    result = refresh_pending_pdufa_data()
    if result["status"] == "success":
        return jsonify({"status": "success", "last_refresh_time": result["last_refresh_time"]})
    else:
       return jsonify(result)

@app.route('/static/pdfs/<filename>')
def download_file(filename):
    return send_from_directory('static/pdfs', filename, as_attachment=True)

@app.route("/generateSoc")
def generatesoc():
    return render_template("generateSoc.html")


@app.route("/tppPeakSales")
def tpppeaksales():
    return render_template("tppPeakSales.html")

@app.route("/download")
def download():
    """Generate and return a CSV file"""
    filename = "latest_combined_data_output.csv"
    file_path = "final_output_json_master.json"

    try:
        with open(file_path, 'r') as file:
            latest_data = json.load(file)
        
        with open(filename, "w", newline="", encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'AA_Date', 'Drug_Name', 'Brand_Name', 'Active_Ingredient',
                             'Approved_Indication', 'Company_Name', 'Relevant Expedited Programs',
                             'Approval_Details', 'Efficacy and Safety', 'Broad Tumor', 'Biomarker', 'URL Link'])
            
            for record in latest_data:
                timestamp_s = record['AA_Date'] / 1000
                date_time = datetime.datetime.fromtimestamp(timestamp_s)
                formatted_date = date_time.strftime('%Y/%m/%d')

                writer.writerow([record["index"], formatted_date, record["Drug_Name"], record["Brand_Name"],
                                 record["Active_Ingredient"], record["Approved_Indication"], record["Company_name"],
                                 record["Relevant Expedited Programs"], record["Approval_Details"],
                                 record["Efficacy and Safety"], record["Broad_Tumor"], record["Biomarker"],
                                 record["url_name"]])

        return send_file(filename, as_attachment=True)
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True)