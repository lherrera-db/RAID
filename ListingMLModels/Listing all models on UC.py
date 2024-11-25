# Databricks notebook source
import mlflow
from mlflow.client import MlflowClient
from pprint import pprint

# Set the registry URI to use Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Create an MLflow client
client = MlflowClient()

# Search for all registered models
models = client.search_registered_models()

# Create a list to store model information
model_info_list = []

# Iterate through each model and gather detailed information
for model in models:
    model_info = {
        "Name": model.name,
        "Latest Version": model.latest_versions[0].version if model.latest_versions else "N/A",
        "Description": model.description,
        "Versions": []
    }
    
    # Get all versions of the model
    versions = client.search_model_versions(f"name = '{model.name}'")
    
    for version in versions:
        version_info = {
            "Version": version.version,
            "Status": version.status,
            "Description": version.description,
            "Creation Timestamp": version.creation_timestamp,
            "Last Updated Timestamp": version.last_updated_timestamp,
            "User": version.user_id,
            "Source": version.source,
            "Run ID": version.run_id,
            "Tags": version.tags
        }
        model_info["Versions"].append(version_info)
    
    model_info_list.append(model_info)

# Print the gathered information (for verification)
for model_info in model_info_list:
    pprint(model_info)

# COMMAND ----------

# MAGIC %pip install reportlab

# COMMAND ----------

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(model_info_list, filename="models_report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    for model_info in model_info_list:
        # Add model name as a header
        elements.append(Paragraph(f"Model: {model_info['Name']}", styles['Heading1']))
        
        # Add model description
        elements.append(Paragraph(f"Description: {model_info['Description']}", styles['Normal']))
        
        # Add latest version
        elements.append(Paragraph(f"Latest Version: {model_info['Latest Version']}", styles['Normal']))
        
        # Add version details
        for version in model_info['Versions']:
            elements.append(Paragraph(f"Version {version['Version']}", styles['Heading2']))
            
            data = [
                ['Attribute', 'Value'],
                ['Status', version['Status']],
                ['Description', version['Description']],
                ['Creation Timestamp', str(version['Creation Timestamp'])],
                ['Last Updated Timestamp', str(version['Last Updated Timestamp'])],
                ['User', version['User']],
                ['Source', version['Source']],
                ['Run ID', version['Run ID']]
            ]
            
            # Add tags if present
            if version['Tags']:
                for tag_key, tag_value in version['Tags'].items():
                    data.append([f'Tag: {tag_key}', tag_value])
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('TOPPADDING', (0, 1), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
        
        elements.append(Paragraph("<br/><br/>", styles['Normal']))

    doc.build(elements)

# Generate the PDF report
generate_pdf(model_info_list)
