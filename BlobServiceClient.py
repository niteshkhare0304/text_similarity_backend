import streamlit as st
from azure.storage.blob import BlobServiceClient

# --- Azure Blob Storage Configuration ---
# Replace these placeholders with your actual Azure credentials
container_name = "test"  # st.secrets["CONTAINER_NAME"]

def list_blobs(container_client):
    blobs = container_client.list_blobs()
    return [blob.name for blob in blobs]

# --- Streamlit Application ---
st.title("Azure Blob Storage File Explorer")

# Input: Azure Blob Storage Link
blob_link = st.text_input("Enter Azure Blob Storage Link:")

# Use the key directly in the code (not recommended for production)
connection_string = st.text_input(
    "Enter Azure Blob Storage Connection String:", 
    'DefaultEndpointsProtocol=https;AccountName=testlakshay;AccountKey=bFR7nH1OhHhn85T1JWJLYgc14PRD+d+h9g0cf7KJPAuICJ4ZtQCKwjZp9ROIO6s/o5WVLcNf2kRf+AStAwn4/A==;EndpointSuffix=core.windows.net'
)  # st.secrets["AZURE_CONNECTION_STRING"]

if blob_link and connection_string and container_name:
    # Extract container and path from the link
    try:
        # Example link format: https://<account_name>.blob.core.windows.net/<container_name>/<blob_name>
        link_parts = blob_link.split("/")
        container_name = link_parts[-2]
        blob_path = "/".join(link_parts[-2:])  # Include container and path

        # Connect to Blob Storage
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        # List files and display in a table
        files = list_blobs(container_client)

        st.subheader("Files in Blob Storage:")
        if files:
            st.table(files)
        else:
            st.write("No files found in the specified container.")
    except Exception as e:
        st.error(f"Error connecting to Blob Storage or listing files: {e}")
