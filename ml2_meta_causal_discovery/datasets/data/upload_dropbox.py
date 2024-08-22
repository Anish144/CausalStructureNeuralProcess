import dropbox

# Initialize a Dropbox object instance
dbx = dropbox.Dropbox('sl.B0vhx9xi0GzhzrRef4D_xxS8UVCRQLJd_wC-RFeE5Gpz7ephRiQWeTxh9Z75A-7aKEXySfjcg7vnv0mq309ylCdUzOSED9ZkP5bWj7QqatK5ZQG3ayGvz3oj2TtSD2wp45rqGVI4S3JdGdkTp1DQ')

# Function to upload a file
def upload_file(file_path, target_path):
    with open(file_path, "rb") as f:
        try:
            dbx.files_upload(f.read(), target_path, mode=dropbox.files.WriteMode("overwrite"))
            print("File uploaded successfully!")
        except Exception as e:
            print("Error uploading file: ", e)

# Example usage
import time
start = time.time()
upload_file(
    './synth_training_data/gpl.z01',
    '/ml2_meta_causal_discovery/'
)
print("Time taken: ", time.time() - start)

