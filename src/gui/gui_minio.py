import os
import sys
from tkinter import Tk, Listbox, Button, Scrollbar, VERTICAL, RIGHT, Y, BOTH, Frame, Label, messagebox, MULTIPLE
from tkinter.filedialog import askdirectory
from minio import Minio
from dotenv import load_dotenv
from minio.error import S3Error

# Load environment variables
load_dotenv()


def make_minio_client():
    access_key = os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("MINIO_ROOT_PASSWORD")
    minio_uri = os.getenv("MINIO_URI")

    if not access_key or not secret_key or not minio_uri:
        raise ValueError("MinIO credentials or URI not found in environment variables.")

    return Minio(
        minio_uri,
        secure=False,
        access_key=access_key,
        secret_key=secret_key,
    )


def update_current_path_label():
    """Update the label to show the current MinIO folder location."""
    if current_bucket:
        current_path_label.config(text=f"Current Path: {current_bucket}/{current_path}")
    else:
        current_path_label.config(text="Current Path: /")


def list_buckets():
    """List all buckets"""
    global current_bucket, current_path
    current_bucket = None
    current_path = ""
    listbox.delete(0, "end")
    update_current_path_label()
    try:
        buckets = client.list_buckets()
        for bucket in buckets:
            listbox.insert("end", f"[Bucket] {bucket.name}")
        back_button.pack_forget()  # Hide the back button at the root level
    except S3Error as e:
        messagebox.showerror("Error", f"Unable to list buckets: {e}")


def list_objects(bucket_name, prefix=""):
    """List top-level folders and files in the specified bucket and prefix."""
    global current_bucket, current_path
    current_bucket = bucket_name
    current_path = prefix

    update_current_path_label()

    listbox.delete(0, "end")
    try:
        objects = client.list_objects(bucket_name, prefix=prefix, recursive=False)
        seen_folders = set()
        for obj in objects:
            if obj.object_name.endswith("/"):
                folder_name = obj.object_name[len(prefix):].strip("/")
                if folder_name and folder_name not in seen_folders:
                    listbox.insert("end", f"[Folder] {folder_name}")
                    seen_folders.add(folder_name)
            else:
                file_name = obj.object_name[len(prefix):]
                listbox.insert("end", file_name)
        back_button.pack(side="top", pady=5) if prefix or bucket_name else back_button.pack_forget()
    except S3Error as e:
        messagebox.showerror("Error", f"Unable to list objects: {e}")


def on_item_click(event):
    """Handle single-click event"""
    if not listbox.curselection():  # Prevent error when clicking on an empty area
        return


def on_item_double_click(event):
    """Handle double-click event (open folder)."""
    selection = listbox.curselection()
    if not selection:
        return
    selected_item = listbox.get(selection[0])
    if selected_item.startswith("[Bucket]"):
        bucket_name = selected_item.replace("[Bucket] ", "")
        list_objects(bucket_name)
    elif selected_item.startswith("[Folder]"):
        folder_name = selected_item.replace("[Folder] ", "")
        list_objects(current_bucket, prefix=f"{current_path}{folder_name}/")


def go_back():
    """Go back to the previous path or bucket list."""
    if current_path:
        parent_path = "/".join(current_path.strip("/").split("/")[:-1])
        parent_path = f"{parent_path}/" if parent_path else ""
        list_objects(current_bucket, prefix=parent_path)
    else:
        list_buckets()
    update_current_path_label()


def download_selected():
    """Download selected files or folders."""
    selections = listbox.curselection()
    if not selections:
        messagebox.showwarning("Warning", "Please select files or folders first.")
        return

    # Ask the user to select the save directory
    save_path = askdirectory(title="Select download directory")
    if not save_path:
        return

    is_single_folder = (
        len(selections) == 1 and listbox.get(selections[0]).startswith("[Folder]")
    )

    for idx in selections:
        item = listbox.get(idx)
        if item.startswith("[Folder]"):
            folder_name = item.replace("[Folder] ", "")
            folder_save_path = os.path.join(save_path, folder_name)
            # Ensure target folder exists
            os.makedirs(folder_save_path, exist_ok=True)
            # Download the folder
            download_folder(folder_name, folder_save_path)
        else:
            download_file(item, save_path)

    if is_single_folder:
        print(f"{save_path}/{folder_name}")
    else:
        print(f"{save_path}")

    messagebox.showinfo("Success", "Files downloaded successfully!")




def download_file(file_name, save_path):
    """Download a single file."""
    try:
        local_file_path = os.path.join(save_path, file_name.split("/")[-1])
        client.fget_object(current_bucket, f"{current_path}{file_name}", local_file_path)
    except S3Error as e:
        messagebox.showerror("Error", f"Failed to download file: {e}")


def download_folder(folder_name, save_path):
    """Recursively download folder contents."""
    try:
        # List all objects in the specified folder
        objects = client.list_objects(current_bucket, prefix=f"{current_path}{folder_name}/", recursive=True)

        # Ensure the target directory exists
        os.makedirs(save_path, exist_ok=True)

        for obj in objects:
            # Remove the parent folder prefix to get the relative path
            relative_path = obj.object_name[len(f"{current_path}{folder_name}/"):]

            # Construct the local file path using save_path and relative_path
            local_file_path = os.path.join(save_path, relative_path)

            # Ensure subdirectories exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the file
            client.fget_object(current_bucket, obj.object_name, local_file_path)
    except S3Error as e:
        messagebox.showerror("Error", f"Failed to download folder: {e}")


# Initialize MinIO client
try:
    client = make_minio_client()
except ValueError as e:
    messagebox.showerror("Error", str(e))
    exit(1)

current_bucket = None
current_path = ""

# Create the main window
root = Tk()
root.title("MinIO File Browser")

# Create a label to display the current MinIO folder location
current_path_label = Label(root, text="Current Path: /", font=("Helvetica", 10), anchor="w", wraplength=600)
current_path_label.pack(fill="x", padx=10, pady=5)

# Create a frame to hold the listbox and scrollbar
frame = Frame(root)
frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

# Create the listbox and scrollbar
scrollbar = Scrollbar(frame, orient=VERTICAL)
listbox = Listbox(frame, width=80, height=20, yscrollcommand=scrollbar.set, selectmode=MULTIPLE)
scrollbar.config(command=listbox.yview)

# Layout the scrollbar and listbox
listbox.pack(side="left", fill=BOTH, expand=True)
scrollbar.pack(side=RIGHT, fill=Y)

# Bind single-click and double-click events
listbox.bind("<<ListboxSelect>>", on_item_click)
listbox.bind("<Double-Button-1>", on_item_double_click)

# Create the back button
back_button = Button(root, text="Go Back", command=go_back)
back_button.pack(side="top", pady=5)
back_button.pack_forget()  # Hidden by default

# Create the download button
download_button = Button(root, text="Download Selected Files/Folders", command=download_selected)
download_button.pack(pady=5)

# Check if bucket and prefix are passed as arguments
if len(sys.argv) > 1:
    bucket_name = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else ""
    list_objects(bucket_name, prefix=prefix)
else:
    list_buckets()

# Run the main loop
root.mainloop()
