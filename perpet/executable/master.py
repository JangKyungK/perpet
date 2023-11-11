import argparse
import subprocess
import os
import glob
import shutil
import sys


# Set up the argument parser
parser = argparse.ArgumentParser(description="Find folders to run")
parser.add_argument(
    "--input_name", type=str, help="Base directory to create folders in", required=True
)

# Parse command line arguments
args = parser.parse_args()
target_name = args.input_name

# target_name = "eeeooong"

# List of folder names to be created
folders = ["class", "img", "log", "model", "raw"]
script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(script_directory, os.pardir))
kohya_directory = os.path.join(parent_directory, "kohya_ss")

sys.path.append(kohya_directory)

import re

target_directory = os.path.join(parent_directory, "customer", target_name)

jpg_files = glob.glob(os.path.join(target_directory, "*.jpg"))
png_files = glob.glob(os.path.join(target_directory, "*.png"))
all_files = jpg_files + png_files

jpg_count = len(jpg_files)
png_count = len(png_files)

total_count = jpg_count + png_count
if total_count == 0:
    print("No photo in base folder!")
else:
    repeat_count = round(1500 / total_count)

raw_folder_path = ""
sub_img_path = ""

if os.path.exists(target_directory):
    # Iterate over the folder names
    for folder in folders:
        # Define the folder path
        folder_path = os.path.join(target_directory, folder)
        if os.path.exists(folder_path):
            continue
        else:
            # Use subprocess to run the mkdir command
            subprocess.run(["mkdir", folder_path], shell=True)
            if folder == "img":
                sub_img_folder = str(repeat_count) + "_xyzctf"
                sub_img_path = os.path.join(folder_path, sub_img_folder)
                subprocess.run(["mkdir", sub_img_path], shell=True)
            elif folder == "raw":
                for file_path in all_files:
                    # Get the base name of the file (i.e., just the file name without the source directory)
                    file_name = os.path.basename(file_path)

                    # Set the new file path in the destination directory
                    raw_folder_path = os.path.join(folder_path, file_name)

                    # Move the file
                    shutil.move(file_path, raw_folder_path)
                    resize_folder_path = os.path.join(folder_path, "resized")
                    if os.path.exists(resize_folder_path):
                        continue
                    else:
                        subprocess.run(["mkdir", resize_folder_path], shell=True)
else:
    print("No Target directory found")

img_folder_path = os.path.join(target_directory, "img")

# ============================== Preprossing images ===========================
import preprocess_imgs

try:
    preprocess_imgs.main([target_name])  # Replace with actual arguments
except SystemExit as e:
    print(f"Script exited with code {e.code}")

# preprocess_commend = "python preprocess_imgs.py --input_name " + target_name
# prep_result = subprocess.run(preprocess_commend, shell=True)

# if prep_result.returncode == 0:
#     print("Preprossing executed successfully")
#     # You can also access the output with result.stdout
# else:
#     print("Error in Preprossing execution")
#     print(prep_result.stderr)


#  ============================== tag img ====================================
print("Tag in process")


entries = os.listdir(img_folder_path)
img_folder_list = [
    entry for entry in entries if os.path.isdir(os.path.join(img_folder_path, entry))
]

frequency_tags_path = os.path.join(img_folder_path, img_folder_list[0])

tag_command = [
    "accelerate",
    "launch",
    "./kohya_ss/finetune/tag_images_by_wd14_tagger.py",
    "--batch_size=8",
    "--general_threshold=0.35",
    "--character_threshold=0.35",
    "--caption_extension=.txt",  # No need for quotes here
    "--model=SmilingWolf/wd-v1-4-convnextv2-tagger-v2",  # No need for quotes here either
    "--max_data_loader_n_workers=2",  # No need for quotes
    "--debug",
    "--remove_underscore",
    "--frequency_tags",
    frequency_tags_path,  # Variable included in the command
]

# Run the command
tag_process = subprocess.Popen(
    tag_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
)

# Check if the command was successful
for line in tag_process.stdout:
    print(line, end="")

tag_process.wait

# ============================== tag transform ====================================
print("Tag transform in process")
import tag_transform_cat

try:
    tag_transform_cat.main([target_name])  # Replace with actual arguments
except SystemExit as e:
    print(f"Script exited with code {e.code}")

# =============================== activate venv ===============================
# venv_commend = "./kohya_ss/venv/Scripts/activate"
# subprocess.run(venv_commend, shell=True)

# venv_path = "./kohya_ss/venv/Scripts/activate"

# # Add the venv's Python interpreter to the PATH environment variable
# sys.path.insert(0, os.path.join(venv_path, "bin"))

# # Activate the venv
# os.environ["VIRTUAL_ENV"] = venv_path

# # Run the Python script

# ============================== LoRA Trainging ===================================
# import train_network

log_folder_path = os.path.join(target_directory, "log")
model_folder_path = os.path.join(target_directory, "model")
ckpt_path = "C:\\Users\\lukes\\Documents\\AiProject\\stable-diffusion-webui\\models\\Stable-diffusion\\photon_v1.safetensors"

output_dir = "--output_dir=" + model_folder_path
logging_dir = "--logging_dir=" + log_folder_path
train_data_dir = "--train_data_dir=" + img_folder_path
pretrained_model_name_or_path = "--pretrained_model_name_or_path=" + ckpt_path
output_name = "--output_name=" + target_name
print("LoRA Training!!")

lora_command = [
    "accelerate",
    "launch",
    "--num_cpu_threads_per_process=2",
    "./kohya_ss/train_network.py",
    pretrained_model_name_or_path,
    train_data_dir,
    "--resolution=512,512",
    output_dir,
    logging_dir,
    "--network_alpha=1",
    "--save_model_as=safetensors",
    "--network_module=networks.lora",
    "--text_encoder_lr=5e-05",
    "--unet_lr=0.0001",
    "--network_dim=96",
    output_name,
    "--lr_scheduler_num_cycles=1",  # Removed quotes around the number
    "--no_half_vae",
    "--learning_rate=0.0001",
    "--lr_scheduler=cosine",
    "--lr_warmup_steps=135",
    "--train_batch_size=1",  # Removed quotes around the number
    "--max_train_steps=1350",  # Removed quotes around the number
    "--save_every_n_epochs=1",  # Removed quotes around the number
    "--mixed_precision=bf16",
    "--save_precision=bf16",
    '--caption_extension=".txt"',
    "--cache_latents",
    "--optimizer_type=AdamW8bit",
    "--max_data_loader_n_workers=0",  # Removed quotes around the number
    "--bucket_reso_steps=64",  # Removed quotes around the number
    "--xformers",
    "--bucket_no_upscale",
    "--noise_offset=0.0",
]

lora_prompt = " ".join(lora_command)

print(lora_prompt)

os.system(lora_prompt)

# lora_process = subprocess.Popen(
#     lora_command,
#     stdout=subprocess.PIPE,
#     stderr=subprocess.PIPE,
#     text=True,
#     encoding="utf-8",
#     errors="replace",
#     env=os.environ,  # Pass current environment variables
#     shell=True,
# )

# for line in lora_process.stdout:
#     print(line, end="")

# stdout, stderr = lora_process.communicate()

# print(stdout)
# if stderr:
#     print(stderr)

# # Wait for the process to finish
# lora_process.wait()
