import re


def remove_asian_characters(file_path):
    # Regular expression pattern for matching Chinese and Japanese characters
    pattern = r"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]"

    # Read the original file
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Remove Chinese and Japanese characters
    cleaned_content = re.sub(pattern, "", content)

    # Write the cleaned content back to the file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(cleaned_content)


# Replace 'your_script.py' with the path to your Python script
remove_asian_characters("C:\\Users\\lukes\\Documents\\LoRA_Training\\kohya_ss")
