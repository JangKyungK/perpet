import os
import argparse


def main(args=None):
    parser = argparse.ArgumentParser(description="Find folders to run")
    parser.add_argument(
        "target_name", type=str, help="Base directory to create folders in"
    )

    # Parse command line arguments
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    target_name = args.target_name

    script_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.abspath(os.path.join(script_directory, os.pardir))
    target_directory = os.path.join(parent_directory, "customer", target_name)
    img_directory = os.path.join(target_directory, "img")

    entries = os.listdir(img_directory)
    img_folder_list = [
        entry for entry in entries if os.path.isdir(os.path.join(img_directory, entry))
    ]

    directory = os.path.join(img_directory, img_folder_list[0])
    # Set the directory where the text files are located

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)

            # Read the content of the file
            with open(filepath, "r") as file:
                content = file.read()

            # Perform the text transformations
            words = content.split(", ")
            new_words = ["xyzctf"]  # Add 'xyz' at the beginning
            if "cat" in words:
                new_words.append("cat")  # Place 'cat' second
                words.remove("cat")
            if "human" in words:
                words.remove("human")
            if "dog" in words:
                words.remove("dog")
            if "weibo username" in words:
                words.remove("weibo username")
            if "weibo logo" in words:
                words.remove("weibo logo")
            # 'human' is removed, so no need to add it

            # Remaining words are added at the end
            new_words.extend(words)

            # Join the transformed words back into a string
            new_content = ", ".join(new_words)

            # Overwrite the file with the new content
            with open(filepath, "w") as file:
                file.write(new_content)

    print("Text transformation complete.")


if __name__ == "__main__":
    main()
