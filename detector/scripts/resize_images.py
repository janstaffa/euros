from PIL import Image
import os
import argparse
import filetype


def main():
    parser = argparse.ArgumentParser(prog='Image resizer')
    parser.add_argument('-dir', '--directory')
    parser.add_argument('-out', '--output')
    args = parser.parse_args()
    
    if args.directory == None or args.output == None:
        print("Missing argument/s")
        exit(1)
        
        
    for file in os.listdir(args.directory):
        path = os.path.join(args.directory, file)
        
        if not filetype.is_image(path):
            continue
        
        img = Image.open(path)
        
        resized = img.resize((960, 720)).rotate(-90, expand=True)
        new_path = os.path.join(args.output, file)
        resized.save(new_path)
        print("Saved", new_path)


if __name__ == "__main__":
    main()