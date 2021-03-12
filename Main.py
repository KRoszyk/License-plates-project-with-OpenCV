import argparse
import json
from pathlib import Path
import cv2 as cv
from processing.processing import perform_processing
import os


# Function responsible for loading sample letters to the list of fonts.
def fonts():
    predict_fonts = os.listdir("match_data")
    predict_fonts.sort()
    fonts_list = []
    for font in predict_fonts:
        fonts_list.append(cv.cvtColor(cv.imread("match_data/"+str(font)), cv.COLOR_BGR2GRAY))
    return predict_fonts, fonts_list


def main():
    # Operations to load photos from a folder
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()
    images_dir = Path(args.images_dir)
    results_file = Path(args.results_file)
    images_paths = sorted([image_path for image_path in images_dir.iterdir() if image_path.name.endswith('.jpg')])
    results = {}

    # Calling function responsible for creating list of font images
    predict_fonts, fonts_list = fonts()
    for image_path in images_paths:
        image = cv.imread(str(image_path))
        print(image_path)
        if image is None:
            print(f'Error loading image {image_path}')
            continue

        # Calling the function responsible for detecting license plate signs
        results[image_path.name] = perform_processing(predict_fonts, fonts_list, image)

        # Writing results to json file
        with results_file.open('w') as output_file:
            json.dump(results, output_file, indent=4)


if __name__ == '__main__':
    main()
