import argparse
import json
from pathlib import Path
import cv2 as cv
from processing.processing import perform_processing

# Function responsible for loading sample letters to the list of fonts.
def fonts():

    fonts_list = [cv.cvtColor(cv.imread("match_data/A.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/B.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/C.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/D.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/E.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/F.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/G.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/H.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/I.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/J.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/K.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/L.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/M.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/N.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/O.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/P.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/Q.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/R.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/S.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/T.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/U.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/V.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/W.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/X.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/Y.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/Z.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/0.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/1.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/2.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/3.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/4.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/5.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/6.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/7.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/8.png"), cv.COLOR_BGR2GRAY),
                  cv.cvtColor(cv.imread("match_data/9.png"), cv.COLOR_BGR2GRAY)]
    return fonts_list


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
    fonts_list = fonts()
    for image_path in images_paths:
        image = cv.imread(str(image_path))
        if image is None:
            print(f'Error loading image {image_path}')
            continue

        # Calling the function responsible for detecting license plate signs
        results[image_path.name] = perform_processing(fonts_list, image)

        # Writing results to json file
        with results_file.open('w') as output_file:
            json.dump(results, output_file, indent=4)


if __name__ == '__main__':
    main()
