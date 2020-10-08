# Detection of license plate characters using the OpenCV library 

The main goal of the task was to read the characters from the Polish license plates.
Images of the plates were taken with a standard definition phone. 

Sample photos have been placed in the **train** folder.
Moreover, character samples were placed in the **match_data** folder. 

These samples will later be used to check the structural similarity of the detected characters.


### The main components of the code

The code was written in Python. I split it into two main files. The **Main.py** file includes downloading photos from a folder and their handling, as well as calling the processing function and saving the results to a json file.

The **processing.py** file has been placed in the processing folder and it contains the main image operations. We will focus on it later in the description.

We will process images similar to the one below.

![Screenshot](https://github.com/KRoszyk/License_plates_OpenCV_project/blob/main/openCV__smaller_images/original.PNG)
### Image processing using OpenCV

As mentioned before, image processing is included in the **processing.py**. 
Below is a description of the individual functions and the results of their image processing.

- **perform_processing function** - This main function is used to invoke other side functions.

- **preprocessing function** - This function resizes the original image, converts it to grayscale and looks for outlines to find the license plate in the image. 
All operations have been commented out in detail in the code. The results of these operations are shown below.

![Screenshot](https://github.com/KRoszyk/License_plates_OpenCV_project/blob/main/openCV__smaller_images/resized_image.PNG)

![Screenshot](https://github.com/KRoszyk/License_plates_OpenCV_project/blob/main/openCV__smaller_images/outlines.PNG)

- **plate_transform function** - This function uses a perspective transformation to straighten outlines that are possibly a license plate. 
This will be useful later in processing the characters. The result is shown below.

![Screenshot](https://github.com/KRoszyk/License_plates_OpenCV_project/blob/main/openCV__smaller_images/trans_plate.PNG)

- **plate_processing function** - This function uses conversion from BGR to HSV color space to remove the blue stripe with country mark. What is more, it performs operations to find the contours of the letters on the license plate and uses hierarchy to remove inside outlines. As output, the function returns a list of boxes containing letters from the plate.

- **letters_processing function** - This function uses perspective transformation of the letters to straighten them.  After this operation it is checked whether 7 characters have been detected (number for the standard Polish license plate). In addition, morphological operations are performed to remove all disturbances and noises. After this processing, ready character images are saved to the list. An example of the letter output from the list is shown below. 

![Screenshot](https://github.com/KRoszyk/License_plates_OpenCV_project/blob/main/openCV__smaller_images/letter.PNG)

- **final_processing function** - In this function the structural similarity of letters is examined together with appropriately prepared match data from the **match_data** folder. the **scikit-image** library is used to test for structural similarity. Each matching character has its own index in the list and if a given character has a higher probability of matching than the previous one, its index overwrites the previous value of the index variable.

  ```python
    # Checking the best match for the cut letters and those from the reference list using the scikit-image library
    for original in adapt_letter:
        result = 0
        index = 0
        for i, font in enumerate(fonts_list):
            score = ssim(original, font, multichannel=False)
            if score > result:
                result = score
                index = i
    ```

### Summary and conclusions

If you want to use this code,remember to declare parameters in the Run configurations.
This code works fine, however it is about 85% efficient for character recognition from random images. If you need more, you can try to use other morphological operations from OpenCV library or convolutional neural network. 

If you have any questions, just leave a comment or write to me a message. :)
