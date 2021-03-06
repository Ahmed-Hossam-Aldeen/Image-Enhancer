# Image-Enhancer
 Increase the resolution of your images, using the latest algorithms, with a simple-to-use function in the OpenCV library.
 
 
![Capture](https://user-images.githubusercontent.com/61332730/155848372-c9a3afce-72d1-435e-a386-d4f96b6a2022.PNG)


## Models used
1. **ESPCN**: This is a small model with fast and good inference. It can do real-time video upscaling (depending on image size).

2. **FSRCNN**: This is also small model with fast and accurate inference. Can also do real-time video upscaling.

3. **LapSRN**: This is a medium sized model that can upscale by a factor as high as 8.


### ESPCN Samples
![1_A8yToxEh-f0_1Up8u51aHQ](https://user-images.githubusercontent.com/61332730/155812372-4022571b-b2f3-4284-9794-38a0bb4ed805.png)

original image

![1_U7fbkyr4gvceALewEOi3Dw](https://user-images.githubusercontent.com/61332730/155812374-06933b3e-3c69-4d17-a94b-46f84141dfa9.png)

Upscaled by ESPCN (factor of 2)


![1_swfxLCsjlBq7Y6etJWdleQ](https://user-images.githubusercontent.com/61332730/155812384-4aa1d077-2575-4a59-8303-ed433ada25ab.png)

Upscaled by ESPCN (factor of 4)

### FSRCNN Samples

![1_7sqZ6SIRlyR6ex1IbFjtFg](https://user-images.githubusercontent.com/61332730/155812835-c495b6ef-11ed-476a-b856-febca2f577f0.png)

original image

![1__vSikPDJavzxRXMI-wo6vQ](https://user-images.githubusercontent.com/61332730/155812840-02eabfb4-0281-4279-865c-bc9e9b89bf44.png)

Upscaled by FSRCNN (factor of 2)

![1_QYeg9TBGItkfmyO2NKEFGA](https://user-images.githubusercontent.com/61332730/155812861-3230c823-8de8-407a-a9eb-f9fdddd897bc.png)

Upscaled by FSRCNN (factor of 4)
