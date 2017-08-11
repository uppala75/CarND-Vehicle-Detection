
## Vehicle Detection Project

### The goals for this project are the following:

### - Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
### - Optionally, apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
### - Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
### - Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
### - Estimate a bounding box for vehicles detected.

### Import the necessary libraries and functions


```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

```

### Load the Car/Not Car images


```python
import os
import glob

# Load the vehicle and non-vehicle image data
basedir = 'vehicles/'
image_types = os.listdir(basedir)
cars = []
for imtype in image_types:
    cars.extend(glob.glob(basedir+imtype+'/*'))
    
print('Number of Vehicle Images found:', len(cars))
with open("cars.txt", 'w') as f:
    for fn in cars:
        f.write(fn+'\n')

basedir = 'non-vehicles/'
image_types = os.listdir(basedir)
notcars = []
for imtype in image_types:
    notcars.extend(glob.glob(basedir+imtype+'/*'))
    
print('Number of Non-Vehicle Images found:', len(notcars))
with open("notcars.txt", 'w') as f:
    for fn in notcars:
        f.write(fn+'\n')


```

    Number of Vehicle Images found: 8792
    Number of Non-Vehicle Images found: 8968
    

### List of useful functions for the project adapted from the lessons


```python
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                     vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=feature_vec)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features
    

def bin_spatial(image, size=(32, 32)):
    color1 = cv2.resize(image[:,:,0], size).ravel()
    color2 = cv2.resize(image[:,:,1], size).ravel() 
    color3 = cv2.resize(image[:,:,2], size).ravel() 
    return np.hstack((color1, color2, color3))

# Define a function to compute color histogram features  
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                    hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)
        
        if spatial_feat == True:
            # Apply bin_spatial() to get spatial color features
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False,
                                                feature_vec=True)
            #Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # return list of feature vectors
    return features
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y NO BUFFER IN UTUBE VID
    #nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    #ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Draws a bounding box given a set of coordinates of opposite rectangle corners
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, vis=False):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.concatenate(hog_features)
        else:
            if vis == True:
                hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=True, feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    if vis == True:
        return np.concatenate(img_features), hog_image
    else:
        return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

def visualize(fig, rows, cols, imgs, titles):
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.title(i+1)
        img_dims = len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap='hot')
            plt.title(titles[i])
        else:
            plt.imshow(img)
            plt.title(titles[i])
            
```

### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

### The get_hog_features() function in the previous cell was used to extract HOG features from the training images. The extract_features() function combines the features from HOG along with spatial features and color histogram features as seen in the cell below. While the extract_features() function uses multiple images as input, the function single_img_features() below uses a single image for visualization purposes. Otherwise, the functions are the same.


```python
%matplotlib inline

#choose random indices in the set of images
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))

# read in car & non car images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])

# define feature parameters
color_space = 'RGB'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True

car_features, car_hog_image = single_img_features(car_image, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat, vis=True)
notcar_features, notcar_hog_image = single_img_features(notcar_image, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat, vis=True)

images = [car_image, car_hog_image, notcar_image, notcar_hog_image]
titles = ['car image', 'car HOG image', 'notcar_image', 'notcar HOG image']
fig = plt.figure(figsize=(12,3))#,dpi=80
visualize(fig, 1, 4, images, titles)


```


![png](output_8_0.png)


### 2. Explain how you settled on your final choice of HOG parameters.
### In the cell below, I used all hog channels by keeping the orientation, pix_per_cell, cell_per_block constant and derived the HOG features. I also kept the spatial size and color histogram bins constant to derive these features. The extract_features() function extracts all the above features together. As a first step, I experimented with different color spaces.  My thought process was to check for a balance between the test accuracy and the prediction time on the classifier with different color spaces. Once I pick the best option and look at the end result, I can revisit and review if further tuning is needed for the HOG parameters and the parameters for the other features. As seen in the results of the cell below, the color space 'YCrCb' shows a good balance between accuracy among all the color spaces and low prediction time. Furthermore, the time taken to train the SVC and to compute the features is quite low as well when compared to the other color spaces. 


```python
#Define feature parameters
color_space = ['RGB', 'HSV', 'YCrCb', 'LUV', 'YUV', 'HLS']

for each_cs in color_space:
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'
    spatial_size = (32, 32)
    hist_bins = 32
    spatial_feat = True
    hist_feat = True
    hog_feat = True

    t=time.time()
    n_samples = 1000
    random_idxs = np.random.randint(0, len(cars), n_samples)
    test_cars = cars #np.array(cars)[random_idxs]
    test_notcars = notcars #np.array(notcars)[random_idxs]

    car_features = extract_features(test_cars, cspace=each_cs, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(test_notcars, cspace=each_cs, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

    print(time.time()-t, 'Seconds: Time taken in seconds to compute features for color space ', each_cs)
    
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0,100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, 
                                                    random_state=rand_state)

    print('Using:',orient,'orientations and',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block and', 
     hist_bins, 'histogram bins, and', spatial_size, 'spatial sampling')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC for color space', each_cs)
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4), 'for color space', each_cs)
    
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC for colorspace', each_cs)
    print('-------------------------------------------------')
```

    220.13664174079895 Seconds: Time taken in seconds to compute features for color space  RGB
    Using: 9 orientations and 8 pixels per cell and 2 cells per block and 32 histogram bins, and (32, 32) spatial sampling
    Feature vector length: 8460
    24.36 Seconds to train SVC for color space RGB
    Test Accuracy of SVC =  0.9868 for color space RGB
    My SVC predicts:  [ 1.  1.  1.  1.  1.  0.  1.  1.  0.  1.]
    For these 10 labels:  [ 1.  1.  1.  1.  1.  0.  1.  1.  0.  1.]
    0.01504 Seconds to predict 10 labels with SVC for colorspace RGB
    -------------------------------------------------
    162.76724433898926 Seconds: Time taken in seconds to compute features for color space  HSV
    Using: 9 orientations and 8 pixels per cell and 2 cells per block and 32 histogram bins, and (32, 32) spatial sampling
    Feature vector length: 8460
    5.83 Seconds to train SVC for color space HSV
    Test Accuracy of SVC =  0.9907 for color space HSV
    My SVC predicts:  [ 0.  0.  1.  0.  0.  0.  1.  1.  0.  1.]
    For these 10 labels:  [ 0.  0.  1.  0.  0.  0.  1.  1.  0.  1.]
    0.01404 Seconds to predict 10 labels with SVC for colorspace HSV
    -------------------------------------------------
    113.3538019657135 Seconds: Time taken in seconds to compute features for color space  YCrCb
    Using: 9 orientations and 8 pixels per cell and 2 cells per block and 32 histogram bins, and (32, 32) spatial sampling
    Feature vector length: 8460
    6.31 Seconds to train SVC for color space YCrCb
    Test Accuracy of SVC =  0.9901 for color space YCrCb
    My SVC predicts:  [ 0.  0.  1.  0.  0.  0.  1.  1.  0.  1.]
    For these 10 labels:  [ 0.  0.  1.  0.  0.  0.  1.  1.  0.  1.]
    0.01404 Seconds to predict 10 labels with SVC for colorspace YCrCb
    -------------------------------------------------
    93.50146985054016 Seconds: Time taken in seconds to compute features for color space  LUV
    Using: 9 orientations and 8 pixels per cell and 2 cells per block and 32 histogram bins, and (32, 32) spatial sampling
    Feature vector length: 8460
    7.86 Seconds to train SVC for color space LUV
    Test Accuracy of SVC =  0.9921 for color space LUV
    My SVC predicts:  [ 0.  1.  1.  1.  0.  1.  0.  0.  0.  1.]
    For these 10 labels:  [ 0.  1.  1.  1.  0.  1.  0.  0.  0.  1.]
    0.01504 Seconds to predict 10 labels with SVC for colorspace LUV
    -------------------------------------------------
    110.48069190979004 Seconds: Time taken in seconds to compute features for color space  YUV
    Using: 9 orientations and 8 pixels per cell and 2 cells per block and 32 histogram bins, and (32, 32) spatial sampling
    Feature vector length: 8460
    6.29 Seconds to train SVC for color space YUV
    Test Accuracy of SVC =  0.9938 for color space YUV
    My SVC predicts:  [ 0.  0.  1.  0.  0.  1.  0.  1.  1.  0.]
    For these 10 labels:  [ 0.  0.  1.  0.  0.  1.  0.  1.  1.  0.]
    0.01404 Seconds to predict 10 labels with SVC for colorspace YUV
    -------------------------------------------------
    96.6451723575592 Seconds: Time taken in seconds to compute features for color space  HLS
    Using: 9 orientations and 8 pixels per cell and 2 cells per block and 32 histogram bins, and (32, 32) spatial sampling
    Feature vector length: 8460
    5.22 Seconds to train SVC for color space HLS
    Test Accuracy of SVC =  0.9913 for color space HLS
    My SVC predicts:  [ 0.  0.  1.  1.  1.  0.  1.  1.  1.  0.]
    For these 10 labels:  [ 0.  0.  1.  1.  1.  1.  1.  1.  1.  0.]
    0.01404 Seconds to predict 10 labels with SVC for colorspace HLS
    -------------------------------------------------
    

### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them)
### Based on the results in the previous cell, I used the chosen HOG features, Color features (YCrCb color space), Spatial and Histogram features to train the classifier in the cell below


```python
color_space = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True

test_cars = cars 
test_notcars = notcars

car_features = extract_features(test_cars, cspace=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(test_notcars, cspace=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0,100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, 
                                                    random_state=rand_state)

print('Using:',orient,'orientations and',pix_per_cell,
'pixels per cell and', cell_per_block,'cells per block and', 
hist_bins, 'histogram bins, and', spatial_size, 'spatial sampling')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
svc.fit(X_train, y_train)
```

    Using: 9 orientations and 8 pixels per cell and 2 cells per block and 32 histogram bins, and (32, 32) spatial sampling
    Feature vector length: 8460
    




    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=1000,
         multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
         verbose=0)



### 4. Sliding Window Search

### a. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

### Test the classifier on sample images by narrowing down the search region to create a region of interest. This was achieved by removing the sky and tree tops to avoid false positives that may occur outside the region of interest. By doing this, we also reduce the number of search windows which in turn makes the search for cars faster.

### After experimenting with different windows sizes and overlap, the combination of 50% overlap and a window size of 96x96 pixels worked the best in identifying all cars in the test images. However, there was also one false positive as seen in the results of the cell below


```python
searchpath = 'test_images/*'
example_images = glob.glob(searchpath)
images = []
titles = []
y_start_stop = [400, 656] # Min & Max in y to search in slide_window()
overlap = 0.5
for img_src in example_images:
    t1 = time.time()
    img = mpimg.imread(img_src)
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    print(np.min(img), np.max(img))
        
    windows = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop,
                               xy_window=(96,96), xy_overlap=(overlap,overlap))
    hot_windows = search_windows(img, windows, svc, X_scaler, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins, orient=orient,
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
        
    window_img = draw_boxes(draw_img, hot_windows, color=(0, 0 ,255), thick=6)
    images.append(window_img)
    titles.append('')
    print(time.time()-t1, ': Time taken in seconds to process one image searching', len(windows), 'windows')
fig = plt.figure(figsize=(12,18), dpi=300)
visualize(fig,3,2,images,titles)

```

    0.0 1.0
    

    C:\Users\raj_u\Miniconda3\envs\carnd-term1\lib\site-packages\skimage\feature\_hog.py:119: skimage_deprecation: Default value of `block_norm`==`L1` is deprecated and will be changed to `L2-Hys` in v0.15
      'be changed to `L2-Hys` in v0.15', skimage_deprecation)
    

    0.47393226623535156 : Time taken in seconds to process one image searching 100 windows
    0.0 1.0
    0.4397542476654053 : Time taken in seconds to process one image searching 100 windows
    0.0 1.0
    0.49836301803588867 : Time taken in seconds to process one image searching 100 windows
    0.0 1.0
    0.4642925262451172 : Time taken in seconds to process one image searching 100 windows
    0.0 1.0
    0.45792365074157715 : Time taken in seconds to process one image searching 100 windows
    0.0 1.0
    0.5208570957183838 : Time taken in seconds to process one image searching 100 windows
    


![png](output_14_3.png)


### 4. Sliding Window Search

### b. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

### The previous cell demonstrates that the basic pipeline works. However, it still shows a false positive and it takes nearly half a second to process each image. To optimize the performance of the classifer, the code in the cell below extracts hog features only once and then sub-samples to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells. The overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75%, which also happen to be the parameters that I chose. A scaling factor of 1.5 was chosen as well. Furthermore, a heatmap was developed where I add "heat" (+=1) for all pixels within windows where a positive detection is reported by the classifier. The individual heat-maps for the example images are shown as the output of next two cells below. Once I use a threshold on the heat-maps, the results in the cells further below demonstrate that the false positives are eliminated.


```python
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)    
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
```


```python
out_images = []
out_maps = []
out_titles = []
out_boxes = []
#Consider a narrower swath in y
ystart = 400
ystop = 656
scale = 1.5
color_space = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (32, 32)
hist_bins = 32

#Iterate over test images
for img_src in example_images:
    img_boxes = []
    t=time.time()
    count=0
    img = mpimg.imread(img_src)
    draw_img = np.copy(img)
    # Make a heat map of zeros
    heatmap=np.zeros_like(img[:,:,0])
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    ch1=ctrans_tosearch[:,:,0]
    ch2=ctrans_tosearch[:,:,1]
    ch3=ctrans_tosearch[:,:,2]

    #define blocks and steops as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient*cell_per_block**2
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2 # Instead of overlap, define how man cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    #Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            count += 1
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
            #Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            
            #Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            
            #Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart),(0,0,255),6)
                img_boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] += 1
                
    print(time.time()-t, 'seconds to run, total windows = ', count)
        
    out_images.append(draw_img)
        
    out_titles.append(img_src[-9:])
    out_titles.append(img_src[-9:])
    out_images.append(heatmap)
    out_maps.append(heatmap)
    out_boxes.append(img_boxes)

fig = plt.figure(figsize=(12,24))
visualize(fig, 6, 2, out_images, out_titles)
```

    0.35599803924560547 seconds to run, total windows =  294
    0.3495948314666748 seconds to run, total windows =  294
    0.3395695686340332 seconds to run, total windows =  294
    0.34881019592285156 seconds to run, total windows =  294
    0.3562772274017334 seconds to run, total windows =  294
    0.3551170825958252 seconds to run, total windows =  294
    


![png](output_17_1.png)


#### The cell below uses the code in the previous cell to create a find_cars() function


```python
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, scale):
    img_boxes = []
    draw_img = np.copy(img)
    # Make a heat map of zeros
    heatmap=np.zeros_like(img[:,:,0])
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    ch1=ctrans_tosearch[:,:,0]
    ch2=ctrans_tosearch[:,:,1]
    ch3=ctrans_tosearch[:,:,2]

    #define blocks and steops as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient*cell_per_block**2
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2 # Instead of overlap, define how man cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    #Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
            #Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            
            #Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            
            #Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                img_boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                        
    return img_boxes
```

### Some useful functions that create a heat map of recurring detections frame by frame to reject outliers using a threshold and follow detected vehicles, and to estimate a bounding box for vehicles detected.


```python
from scipy.ndimage.measurements import label

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
```

### Test the bounding box estimation on the example images along with the heatmap. The results demonstrate that applying a threshold on the heatmap removes the false positives.


```python
ystart = 400
ystop = 656
scale = 1.5

#Iterate over test images
for img_src in example_images:
    img = mpimg.imread(img_src)
    out_img = find_cars(img, scale)

    heat_map = np.zeros_like(img[:,:,0]).astype(np.float)
    #out_img, heat_map = find_cars(img, scale)
    add_heat(heat_map, out_img)
    heat_map = apply_threshold(heat_map, 1)
    heat_map = np.clip(heat_map, 0 , 255)
    labels = label(heat_map)
    #Draw bounding boxes on a copy of the image
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    #out_images.append(draw_img)
    #out_images.append(heat_map)
    
    plt.figure(figsize=(12,18))
    plt.subplot(121)
    plt.imshow(draw_img)
    plt.subplot(122)
    plt.imshow(heat_map, cmap='hot')

```


![png](output_23_0.png)



![png](output_23_1.png)



![png](output_23_2.png)



![png](output_23_3.png)



![png](output_23_4.png)



![png](output_23_5.png)


### Pipeline for the video to detect recurring cars frame by frame and reject outliers using a threshold. Multiple frames were integrated and a threshold was applied to ensure there are no false positives


```python
from collections import deque
b_boxes_deque = deque(maxlen=30)

def add_heat_video(heatmap, b_boxes_deque):
    # Iterate through list of bboxes
    for bbox_list in b_boxes_deque:
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap

def process_image(img):
    
    b_boxes = find_cars(img, scale)
    b_boxes_deque.append(b_boxes)
    
    heat = np.zeros_like(img[:,:,0]).astype(np.float)

    add_heat_video(heat, b_boxes_deque)

    # Apply threshold to remove false positive and ensure only recurring cars in the frame are detected
    heat = apply_threshold(heat,22)

    heatmap = np.clip(heat, 0, 255)

    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    return draw_img
```


```python
# Import stuff needed for video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

proj_output = 'project_video_output.mp4'
clip = VideoFileClip("project_video.mp4")
output_clip = clip.fl_image(process_image)
output_clip.write_videofile(proj_output, audio=False)
```

    [MoviePy] >>>> Building video project_video_output.mp4
    [MoviePy] Writing video project_video_output.mp4
    

    100%|█████████████████████████████████████████████████████████████████████████████▉| 1260/1261 [07:57<00:00,  2.37it/s]
    

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: project_video_output.mp4 
    
    


```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(proj_output))
```





<video width="960" height="540" controls>
  <source src="project_video_output.mp4">
</video>




### Summary:
### Although the SVM classifier method works for this test case and is a great way to understand the concepts on what it takes to detect and track vehicles, it is also quite slow i.e., the time taken to search through the windows in the region of interest within an image takes ~0.35 seconds per image. This may not be applicable in real world situations when there are many different types of vehicles, pedestrians, bicycles etc. A better approach would be to use a neural network or a YOLO (You Only Look Once) classifier approach (https://www.ted.com/talks/joseph_redmon_how_a_computer_learns_to_recognize_objects_instantly) which works in real time. Another issue is that a lot more data is needed to train the SVM to avoid false positives and that would make it even slower. Different window sizes would also need to be combined if cars further away need to be detected, adding to the processing time and computing resources. In the case of freeway driving similar to the one in the project video, the divider separates oncoming traffic. However, we would also need to consider oncoming traffic where there are no dividers on highways. To address this situation, we would need HOG data for oncoming vehicles to train the classifier, which (you guessed it) adds to the processing time as well.


```python

```
