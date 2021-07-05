# Task 1 
discriminator.ipynb is the code for Task 1. The trained discriminator is stored in file discriminator_model, and matched_images_800.pkl stores 800 matched images (not complete due to computational limitation). The figure ExampleOutput.png is some plotting of the output images.
## Overall Idea : 
Train a binary discriminator s.t. the prediction for true image is close to 1 and prediction for fake image is close to 0. Then for all of the possible mathcing of half-pieces, if the score is close to 1 (> 0.98 in my experiment setting), we claim that image is a 'good' match and we put that matched image into our output.
### Step 1 - Training Data Preparation : 
Design a 'fake' training set by cutting X_train in half vertically, and paste them in wrong ways (for the same labeled ones, inverse left & right; paste the different labeled ones). Combine original X_train with our fake image set. For original images, label them as 1, for fake images, label them as 0.

### Step 2 - Build a Discriminator
Input : Image of size 28x28x1 . Output : binary (use a sigmoid activation function). Use three layers of neural network, for each layer, use a 2D convolutional neural network, then use LeakyReLu as activation function (advantage of LeakyReLu is that it doesn’t have zero-slope parts, thus preventing the dying Relu problem. It also speeds up training.) Then flatten to 1D, and dropout some features to prevent overfitting, then dense the layer to dim 1 (the output dim).

Train the discriminator by the extended training dataset, and the validation accuracy (test data by splitting from the training set) is above 90%, the training accuracy is above 98%.

### Step 3 - Test Data : 
Cut the test images vertically in half and shuffle them. I thought of two ways of '1-1 match' the half-pieces. One is to separate the 'half image set' into two equal sized sets, and predict the accuracy of discriminator on each of those sets, terminate when the accuracy is > 90. But this is computationally expensive since (1) the number of all possible 'two equal sized subsets' is large (all possible matches could be computed using dynamic programming), when running on google colab, it will reach the RAM limit. (2) the time for predicting a large dataset (in this case, of size 10000) is long.

Hence I match the half-images 1-1 in this way : for the first piece, iterate over all of the other pieces, and halt when there is a match s.t. prediction > 0.98, put the matched image into our output, and delete those two half-pieces from the half-images-set. 

Note that the computation could be accelerated if running parallelly on GPU.

## Further Possible Improvements
### Produce more fake images by cutting and pasting
### Train image generators to generate true & fake images (could use Gaussian Mixture Models, LDA). i.e., train GAN model architecture : a generator model for generating new examples and a discriminator model for judging whether generated examples are real.

# Task 2
SemiSupervised.ipynb is the code for Task 2. 
## Overall Idea : 
Hinted by Task 1, since we've already got many 'fake' images, we can train a semi-supervised model, where the supervised discriminator and the unsupervised discriminator share layers. The supervised discriminator is trained to 'classify' and the unsupervised discriminator is trained to 'judge true image'.
### Step 1 - Training Data Preparation : 
#### Part 1 - Labeled Data 
For each label, randomly choose 20 images of that label
#### Part 2 - Unlabeled Data
Same as Task 1 Step 1

### Step 2 - Define Discriminators 
The supervised discriminator and unsupervised discriminator share layers, they both take images (size 28x28x1) as input, use 3 layers of 2D convolutional neural network + LeakyReLu activation function, then flatten and dropout features, the supervised discriminator output layer is 1x10 since it aims to classify images, and the unsupervised discriminator output layer is 1x1 since it aims to judge true&fake images. 

### Step 3 - Train Models
In each epoch, train the supervised discriminator using batches of labeled data, and train unsupervised discriminator using batches of unlabeled data (from the expanded dataset). 

## Further Possible Improvements
### Train image generators to generate true & fake images 
