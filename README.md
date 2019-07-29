# Image-Compression 
Image compression using K-mean clustering algorithm

I have used K-mean Clustering to compress the given image(bird_small.png).

The given file is a 24 bit RGB image of 128*128 size. This means each pixel has a size of 24 bits. 

And the total size of the image is 128 * 128 * 24 = 393,216 bits. 

Each pixel is a combination of 3 input color channels(R , G and B). The input channels have a length of 
8 bits each.
Hence, total possible colors = 2^24=16777216.

And we want to reduce this to only 16 colors i.e each pixel having a size of 4 bits.

I clustered the entire data using K-means algo to form 16 clusters and ran it several times to get unique clusters having
centroids.

Then for the compressing part I assigned each pixel location of the original data sample to the corresponding Cluster-center
assigned to it.

You can see the compressed image in Compressed.png.

And observe the 16 clusters in the 3D Scatter plot provided.

[The red stars in the png file are the final positions of the cluster centers].
