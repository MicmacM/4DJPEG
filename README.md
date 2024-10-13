# 4DJPEG
Using JPEG like technique to compress a video efficiently
## About
Upon seeing this amazing video : [the jpeg algorithm](https://youtu.be/0me3guauqOU?si=WSZwbFdwA-Anfzx7), i wondered "if jpeg applies 3 2D-DCT along the two spatial axis and the 3 colors, can't we apply 3 3D-DCT along the two spatial axis and the temporal axis, by viewing a pixel as a time series ?". And so this is what I'm trying to do, apply it to a video and :

- Check if the decompressed video is still close to the original
- Compare the compression to a standard compression using only a series of jpeg image, and then to the MPEG standart

## How it works
### The JPEG algorithm

The JPEG algorithm consist of 3 main parts

1) It applies a Discrete Cosine Transform (DCT) along the x axis, and then along the y axis of an image, treating the pixels intensity as a signal. A property of the DCT is that it tends to group the higher valued coefficient on the top left corner of the matrix, this will be usefull in part 2, and this is why we use DCT and not FFT. The coefficient are then divided by a quantization factor and floored, this way, the higher frequency coefficient (which the human eye struggle to identify) will have lower value according to what was said earlier, and thus will get rounded to 0.

2) Then by using a zigzag order we iterate through the coefficient and compress them in a list using delta and run-length encoding

3) A Huffman tree is then constructed in order to compress the coefficient even further.

### The 3D DCT version
I choose not to implement the Huffman tree part as it would give the same result regarding the two methods discussed in the 'About' section. I have thus decided to create a $(H,W,T,3)$ array, where each slice $(H,W,t,3), t\in \{0...T\}$ is an image of the video. Then spatio-temporal chunks of size $(8,8,8,3)$ are created and 3 DCT are applied 

#### The Quantization matrix
For the time being I didn't really know how to define my quantization matrix, as they are often the result of several experimentation. Thus I tried a formula using a 2D quantization matrix, with value increasing as one gets further from the DC coefficient, located in $(0,0,0)$ (the coefficient are supposed to be lower the further you go, and we want to nullify the low valued coefficient).

#### The 3D zigzag
In 3D it's rather easy to do a zigzag search because there is only two direction possible : top left or down right (and up and down to switch from one diagonal to another but that's stille few), in 3D there are way more, so instead I'm not really doing a zigzag search but a manhattan distance ($d_{manhattan}(x,y,z) = x+y+z$) one ! Indeed, zigzag isn't necessary, since we only want to group together near valued coefficient in the zigzag search, and such coefficient are the ones that share a common manhattan distance from the origin. This way it's much easier to code

## Results

The decompressed video is nearly identical to the input video, except when there are sudden change in the pixel value (this is linked to the high frequency of the temporal DCT, thus they were removed during the quantization process), where each chunk becomes clearly visible.

The compression ratio achieved is between 97,5 and 99%, this is really nice I think !

## To Do
- Experiment against the version that only uses a series of jpeg image
- Try selecting the length of the temporal chunks according to the temporal variation
- Compare it against MPEG
- Use a better quantization matrix


## Aknowledgment

I used [Understanding and Decoding a JPEG Image using Python](https://yasoob.me/posts/understanding-and-writing-jpeg-decoder-in-python/) article to understand the jpeg algorithm in more details than in the video linked above, as well as the wikipedia pages