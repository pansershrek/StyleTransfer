# StyleTransfer
Implementation of this article [Image Style Transfer Using CNN](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) with mmodification from this [course](https://www.udacity.com/course/deep-learning-pytorch--ud188).

# How to start?
If you have GPU you should remove Dockerfile and rename Dockerfile_gpu to Dockerfile.
To run services you only have to run this command:
```
# Build docker image
docker build -t style_transfer .

# Start docker image
docker run -it --rm -p 80:80 style_transfer
```
