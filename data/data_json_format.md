# dataset annotation format
```
{
    "info":"for vnect training",

    "annotations":[{"img_id":0,
                    "2D-keypoints":[x1,y1,x2,y2,...],
                    "3D-keypoints":[x1,y1,z1,x2,y2,z2,...]
                },{},...],

    "images":{0:"image_name1,
            1:image_name1,....}

}
```
able to add items in annotation for per person