# Image Classification

This repo creates an image classification model in the form of an h5py file. 

### Current nesting system:
```
.
├── training
│   ├── train
│   └── train.csv
└── testing
    ├── test
    └── test.csv
```

Formatting of training csv file:
filename,label
where filename is the name of the photo and label is an int showing the class it belongs to.

Formatting of testing csv file:
id
where id is the name of them image represented by an integer. This is to get the prediction.
*Optional*: you could add a column to the sample df that writes the proper classification
