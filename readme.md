## Master Thesis: Advanced data science solutions - data analyitics program

I have created a data analyitics program using tensorflow/keras as part of my graduate level thesis work.
Some of the functionalities and usage info are described below.

* **Functionalities**
  - data processing for time-series and regular data (sequence-length for input tensor)
  - data normalization and standardization capability
  - dynamic keras Sequential model building capability from config.json file
    - Layer types for: LSTM & Dense
    - configuring initial weight method
    - activation func
    - neuron number
    - layer number
  - model serialization and output file creation + training data attributes (normalization high & low values)
  - matplotlib vizualization for performance and epoch-wise training
  - functional testing with example data (square, root, sinus func.)

* **Instructions for PROD-stage**
  - Create a REST-API which calls the model.predict() method
  - don't forget to normalize input data before you call the predict() method (or if you used one-hot encode get the input data to same structure as the training data was) 

* **Short Comings**
  - many layer type, optimization possibility is left out
    - like dropout between layers
    - like batchnormalization layers
    - other optimization types, algorithm wise methods etc.

* **Getting Started**
  - git clone ** this repo **
  - activate python venv
  - pip install -r requirements.txt
  - python -m test
