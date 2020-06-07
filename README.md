[![facialAnalysisDetection2D Homepage](https://img.shields.io/badge/facialAnalysisDetection2D-develop-orange.svg)](https://github.com/davidvelascogarcia/facialAnalysisDetection2D/tree/develop/programs) [![Latest Release](https://img.shields.io/github/tag/davidvelascogarcia/facialAnalysisDetection2D.svg?label=Latest%20Release)](https://github.com/davidvelascogarcia/facialAnalysisDetection2D/tags) [![Build Status](https://travis-ci.org/davidvelascogarcia/facialAnalysisDetection2D.svg?branch=develop)](https://travis-ci.org/davidvelascogarcia/facialAnalysisDetection2D)

# Facial Analysis: Detector 2D (Python API)

- [Introduction](#introduction)
- [Trained Models](#trained-models)
- [Requirements](#requirements)
- [Status](#status)
- [Related projects](#related-projects)


## Introduction

`facialAnalysisDetection2D` module use `deepFace` `python` API. The module analyze faces using pre-trained models and adds facial analysis doing prediction about some features like gender, age, race and emotions. Also use `YARP` to send video source pre and post-procesed. Also admits `YARP` source video like input. This module also publish detection results in `YARP` port.


## Trained Models

`facialAnalysisDetection2D` requires images source to detect. First run program will download pre-trained models about features to detect:

1. Execute [programs/facialAnalysisDetection2D.py](./programs), to start the program.
```python
python3 facialAnalysisDetection2D.py
```
3. Connect video source to `facialAnalysisDetection2D`.
```bash
yarp connect /videoSource /facialAnalysisDetection2D/img:i
```

NOTE:

- Video results are published on `/facialAnalysisDetection2D/img:o`
- Data results are published on `/facialAnalysisDetection2D/data:o`

## Requirements

`facialRecognitionDetection2D` requires:

* [Install OpenCV 3.0.0+](https://github.com/roboticslab-uc3m/installation-guides/blob/master/install-opencv.md)
* [Install YARP 2.3.XX+](https://github.com/roboticslab-uc3m/installation-guides/blob/master/install-yarp.md)
* [Install pip](https://github.com/roboticslab-uc3m/installation-guides/blob/master/install-pip.md)
* Install deepFace:
```bash
pip3 install deepface
```

**NOTE:**
This module use `Tensorflow` backend.

**Possible errors:**
`deepFace` requires `Python 3.5.5` but `Ubuntu 16.04` use `Python 3.5.2` by default. You can add `Python 3.6` with PPA repository like:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6
```
To install `pip` in `Python 3.6`:

```bash
curl https://bootstrap.pypa.io/get-pip.py | sudo python3.6
```

You can install the package with:

```bash
python3.6 -m pip install deepface
```
If `pip` change the link to `Python 3.6` by default, you can revert to link `pip` to `Python 3.5` with:

```bash
python3 -m pip install --upgrade --force pip
```

Tested on: `ubuntu 14.04`, `ubuntu 16.04`, `ubuntu 18.04`, `lubuntu 18.04` and `raspbian`.


## Status

[![Build Status](https://travis-ci.org/davidvelascogarcia/facialAnalysisDetection2D.svg?branch=develop)](https://travis-ci.org/davidvelascogarcia/facialAnalysisDetection2D)

[![Issues](https://img.shields.io/github/issues/davidvelascogarcia/facialAnalysisDetection2D.svg?label=Issues)](https://github.com/davidvelascogarcia/facialAnalysisDetection2D/issues)

## Related projects

* [serengil: deepFace project](https://github.com/serengil/deepface)

