## Hi 👋

Come is come, don’t you star this project? & Forgive my pool English.

Welcome to star this repo!

Fedrated Learning-Smart Medical Introduction Video [<a href="https://www.bilibili.com/video/BV1Da4y1W7K3">Intro</a>]

Fedrated Learning-Smart Medical Demo Video [<a href="https://www.bilibili.com/video/BV">Demo</a>]

README [<a href="README.md">EN</a>|<a href="README_CN.md">CN</a>]

Github Page: See <a href="https://beiyuouo.github.io/paddle-fl-gui/">Here</a>

## Description

This is a simple GUI interface developed for operation by medical staff. Server parameters and client parameters can be configured in `config/config_server.yaml` and `config/config_client.yaml`

- Server main interface

![](imgs/serverFrame.png)

- Server main control interface

![](imgs/serverControlFrame.png)

- Client interface

![](imgs/clientFrame.png)

- Test interface

![](imgs/testFrame.png)

- Report generation page

![](imgs/report.png)
<a href="https://beiyuouo.github.io/paddle-fl-gui/report/report">Online Demo</a>


## Getting start

### Installation

- Install <a href="https://paddlefl.bj.bcebos.com/redis-stable.tar">Redis</a> & <a href="https://github.com/PaddlePaddle/PaddleFL"> PaddleFL</a>
- Install requirements `pip install -r requirements.txt`


### How to run

```sh
python serverFrame.py
python clientFrame.py 0
python clientFrame.py 1
python clientFrame.py 2
```


## Changelog

### v1.0.1_210209_beta
- Add qss.

### v1.0.0_201130_beta
- The demo open source.
