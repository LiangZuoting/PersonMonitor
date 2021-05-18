# PersonMonitor
part of my Smart Home solution. Using Paddle-Lite to detect person.

模型用的是 [MobileNetV3 Small](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0/static/configs/mobile/README.md) ，下载后用 [opt 工具](https://paddle-lite.readthedocs.io/zh/latest/user_guides/model_optimize_tool.html) 转换成 Paddle-Lite 模型。